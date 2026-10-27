/*!
 * \file tl/hexagon/transform/broadcast_from_registers.cc
 * \brief Broadcast per-row scalars from registers loaded once per strip.
 *
 * A vectorized map over the rows of a tile often broadcasts one scalar per
 * row: softmax subtracts rmax[i] from row i, a norm scales row i by inv[i].
 * Loading that scalar inside the row loop puts a scalar VTCM load at the head
 * of every iteration, and LLVM cannot tell it from the row's stores, so it does
 * not software-pipeline the loop. On the v79 device, the exp map of flash
 * attention spent half its time there.
 *
 * This pass splits such a loop into strips of one register's worth of rows.
 * Each strip loads the scalars of its rows into a register once
 * (tl_hvx_load_lanes); each row broadcasts its lane from that register
 * (tl_hvx_broadcast_lane, a rotate and a delta-network copy). The row loop
 * then reads only the tile itself. A T.Parallel(M, N) loop wider than a
 * register is one loop over the vectors of all rows, whose scalar sits at
 * i // K for K vectors per row; it splits into rows of K vectors first.
 *
 * Runs after VectorizeLoop, on flattened buffers.
 */

#include "hexagon/op/builtin.h"

#include <tvm/arith/analyzer.h>
#include <tvm/arith/pattern.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <algorithm>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

namespace {

constexpr int kRegisterBytes = 128;

// A buffer that a loop reads one scalar of per iteration, at loop_var + base,
// and broadcasts into vectors.
struct Source {
  Buffer buffer;
  PrimExpr base;
};

// The buffers a statement may write: stores, and the buffers of any pointer a
// call takes (builtins write through them).
std::unordered_set<const VarNode *> WrittenBuffers(const Stmt &body) {
  std::unordered_set<const VarNode *> written;
  PostOrderVisit(body, [&](const ObjectRef &node) {
    if (const auto *store = node.as<BufferStoreNode>()) {
      written.insert(store->buffer->data.get());
    } else if (const auto *call = node.as<CallNode>()) {
      if (call->op.same_as(builtin::address_of())) {
        if (const auto *load = call->args[0].as<BufferLoadNode>())
          written.insert(load->buffer->data.get());
      } else if (call->op.same_as(builtin::tvm_access_ptr())) {
        if (const auto *data = call->args[1].as<VarNode>())
          written.insert(data);
      }
    }
  });
  return written;
}

// The variables a statement defines: its loop and bound variables.
std::unordered_set<const VarNode *> DefinedVars(const Stmt &body) {
  std::unordered_set<const VarNode *> defined;
  PostOrderVisit(body, [&](const ObjectRef &node) {
    if (const auto *loop = node.as<ForNode>())
      defined.insert(loop->loop_var.get());
    else if (const auto *bind = node.as<BindNode>())
      defined.insert(bind->var.get());
    else if (const auto *let = node.as<LetNode>())
      defined.insert(let->var.get());
  });
  return defined;
}

// The broadcast scalars of `body` that step through a buffer with `row`, at
// row + base, from buffers the body does not write.
std::vector<Source> CollectSources(const Stmt &body, const Var &row) {
  std::unordered_set<const VarNode *> written = WrittenBuffers(body);
  std::unordered_set<const VarNode *> defined = DefinedVars(body);
  std::vector<Source> sources;
  PostOrderVisit(body, [&](const ObjectRef &node) {
    const auto *broadcast = node.as<BroadcastNode>();
    if (!broadcast)
      return;
    const auto *load = broadcast->value.as<BufferLoadNode>();
    if (!load || load->indices.size() != 1 || load->dtype.lanes() != 1 ||
        written.count(load->buffer->data.get()))
      return;
    Array<PrimExpr> coef = arith::DetectLinearEquation(load->indices[0], {row});
    if (coef.size() != 2 || !is_one(coef[0]))
      return;
    PrimExpr base = coef[1];
    bool invariant =
        !UsesVar(base, [&](const VarNode *v) { return defined.count(v) > 0; });
    if (!invariant)
      return;
    for (const Source &s : sources) {
      if (s.buffer.same_as(load->buffer) && StructuralEqual()(s.base, base))
        return;
    }
    sources.push_back({load->buffer, base});
  });
  return sources;
}

// A loop over the vectors of whole rows, T.Parallel(M, N) fused, steps its
// row every K iterations: its broadcast scalars sit at base + i // K. The
// smallest such K of a broadcast scalar of `loop`, or 1.
int64_t RowPeriod(const For &loop) {
  int64_t period = 1;
  PostOrderVisit(loop->body, [&](const ObjectRef &node) {
    const auto *broadcast = node.as<BroadcastNode>();
    const auto *load =
        broadcast ? broadcast->value.as<BufferLoadNode>() : nullptr;
    if (!load || load->indices.size() != 1)
      return;
    PostOrderVisit(load->indices[0], [&](const ObjectRef &e) {
      const auto *div = e.as<FloorDivNode>();
      const auto *k = div ? div->b.as<IntImmNode>() : nullptr;
      if (k && k->value > 1 && div->a.same_as(loop->loop_var) &&
          (period == 1 || k->value < period))
        period = k->value;
    });
  });
  return period;
}

// Simplifies the buffer indices of a statement under the analyzer's bounds.
class IndexSimplifier : public StmtExprMutator {
public:
  explicit IndexSimplifier(arith::Analyzer *analyzer) : analyzer_(analyzer) {}

private:
  Array<PrimExpr> Simplify(const Array<PrimExpr> &indices) {
    return indices.Map(
        [this](const PrimExpr &i) { return analyzer_->Simplify(i); });
  }
  PrimExpr VisitExpr_(const BufferLoadNode *op) final {
    BufferLoad load = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(op));
    load.CopyOnWrite()->indices = Simplify(load->indices);
    return load;
  }
  Stmt VisitStmt_(const BufferStoreNode *op) final {
    BufferStore store = Downcast<BufferStore>(StmtExprMutator::VisitStmt_(op));
    store.CopyOnWrite()->indices = Simplify(store->indices);
    return store;
  }

  arith::Analyzer *analyzer_;
};

// Replaces the broadcasts of the sources with lane broadcasts from their
// registers.
class BroadcastRewriter : public StmtExprMutator {
public:
  BroadcastRewriter(const Var &loop_var, const std::vector<Source> &sources,
                    const std::vector<Var> &registers, const Var &lane)
      : loop_var_(loop_var), sources_(sources), registers_(registers),
        lane_(lane) {}

private:
  PrimExpr VisitExpr_(const BroadcastNode *op) final {
    if (const auto *load = op->value.as<BufferLoadNode>()) {
      for (size_t k = 0; k < sources_.size(); ++k) {
        if (!load->buffer.same_as(sources_[k].buffer))
          continue;
        Array<PrimExpr> coef =
            arith::DetectLinearEquation(load->indices[0], {loop_var_});
        if (coef.size() == 2 && is_one(coef[0]) &&
            StructuralEqual()(coef[1], sources_[k].base)) {
          int lanes = op->dtype.lanes();
          return Call(op->dtype, hexagon_broadcast_lane(),
                      {StringImm(std::to_string(lanes)), registers_[k], lane_});
        }
      }
    }
    return StmtExprMutator::VisitExpr_(op);
  }

  const Var &loop_var_;
  const std::vector<Source> &sources_;
  const std::vector<Var> &registers_;
  const Var &lane_;
};

class BroadcastStripper : public StmtExprMutator {
private:
  Stmt VisitStmt_(const ForNode *op) final {
    For loop = Downcast<For>(StmtExprMutator::VisitStmt_(op));
    if (loop->kind != ForKind::kSerial && loop->kind != ForKind::kUnrolled)
      return loop;
    if (is_one(loop->extent) || is_zero(loop->extent))
      return loop;
    arith::Analyzer analyzer;
    DataType index_t = loop->loop_var.dtype();

    // The rows: the loop itself, or, when it walks K vectors per row, an
    // outer loop over rows around one over each row's K vectors.
    Var row = loop->loop_var;
    PrimExpr row_min = loop->min, rows = loop->extent;
    Stmt row_body = loop->body;
    int64_t period = RowPeriod(loop);
    if (period > 1) {
      PrimExpr k = make_const(index_t, period);
      if (!is_zero(loop->min) ||
          !analyzer.CanProveEqual(floormod(loop->extent, k), 0))
        return loop;
      row = Var(loop->loop_var->name_hint + "_row", index_t);
      Var col(loop->loop_var->name_hint + "_col", index_t);
      rows = analyzer.Simplify(floordiv(loop->extent, k));
      analyzer.Bind(row, Range::FromMinExtent(make_zero(index_t), rows));
      analyzer.Bind(col, Range::FromMinExtent(make_zero(index_t), k));
      Stmt body = Substitute(loop->body, {{loop->loop_var, row * k + col}});
      body = IndexSimplifier(&analyzer)(body);
      row_min = make_zero(index_t);
      row_body = For(col, make_zero(index_t), k, ForKind::kSerial, body,
                     std::nullopt, loop->annotations);
    }
    std::vector<Source> sources = CollectSources(row_body, row);
    if (sources.empty())
      return loop;

    // A strip is one register of the widest source type.
    int lanes = kRegisterBytes;
    for (const Source &s : sources)
      lanes = std::min(lanes, kRegisterBytes / s.buffer->dtype.bytes());
    PrimExpr width = make_const(index_t, lanes);
    Var strip(loop->loop_var->name_hint + "_strip", index_t);
    Var lane(loop->loop_var->name_hint + "_lane", index_t);
    PrimExpr first = strip * width;
    // The rows of this strip: a whole register, or what remains.
    bool whole = analyzer.CanProveEqual(floormod(rows, width), 0);
    PrimExpr count = whole ? width : Min(width, rows - first);

    std::vector<Var> registers;
    for (const Source &s : sources) {
      registers.emplace_back(s.buffer->name + "_lanes",
                             s.buffer->dtype.with_lanes(lanes));
    }
    Stmt body = BroadcastRewriter(row, sources, registers, lane)(row_body);
    body = Substitute(body, {{row, row_min + first + lane}});
    Array<Stmt> strip_body;
    for (size_t k = 0; k < sources.size(); ++k) {
      const Source &s = sources[k];
      PrimExpr at = s.base + row_min + first;
      strip_body.push_back(Bind(
          registers[k], Call(registers[k].dtype(), hexagon_load_lanes(),
                             {StringImm(std::to_string(lanes)),
                              Call(DataType::Handle(), builtin::address_of(),
                                   {BufferLoad(s.buffer, {at})}),
                              count})));
    }
    // Serial, as a whole strip unrolled would leave nothing to pipeline.
    strip_body.push_back(For(lane, make_zero(index_t), count, ForKind::kSerial,
                             body, std::nullopt, loop->annotations));
    PrimExpr strips = floordiv(rows + width - 1, width);
    return For(strip, make_zero(index_t), analyzer.Simplify(strips),
               ForKind::kSerial, SeqStmt(strip_body));
  }
};

} // namespace

namespace transform {

tvm::transform::Pass BroadcastFromRegisters() {
  auto pass_func = [](PrimFunc func, const IRModule &,
                      const tvm::transform::PassContext &) -> PrimFunc {
    func.CopyOnWrite()->body = BroadcastStripper()(func->body);
    return func;
  };
  return tirx::transform::CreatePrimFuncPass(
      pass_func, 0, "tl.hexagon.BroadcastFromRegisters", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.hexagon.transform.BroadcastFromRegisters",
                        BroadcastFromRegisters);
}

} // namespace transform
} // namespace tl
} // namespace tvm
