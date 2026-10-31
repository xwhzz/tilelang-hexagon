/*!
 * \file tl/hexagon/transform/insert_write_through_writebacks.cc
 * \brief Each write of a write-through buffer is written back right away.
 *
 * The DSP's stores to DDR stay in its caches, and DMA, which reads DDR, would
 * not see them; so a DMA load writes its source's lines back first, about a
 * cycle every 16 bytes it moves (dma.h). A buffer that T.annotate_write_through
 * declares is written back where it is written instead: after each T.copy into
 * it, each T.Parallel loop that stores into it, and each other store, this pass
 * puts a writeback of the region written (tl.tileop.cache_writeback). Its DMA
 * loads then skip their own (MarkDMACacheDirections). A buffer that is written
 * once and read many times, such as a KV cache, saves most of that cost.
 *
 * Runs on tile programs, before LayoutInference: the writeback's lowering maps
 * the region through the buffer's layout.
 */

#include "hexagon/op/dma.h"
#include "hexagon/op/utils.h"
#include "op/copy.h"
#include "op/operator.h"

#include <tvm/arith/int_set.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

namespace {

// The block annotation of T.annotate_write_through: the buffers' data vars.
constexpr const char *kWriteThrough = "tl.hexagon_write_through";

class WriteThroughCollector : public StmtVisitor {
public:
  std::unordered_set<const VarNode *> vars;

private:
  void VisitStmt_(const SBlockNode *op) final {
    if (auto declared = op->annotations.Get(kWriteThrough))
      for (const Var &var : Downcast<Array<Var>>(declared.value()))
        vars.insert(var.get());
    StmtVisitor::VisitStmt_(op);
  }
};

class WritebackInserter : public StmtExprMutator {
public:
  explicit WritebackInserter(std::unordered_set<const VarNode *> vars)
      : vars_(std::move(vars)) {}

private:
  static Stmt Writeback(const Buffer &buffer, const Array<Range> &region) {
    return Evaluate(
        Call(DataType::Handle(), HexagonCacheWriteback::Get(),
             {hexagon::RegionArg(buffer, region, kAccessReadWrite)}));
  }

  bool WriteThrough(const Buffer &buffer) const {
    return vars_.count(buffer->data.get()) != 0;
  }

  Stmt VisitStmt_(const EvaluateNode *op) final {
    const auto *call = op->value.as<CallNode>();
    TileOperator tile =
        call != nullptr ? ParseOperator(GetRef<Call>(call)) : TileOperator();
    // DMA bypasses the caches; a writeback writes nothing.
    if (!tile.defined() || tile.as<HexagonDmaCopyNode>() ||
        tile.as<HexagonCacheWritebackNode>())
      return StmtExprMutator::VisitStmt_(op);
    Array<Stmt> seq{GetRef<Stmt>(op)};
    for (const BufferRegion &write : tile->GetAccessRegions().writes) {
      if (!WriteThrough(write->buffer))
        continue;
      ICHECK(tile.as<CopyNode>())
          << "T.annotate_write_through: `" << write->buffer->name
          << "` is written by " << call->op
          << ", whose writes the compiler does not write back; write it with "
             "T.copy or a T.Parallel loop";
      seq.push_back(Writeback(write->buffer, write->region));
    }
    return seq.size() == 1 ? seq[0] : SeqStmt(seq);
  }

  Stmt VisitStmt_(const ForNode *op) final {
    if (op->kind != ForKind::kParallel)
      return StmtExprMutator::VisitStmt_(op);
    // A T.Parallel nest: the region of each write-through buffer that its
    // stores cover, over the nest's loop variables.
    Map<Var, arith::IntSet> domain;
    std::vector<Buffer> order;
    std::unordered_map<const BufferNode *, std::vector<Array<PrimExpr>>> stores;
    PostOrderVisit(GetRef<Stmt>(op), [&](const ObjectRef &node) {
      if (const auto *loop = node.as<ForNode>()) {
        domain.Set(loop->loop_var,
                   arith::IntSet::FromMinExtent(loop->min, loop->extent));
      } else if (const auto *store = node.as<BufferStoreNode>()) {
        if (!WriteThrough(store->buffer))
          return;
        if (!stores.count(store->buffer.get()))
          order.push_back(store->buffer);
        stores[store->buffer.get()].push_back(store->indices);
      }
    });
    if (order.empty())
      return GetRef<Stmt>(op);
    Array<Stmt> seq{GetRef<Stmt>(op)};
    for (const Buffer &buffer : order) {
      Array<Range> region;
      for (size_t i = 0; i < buffer->shape.size(); ++i) {
        Array<arith::IntSet> sets;
        for (const Array<PrimExpr> &indices : stores[buffer.get()])
          sets.push_back(arith::EvalSet(indices[i], domain));
        arith::IntSet set = arith::Union(sets);
        ICHECK(set.HasLowerBound() && set.HasUpperBound())
            << "T.annotate_write_through: the stores into `" << buffer->name
            << "` of a T.Parallel loop need bounded indices";
        region.push_back(Range::FromMinExtent(
            set.min(), analyzer_.Simplify(set.max() - set.min() + 1)));
      }
      seq.push_back(Writeback(buffer, region));
    }
    return SeqStmt(seq);
  }

  Stmt VisitStmt_(const BufferStoreNode *op) final {
    if (!WriteThrough(op->buffer))
      return StmtExprMutator::VisitStmt_(op);
    // A store of its own: its element.
    Array<Range> region;
    for (const PrimExpr &index : op->indices)
      region.push_back(Range::FromMinExtent(index, 1));
    return SeqStmt({GetRef<Stmt>(op), Writeback(op->buffer, region)});
  }

  std::unordered_set<const VarNode *> vars_;
  arith::Analyzer analyzer_;
};

} // namespace

tvm::transform::Pass InsertWriteThroughWritebacks() {
  using namespace tirx::transform;
  auto pass_func = [](PrimFunc f, const IRModule &, const PassContext &) {
    WriteThroughCollector collector;
    collector(f->body);
    if (collector.vars.empty())
      return f;
    std::unordered_set<const VarNode *> params;
    for (const auto &kv : f->buffer_map)
      if (kv.second.scope() == "global")
        params.insert(kv.second->data.get());
    for (const VarNode *var : collector.vars)
      ICHECK(params.count(var))
          << "T.annotate_write_through takes the kernel's DDR buffer "
             "parameters; `"
          << var->name_hint << "` is not one";
    f.CopyOnWrite()->body =
        WritebackInserter(std::move(collector.vars))(f->body);
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0,
                            "tl.hexagon.InsertWriteThroughWritebacks", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.hexagon.transform.InsertWriteThroughWritebacks",
                        InsertWriteThroughWritebacks);
}

} // namespace tl
} // namespace tvm
