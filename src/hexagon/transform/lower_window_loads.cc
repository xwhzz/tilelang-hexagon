/*!
 * \file tl/hexagon/transform/lower_window_loads.cc
 * \brief Lower vector loads of small windows, or of two interleaved runs, to
 * loads and a lane shuffle.
 *
 * A map over a Crouton tile runs in the tile's physical order, where register
 * lane 2c + e holds row 2q + e and column c of a row pair. A vector of the
 * rows' values then reads [r0, r1, r0, r1, ...], a vector of the columns'
 * values [c0, c0, c1, c1, ...], and a vector of a row-major matrix's elements
 * at the same rows and columns two runs, one of each row, interleaved. The
 * vectorizer accepts such loads on Hexagon (DetectVectorWindow) and leaves
 * them as per-lane gathers. This pass loads the window, or each run, and
 * shuffles the lanes: tl_hvx_load_replicate copies the window over the
 * register with the delta network, tl_hvx_load_repeat repeats each element by
 * shuffling the register with itself, and tl_hvx_load_interleave shuffles the
 * two runs together.
 *
 * Runs after VectorizeLoop, on flattened buffers.
 */

#include "hexagon/op/builtin.h"

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <optional>
#include <string>
#include <vector>

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

namespace {

constexpr int kRegisterBytes = 128;

// One lane of a vector expression, as a scalar expression. `ok` turns false
// on a vector node it cannot take apart, such as a vector load or a shuffle.
class LaneValue : public ExprMutator {
public:
  explicit LaneValue(int lane) : lane_(lane) {}
  bool ok = true;

private:
  PrimExpr VisitExpr_(const RampNode *op) final {
    PrimExpr base = VisitExpr(op->base);
    return base + VisitExpr(op->stride) * make_const(base.dtype(), lane_);
  }
  PrimExpr VisitExpr_(const BroadcastNode *op) final {
    return VisitExpr(op->value);
  }
  PrimExpr VisitExpr_(const CastNode *op) final {
    return cast(op->dtype.element_of(), VisitExpr(op->value));
  }
  PrimExpr VisitExpr_(const BufferLoadNode *op) final {
    ok = ok && op->dtype.lanes() == 1;
    return GetRef<PrimExpr>(op);
  }
  PrimExpr VisitExpr_(const CallNode *op) final {
    ok = ok && op->dtype.lanes() == 1;
    return GetRef<PrimExpr>(op);
  }
  PrimExpr VisitExpr_(const ShuffleNode *op) final {
    ok = false;
    return GetRef<PrimExpr>(op);
  }

  int lane_;
};

struct Window {
  const Op *op;
  int period;
  PrimExpr first;
  int64_t stride = 0; // interleave: the elements between the runs
};

// The window a vector index reads: every lane's offset from lane 0 is a
// constant, l % period (replicate), l / period (repeat), or
// (l % 2) * stride + l / 2 (interleave).
std::optional<Window> WindowOf(const PrimExpr &index,
                               arith::Analyzer *analyzer) {
  int lanes = index.dtype().lanes();
  std::vector<int64_t> offsets(lanes);
  PrimExpr first;
  for (int l = 0; l < lanes; ++l) {
    LaneValue lane(l);
    PrimExpr value = analyzer->Simplify(lane(index));
    if (!lane.ok)
      return std::nullopt;
    if (l == 0) {
      first = value;
      continue;
    }
    PrimExpr offset = analyzer->Simplify(value - first);
    const auto *imm = offset.as<IntImmNode>();
    if (imm == nullptr)
      return std::nullopt;
    offsets[l] = imm->value;
  }
  for (int period = 2; period < lanes; period *= 2) {
    bool replicate = true, repeat = true;
    for (int l = 0; l < lanes; ++l) {
      replicate = replicate && offsets[l] == l % period;
      repeat = repeat && offsets[l] == l / period;
    }
    if (replicate)
      return Window{&hexagon_load_replicate(), period, first};
    if (repeat)
      return Window{&hexagon_load_repeat(), period, first};
  }
  bool interleave = lanes > 2 && offsets[1] > 0;
  for (int l = 0; interleave && l < lanes; ++l)
    interleave = offsets[l] == l % 2 * offsets[1] + l / 2;
  if (interleave)
    return Window{&hexagon_load_interleave(), 2, first, offsets[1]};
  return std::nullopt;
}

class WindowLoadLowerer : public StmtExprMutator {
private:
  PrimExpr VisitExpr_(const BufferLoadNode *op) final {
    PrimExpr expr = StmtExprMutator::VisitExpr_(op);
    const auto *load = expr.as<BufferLoadNode>();
    int lanes = load->dtype.lanes();
    if (lanes == 1 || load->indices.size() != 1 ||
        load->indices[0].as<RampNode>() ||
        load->dtype.bytes() > 2 * kRegisterBytes)
      return expr;
    std::optional<Window> window = WindowOf(load->indices[0], &analyzer_);
    if (!window)
      return expr;
    // A window, or a run, that starts on a multiple of its size lies in one
    // aligned block (buffers start on whole registers).
    int elements = window->op == &hexagon_load_replicate()
                       ? window->period
                       : lanes / window->period;
    PrimExpr size = make_const(window->first.dtype(), elements);
    bool aligned = analyzer_.CanProveEqual(floormod(window->first, size), 0) &&
                   window->stride % elements == 0;
    PrimExpr ptr = Call(DataType::Handle(), builtin::address_of(),
                        {BufferLoad(load->buffer, {window->first})});
    Array<PrimExpr> args = {StringImm(std::to_string(window->period)),
                            StringImm(std::to_string(lanes)),
                            StringImm(aligned ? "true" : "false"), ptr};
    if (window->op == &hexagon_load_interleave())
      args.push_back(make_const(DataType::Int(32), window->stride));
    return Call(load->dtype, *window->op, args);
  }

  Stmt VisitStmt_(const ForNode *op) final {
    analyzer_.Bind(op->loop_var, Range::FromMinExtent(op->min, op->extent));
    return StmtExprMutator::VisitStmt_(op);
  }

  arith::Analyzer analyzer_;
};

} // namespace

tvm::transform::Pass LowerWindowLoads() {
  using namespace tirx::transform;
  auto pass_func = [](PrimFunc f, const IRModule &, const PassContext &) {
    f.CopyOnWrite()->body = WindowLoadLowerer()(f->body);
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.hexagon.LowerWindowLoads", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.hexagon.transform.LowerWindowLoads",
                        LowerWindowLoads);
}

} // namespace tl
} // namespace tvm
