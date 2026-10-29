/*!
 * \file tl/hexagon/transform/select_guarded_loads.cc
 * \brief Load the vectors that a scalar condition guards from a safe address,
 * and select.
 *
 * LegalizeSafeMemoryAccess guards a load it cannot prove in bounds,
 * if_then_else(cond, load, 0). C evaluates only the branch taken, so the
 * codegen prints it as an if around the load, and a vector loop with one is
 * not a single basic block, which LLVM does not software-pipeline: the LFM2
 * prefill's RoPE, which reads its tables at row base + i, ran one iteration
 * at a time.
 *
 * When the condition is a scalar and the load is of one vector, contiguous or
 * strided, or of one element, of a buffer at least as long as the access, this
 * pass loads from the same address when the condition holds and from the
 * buffer's start when it does not, then selects: the load reads memory of the
 * buffer either way, and the loop has no branch.
 *
 * Runs after UnrollLoop: the Let that holds a loaded vector binds its
 * variable once, which unrolling a loop around it would repeat.
 */

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/op_attr_types.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

namespace {

// The elements of a buffer, or -1 when its shape is not constant.
int64_t Elements(const Buffer &buffer) {
  int64_t elements = 1;
  for (const PrimExpr &dim : buffer->shape) {
    const auto *n = dim.as<IntImmNode>();
    if (!n)
      return -1;
    elements *= n->value;
  }
  return elements;
}

// Whether an expression reads no memory and has no effect.
bool IsPureValue(const PrimExpr &expr) {
  bool pure = true;
  PostOrderVisit(expr, [&](const ObjectRef &node) {
    if (node.as<BufferLoadNode>())
      pure = false;
    else if (const auto *call = node.as<CallNode>())
      pure =
          pure && SideEffect(GetRef<PrimExpr>(call)) <= CallEffectKind::kPure;
  });
  return pure;
}

class GuardedLoadSelector : public StmtExprMutator {
private:
  PrimExpr VisitExpr_(const CallNode *op) final {
    PrimExpr expr = StmtExprMutator::VisitExpr_(op);
    const auto *call = expr.as<CallNode>();
    if (!call || !call->op.same_as(builtin::if_then_else()))
      return expr;
    const PrimExpr &cond = call->args[0];
    const auto *load = call->args[1].as<BufferLoadNode>();
    const PrimExpr &other = call->args[2];
    if (cond.dtype().lanes() != 1 || !load || load->indices.size() != 1 ||
        load->predicate.defined() || !IsPureValue(other))
      return expr;
    int64_t elements = Elements(load->buffer);
    if (elements <= 0)
      return expr;
    // The access from the buffer's start: one element, or a ramp.
    PrimExpr index = analyzer_.Simplify(load->indices[0]);
    PrimExpr safe;
    if (index.dtype().lanes() == 1) {
      safe = Select(cond, index, make_zero(index.dtype()));
    } else if (const auto *ramp = index.as<RampNode>()) {
      const auto *stride = ramp->stride.as<IntImmNode>();
      const auto *lanes = ramp->lanes.as<IntImmNode>();
      if (!stride || !lanes || stride->value < 0 ||
          (lanes->value - 1) * stride->value + 1 > elements)
        return expr;
      safe = Ramp(Select(cond, ramp->base, make_zero(ramp->base.dtype())),
                  ramp->stride, ramp->lanes);
    } else {
      return expr;
    }
    Var value("guarded", load->dtype);
    return Let(value, BufferLoad(load->buffer, {safe}),
               Select(cond, value, other));
  }

  arith::Analyzer analyzer_;
};

} // namespace

namespace transform {

tvm::transform::Pass SelectGuardedLoads() {
  auto pass_func = [](PrimFunc func, const IRModule &,
                      const tvm::transform::PassContext &) -> PrimFunc {
    func.CopyOnWrite()->body = GuardedLoadSelector()(func->body);
    return func;
  };
  return tirx::transform::CreatePrimFuncPass(
      pass_func, 0, "tl.hexagon.SelectGuardedLoads", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.hexagon.transform.SelectGuardedLoads",
                        SelectGuardedLoads);
}

} // namespace transform
} // namespace tl
} // namespace tvm
