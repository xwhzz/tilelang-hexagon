/*!
 * \file tl/hexagon/transform/lower_math.cc
 * \brief Lower fp16/fp32 math intrinsics to the HVX math builtins.
 *
 * Runs before layout inference, which plans the vector width of every
 * T.Parallel loop. TVM does not mark every math intrinsic vectorizable
 * (tirx.rsqrt, tirx.erf), and its default lowering has no vector form, so the
 * fp16/fp32 intrinsics become Hexagon builtins that vectorize and print as the
 * hvx_math.h functions. Other dtypes keep the default lowering.
 */

#include "hexagon/op/builtin.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <utility>
#include <vector>

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

namespace {

const Op *HexagonMathOp(const CallNode *call) {
  // Built on first use: the Ops are registered by static initializers.
  static const std::vector<std::pair<Op, Op>> ops = {
      {Op::Get("tirx.exp"), hexagon_exp()},
      {Op::Get("tirx.exp2"), hexagon_exp2()},
      {Op::Get("tirx.log"), hexagon_log()},
      {Op::Get("tirx.log2"), hexagon_log2()},
      {Op::Get("tirx.sqrt"), hexagon_sqrt()},
      {Op::Get("tirx.rsqrt"), hexagon_rsqrt()},
      {Op::Get("tirx.tanh"), hexagon_tanh()},
      {Op::Get("tirx.sigmoid"), hexagon_sigmoid()},
      {Op::Get("tirx.erf"), hexagon_erf()},
      {Op::Get("tirx.fabs"), hexagon_fabs()},
  };
  DataType t = call->dtype.element_of();
  if (!t.is_float() || (t.bits() != 16 && t.bits() != 32))
    return nullptr;
  for (const auto &[from, to] : ops) {
    if (call->op.same_as(from))
      return &to;
  }
  return nullptr;
}

class MathLowerer : public StmtExprMutator {
private:
  PrimExpr VisitExpr_(const CallNode *op) final {
    Call call = Downcast<Call>(StmtExprMutator::VisitExpr_(op));
    if (const Op *math = HexagonMathOp(call.get()))
      return Call(call->dtype, *math, call->args);
    return call;
  }
};

} // namespace

namespace transform {

tvm::transform::Pass LowerMath() {
  auto pass_func = [](PrimFunc func, const IRModule &,
                      const tvm::transform::PassContext &) -> PrimFunc {
    func.CopyOnWrite()->body = MathLowerer()(func->body);
    return func;
  };
  return tirx::transform::CreatePrimFuncPass(pass_func, 0,
                                             "tl.hexagon.LowerMath", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.hexagon.transform.LowerMath", LowerMath);
}

} // namespace transform
} // namespace tl
} // namespace tvm
