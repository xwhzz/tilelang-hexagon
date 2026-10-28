/*!
 * \file tl/hexagon/transform/lower_math.cc
 * \brief Lower fp16/fp32 math intrinsics to the HVX math builtins.
 *
 * Runs before layout inference, which plans the vector width of every
 * T.Parallel loop. TVM does not mark every math intrinsic vectorizable
 * (tirx.rsqrt, tirx.erf), and its default lowering has no vector form, so the
 * fp16/fp32 intrinsics become Hexagon builtins that vectorize and print as the
 * hvx_math.h functions. Other dtypes keep the default lowering.
 *
 * Under tl.enable_fast_math they become the fast variants instead, and a float
 * division becomes hexagon_div_fast: fp16-grade accuracy (hvx_math.h).
 */

#include "hexagon/op/builtin.h"
#include "op/builtin.h" // kEnableFastMath

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

bool HexagonFloat(DataType t) {
  t = t.element_of();
  return t.is_float() && (t.bits() == 16 || t.bits() == 32);
}

const Op *HexagonMathOp(const CallNode *call, bool fast) {
  // Built on first use: the Ops are registered by static initializers. Each
  // intrinsic, its builtin and its fast builtin; fabs is exact either way.
  struct Entry {
    Op from, accurate, fast;
  };
  static const std::vector<Entry> ops = {
      {Op::Get("tirx.exp"), hexagon_exp(), hexagon_exp_fast()},
      {Op::Get("tirx.exp2"), hexagon_exp2(), hexagon_exp2_fast()},
      {Op::Get("tirx.log"), hexagon_log(), hexagon_log_fast()},
      {Op::Get("tirx.log2"), hexagon_log2(), hexagon_log2_fast()},
      {Op::Get("tirx.sqrt"), hexagon_sqrt(), hexagon_sqrt_fast()},
      {Op::Get("tirx.rsqrt"), hexagon_rsqrt(), hexagon_rsqrt_fast()},
      {Op::Get("tirx.tanh"), hexagon_tanh(), hexagon_tanh_fast()},
      {Op::Get("tirx.sigmoid"), hexagon_sigmoid(), hexagon_sigmoid_fast()},
      {Op::Get("tirx.erf"), hexagon_erf(), hexagon_erf_fast()},
      {Op::Get("tirx.fabs"), hexagon_fabs(), hexagon_fabs()},
  };
  if (!HexagonFloat(call->dtype))
    return nullptr;
  for (const auto &entry : ops) {
    if (call->op.same_as(entry.from))
      return fast ? &entry.fast : &entry.accurate;
  }
  return nullptr;
}

class MathLowerer : public StmtExprMutator {
public:
  explicit MathLowerer(bool fast) : fast_(fast) {}

private:
  PrimExpr VisitExpr_(const CallNode *op) final {
    Call call = Downcast<Call>(StmtExprMutator::VisitExpr_(op));
    if (const Op *math = HexagonMathOp(call.get(), fast_))
      return Call(call->dtype, *math, call->args);
    return call;
  }

  PrimExpr VisitExpr_(const DivNode *op) final {
    PrimExpr div = StmtExprMutator::VisitExpr_(op);
    const auto *node = div.as<DivNode>();
    if (!fast_ || node == nullptr || !HexagonFloat(node->dtype))
      return div;
    return Call(node->dtype, hexagon_div_fast(), {node->a, node->b});
  }

  bool fast_;
};

} // namespace

namespace transform {

tvm::transform::Pass LowerMath() {
  auto pass_func = [](PrimFunc func, const IRModule &,
                      const tvm::transform::PassContext &ctx) -> PrimFunc {
    bool fast = ctx->GetConfig<Bool>(kEnableFastMath, Bool(false)).value();
    func.CopyOnWrite()->body = MathLowerer(fast)(func->body);
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
