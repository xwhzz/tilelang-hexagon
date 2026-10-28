/*!
 * \file tl/hexagon/transform/interleave_maps.cc
 * \brief Two vectors of a heavy map per loop iteration.
 *
 * A vectorized map whose elements compute an HVX math function (exp, sigmoid,
 * a division, ...) runs a chain of a few dozen dependent instructions per
 * vector. LLVM software-pipelines a loop only up to an initiation interval it
 * bounds (-pipeliner-max-mii), which such a chain can exceed: SwiGLU's
 * x * sigmoid(x) * y needs 47 cycles, against a bound of 27, and its loop then
 * runs one chain at a time. Unrolled by two, an iteration interleaves two
 * independent chains: on v79 SwiGLU over 128 x 128 fp32 went from 126 to 64
 * cycles a vector, and four chains gained nothing more.
 *
 * Marks such loops, innermost and storing one vector an iteration, for
 * the codegen, which prints them unrolled by two (`#pragma clang loop
 * unroll_count`). Unrolling keeps the iterations in order, so any loop may.
 */

#include "hexagon/op/builtin.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <vector>

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

namespace {

bool IsMathFunction(const CallNode *call) {
  // Built on first use: the Ops are registered by static initializers.
  static const std::vector<Op> functions = {
      hexagon_exp(),       hexagon_exp2(),         hexagon_log(),
      hexagon_log2(),      hexagon_sqrt(),         hexagon_rsqrt(),
      hexagon_tanh(),      hexagon_sigmoid(),      hexagon_erf(),
      hexagon_exp_fast(),  hexagon_exp2_fast(),    hexagon_log_fast(),
      hexagon_log2_fast(), hexagon_sqrt_fast(),    hexagon_rsqrt_fast(),
      hexagon_tanh_fast(), hexagon_sigmoid_fast(), hexagon_erf_fast(),
      hexagon_div_fast()};
  for (const Op &op : functions)
    if (call->op.same_as(op))
      return true;
  return false;
}

// One iteration of a heavy map: no loop, one vector stored, and an HVX math
// function or a float division of vectors on the way.
bool IsHeavyMapIteration(const Stmt &body) {
  int stores = 0;
  bool loop = false, heavy = false;
  PostOrderVisit(body, [&](const ObjectRef &node) {
    if (node.as<ForNode>()) {
      loop = true;
    } else if (const auto *store = node.as<BufferStoreNode>()) {
      stores += store->value.dtype().lanes() > 1;
    } else if (const auto *call = node.as<CallNode>()) {
      heavy |= call->dtype.lanes() > 1 && IsMathFunction(call);
    } else if (const auto *div = node.as<DivNode>()) {
      heavy |= div->dtype.lanes() > 1 && div->dtype.is_float();
    }
  });
  return !loop && stores == 1 && heavy;
}

class MapInterleaver : public StmtExprMutator {
private:
  Stmt VisitStmt_(const ForNode *op) final {
    For loop = Downcast<For>(StmtExprMutator::VisitStmt_(op));
    // The loops over the vectors of a map are serial, or unrolled kind (which
    // the codegen prints as plain loops).
    if ((loop->kind != ForKind::kSerial && loop->kind != ForKind::kUnrolled) ||
        !IsHeavyMapIteration(loop->body))
      return loop;
    if (const auto *n = loop->extent.as<IntImmNode>(); n && n->value < 2)
      return loop;
    loop.CopyOnWrite()->annotations.Set(attr::kHexagonUnrollCount, Integer(2));
    return loop;
  }
};

} // namespace

namespace transform {

tvm::transform::Pass InterleaveMaps() {
  auto pass_func = [](PrimFunc func, const IRModule &,
                      const tvm::transform::PassContext &) -> PrimFunc {
    func.CopyOnWrite()->body = MapInterleaver()(func->body);
    return func;
  };
  return tirx::transform::CreatePrimFuncPass(pass_func, 0,
                                             "tl.hexagon.InterleaveMaps", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.hexagon.transform.InterleaveMaps", InterleaveMaps);
}

} // namespace transform
} // namespace tl
} // namespace tvm
