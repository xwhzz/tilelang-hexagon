/*!
 * \file tl/hexagon/transform/mark_unwritten_dma_sources.cc
 * \brief DMA loads from a kernel's read-only arguments skip the writeback.
 *
 * DMA reads DDR, so before a DMA load the runtime writes the source's cache
 * lines back, in case the DSP wrote newer bytes (dma.h). A kernel argument
 * that the kernel never writes (AnnotateReadOnlyParams) has no such lines:
 * the DSP gets its arguments with clean caches, and arguments do not alias
 * (the kernel is noalias). Its loads use the direction that skips the
 * writeback, TL_HEXAGON_DMA_DDR_TO_VTCM_UNWRITTEN, which spares a dccleana per
 * 64 bytes and a syncht per transfer.
 *
 * Runs after AnnotateReadOnlyParams, on device functions.
 */

#include "hexagon/op/builtin.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <unordered_set>

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

namespace {

// dma.h's tl_hexagon_dma_direction.
constexpr int64_t kDDRToVTCM = 0;
constexpr int64_t kDDRToVTCMUnwritten = 3;

// The data var a pointer argument addresses: tvm_access_ptr's buffer, or an
// address_of's BufferLoad.
const VarNode *PointedVar(const PrimExpr &ptr) {
  const auto *call = ptr.as<CallNode>();
  if (call == nullptr)
    return ptr.as<VarNode>();
  if (call->op.same_as(builtin::tvm_access_ptr()))
    return call->args[1].as<VarNode>();
  if (call->op.same_as(builtin::address_of()))
    if (const auto *load = call->args[0].as<BufferLoadNode>())
      return load->buffer->data.get();
  return nullptr;
}

class UnwrittenSourceMarker : public StmtExprMutator {
public:
  explicit UnwrittenSourceMarker(std::unordered_set<const VarNode *> readonly)
      : readonly_(std::move(readonly)) {}

private:
  PrimExpr VisitExpr_(const CallNode *op) final {
    if (!op->op.same_as(hexagon_dma_submit()))
      return StmtExprMutator::VisitExpr_(op);
    // (dst, src, dst_stride, src_stride, row_bytes, rows, direction)
    const auto *direction = op->args[6].as<IntImmNode>();
    if (direction == nullptr || direction->value != kDDRToVTCM ||
        !readonly_.count(PointedVar(op->args[1])))
      return StmtExprMutator::VisitExpr_(op);
    Array<PrimExpr> args = op->args;
    args.Set(6, IntImm(direction->dtype, kDDRToVTCMUnwritten));
    return Call(op->dtype, op->op, args, op->annotations, op->span);
  }

  std::unordered_set<const VarNode *> readonly_;
};

} // namespace

tvm::transform::Pass MarkUnwrittenDMASources() {
  using namespace tirx::transform;
  auto pass_func = [](PrimFunc f, const IRModule &, const PassContext &) {
    auto indices = f->GetAttr<Array<Integer>>("tl.readonly_param_indices");
    if (!indices)
      return f;
    std::unordered_set<const VarNode *> readonly;
    for (const Integer &i : indices.value()) {
      const Var &param = f->params[i->value];
      readonly.insert(param.get());
      if (auto buffer = f->buffer_map.Get(param))
        readonly.insert(buffer.value()->data.get());
    }
    f.CopyOnWrite()->body = UnwrittenSourceMarker(std::move(readonly))(f->body);
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.hexagon.MarkUnwrittenDMASources",
                            {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.hexagon.transform.MarkUnwrittenDMASources",
                        MarkUnwrittenDMASources);
}

} // namespace tl
} // namespace tvm
