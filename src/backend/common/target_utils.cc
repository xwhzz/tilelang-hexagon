/*!
 * \file tl/backend/common/target_utils.cc
 * \brief Common target helper dispatch.
 */

#include "backend/common/target_utils.h"

#include <tvm/ffi/reflection/registry.h>

namespace tvm {
namespace tl {

bool TargetHasAsyncCopy(Target target) {
  // A target that declares async copies lowers a software pipeline's async
  // copies itself, and names the statements that commit and wait for them
  // ("async_copy_commit", "async_copy_wait"; PTX cp.async groups without).
  if (auto declared = target->GetAttr<Bool>("supports_async_copy")) {
    return declared.value();
  }
  if (TargetIsCuda(target)) {
    return TargetCudaHasAsyncCopy(target);
  }
  if (TargetIsRocm(target)) {
    return TargetRocmHasAsyncCopy(target);
  }
  return false;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.TargetHasAsyncCopy", [](Target target) {
    return TargetHasAsyncCopy(target);
  });
}

} // namespace tl
} // namespace tvm
