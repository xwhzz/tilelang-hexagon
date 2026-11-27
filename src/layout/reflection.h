// Utilities for restoring the bound variables of reflected layout fields.
#ifndef TVM_TL_LAYOUT_REFLECTION_H_
#define TVM_TL_LAYOUT_REFLECTION_H_

#include <tvm/ffi/reflection/registry.h>

namespace tvm::tl {

// The graph loader sets inherited fields first, in declaration order. Rebuild
// derived state after the field that supplies its final input is assigned.
template <auto Member, auto Restore> struct RestoreAfterSet;
template <typename Node, typename T, T Node::*Member, void (Node::*Restore)()>
struct RestoreAfterSet<Member, Restore> : ffi::reflection::InfoTrait {
  void Apply(TVMFFIFieldInfo *info) const {
    info->setter = reinterpret_cast<void *>(Set);
  }

  static int Set(void *address, const TVMFFIAny *value) {
    TVM_FFI_SAFE_CALL_BEGIN();
    *static_cast<T *>(address) =
        ffi::AnyView::CopyFromTVMFFIAny(*value).cast<T>();
    auto offset = ffi::reflection::GetFieldByteOffsetToObject(Member);
    auto *object =
        reinterpret_cast<ffi::Object *>(static_cast<char *>(address) - offset);
    (static_cast<Node *>(object)->*Restore)();
    TVM_FFI_SAFE_CALL_END();
  }
};

} // namespace tvm::tl
#endif // TVM_TL_LAYOUT_REFLECTION_H_
