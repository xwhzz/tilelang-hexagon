/*!
 * \file tl/hexagon/target_utils.cc
 * \brief Hexagon target attribute helpers.
 */

#include "hexagon/target_utils.h"

#include <tvm/ffi/reflection/registry.h>

#include "dlpack/dlpack.h"

namespace tvm {
namespace tl {

bool TargetIsHexagon(Target target) {
  return target->GetTargetDeviceType() == kDLHexagon;
}

int HexagonArchVersion(Target target) {
  auto mcpu = target->GetAttr<ffi::String>("mcpu");
  if (!mcpu)
    return 0;
  std::string value = mcpu.value();
  const std::string prefix = "hexagonv";
  if (value.rfind(prefix, 0) != 0)
    return 0;
  int version = 0;
  for (size_t i = prefix.size(); i < value.size(); ++i) {
    if (value[i] < '0' || value[i] > '9')
      return 0;
    version = version * 10 + (value[i] - '0');
  }
  return version;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.TargetIsHexagon",
                        [](Target target) { return TargetIsHexagon(target); });
}

} // namespace tl
} // namespace tvm
