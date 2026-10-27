/*!
 * \file tl/hexagon/target.cc
 * \brief TileLang capability attributes of the Hexagon target kind.
 *
 * TVM registers the "hexagon" target kind. TileLang adds the attributes its
 * shared passes read, so those passes never check the target kind.
 */

#include <tvm/ffi/container/array.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/target/target.h>
#include <tvm/target/target_kind.h>

namespace tvm {
namespace {

namespace refl = tvm::ffi::reflection;

// HVX registers are 1024 bits wide (64 fp16 or 128 int8 lanes). The
// vectorizer reads this instead of its GPU-sized default.
[[maybe_unused]] TargetKindRegEntry &hexagon_kind =
    TargetKindRegEntry::RegisterOrGet("hexagon").add_attr_option<int64_t>(
        "max_vector_bits", refl::DefaultValue(1024));

} // namespace
} // namespace tvm
