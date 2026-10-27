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

// HVX registers are 1024 bits wide (64 fp16 or 128 int8 lanes), and every
// HVX instruction works on whole registers. The vectorizer reads both limits
// instead of its GPU-sized default. LowerTileOp places fragments in the
// storage scope `fragment_scope` names; the target normalizer sets it, since a
// string default registered here does not survive static initialization. With
// `vector_window_loads` the vectorizer accepts loads of small windows (a
// Crouton tile's row and column operands), which LowerWindowLoads lowers to a
// window load and a lane shuffle.
[[maybe_unused]] TargetKindRegEntry &hexagon_kind =
    TargetKindRegEntry::RegisterOrGet("hexagon")
        .add_attr_option<int64_t>("max_vector_bits", refl::DefaultValue(1024))
        .add_attr_option<int64_t>("min_vector_bits", refl::DefaultValue(1024))
        .add_attr_option<ffi::String>("fragment_scope")
        .add_attr_option<bool>("vector_window_loads", refl::DefaultValue(true));

} // namespace
} // namespace tvm
