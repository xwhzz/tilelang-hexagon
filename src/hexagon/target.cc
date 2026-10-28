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
// window load and a lane shuffle. `supports_vector_predicate`: a select picks
// lane by lane (tl_vec_select), so selects whose condition varies across
// lanes vectorize, as do if_then_else's whose branches are safe to evaluate.
// `supports_unaligned_vector_access`: vector loads and stores take any address
// (vmemu), so an access whose alignment the vectorizer cannot prove still
// vectorizes, and the codegen prints it unaligned unless it proves it aligned.
// `supports_async_copy`: a software pipeline's copies from DDR become DMA
// transfers, whose stages commit and wait as groups of the tile program's DMA
// queue (`async_copy_commit` and `async_copy_wait`, set by the normalizer).
[[maybe_unused]] TargetKindRegEntry &hexagon_kind =
    TargetKindRegEntry::RegisterOrGet("hexagon")
        .add_attr_option<int64_t>("max_vector_bits", refl::DefaultValue(1024))
        .add_attr_option<int64_t>("min_vector_bits", refl::DefaultValue(1024))
        .add_attr_option<ffi::String>("fragment_scope")
        .add_attr_option<bool>("vector_window_loads", refl::DefaultValue(true))
        .add_attr_option<bool>("supports_vector_predicate",
                               refl::DefaultValue(true))
        .add_attr_option<bool>("supports_unaligned_vector_access",
                               refl::DefaultValue(true))
        .add_attr_option<bool>("supports_async_copy", refl::DefaultValue(true))
        .add_attr_option<ffi::String>("async_copy_commit")
        .add_attr_option<ffi::String>("async_copy_wait");

} // namespace
} // namespace tvm
