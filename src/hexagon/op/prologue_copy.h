/*!
 * \file tl/hexagon/op/prologue_copy.h
 * \brief Opt-in synchronous copies in an automatic pipeline's prologue.
 */
#ifndef TVM_TL_HEXAGON_OP_PROLOGUE_COPY_H_
#define TVM_TL_HEXAGON_OP_PROLOGUE_COPY_H_

namespace tvm {
namespace tl {
namespace hexagon {

constexpr const char *kPrologueCopyPolicy = "hexagon.prologue_copy_policy";
// Internal candidate marker. The copy lowerer must still prove that the
// actual layout, region and cache contract support an ordinary sync copy.
constexpr const char *kPrologueCopyCandidate =
    "tl.hexagon.prologue_copy_candidate";

} // namespace hexagon
} // namespace tl
} // namespace tvm

#endif // TVM_TL_HEXAGON_OP_PROLOGUE_COPY_H_
