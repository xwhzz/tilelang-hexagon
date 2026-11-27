/*!
 * \file tl/hexagon/transform/vectorize_map_tail.h
 * \brief Vectorize the full-register prefix of a pointwise parallel map.
 */
#ifndef TVM_TL_HEXAGON_TRANSFORM_VECTORIZE_MAP_TAIL_H_
#define TVM_TL_HEXAGON_TRANSFORM_VECTORIZE_MAP_TAIL_H_

#include "transform/loop_vectorize.h"

namespace tvm::tl::hexagon {

ffi::Optional<tirx::Stmt> VectorizeMapTail(const tirx::For &loop,
                                           const arith::Analyzer *analyzer,
                                           const LayoutMap &layouts,
                                           const Target &target);

} // namespace tvm::tl::hexagon
#endif // TVM_TL_HEXAGON_TRANSFORM_VECTORIZE_MAP_TAIL_H_
