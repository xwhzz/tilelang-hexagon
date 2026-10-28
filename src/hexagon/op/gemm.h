/*!
 * \file tl/hexagon/op/gemm.h
 * \brief Whether a T.gemm runs on HMX.
 */

#ifndef TVM_TL_HEXAGON_OP_GEMM_H_
#define TVM_TL_HEXAGON_OP_GEMM_H_

#include "op/gemm.h"

#include <string>

namespace tvm {
namespace tl {
namespace hexagon {

/*!
 * \brief Why HMX cannot run `op`, or an empty string when it can: A and B
 * fp16 and C fp16 or fp32, all in VTCM (shared buffers or fragments, which are
 * VTCM on Hexagon); M, N and K multiples of 32; and each operand a matrix of
 * whole 32 x 32 tiles: its region starts on a tile, its leading dimensions are
 * single indices, and A's and B's matrices, which take Crouton layouts, are
 * whole tiles too.
 */
std::string HMXRejection(const GemmNode &op);

} // namespace hexagon
} // namespace tl
} // namespace tvm

#endif // TVM_TL_HEXAGON_OP_GEMM_H_
