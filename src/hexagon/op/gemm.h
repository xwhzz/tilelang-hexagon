/*!
 * \file tl/hexagon/op/gemm.h
 * \brief Whether a T.gemm runs on HMX or on HVX.
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

/*!
 * \brief Why HVX cannot run `op` (tl_hvx_gemm, hvx_gemm.h), or an empty string
 * when it can: A and B both fp16 into an fp16 or fp32 C, or both fp32 into an
 * fp32 C; at most one of them transposed; and each operand a matrix of
 * contiguous rows a constant stride apart, its leading dimensions single
 * indices. GemmHVX also needs the operands row-major, which their layouts
 * decide after selection.
 */
std::string HVXRejection(const GemmNode &op);

} // namespace hexagon
} // namespace tl
} // namespace tvm

#endif // TVM_TL_HEXAGON_OP_GEMM_H_
