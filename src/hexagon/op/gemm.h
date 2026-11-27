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

// An explicit row-major A or B chose the HVX layout before pipeline staging.
constexpr const char *kRowMajorGemm = "hexagon.row_major_gemm";

/*!
 * \brief Why HMX cannot run `op`, or an empty string when it can: A and B
 * fp16 and C fp16 or fp32, all in VTCM (shared buffers or fragments, which are
 * VTCM on Hexagon); each operand a matrix whose region starts on a 32 x 32
 * tile and whose leading dimensions are single indices; and A's and B's
 * buffers, which take Crouton layouts, whole tiles. M, N and K need not be
 * multiples of 32: HMX multiplies zero-padded tiles (PadHMXOperands pads the
 * buffers), when M is at least kHMXMinRows or a multiple of 32.
 *
 * With `pad`, A's and B's buffers need not be whole tiles yet: whether HMX
 * runs `op` once PadHMXOperands pads them.
 */
std::string HMXRejection(const GemmNode &op, bool pad = false);

// Validate an integer HMX tile call before layout remapping and flattening.
void ValidateHMXGemmU8I8(const tirx::CallNode &op, const Target &target,
                         const LayoutMap &layouts, arith::Analyzer *analyzer);

/*!
 * \brief The fewest rows of a GEMM off whole tiles that HMX runs. A GEMV
 * (M = 1) stays on HVX: padding its A and C to 32 rows would take 32 times
 * their VTCM. HMX would still be faster (on v79 at N = K = 512, 1-1.5 against
 * 7.9 us a call with B in VTCM, 7.3 against 11.8 us when B is copied in each
 * call, #84): a GEMV that can spare the VTCM can multiply 2 rows instead.
 */
constexpr int kHMXMinRows = 2;

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
