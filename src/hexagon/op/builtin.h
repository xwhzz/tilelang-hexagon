/*!
 * \file tl/hexagon/op/builtin.h
 * \brief Hexagon-specific TileLang intrinsic Ops.
 */

#ifndef TVM_TL_HEXAGON_OP_BUILTIN_H_
#define TVM_TL_HEXAGON_OP_BUILTIN_H_

#include "op/builtin.h"

namespace tvm {
namespace tl {

namespace attr {

// Grid axis of a launch loop, in outer-to-inner order. Set by
// LowerKernelLaunch and consumed by MaterializeGrid.
constexpr const char *kHexagonGridDim = "hexagon.grid_dim";

// Worker count of a launch: on the launch block (T.Kernel(num_workers=)), then
// on the outermost grid loop, and finally on the parallel grid loop.
constexpr const char *kHexagonNumWorkers = "hexagon.num_workers";

} // namespace attr

/*! \brief Acquire exclusive use of the implicit HMX accumulator state. */
TVM_DLL const Op &hexagon_hmx_acquire();

/*! \brief Release exclusive use of the implicit HMX accumulator state. */
TVM_DLL const Op &hexagon_hmx_release();

/*! \brief Load the HMX bias/scale register from a VTCM memory operand. */
TVM_DLL const Op &hexagon_hmx_load_bias();

/*! \brief Clear the implicit HMX accumulator. */
TVM_DLL const Op &hexagon_hmx_clear();

/*!
 * \brief Execute one HMX deep multiply packet for 1 to 32 contiguous K tiles.
 *
 * The memory operands are activation and weight Crouton tiles in VTCM.  The
 * result accumulates into the implicit accumulator state token.
 */
TVM_DLL const Op &hexagon_hmx_mma();

/*! \brief Convert the implicit HMX accumulator into convert state. */
TVM_DLL const Op &hexagon_hmx_convert();

/*! \brief Store HMX convert state to an output Crouton tile in VTCM. */
TVM_DLL const Op &hexagon_hmx_store();

// Hexagon User-DMA. A tile program owns one queue of descriptors, completed in
// commit groups like cp.async groups: T.dma_copy is submits plus one commit,
// and T.dma_wait(n) waits until at most n committed groups are pending.

/*!
 * \brief Declare the tile program's DMA queue: hexagon_dma_queue(descriptors,
 * groups), its capacities. PlanDMAQueue places it at the start of the program.
 */
TVM_DLL const Op &hexagon_dma_queue();

/*!
 * \brief Submit one DMA descriptor: hexagon_dma_submit(dst, src, dst_stride,
 * src_stride, row_bytes, rows, direction). Strides and row size are in bytes;
 * direction is 0 for DDR to VTCM and 1 for VTCM to DDR.
 */
TVM_DLL const Op &hexagon_dma_submit();

/*! \brief Close the open group: the descriptors submitted since the last
 * commit. */
TVM_DLL const Op &hexagon_dma_commit();

/*! \brief Wait until at most `groups` committed groups are pending:
 * hexagon_dma_wait(groups), with a constant count. */
TVM_DLL const Op &hexagon_dma_wait();

// HVX math. LowerMath rewrites the fp16/fp32 tirx math intrinsics into these
// before vectorization: they are vectorizable, and the codegen prints scalar
// and vector forms as the same hvx_math.h function, so a result does not
// depend on whether its loop vectorized.
TVM_DLL const Op &hexagon_exp();
TVM_DLL const Op &hexagon_exp2();
TVM_DLL const Op &hexagon_log();
TVM_DLL const Op &hexagon_log2();
TVM_DLL const Op &hexagon_sqrt();
TVM_DLL const Op &hexagon_rsqrt();
TVM_DLL const Op &hexagon_tanh();
TVM_DLL const Op &hexagon_sigmoid();
TVM_DLL const Op &hexagon_erf();
TVM_DLL const Op &hexagon_fabs();

} // namespace tl
} // namespace tvm

#endif // TVM_TL_HEXAGON_OP_BUILTIN_H_
