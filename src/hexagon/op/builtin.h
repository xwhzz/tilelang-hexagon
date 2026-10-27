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

/*!
 * \brief Wait until at most `groups` committed groups are pending:
 * hexagon_dma_wait(groups, buffers...), with a constant count. PlanDMAQueue
 * appends the data vars of the VTCM buffers whose transfers the wait may
 * complete; they keep those buffers live in shared memory planning, and the
 * codegen does not print them.
 */
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

// tl_hvx_reduce_rows<kind, acc, aligned>(dst, src, rows, cols, src_stride,
// dst_stride, clear) (hvx_reduce.h): reduces `rows` rows of `cols` elements
// into one element each, accumulating in `acc`. The kind names a TL_REDUCE_*
// constant, `acc` is a type annotation and `aligned` ("true" or "false") says
// whether every row starts on a 128-byte boundary; the codegen prints the three
// as template arguments.
TVM_DLL const Op &hexagon_reduce_rows();

// Lanes of HVX registers (hvx_math.h); the first argument, the vector's lane
// count, prints as a template argument.
// tl_hvx_load_lanes<lanes>(ptr, count): the `count` elements at `ptr` in the
// low lanes of a vector of `lanes`; reads no memory past them.
TVM_DLL const Op &hexagon_load_lanes();
// tl_hvx_broadcast_lane<lanes>(vector, i): lane i of a register, in every lane
// of a vector of `lanes`.
TVM_DLL const Op &hexagon_broadcast_lane();

// tl_crouton_copy<pack, column_tiles>(crouton, row_tiles, col_tiles, r0, c0,
// rows, cols, matrix, rs, cs) (crouton.h): copies the region [r0, r0 + rows) x
// [c0, c0 + cols) of a Crouton matrix to (pack: from) the strided matrix whose
// element (r0 + i, c0 + j) is matrix[i * rs + j * cs]. `pack` and
// `column_tiles` ("true" or "false") print as template arguments.
TVM_DLL const Op &hexagon_crouton_copy();

// tl_vtcm_exclusive(ptr) (vtcm.h): the pointer of a VTCM buffer that shares no
// memory with its arena's other buffers, as its own allocation to LLVM.
TVM_DLL const Op &hexagon_exclusive_pointer();

} // namespace tl
} // namespace tvm

#endif // TVM_TL_HEXAGON_OP_BUILTIN_H_
