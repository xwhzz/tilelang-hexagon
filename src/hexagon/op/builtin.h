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
// on the outermost grid loop, and finally on the parallel grid loop. Also on
// the T.gemm calls of a pool of more than one worker (GemmHMX).
constexpr const char *kHexagonNumWorkers = "hexagon.num_workers";

// How many iterations of a loop the codegen unrolls into one (#pragma clang
// loop unroll_count). Set by InterleaveMaps.
constexpr const char *kHexagonUnrollCount = "hexagon.unroll_count";

} // namespace attr

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
 * direction is dma.h's tl_hexagon_dma_direction, which picks the DDR side's
 * cache maintenance: 0 for DDR to VTCM and 1 for VTCM to DDR, as lowered; 2,
 * none, for a buffer only DMA moves (MarkDMAOnlyBuffers); 3 for DDR to VTCM
 * from an argument the kernel never writes (MarkUnwrittenDMASources).
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
// The fast variants (tl.enable_fast_math), hvx_math.h's tl_hvx_<fn>_fast, and
// a float division a / b as hexagon_div_fast(a, b).
TVM_DLL const Op &hexagon_exp_fast();
TVM_DLL const Op &hexagon_exp2_fast();
TVM_DLL const Op &hexagon_log_fast();
TVM_DLL const Op &hexagon_log2_fast();
TVM_DLL const Op &hexagon_sqrt_fast();
TVM_DLL const Op &hexagon_rsqrt_fast();
TVM_DLL const Op &hexagon_tanh_fast();
TVM_DLL const Op &hexagon_sigmoid_fast();
TVM_DLL const Op &hexagon_erf_fast();
TVM_DLL const Op &hexagon_div_fast();

// tl_hvx_reduce_rows<kind, acc, aligned>(dst, src, rows, cols, src_stride,
// dst_stride, clear) (hvx_reduce.h): reduces `rows` rows of `cols` elements
// into one element each, accumulating in `acc`. The kind names a TL_REDUCE_*
// constant, `acc` is a type annotation and `aligned` ("true" or "false") says
// whether every row starts on a 128-byte boundary; the codegen prints the three
// as template arguments.
TVM_DLL const Op &hexagon_reduce_rows();

// tl_crouton_reduce<kind, acc, rows, column_tiles>(dst, crouton, row_tiles,
// col_tiles, dst_stride, clear) (crouton.h): reduces a Crouton matrix along
// its rows (rows "true": one result per row) or its columns into
// dst[i * dst_stride], accumulating in `acc`. The kind, acc, rows and
// column_tiles print as template arguments, as in hexagon_reduce_rows.
TVM_DLL const Op &hexagon_crouton_reduce();

// Lanes of HVX registers (hvx_math.h); the first argument, the vector's lane
// count, prints as a template argument.
// tl_hvx_load_lanes<lanes>(ptr, count): the `count` elements at `ptr` in the
// low lanes of a vector of `lanes`; reads no memory past them.
TVM_DLL const Op &hexagon_load_lanes();
// tl_hvx_broadcast_lane<lanes, group>(vector, i): lane group i of a register,
// the `group` lanes from lane i * group, repeated over a vector of `lanes`; the
// group size prints as a second template argument.
TVM_DLL const Op &hexagon_broadcast_lane();

// Vector loads of small windows (hvx_lanes.h); the first three arguments, the
// period, the vector's lane count and whether the window starts on a multiple
// of its size ("true" or "false"), print as template arguments.
// tl_hvx_load_replicate<period, lanes, aligned>(ptr): lane l is
// ptr[l % period].
TVM_DLL const Op &hexagon_load_replicate();
// tl_hvx_load_repeat<period, lanes, aligned>(ptr): lane l is ptr[l / period].
TVM_DLL const Op &hexagon_load_repeat();
// tl_hvx_load_interleave<2, lanes, aligned>(ptr, stride): lane l is
// ptr[(l % 2) * stride + l / 2], two runs interleaved; aligned: both runs.
TVM_DLL const Op &hexagon_load_interleave();

// tl_crouton_copy<pack, column_tiles>(crouton, row_tiles, col_tiles, r0, c0,
// rows, cols, matrix, rs, cs) (crouton.h): copies the region [r0, r0 + rows) x
// [c0, c0 + cols) of a Crouton matrix to (pack: from) the strided matrix whose
// element (r0 + i, c0 + j) is matrix[i * rs + j * cs]. `pack` and
// `column_tiles` ("true" or "false") print as template arguments.
TVM_DLL const Op &hexagon_crouton_copy();

// tl_crouton_move_tiles<column_tiles>(dst, dst_row_tiles, dst_col_tiles, dr0,
// dc0, src, src_row_tiles, src_col_tiles, sr0, sc0, rows, cols) (crouton.h):
// copies a region of whole tiles between two Crouton matrices of the same tile
// order, from (sr0, sc0) of src to (dr0, dc0) of dst. `column_tiles` ("true"
// or "false") prints as a template argument.
TVM_DLL const Op &hexagon_crouton_move_tiles();

// tl_crouton_tile_store<accumulate, rows, aligned, scaled>(dst, ldc, tile[,
// scale, ld]) (crouton.h): writes the fp16 Crouton tile `tile`, an HMX output,
// into a 32 x 32 block of a matrix, or adds it to the block (accumulate): a
// row-major block whose element (i, j) is dst[i * ldc + j] (rows), fp16 or
// fp32, or the Crouton tile dst. `aligned`: the rows start on register
// boundaries (fp32) or half-register ones (fp16). `scaled`: row i is
// multiplied by the fp32 scale[i * ld] first. The four print as template
// arguments.
TVM_DLL const Op &hexagon_crouton_tile_store();

// tl_hexagon_hmx_gemm<rows, aligned, shared, direct>(M, N, K, c, ldc, a,
// a_row_tile, b, b_column_tile, clear, stage, config) (hmx_gemm.h): C (+)= A
// B^T on HMX for fp16 A and B in the HMX activation and weight layouts, column
// tile nt of B at b + nt * b_column_tile; `stage` and `config` are its staging
// tiles and conversion config. `direct`: an overwritten Crouton C takes HMX's
// tiles without staging. The four flags print as template arguments.
TVM_DLL const Op &hexagon_hmx_gemm();

// tl_hexagon_hmx_gemm_blockscaled<signed, rows, aligned, scaled, shared>(M, N,
// K, c, ldc, a, a_row_tile, b, panel_bytes, sfb, ld_sfb, gb, sfa, ld_sfa,
// clear, w, stage, config) (hmx_gemm.h): C (+)= (A * SFA) (B * SFB)^T on HMX
// for an fp16 A in the HMX activation layout and a 4-bit B in the panel
// layout, B dequantized on HVX into the weight tiles `w`; `stage` and
// `config` are its staging tiles and conversion config. The five flags print
// as template arguments.
TVM_DLL const Op &hexagon_hmx_gemm_blockscaled();

// tl_hvx_gemm<M, N, K, trans_A, trans_B, lda, ldb, ldc, align_a, align_b,
// align_c>(C, A, B, clear) (hvx_gemm.h): C = A B, or C += A B unless clear,
// on HVX for the M x N x K GEMM whose operands start at C, A and B, their rows
// lda, ldb and ldc elements apart; align_x: x starts on a register boundary.
// The eleven integers and booleans print as template arguments.
TVM_DLL const Op &hexagon_hvx_gemm();

// tl_hvx_gemm_weight<M, N, K, lda, ldc, align_a, panel_stride>(C, A, B, clear)
// (hvx_gemm.h): C = A B, or C += A B unless clear, on HVX for an fp16 A with
// rows lda elements apart and the fp16 B [K, N] of the HMX weight operand, in
// Crouton tiles, its panels of 32 columns panel_stride elements apart; C is
// fp32 or fp16 with rows ldc elements apart. align_a: A starts on a register
// boundary. The seven integers and booleans print as template arguments.
TVM_DLL const Op &hexagon_hvx_gemm_weight();

// tl_hvx_gemm_int<M, N, K, b_bits, b_signed, lda, ldc, align_c>(C, A, B,
// panel_bytes, clear) (hvx_gemm.h): C = A B^T exactly in int32, or C += A B^T
// unless clear, on HVX: A int8, uint8 or int16 with rows lda elements apart, B
// b_bits-bit (8 or 4) in the panel layout with panels panel_bytes bytes
// apart, and C int32 with rows ldc elements apart; align_c: C starts on a
// register boundary. The eight integers and booleans print as template
// arguments.
TVM_DLL const Op &hexagon_hvx_gemm_int();

// tl_hvx_gemm_int_scaled<M, N, K, b_bits, b_signed, lda, ldc, ga, gb,
// align_c>(C, A, B, panel_bytes, sfa, ld_sfa, sfb, ld_sfb, clear) (hvx_gemm.h):
// C (+)= (A * SFA) (B * SFB)^T in fp32, for A and B as tl_hvx_gemm_int's, an
// fp32 SFA of one factor per ga columns of a row (rows ld_sfa apart) and an
// fp16 SFB of one factor per gb columns of a row, group-major (groups ld_sfb
// apart). The ten integers and booleans print as template arguments.
TVM_DLL const Op &hexagon_hvx_gemm_int_scaled();

// tl_exclusive(ptr) (common.h): the pointer of a VTCM buffer that shares no
// memory with its arena's other buffers, as its own allocation to LLVM.
TVM_DLL const Op &hexagon_exclusive_pointer();

} // namespace tl
} // namespace tvm

#endif // TVM_TL_HEXAGON_OP_BUILTIN_H_
