#pragma once
// HMX FP16 matmul runtime for tilelang-generated Hexagon kernels. All the
// HMX/HVX/ VTCM/Crouton hardware intimacy lives here; the codegen stays a thin
// C emitter that just calls into it. Distilled from the verified mini-htp /
// htp-ops-lib recipes.
//
// T.gemm and explicit DSL kernels use stateful HMX instruction atoms.
// Layout-copy helpers transform caller-owned buffers; the runtime owns
// sessions.
//
// Compiled into the FastRPC skel (.cc) with the Hexagon SDK on the include path
// and -mhmx; it pulls in HAP headers, so it is only included by kernels that
// use HMX.
#include <HAP_compute_res.h>
#include <HAP_perf.h> // Generated DSP timing wrappers also use the qtimer API.
#include <HAP_power.h>
#include <hexagon_types.h> // HVX_Vector + Q6_* intrinsics (needs -mhvx)
#include <hmx_hexagon_protos.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <tl_templates/hexagon/qmatmul.h> // reusable FP32/Crouton row-pair atoms
#include <tl_templates/hexagon/vtcm.h> // unified VTCM arena (shared with alloc_shared)
#include <tl_templates/hexagon/worker.h> // HW-thread worker pool (multithreaded HMX/HVX)

#define TL_HMX_INLINE static inline __attribute__((unused, always_inline))

// DDR matrix slices are not guaranteed to retain 128-byte alignment. Keep the
// fused row-major <-> Crouton helpers correct for arbitrary submatrix bases;
// hexagon-clang lowers the aligned(1) accesses to unaligned HVX vmem forms.
typedef HVX_Vector tl_hmx_unaligned_vector __attribute__((aligned(1)));
TL_HMX_INLINE HVX_Vector tl_hmx_loadu(const void *p) {
  return *(const tl_hmx_unaligned_vector *)p;
}
TL_HMX_INLINE void tl_hmx_storeu(void *p, HVX_Vector value) {
  *(tl_hmx_unaligned_vector *)p = value;
}

// ------------------------------ power -------------------------------------
static int tl_hmx_power_ctx;
TL_HMX_INLINE void tl_hmx_power_setup(void) {
  HAP_power_request_t req;
  memset(&req, 0, sizeof(req));
  req.type = HAP_power_set_DCVS_v3;
  req.dcvs_v3.dcvs_enable = TRUE;
  req.dcvs_v3.dcvs_option = HAP_DCVS_V2_PERFORMANCE_MODE;
  req.dcvs_v3.set_latency = TRUE;
  req.dcvs_v3.latency = 100;
  req.dcvs_v3.set_core_params = TRUE;
  req.dcvs_v3.core_params.min_corner = HAP_DCVS_VCORNER_NOM;
  req.dcvs_v3.core_params.max_corner = HAP_DCVS_VCORNER_TURBO_L3;
  req.dcvs_v3.core_params.target_corner = HAP_DCVS_VCORNER_TURBO_L3;
  req.dcvs_v3.set_bus_params = TRUE;
  req.dcvs_v3.bus_params.min_corner = HAP_DCVS_VCORNER_NOM;
  req.dcvs_v3.bus_params.max_corner = HAP_DCVS_VCORNER_TURBO_L3;
  req.dcvs_v3.bus_params.target_corner = HAP_DCVS_VCORNER_TURBO_L3;
  HAP_power_set(&tl_hmx_power_ctx, &req);
  memset(&req, 0, sizeof(req));
  req.type = HAP_power_set_HMX;
  req.hmx.power_up = TRUE;
  HAP_power_set(&tl_hmx_power_ctx, &req);
}
TL_HMX_INLINE void tl_hmx_power_reset(void) {
  HAP_power_request_t req;
  memset(&req, 0, sizeof(req));
  req.type = HAP_power_set_HMX;
  req.hmx.power_up = FALSE;
  HAP_power_set(&tl_hmx_power_ctx, &req);
  HAP_power_set_dcvs_v3_init(&req);
  HAP_power_set(&tl_hmx_power_ctx, &req);
}

// ------------------------------ VTCM --------------------------------------
// Delegated to the shared tl_vtcm arena (vtcm.h) so HMX scratch and
// alloc_shared operands draw from one acquisition — no double-acquire when a
// T.gemm kernel has its operands in VTCM and also drives HMX.
TL_HMX_INLINE void tl_hmx_vtcm_setup(void) { tl_vtcm_acquire(); }
TL_HMX_INLINE void tl_hmx_vtcm_reset(void) { tl_vtcm_release(); }

// ------------------------------ HMX lock ----------------------------------
static int tl_hmx_ctx;
static int tl_hmx_spin;
TL_HMX_INLINE void tl_hmx_setup(void) {
  compute_res_attr_t req;
  HAP_compute_res_attr_init(&req);
  HAP_compute_res_attr_set_hmx_param(&req, 1);
  tl_hmx_ctx = HAP_compute_res_acquire(&req, 10000);
}
TL_HMX_INLINE void tl_hmx_reset(void) {
  if (tl_hmx_ctx)
    HAP_compute_res_release(tl_hmx_ctx);
  tl_hmx_ctx = 0;
}
// Per-thread HMX SHARED enable: each thread that issues HMX MACs must call this
// once. Returns 0 if this thread now holds the lock, non-zero on failure
// (incl. no ctx) so callers can refuse rather than run MACs with HMX unlocked.
TL_HMX_INLINE int tl_hmx_enable(void) {
  if (!tl_hmx_ctx)
    return -1;
  return HAP_compute_res_hmx_lock2(tl_hmx_ctx, HAP_COMPUTE_RES_HMX_SHARED);
}
TL_HMX_INLINE void tl_hmx_disable(void) {
  if (tl_hmx_ctx)
    HAP_compute_res_hmx_unlock2(tl_hmx_ctx, HAP_COMPUTE_RES_HMX_SHARED);
}
TL_HMX_INLINE void tl_hmx_unit_acquire(void) {
  int *lp = &tl_hmx_spin;
  asm volatile("1:  r0 = memw_locked(%0)     \n"
               "    p0 = cmp.eq(r0, #0)      \n"
               "    if (!p0) jump 2f         \n"
               "    memw_locked(%0, p0) = %0 \n"
               "    if (p0) jump 3f          \n"
               "2:  pause(#8)                \n"
               "    jump 1b                  \n"
               "3:"
               : "+r"(lp)::"p0", "r0");
}
// Release with a plain volatile store (matching htp-ops-lib's hmx_mgr). No
// extra barrier is needed: the only state the lock guards is the HMX
// accumulator, which the next holder unconditionally re-clears (mxclracc)
// before use, and every worker reads its accumulator out to its OWN VTCM region
// — so no worker depends on another's in-flight accumulator/VTCM writes being
// visible at release time.
TL_HMX_INLINE void tl_hmx_unit_release(void) {
  *(volatile int *)&tl_hmx_spin = 0;
}

// ------------------------- the HMX matmul ---------------------------------
#define TL_HMX_T 32
#define TL_HMX_TILE_ELMS 1024
#define TL_HMX_TILE_BYTES 2048
#define TL_HMX_SPATIAL_MASK 0x1cu
#define TL_HMX_SPATIAL_RT                                                      \
  ((((TL_HMX_SPATIAL_MASK) >> 1) << 7) | (((TL_HMX_SPATIAL_MASK) & 1u) << 1))
#define TL_HMX_ACTIVATION_RT (TL_HMX_SPATIAL_RT | (31u << 2))

TL_HMX_INLINE void tl_hmx_clear_acc(void) {
  asm volatile("mxclracc.hf" ::: "memory");
}
TL_HMX_INLINE void tl_hmx_set_scales(const void *s) {
  asm volatile("bias = mxmem2(%0)" ::"r"(s) : "memory");
}
// Fill a 256-byte HMX output-scale region: per-column scale = fp16 1.0 (u32 low
// half == 0x3c00), per-column bias = 0. Single source for the u32 scale-word
// layout, initialized by standalone sessions and the embedding bridge.
TL_HMX_INLINE void tl_hmx_fill_unit_scales(uint32_t *s) {
  for (int i = 0; i < 32; ++i)
    s[i] = 0x00003c00u;
  for (int i = 32; i < 64; ++i)
    s[i] = 0u;
}
TL_HMX_INLINE void tl_hmx_mac_tiles(const __fp16 *a, const __fp16 *w,
                                    size_t n) {
  // Caller guarantees 1 <= n <= 32 and contiguous native tiles.
  const uint32_t act_rt = TL_HMX_ACTIVATION_RT | ((n - 1) << 11);
  const uint32_t weight_rt = n * TL_HMX_TILE_BYTES - 1;
  asm volatile("{ activation.hf = mxmem(%0, %1):deep\n"
               "  weight.hf = mxmem(%2, %3) }\n" ::"r"(a),
               "r"(act_rt), "r"(w), "r"(weight_rt)
               : "memory");
}
TL_HMX_INLINE void tl_hmx_convert_acc(int mode) {
  asm volatile("cvt.hf = acc(%0)" ::"r"(mode) : "memory");
}
TL_HMX_INLINE void tl_hmx_store_cvt(__fp16 *o) {
  asm volatile("mxmem(%0, %1) = cvt" ::"r"(o), "r"(TL_HMX_SPATIAL_RT)
               : "memory");
}
// Crouton 32x32 tile layout: (row i, col j) -> (i&~1)*32 + j*2 + (i&1).
TL_HMX_INLINE int tl_hmx_crouton_tile_offset(int i, int j) {
  return (i & ~1) * 32 + j * 2 + (i & 1);
}
// Element offset of operand coord (r,c) in a Crouton-tiled scratch whose tile
// grid is `tile_cols` tiles wide. `transpose_tile` selects the weight
// intra-tile order (column, row) vs the activation/output order (row, column).
// One source of truth for the tile addressing, used by every scalar pack/unpack
// fallback below.
TL_HMX_INLINE size_t tl_hmx_crouton_offset(int r, int c, int tile_cols,
                                           int transpose_tile) {
  size_t tile =
      ((size_t)(r / TL_HMX_T) * tile_cols + c / TL_HMX_T) * TL_HMX_TILE_ELMS;
  int pr = r % TL_HMX_T, pc = c % TL_HMX_T;
  return tile + (transpose_tile ? tl_hmx_crouton_tile_offset(pc, pr)
                                : tl_hmx_crouton_tile_offset(pr, pc));
}

// Scalar Crouton pack: source logical element (i,j) lives at src[i*row_stride +
// j*col_stride], over a rows x cols logical shape, into a tile grid `tile_cols`
// wide; `transpose_tile` picks the weight intra-tile order. The
// activation/weight HVX fast paths handle the common 64-multiple case; this
// covers the remainder AND the transposed operands (whose strided source would
// need gathered HVX loads), sharing one scalar fallback body.
TL_HMX_INLINE void
tl_hmx_pack_crouton_scalar_f16(__fp16 *dst, const __fp16 *src, int rows,
                               int cols, int tile_cols, int row_stride,
                               int col_stride, int transpose_tile) {
  for (int i = 0; i < rows; ++i)
    for (int j = 0; j < cols; ++j)
      dst[tl_hmx_crouton_offset(i, j, tile_cols, transpose_tile)] =
          src[(size_t)i * row_stride + (size_t)j * col_stride];
}
// Crouton's intra-tile layout IS a row-pair interleave, so one HVX `vshuff` of
// two source rows produces a whole 64-element row-pair span (lo half -> the
// even tile, hi half -> the next tile). The 64-column chunk spans 2 tiles, so
// the contiguous dim must be a 64-multiple; otherwise fall back to the scalar
// pack.
TL_HMX_INLINE void tl_hmx_pack_activation_strided_f16(__fp16 *dst,
                                                      const __fp16 *src, int M,
                                                      int K, int row_stride,
                                                      int col_stride) {
  int KT = K / TL_HMX_T;
  if (col_stride == 1 && (K & 63) == 0) {
    for (int mt = 0; mt < M / TL_HMX_T; ++mt)
      for (int r = 0; r < TL_HMX_T / 2; ++r) { // 16 row-pairs per tile
        for (int kc = 0; kc < K / 64; ++kc) {  // 64 cols == 2 k-tiles
          const __fp16 *r0 =
              src + (size_t)(mt * TL_HMX_T + 2 * r) * row_stride + kc * 64;
          const __fp16 *r1 =
              src + (size_t)(mt * TL_HMX_T + 2 * r + 1) * row_stride + kc * 64;
          HVX_VectorPair vp =
              Q6_W_vshuff_VVR(tl_hmx_loadu(r1), tl_hmx_loadu(r0), -2);
          tl_hmx_storeu(dst + (size_t)(mt * KT + 2 * kc) * TL_HMX_TILE_ELMS +
                            r * 64,
                        Q6_V_lo_W(vp));
          tl_hmx_storeu(
              dst + (size_t)(mt * KT + 2 * kc + 1) * TL_HMX_TILE_ELMS + r * 64,
              Q6_V_hi_W(vp));
        }
      }
    return;
  }
  tl_hmx_pack_crouton_scalar_f16(dst, src, M, K, KT, row_stride, col_stride, 0);
}
TL_HMX_INLINE void tl_hmx_pack_weight_strided_f16(__fp16 *dst,
                                                  const __fp16 *src, int K,
                                                  int N, int row_stride,
                                                  int col_stride) {
  int KT = K / TL_HMX_T;
  if (col_stride == 1 && (N & 63) == 0) {
    for (int kt = 0; kt < KT; ++kt)
      for (int r = 0; r < TL_HMX_T / 2; ++r) { // 16 k-row-pairs per k-tile
        for (int nc = 0; nc < N / 64; ++nc) {  // 64 cols == 2 n-tiles
          const __fp16 *r0 =
              src + (size_t)(kt * TL_HMX_T + 2 * r) * row_stride + nc * 64;
          const __fp16 *r1 =
              src + (size_t)(kt * TL_HMX_T + 2 * r + 1) * row_stride + nc * 64;
          HVX_VectorPair vp =
              Q6_W_vshuff_VVR(tl_hmx_loadu(r1), tl_hmx_loadu(r0), -2);
          tl_hmx_storeu(dst + (size_t)((2 * nc) * KT + kt) * TL_HMX_TILE_ELMS +
                            r * 64,
                        Q6_V_lo_W(vp));
          tl_hmx_storeu(
              dst + (size_t)((2 * nc + 1) * KT + kt) * TL_HMX_TILE_ELMS +
                  r * 64,
              Q6_V_hi_W(vp));
        }
      }
    return;
  }
  tl_hmx_pack_crouton_scalar_f16(dst, src, N, K, KT, col_stride, row_stride, 1);
}
TL_HMX_INLINE void tl_hmx_unpack_output_strided_f16(__fp16 *dst,
                                                    const __fp16 *src, int M,
                                                    int N, int row_stride,
                                                    int col_stride) {
  int NT = N / TL_HMX_T;
  if (col_stride == 1 && (N & 63) == 0) {
    for (int mt = 0; mt < M / TL_HMX_T; ++mt)
      for (int r = 0; r < TL_HMX_T / 2; ++r) {
        for (int nc = 0; nc < N / 64; ++nc) {
          HVX_Vector lo = tl_hmx_loadu(
              src + (size_t)(mt * NT + 2 * nc) * TL_HMX_TILE_ELMS + r * 64);
          HVX_Vector hi = tl_hmx_loadu(
              src + (size_t)(mt * NT + 2 * nc + 1) * TL_HMX_TILE_ELMS + r * 64);
          HVX_VectorPair vp = Q6_W_vdeal_VVR(hi, lo, -2);
          tl_hmx_storeu(dst + (size_t)(mt * TL_HMX_T + 2 * r) * row_stride +
                            nc * 64,
                        Q6_V_lo_W(vp));
          tl_hmx_storeu(dst + (size_t)(mt * TL_HMX_T + 2 * r + 1) * row_stride +
                            nc * 64,
                        Q6_V_hi_W(vp));
        }
      }
    return;
  }
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n)
      dst[(size_t)m * row_stride + (size_t)n * col_stride] =
          src[tl_hmx_crouton_offset(m, n, NT, 0)];
}
// One butterfly stage of the packed-FP16 transpose. Keep the shuffle
// width and all vector indices constant after inlining and loop unrolling.
template <int Step>
TL_HMX_INLINE void tl_hmx_transpose_pairs_stage_f16(HVX_Vector (&v)[32]) {
#pragma unroll
  for (int base = 0; base < 32; base += 2 * Step) {
#pragma unroll
    for (int j = 0; j < Step; ++j) {
      HVX_VectorPair p =
          Q6_W_vshuff_VVR(v[base + j + Step], v[base + j], -4 * Step);
      v[base + j] = Q6_V_lo_W(p);
      v[base + j + Step] = Q6_V_hi_W(p);
    }
  }
}

// Transpose a 32x64 FP16 source rectangle as 32x32 packed pairs. Each
// resulting vector holds one Crouton row pair across 32 columns. The first
// 16 vectors form one tile, the next 16 form a second tile. dst_tile_stride
// is measured in FP16 elements: weight tiles are adjacent, whereas activation
// tiles are separated by a complete K tile row. No source padding is read.
TL_HMX_INLINE void tl_hmx_pack_transposed_32x64_f16(__fp16 *dst,
                                                    const __fp16 *src,
                                                    int row_stride,
                                                    size_t dst_tile_stride) {
  HVX_Vector v[32];
#pragma unroll
  for (int n = 0; n < 32; ++n)
    v[n] = *(const HVX_Vector *)(src + (size_t)n * row_stride);

  tl_hmx_transpose_pairs_stage_f16<1>(v);
  tl_hmx_transpose_pairs_stage_f16<2>(v);
  tl_hmx_transpose_pairs_stage_f16<4>(v);
  tl_hmx_transpose_pairs_stage_f16<8>(v);
  tl_hmx_transpose_pairs_stage_f16<16>(v);

#pragma unroll
  for (int kp = 0; kp < 32; ++kp) {
    // Butterfly register indices are the five-bit reversal of K-pair index.
    int index = ((kp & 1) << 4) | ((kp & 2) << 2) | (kp & 4) | ((kp & 8) >> 2) |
                ((kp & 16) >> 4);
    *(HVX_Vector *)(dst + (kp / 16) * dst_tile_stride + (kp % 16) * 64) =
        v[index];
  }
}

// Pack row-major W[N,K] to weight Crouton in caller-owned VTCM. A 64-half
// source load contains two K tiles. Register shuffles transpose packed pairs;
// every load stays inside the logical row, with no readable padding required.
// Noncontiguous, unaligned and 32-column remainder layouts retain the scalar
// path. This is a layout-copy primitive, not a GEMM schedule or worker task.
TL_HMX_INLINE void tl_hmx_pack_weight_transposed_strided_f16(__fp16 *dst,
                                                             const __fp16 *src,
                                                             int N, int K,
                                                             int row_stride,
                                                             int col_stride) {
  if (col_stride == 1 && K % 64 == 0 && row_stride >= K &&
      row_stride % 64 == 0 && ((uintptr_t)src & 127u) == 0 &&
      ((uintptr_t)dst & 127u) == 0) {
    for (int nt = 0; nt < N / 32; ++nt)
      for (int kc = 0; kc < K / 64; ++kc)
        tl_hmx_pack_transposed_32x64_f16(
            dst + ((size_t)nt * (K / 32) + 2 * kc) * TL_HMX_TILE_ELMS,
            src + (size_t)nt * 32 * row_stride + kc * 64, row_stride,
            TL_HMX_TILE_ELMS);
    return;
  }
  tl_hmx_pack_crouton_scalar_f16(dst, src, N, K, K / TL_HMX_T, row_stride,
                                 col_stride, 1);
}

// Pack transposed activation storage [K,M] into activation Crouton [mt,kt].
// A 32x64 source rectangle contributes two M tiles at the same K tile index.
// Unaligned, noncontiguous and non-64-wide sources retain the scalar fallback.
TL_HMX_INLINE void
tl_hmx_pack_activation_transposed_strided_f16(__fp16 *dst, const __fp16 *src,
                                              int K, int M, int row_stride,
                                              int col_stride) {
  const int k_tiles = K / TL_HMX_T;
  if (col_stride == 1 && M % 64 == 0 && row_stride >= M &&
      row_stride % 64 == 0 && ((uintptr_t)src & 127u) == 0 &&
      ((uintptr_t)dst & 127u) == 0) {
    for (int mc = 0; mc < M / 64; ++mc)
      for (int kt = 0; kt < k_tiles; ++kt)
        tl_hmx_pack_transposed_32x64_f16(
            dst + ((size_t)(2 * mc) * k_tiles + kt) * TL_HMX_TILE_ELMS,
            src + (size_t)kt * TL_HMX_T * row_stride + mc * 64, row_stride,
            (size_t)k_tiles * TL_HMX_TILE_ELMS);
    return;
  }
  tl_hmx_pack_crouton_scalar_f16(dst, src, M, K, k_tiles, col_stride,
                                 row_stride, 0);
}

// Layout kinds shared with src/hexagon/op/copy.cc. Public pack/unpack wrappers
// lower one logical T.copy. The row-major endpoint may be a strided DDR/VTCM
// matrix region; the Crouton endpoint is always a complete caller-owned VTCM
// buffer. A future DMA path can still stage a row-major region before
// invoking the same strided transform without changing this interface.
enum {
  TL_HMX_LAYOUT_ACTIVATION = 0,
  TL_HMX_LAYOUT_WEIGHT = 1,
  TL_HMX_LAYOUT_ACTIVATION_TRANSPOSED = 2,
  TL_HMX_LAYOUT_WEIGHT_TRANSPOSED = 3,
};

TL_HMX_INLINE void tl_hexagon_hmx_pack_crouton(__fp16 *dst, const __fp16 *src,
                                               int rows, int cols,
                                               int src_stride0, int src_stride1,
                                               int layout_kind) {
  switch (layout_kind) {
  case TL_HMX_LAYOUT_ACTIVATION:
    tl_hmx_pack_activation_strided_f16(dst, src, rows, cols, src_stride0,
                                       src_stride1);
    return;
  case TL_HMX_LAYOUT_WEIGHT:
    tl_hmx_pack_weight_strided_f16(dst, src, rows, cols, src_stride0,
                                   src_stride1);
    return;
  case TL_HMX_LAYOUT_ACTIVATION_TRANSPOSED:
    // Storage is [K,M], while the physical activation coordinates are [M,K].
    tl_hmx_pack_activation_transposed_strided_f16(dst, src, rows, cols,
                                                  src_stride0, src_stride1);
    return;
  case TL_HMX_LAYOUT_WEIGHT_TRANSPOSED:
    // Storage and physical outer coordinates are both [N,K].
    tl_hmx_pack_weight_transposed_strided_f16(dst, src, rows, cols, src_stride0,
                                              src_stride1);
    return;
  default:
    return;
  }
}

TL_HMX_INLINE void tl_hexagon_hmx_unpack_crouton(__fp16 *dst, const __fp16 *src,
                                                 int rows, int cols,
                                                 int dst_stride0,
                                                 int dst_stride1,
                                                 int layout_kind) {
  if (layout_kind == TL_HMX_LAYOUT_ACTIVATION) {
    tl_hmx_unpack_output_strided_f16(dst, src, rows, cols, dst_stride0,
                                     dst_stride1);
    return;
  }

  if (layout_kind == TL_HMX_LAYOUT_WEIGHT) {
    // Logical dst is weight[K,N]; physical src tiles are [nt,kt,cpos(k,n)].
    for (int k = 0; k < rows; ++k)
      for (int n = 0; n < cols; ++n)
        dst[(size_t)k * dst_stride0 + (size_t)n * dst_stride1] =
            src[tl_hmx_crouton_offset(n, k, rows / TL_HMX_T, 1)];
    return;
  }

  if (layout_kind == TL_HMX_LAYOUT_ACTIVATION_TRANSPOSED) {
    // Logical dst is activation_storage[K,M]; physical src coordinates are
    // [m,k].
    for (int k = 0; k < rows; ++k)
      for (int m = 0; m < cols; ++m)
        dst[(size_t)k * dst_stride0 + (size_t)m * dst_stride1] =
            src[tl_hmx_crouton_offset(m, k, rows / TL_HMX_T, 0)];
    return;
  }

  if (layout_kind == TL_HMX_LAYOUT_WEIGHT_TRANSPOSED) {
    // Logical dst is weight_storage[N,K]; physical src coordinates are [nt,kt].
    for (int n = 0; n < rows; ++n)
      for (int k = 0; k < cols; ++k)
        dst[(size_t)n * dst_stride0 + (size_t)k * dst_stride1] =
            src[tl_hmx_crouton_offset(n, k, cols / TL_HMX_T, 1)];
  }
}

// Fused numeric conversion and activation/output layout copy. These primitives
// visit row pairs, never a GEMM or reduction. Scalar fallback retains arbitrary
// source/destination strides; full 32-column row pairs use two HVX registers.
TL_HMX_INLINE void tl_hexagon_hmx_pack_crouton_f32(__fp16 *dst,
                                                   const float *src, int rows,
                                                   int cols, int row_stride,
                                                   int col_stride, int kind) {
  (void)kind; // Lowering only selects activation/output Crouton (kind 0).
  if (col_stride == 1) {
    for (int mt = 0; mt < rows / 32; ++mt)
      for (int kt = 0; kt < cols / 32; ++kt)
        for (int pair = 0; pair < 16; ++pair)
          tl_hexagon_hmx_pack_a_f32_pair_k32(
              dst + (size_t)(mt * (cols / 32) + kt) * 1024 + pair * 64,
              src + (size_t)(mt * 32 + pair * 2) * row_stride + kt * 32,
              src + (size_t)(mt * 32 + pair * 2 + 1) * row_stride + kt * 32);
    return;
  }
  for (int r = 0; r < rows; ++r)
    for (int c = 0; c < cols; ++c)
      dst[tl_hmx_crouton_offset(r, c, cols / 32, 0)] =
          (__fp16)src[(size_t)r * row_stride + (size_t)c * col_stride];
}

TL_HMX_INLINE void tl_hexagon_hmx_unpack_crouton_f32(float *dst,
                                                     const __fp16 *src,
                                                     int rows, int cols,
                                                     int row_stride,
                                                     int col_stride, int kind) {
  (void)kind;
  if (col_stride == 1) {
    for (int mt = 0; mt < rows / 32; ++mt)
      for (int nt = 0; nt < cols / 32; ++nt)
        for (int pair = 0; pair < 16; ++pair)
          tl_hexagon_hmx_unpack_c_f32_pair_n32(
              dst + (size_t)(mt * 32 + pair * 2) * row_stride + nt * 32,
              dst + (size_t)(mt * 32 + pair * 2 + 1) * row_stride + nt * 32,
              src + (size_t)(mt * (cols / 32) + nt) * 1024 + pair * 64);
    return;
  }
  for (int r = 0; r < rows; ++r)
    for (int c = 0; c < cols; ++c)
      dst[(size_t)r * row_stride + (size_t)c * col_stride] =
          (float)src[tl_hmx_crouton_offset(r, c, cols / 32, 0)];
}

// Synchronous parallel T.copy: workers partition disjoint 32x32/64 tiles in
// row-tile-major storage. It may overlap DMA on the caller's next input slot,
// but returns only after all stripes are visible. No DMA/HMX ownership, scratch
// allocation, or operator scheduling is hidden here. The codegen rejects nested
// pool dispatch; the caller owns both endpoints throughout this copy.
struct tl_hmx_copy_crouton_parallel_args {
  void *dst;
  const void *src;
  int rows, cols, row_stride, col_stride, kind, operation;
};

static int tl_hmx_copy_crouton_parallel_worker(void *opaque, int wid, int nw) {
  const auto &a = *static_cast<tl_hmx_copy_crouton_parallel_args *>(opaque);
  // Two adjacent K/N tiles preserve the FP16 packer's full-HVX source loads.
  // Partition both axes so short-M, long-K copies can use all workers too.
  const int tc = a.cols % 64 == 0 ? 64 : 32;
  const int columns = a.cols / tc;
  for (int tile = wid; tile < (a.rows / 32) * columns; tile += nw) {
    const int rt = tile / columns, ct = tile % columns;
    const size_t native_offset = (size_t)rt * 32 * a.cols + ct * tc * 32;
    const size_t row_offset =
        (size_t)rt * 32 * a.row_stride + (size_t)ct * tc * a.col_stride;
    if (a.operation == 0)
      tl_hexagon_hmx_pack_crouton(static_cast<__fp16 *>(a.dst) + native_offset,
                                  static_cast<const __fp16 *>(a.src) +
                                      row_offset,
                                  32, tc, a.row_stride, a.col_stride, a.kind);
    else if (a.operation == 1)
      tl_hexagon_hmx_unpack_crouton(static_cast<__fp16 *>(a.dst) + row_offset,
                                    static_cast<const __fp16 *>(a.src) +
                                        native_offset,
                                    32, tc, a.row_stride, a.col_stride, a.kind);
    else if (a.operation == 2)
      tl_hexagon_hmx_pack_crouton_f32(
          static_cast<__fp16 *>(a.dst) + native_offset,
          static_cast<const float *>(a.src) + row_offset, 32, tc, a.row_stride,
          a.col_stride, a.kind);
    else
      tl_hexagon_hmx_unpack_crouton_f32(
          static_cast<float *>(a.dst) + row_offset,
          static_cast<const __fp16 *>(a.src) + native_offset, 32, tc,
          a.row_stride, a.col_stride, a.kind);
  }
  return 0;
}

TL_HMX_INLINE void
tl_hexagon_hmx_copy_crouton_parallel(void *dst, const void *src, int rows,
                                     int cols, int row_stride, int col_stride,
                                     int kind, int operation, int workers) {
  tl_hmx_copy_crouton_parallel_args args{
      dst, src, rows, cols, row_stride, col_stride, kind, operation};
  int nw = tl_num_workers();
  if (nw > workers)
    nw = workers;
  int tiles = (rows / 32) * (cols / (cols % 64 == 0 ? 64 : 32));
  if (nw > tiles)
    nw = tiles;
  tl_parallel(tl_hmx_copy_crouton_parallel_worker, &args, nw);
}

// Persistent session resources: power/VTCM/HMX are acquired once and held
// across many matmuls. The generated FastRPC _open handler calls init; _close
// calls deinit. The one-shot runtime also initializes resources before kernel
// execution. NOTE: the check-then-set of these globals is NOT internally
// synchronized; it is safe because every caller invokes it single-threaded
// BEFORE tl_parallel spawns workers. A future persistent/cross-kernel pool that
// could init from two threads must add a lock here.
static int tl_hmx_inited;
static qurt_thread_t
    tl_hmx_session_tid; // thread whose HMX SHARED enable session_init holds
static void tl_hmx_session_init(void) {
  if (tl_hmx_inited)
    return;
  tl_hmx_power_setup();
  tl_hmx_vtcm_setup();
  tl_hmx_setup();
  // Require BOTH the VTCM grant AND a successful per-thread HMX enable (which
  // needs the ctx from tl_hmx_setup): otherwise tear everything down so
  // tl_hmx_session_ok() reports failure loudly, rather than marking the session
  // ready while enable was a silent no-op (ctx==0) and every later MAC runs
  // with HMX never locked.
  if (tl_vtcm_base_ptr && tl_hmx_enable() == 0) {
    // Initialize the reserved scale tile consistently with the embedding
    // bridge. Explicit atoms use their DSL-provided bias buffers and
    // accumulator locks.
    tl_hmx_fill_unit_scales((uint32_t *)tl_vtcm_base_ptr);
    tl_hmx_session_tid =
        qurt_thread_get_id(); // this enable is bound to this thread
    tl_hmx_inited = 1;
  } else {
    // VTCM acquire failed: release the HMX ctx and drop the rail off TURBO so a
    // failed init doesn't leak the unit + pin power for the agent's lifetime.
    tl_hmx_reset();
    tl_hmx_vtcm_reset();
    tl_hmx_power_reset();
  }
}
static void tl_hmx_session_deinit(void) {
  if (!tl_hmx_inited)
    return;
  tl_hmx_disable();
  tl_hmx_reset();
  tl_hmx_vtcm_reset();
  tl_hmx_power_reset();
  tl_hmx_inited = 0;
}

// Let the FastRPC runtime reject execution if VTCM/HMX acquisition failed.
static int tl_hmx_session_ok(void) { return tl_hmx_inited; }

// tl_parallel_for worker scope for tile programs that issue HMX: a pool thread
// enables HMX for itself around its programs. The session thread already holds
// the enable that tl_hmx_session_init took.
struct tl_hmx_worker_scope {
  int status = TL_OK;
  bool enabled = false;

  tl_hmx_worker_scope() {
    if (qurt_thread_get_id() == tl_hmx_session_tid)
      return;
    if (tl_hmx_enable() != 0)
      status = TL_ERR_HMX; // refuse rather than run MACs with HMX unlocked
    else
      enabled = true;
  }
  ~tl_hmx_worker_scope() {
    if (enabled)
      tl_hmx_disable();
  }
  tl_hmx_worker_scope(const tl_hmx_worker_scope &) = delete;
  tl_hmx_worker_scope &operator=(const tl_hmx_worker_scope &) = delete;
};

// ---- explicit HMX atoms: the DSL owns the K-loop and Crouton storage
// --------- These thin wrappers expose the accumulator protocol one instruction
// at a time. A DSL kernel decides where dequantization sits relative to
// clear/MAC/convert/ store and owns every Crouton buffer through alloc_shared.
// TILE = 32; one tile is 1024 fp16 = 2 KB. On one Q6 thread, a source tile is
// reusable after its final multiply packet; accumulator convert and output
// store do not access A/W. `tl_hexagon_hmx` prefix so runtime/fastrpc.py pulls
// in this header.
//
// The atom wrappers retain the dependency-only TIR state tokens as C
// parameters. Hardware state remains implicit; the tokens only expose ordering
// and ownership to compiler analyses.
TL_HMX_INLINE void tl_hexagon_hmx_acc_acquire(void *acc_state) {
  (void)acc_state;
  tl_hmx_unit_acquire();
}
TL_HMX_INLINE void tl_hexagon_hmx_acc_release(void *acc_state) {
  (void)acc_state;
  tl_hmx_unit_release();
}
TL_HMX_INLINE void tl_hexagon_hmx_load_bias(void *bias_state,
                                            const void *bias_vtcm) {
  (void)bias_state;
  tl_hmx_set_scales(bias_vtcm);
}
TL_HMX_INLINE void tl_hexagon_hmx_clear_acc(void *acc_state) {
  (void)acc_state;
  tl_hmx_clear_acc();
}
TL_HMX_INLINE void tl_hexagon_hmx_mma_atom(void *acc_state,
                                           const __fp16 *activation,
                                           const __fp16 *weight, size_t n = 1) {
  (void)acc_state;
  tl_hmx_mac_tiles(activation, weight, n);
}
TL_HMX_INLINE void tl_hexagon_hmx_convert_acc(void *cvt_state, void *acc_state,
                                              void *bias_state,
                                              const void *bias_source,
                                              int mode) {
  (void)cvt_state;
  (void)acc_state;
  (void)bias_state;
  // The source address is carried through TIR until conversion so the shared
  // memory planner cannot recycle it while mxmem2 is still in flight.
  (void)bias_source;
  tl_hmx_convert_acc(mode);
}
TL_HMX_INLINE void
tl_hexagon_hmx_store_cvt_state(void *cvt_state, __fp16 *output, void *acc_state,
                               void *bias_state, const void *bias_source) {
  (void)cvt_state;
  (void)acc_state;
  (void)bias_state;
  (void)bias_source;
  // Conversion is a separate TileLang atom. This instruction only transfers
  // the explicit HMX convert-state token to its Crouton VTCM destination.
  tl_hmx_store_cvt(output);
}
