#pragma once
// HMX FP16 matmul runtime for tilelang-generated Hexagon kernels.
//
// Provides `tl_hexagon_hmx_matmul_f16(C, A, B, M, N, K)`: a self-contained
// C[M,N] = A[M,K] * B[K,N] on the HMX matrix engine, faithfully distilled from
// the verified mini-htp recipe (power-up, VTCM acquire, HMX lock, Crouton
// tile pack/unpack, MAC over K-tiles).  M, N, K must be multiples of 32 and the
// tiled operands must fit in VTCM.  Resources are acquired/released per call
// (correctness first; hoisting setup is a perf concern).
//
// Compiled as part of the FastRPC skel (.cc) with the Hexagon SDK on the include
// path and -mhmx; it pulls in HAP headers, so it is only included by kernels
// that actually call into HMX.
#include <HAP_compute_res.h>
#include <HAP_farf.h>
#include <HAP_perf.h>
#include <HAP_power.h>
#include <hexagon_types.h> // HVX_Vector + Q6_* intrinsics (needs -mhvx)
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <tl_templates/hexagon/vtcm.h> // unified VTCM arena (shared with alloc_shared)

#define TL_HMX_INLINE static inline __attribute__((unused, always_inline))

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
// Delegated to the shared tl_vtcm arena (vtcm.h) so HMX scratch and alloc_shared
// operands draw from one acquisition — no double-acquire when a T.gemm kernel
// has its operands in VTCM and also drives HMX.
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
TL_HMX_INLINE void tl_hmx_enable(void) {
  if (tl_hmx_ctx)
    HAP_compute_res_hmx_lock2(tl_hmx_ctx, HAP_COMPUTE_RES_HMX_SHARED);
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
TL_HMX_INLINE void tl_hmx_unit_release(void) {
  *(volatile int *)&tl_hmx_spin = 0;
}

// ------------------------- the HMX matmul ---------------------------------
#define TL_HMX_T 32
#define TL_HMX_TILE_ELMS 1024
#define TL_HMX_TILE_BYTES 2048

TL_HMX_INLINE void tl_hmx_clear_acc(void) { asm volatile("mxclracc.hf"); }
TL_HMX_INLINE void tl_hmx_set_scales(const void *s) {
  asm volatile("bias = mxmem2(%0)" ::"r"(s));
}
// Fill a 256-byte HMX output-scale region: per-column scale = fp16 1.0 (u32 low
// half == 0x3c00), per-column bias = 0.  Single source for the u32 scale-word
// layout shared by all three HMX entry points (matmul / mac / bench).
TL_HMX_INLINE void tl_hmx_fill_unit_scales(uint32_t *s) {
  for (int i = 0; i < 32; ++i)
    s[i] = 0x00003c00u;
  for (int i = 32; i < 64; ++i)
    s[i] = 0u;
}
TL_HMX_INLINE void tl_hmx_mac_tiles(const __fp16 *a, const __fp16 *w, size_t n) {
  size_t lim = n * TL_HMX_TILE_BYTES - 1;
  asm volatile("{ activation.hf = mxmem(%0, %1):deep\n"
               "  weight.hf     = mxmem(%2, %3) }\n" ::"r"(a),
               "r"(lim), "r"(w), "r"(lim)
               : "memory");
}
TL_HMX_INLINE void tl_hmx_store_tile(__fp16 *o) {
  asm volatile("cvt.hf = acc(%0)\nmxmem(%1, %2) = cvt\n" ::"r"(2), "r"(o),
               "r"(0)
               : "memory");
}

// Crouton 32x32 tile layout: (row i, col j) -> (i&~1)*32 + j*2 + (i&1).
TL_HMX_INLINE int tl_hmx_cpos(int i, int j) {
  return (i & ~1) * 32 + j * 2 + (i & 1);
}
// Crouton's intra-tile layout IS a row-pair interleave, so one HVX `vshuff` of
// two source rows produces a whole 64-element row-pair span (lo half -> the even
// tile, hi half -> the next tile).  ~64x the scalar element-by-element pack.  The
// 64-column chunk spans 2 tiles, so the contiguous dim must be a 64-multiple;
// otherwise fall back to the scalar pack.
TL_HMX_INLINE void tl_hmx_pack_A(__fp16 *t, const __fp16 *A, int M, int K) {
  int KT = K / TL_HMX_T;
  if ((K & 63) == 0) {
    for (int mt = 0; mt < M / TL_HMX_T; ++mt)
      for (int r = 0; r < TL_HMX_T / 2; ++r) { // 16 row-pairs per tile
        const HVX_Vector *r0 =
            (const HVX_Vector *)(A + (size_t)(mt * TL_HMX_T + 2 * r) * K);
        const HVX_Vector *r1 =
            (const HVX_Vector *)(A + (size_t)(mt * TL_HMX_T + 2 * r + 1) * K);
        for (int kc = 0; kc < K / 64; ++kc) { // 64 cols == 2 k-tiles
          HVX_VectorPair vp = Q6_W_vshuff_VVR(r1[kc], r0[kc], -2);
          ((HVX_Vector *)(t + (size_t)(mt * KT + 2 * kc) * TL_HMX_TILE_ELMS))[r] =
              Q6_V_lo_W(vp);
          ((HVX_Vector *)(t + (size_t)(mt * KT + 2 * kc + 1) * TL_HMX_TILE_ELMS))[r] =
              Q6_V_hi_W(vp);
        }
      }
    return;
  }
  for (int m = 0; m < M; ++m)
    for (int k = 0; k < K; ++k)
      t[((m / TL_HMX_T) * KT + k / TL_HMX_T) * TL_HMX_TILE_ELMS +
        tl_hmx_cpos(m % TL_HMX_T, k % TL_HMX_T)] = A[m * K + k];
}
TL_HMX_INLINE void tl_hmx_pack_B(__fp16 *t, const __fp16 *B, int K, int N) {
  int KT = K / TL_HMX_T;
  if ((N & 63) == 0) { // col-tile-major: cpos(k,n) interleaves K row-pairs over N
    for (int kt = 0; kt < KT; ++kt)
      for (int r = 0; r < TL_HMX_T / 2; ++r) { // 16 k-row-pairs per k-tile
        const HVX_Vector *r0 =
            (const HVX_Vector *)(B + (size_t)(kt * TL_HMX_T + 2 * r) * N);
        const HVX_Vector *r1 =
            (const HVX_Vector *)(B + (size_t)(kt * TL_HMX_T + 2 * r + 1) * N);
        for (int nc = 0; nc < N / 64; ++nc) { // 64 cols == 2 n-tiles
          HVX_VectorPair vp = Q6_W_vshuff_VVR(r1[nc], r0[nc], -2);
          ((HVX_Vector *)(t + (size_t)((2 * nc) * KT + kt) * TL_HMX_TILE_ELMS))[r] =
              Q6_V_lo_W(vp);
          ((HVX_Vector *)(t + (size_t)((2 * nc + 1) * KT + kt) * TL_HMX_TILE_ELMS))[r] =
              Q6_V_hi_W(vp);
        }
      }
    return;
  }
  for (int k = 0; k < K; ++k)
    for (int n = 0; n < N; ++n)
      t[((n / TL_HMX_T) * KT + k / TL_HMX_T) * TL_HMX_TILE_ELMS +
        tl_hmx_cpos(k % TL_HMX_T, n % TL_HMX_T)] = B[k * N + n];
}
TL_HMX_INLINE void tl_hmx_unpack_C(__fp16 *C, const __fp16 *t, int M, int N) {
  int NT = N / TL_HMX_T;
  if ((N & 63) == 0) { // HVX: vdeal de-interleaves a row-pair span back to 2 rows
    for (int mt = 0; mt < M / TL_HMX_T; ++mt)
      for (int r = 0; r < TL_HMX_T / 2; ++r) {
        HVX_Vector *o0 = (HVX_Vector *)(C + (size_t)(mt * TL_HMX_T + 2 * r) * N);
        HVX_Vector *o1 = (HVX_Vector *)(C + (size_t)(mt * TL_HMX_T + 2 * r + 1) * N);
        for (int nc = 0; nc < N / 64; ++nc) {
          HVX_Vector lo = ((const HVX_Vector *)(t + (size_t)(mt * NT + 2 * nc) * TL_HMX_TILE_ELMS))[r];
          HVX_Vector hi = ((const HVX_Vector *)(t + (size_t)(mt * NT + 2 * nc + 1) * TL_HMX_TILE_ELMS))[r];
          HVX_VectorPair vp = Q6_W_vdeal_VVR(hi, lo, -2);
          o0[nc] = Q6_V_lo_W(vp);
          o1[nc] = Q6_V_hi_W(vp);
        }
      }
    return;
  }
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n)
      C[m * N + n] = t[((m / TL_HMX_T) * NT + n / TL_HMX_T) * TL_HMX_TILE_ELMS +
                       tl_hmx_cpos(m % TL_HMX_T, n % TL_HMX_T)];
}
// Scalar transposed packs (HVX would need strided/gathered source loads): A
// stored [K,M] (trans_a) / B stored [N,K] (trans_b).  Common gemms have
// trans_a=false; trans_b=true is the linear-layer / attention QKᵀ form.
TL_HMX_INLINE void tl_hmx_pack_A_T(__fp16 *t, const __fp16 *A, int M, int K) {
  int KT = K / TL_HMX_T;
  for (int m = 0; m < M; ++m)
    for (int k = 0; k < K; ++k)
      t[((m / TL_HMX_T) * KT + k / TL_HMX_T) * TL_HMX_TILE_ELMS +
        tl_hmx_cpos(m % TL_HMX_T, k % TL_HMX_T)] = A[(size_t)k * M + m];
}
TL_HMX_INLINE void tl_hmx_pack_B_T(__fp16 *t, const __fp16 *B, int K, int N) {
  int KT = K / TL_HMX_T;
  for (int k = 0; k < K; ++k)
    for (int n = 0; n < N; ++n)
      t[((n / TL_HMX_T) * KT + k / TL_HMX_T) * TL_HMX_TILE_ELMS +
        tl_hmx_cpos(k % TL_HMX_T, n % TL_HMX_T)] = B[(size_t)n * K + k];
}
TL_HMX_INLINE void tl_hmx_matmul_tiles(__fp16 *c, const __fp16 *a,
                                       const __fp16 *b, int M, int N, int K,
                                       const __fp16 *scales) {
  int MT = M / TL_HMX_T, NT = N / TL_HMX_T, KT = K / TL_HMX_T;
  tl_hmx_clear_acc();
  tl_hmx_set_scales(scales);
  for (int mt = 0; mt < MT; ++mt)
    for (int nt = 0; nt < NT; ++nt) {
      const __fp16 *ar = a + (mt * KT) * TL_HMX_TILE_ELMS;
      const __fp16 *bc = b + (nt * KT) * TL_HMX_TILE_ELMS;
      for (int kt = 0; kt < KT; ++kt)
        tl_hmx_mac_tiles(ar + kt * TL_HMX_TILE_ELMS, bc + kt * TL_HMX_TILE_ELMS,
                         1);
      tl_hmx_store_tile(c + (mt * NT + nt) * TL_HMX_TILE_ELMS);
    }
}

// Persistent session resources: power/VTCM/HMX are acquired once and held across
// many matmuls.  The generated FastRPC _open handler calls init; _close calls
// deinit.  In the per-call (one-shot) path the first matmul inits lazily.
static int tl_hmx_inited;
static void tl_hmx_session_init(void) {
  if (tl_hmx_inited)
    return;
  tl_hmx_power_setup();
  tl_hmx_vtcm_setup();
  tl_hmx_setup();
  if (tl_vtcm_base_ptr) {
    tl_hmx_enable();
    tl_hmx_unit_acquire();
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
  tl_hmx_unit_release();
  tl_hmx_disable();
  tl_hmx_reset();
  tl_hmx_vtcm_reset();
  tl_hmx_power_reset();
  tl_hmx_inited = 0;
}

// Did the session actually acquire VTCM/HMX?  The FastRPC _open uses this to
// fail loudly when (e.g.) a leaked agent still holds VTCM — otherwise the matmul
// returns -2, the generated kernel discards the rc, and the caller gets zeros.
static int tl_hmx_session_ok(void) { return tl_hmx_inited; }

// T.gemm (Level-1) entry: operands are ROW-MAJOR in VTCM (loaded by a fast,
// auto-vectorized T.copy).  HVX-pack them to a Crouton scratch, MAC, then HVX-
// unpack the result back to row-major C — so the surrounding copies AND elementwise
// ops (softmax) stay vectorizable.  The Crouton scratch grows TOP-DOWN from the end
// of VTCM while the row-major alloc_shared tiles grow bottom-up.  A Crouton tile
// holds the same element count as its row-major source, so the scratch is the SAME
// size as the operands and the two together need ~2x the operand bytes — a flat
// "scratch fits in VTCM" check is NOT enough.  We refuse (-3) if the scratch base
// would underflow into the live shared tiles: the codegen publishes their bottom
// high-water (tl_vtcm_shared_high_water), which covers the operands AND any sibling
// alloc_shared buffers, rather than silently corrupt them.
// trans_a/trans_b: operand stored transposed ([K,M] / [N,K]), scalar transposed
// pack.  Returns 0 on ok.
static int tl_hexagon_hmx_gemm(__fp16 *C, const __fp16 *A, const __fp16 *B, int M,
                               int N, int K, int trans_a, int trans_b) {
  if (M <= 0 || N <= 0 || K <= 0 || (M % TL_HMX_T) || (N % TL_HMX_T) ||
      (K % TL_HMX_T))
    return -1;
  tl_hmx_session_init(); // power + HMX lock + ensure VTCM (all idempotent)
  if (!tl_vtcm_base_ptr)
    return -2;
  size_t a_sz = (size_t)(M / TL_HMX_T) * (K / TL_HMX_T) * TL_HMX_TILE_ELMS;
  size_t b_sz = (size_t)(N / TL_HMX_T) * (K / TL_HMX_T) * TL_HMX_TILE_ELMS;
  size_t c_sz = (size_t)(M / TL_HMX_T) * (N / TL_HMX_T) * TL_HMX_TILE_ELMS;
  // Arena top, floored to a 2KB Crouton tile so the scratch tiles stay 128-byte
  // (HVX) aligned even if the granted VTCM total isn't tile-aligned.
  size_t top = (size_t)tl_vtcm_total & ~(size_t)(TL_HMX_TILE_BYTES - 1);
  __fp16 *c_t = (__fp16 *)(tl_vtcm_base_ptr + top) - c_sz;
  __fp16 *b_t = c_t - b_sz;
  __fp16 *a_t = b_t - a_sz;
  // Refuse if the scratch base would underflow into the live shared tiles.  The
  // codegen-published bottom high-water covers the operands and any sibling
  // alloc_shared buffers; the scale tile at base+0 is always reserved.  Catches the
  // large gemm the old scratch-only check missed (e.g. M=N=K>=864 on 8MB VTCM,
  // where operands+scratch exceed VTCM though the scratch alone fits).
  const uint8_t *floor = tl_vtcm_base_ptr + tl_vtcm_shared_high_water;
  if (floor < tl_vtcm_base_ptr + TL_HMX_TILE_BYTES)
    floor = tl_vtcm_base_ptr + TL_HMX_TILE_BYTES;
  if ((const uint8_t *)a_t < floor)
    return -3;
  uint32_t *scales = (uint32_t *)tl_vtcm_base_ptr; // reserved first tile
  tl_hmx_fill_unit_scales(scales);
  if (trans_a)
    tl_hmx_pack_A_T(a_t, A, M, K);
  else
    tl_hmx_pack_A(a_t, A, M, K);
  if (trans_b)
    tl_hmx_pack_B_T(b_t, B, K, N);
  else
    tl_hmx_pack_B(b_t, B, K, N);
  tl_hmx_matmul_tiles(c_t, a_t, b_t, M, N, K, (const __fp16 *)scales);
  tl_hmx_unpack_C(C, c_t, M, N);
  return 0;
}

// Public entry called by generated kernels.  Returns 0 on success.  Acquires the
// session lazily if _open didn't; teardown belongs to the session (at _close).
static int tl_hexagon_hmx_matmul_f16(__fp16 *C, const __fp16 *A,
                                     const __fp16 *B, int M, int N, int K) {
  if (M <= 0 || N <= 0 || K <= 0 || (M % TL_HMX_T) || (N % TL_HMX_T) ||
      (K % TL_HMX_T))
    return -1;

  tl_hmx_session_init();
  if (!tl_vtcm_base_ptr)
    return -2;

  // VTCM arena: A tiles | B tiles | C tiles | scales(256B).
  size_t a_sz = (size_t)(M / TL_HMX_T) * (K / TL_HMX_T) * TL_HMX_TILE_ELMS;
  size_t b_sz = (size_t)(N / TL_HMX_T) * (K / TL_HMX_T) * TL_HMX_TILE_ELMS;
  size_t c_sz = (size_t)(M / TL_HMX_T) * (N / TL_HMX_T) * TL_HMX_TILE_ELMS;
  // Bail before writing if the arena won't fit, else pack_* overruns VTCM.
  if ((a_sz + b_sz + c_sz) * sizeof(__fp16) + 256u > (size_t)tl_vtcm_total)
    return -3;
  __fp16 *a_t = (__fp16 *)tl_vtcm_base_ptr;
  __fp16 *b_t = a_t + a_sz;
  __fp16 *c_t = b_t + b_sz;
  uint32_t *scales = (uint32_t *)(c_t + c_sz);
  tl_hmx_fill_unit_scales(scales);

  tl_hmx_pack_A(a_t, A, M, K);
  tl_hmx_pack_B(b_t, B, K, N);
  tl_hmx_matmul_tiles(c_t, a_t, b_t, M, N, K, (const __fp16 *)scales);
  tl_hmx_unpack_C(C, c_t, M, N);
  return 0;
}

// Benchmark variant: acquire + Crouton-pack once, then time `iters` HMX matmuls
// (the matrix-engine compute only, excluding setup/pack/unpack), and write
// timing[0] = microseconds/iter, timing[1] = GFLOPS.  C holds the correct
// result of the final iteration.  Measured on the DSP via qtimer, so the
// number reflects on-device compute, not the adb/RPC round trip.
static int tl_hexagon_hmx_matmul_f16_bench(__fp16 *C, const __fp16 *A,
                                           const __fp16 *B, int M, int N, int K,
                                           int iters, float *timing) {
  if (iters < 1)
    iters = 1;
  if (M <= 0 || N <= 0 || K <= 0 || (M % TL_HMX_T) || (N % TL_HMX_T) ||
      (K % TL_HMX_T))
    return -1;

  tl_hmx_power_setup();
  tl_hmx_vtcm_setup();
  tl_hmx_setup();
  if (!tl_vtcm_base_ptr) {
    tl_hmx_reset();
    tl_hmx_vtcm_reset();
    tl_hmx_power_reset();
    return -2;
  }

  size_t a_sz = (size_t)(M / TL_HMX_T) * (K / TL_HMX_T) * TL_HMX_TILE_ELMS;
  size_t b_sz = (size_t)(N / TL_HMX_T) * (K / TL_HMX_T) * TL_HMX_TILE_ELMS;
  size_t c_sz = (size_t)(M / TL_HMX_T) * (N / TL_HMX_T) * TL_HMX_TILE_ELMS;
  if ((a_sz + b_sz + c_sz) * sizeof(__fp16) + 256u > (size_t)tl_vtcm_total) {
    tl_hmx_reset();
    tl_hmx_vtcm_reset();
    tl_hmx_power_reset();
    return -3;
  }
  __fp16 *a_t = (__fp16 *)tl_vtcm_base_ptr;
  __fp16 *b_t = a_t + a_sz;
  __fp16 *c_t = b_t + b_sz;
  uint32_t *scales = (uint32_t *)(c_t + c_sz);
  tl_hmx_fill_unit_scales(scales);

  tl_hmx_enable();
  tl_hmx_unit_acquire();
  tl_hmx_pack_A(a_t, A, M, K);
  tl_hmx_pack_B(b_t, B, K, N);

  unsigned long long t0 = HAP_perf_get_qtimer_count();
  for (int it = 0; it < iters; ++it)
    tl_hmx_matmul_tiles(c_t, a_t, b_t, M, N, K, (const __fp16 *)scales);
  unsigned long long t1 = HAP_perf_get_qtimer_count();

  tl_hmx_unpack_C(C, c_t, M, N);
  tl_hmx_unit_release();
  tl_hmx_disable();
  tl_hmx_reset();
  tl_hmx_vtcm_reset();
  tl_hmx_power_reset();

  double total_us = (double)HAP_perf_qtimer_count_to_us(t1 - t0);
  double us = total_us / (double)iters;
  double gflops = (us > 0.0)
                      ? (2.0 * (double)M * (double)N * (double)K) / (us * 1000.0)
                      : 0.0;
  if (timing) {
    timing[0] = (float)us;
    timing[1] = (float)gflops;
  }
  FARF(ALWAYS, "HMX bench %dx%dx%d x%d iters: %.3f us/iter, %.1f GFLOPS", M, N, K,
       iters, us, gflops);
  return 0;
}
