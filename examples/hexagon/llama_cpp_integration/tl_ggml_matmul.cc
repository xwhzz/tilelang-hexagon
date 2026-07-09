// tilelang q4_0 matmul, embedded in ggml-hexagon via the tilelang embedding API.
//
// FAST PATH: the per-call weight dequant is the tilelang-GENERATED whole-register
// HVX dequant (not the earlier hand-C scalar loop). It needs the weight column-major
// (qcm[K/2][N] + compact sc[K/32][N]); ggml's repacked-tile q4_0 format is 32-feature
// tiles (sub-register), so we repack it to column-major ONCE per weight (cached) and
// then the whole-register dequant + the reused HMX gemm run per call.
// See docs/llama_cpp_integration.md and docs/hexagon_dsl_kernels.md.
#include <tl_templates/hexagon/common.h>     // HVX vector types (half128, int16_t128, uint8_t128)
#include <tl_templates/hexagon/tl_embed.h>   // C-safe op ctx + registry API
#include <tl_templates/hexagon/tl_bridge.h>  // tl_bridge_enter/exit (+ hmx.h runtime)
#include <HAP_compute_res.h>
#include "htp-ops.h"                          // HTP_TYPE_Q4_0
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// --------------------------- op registry (one TU owns the table) -------------
int tl_mm_enabled = 0;  // master A/B toggle: 0 = all stock, 1 = tilelang ops eligible

#define TL_MAX_OPS 16
static const struct tl_op_desc *g_tl_ops[TL_MAX_OPS];
static int g_tl_nops = 0;

extern "C" void tl_register_op(const struct tl_op_desc *d) {
  if (g_tl_nops < TL_MAX_OPS) g_tl_ops[g_tl_nops++] = d;
}
extern "C" int tl_dispatch(const struct tl_op_ctx *octx) {
  if (!tl_mm_enabled) return -1;
  for (int i = 0; i < g_tl_nops; ++i) {
    if (g_tl_ops[i]->matches(octx) && g_tl_ops[i]->run(octx) == 0) return 0;
  }
  return -1;
}

// --------------------------- one-time repack (ggml tile -> column-major) ------
// ggml Tile(ct,kt) at (ct*nkt+kt)*576; byte[cp*32+row] packs K=kt*32+2cp (low) and
// +1 (high) for feature ct*32+row; fp16 scale[row] (block kt) at tile+512+row*2.
// Column-major: qcm[j][n] (j=K/2, n=feature) is that byte at kt=j/16,cp=j%16;
// sc[blk][n] (blk=K/32) is the block scale. (Byte-math validated off-device, err=0.)
static void tl_repack_q4_0(const uint8_t *w, uint8_t *qcm, __fp16 *sc, int K, int N) {
  const int KH = K / 2, nkt = K / 32;
  for (int n = 0; n < N; ++n) {
    const int ct = n / 32, row = n % 32;
    for (int j = 0; j < KH; ++j)
      qcm[(size_t)j * N + n] = w[(size_t)(ct * nkt + (j >> 4)) * 576 + (j & 15) * 32 + row];
    for (int blk = 0; blk < nkt; ++blk)
      sc[(size_t)blk * N + n] = *(const __fp16 *)(w + (size_t)(ct * nkt + blk) * 576 + 512 + row * 2);
  }
}

// small resident cache: ggml weight ptr -> repacked {qcm, sc}. Bounded by a byte
// budget; if malloc fails or the budget is hit, decline -> the stock kernel runs.
#define TL_CACHE_MAX 128
static struct { const uint8_t *w; uint8_t *qcm; __fp16 *sc; } g_cache[TL_CACHE_MAX];
static int g_ncache = 0;
static size_t g_cache_bytes = 0;
static const size_t TL_CACHE_BUDGET = (size_t)900 * 1024 * 1024;  // ~0.9 GB

static int tl_get_repacked(const uint8_t *w, int K, int N, uint8_t **qcm, __fp16 **sc) {
  for (int i = 0; i < g_ncache; ++i)
    if (g_cache[i].w == w) { *qcm = g_cache[i].qcm; *sc = g_cache[i].sc; return 0; }
  if (g_ncache >= TL_CACHE_MAX) return -1;
  const size_t qb = (size_t)(K / 2) * N, sb = (size_t)(K / 32) * N * sizeof(__fp16);
  if (g_cache_bytes + qb + sb > TL_CACHE_BUDGET) return -1;
  uint8_t *q = (uint8_t *)memalign(256, qb);
  __fp16 *s = (__fp16 *)memalign(256, sb);
  if (!q || !s) { free(q); free(s); return -1; }
  tl_repack_q4_0(w, q, s, K, N);
  g_cache[g_ncache].w = w; g_cache[g_ncache].qcm = q; g_cache[g_ncache].sc = s;
  g_ncache++; g_cache_bytes += qb + sb;
  *qcm = q; *sc = s;
  return 0;
}

// --------------------------- generated whole-register HVX dequant -------------
// The tilelang-generated body (see emit_embeddable.py), parameterized over K/N: read
// column-major qcm/sc, write Wf[K][nc] row-major. 128-wide = one whole HVX register.
static void tl_dequant_wholereg(const uint8_t *qcm, const __fp16 *sc, __fp16 *Wf,
                                int KH, int Nfull, int n0, int nc) {
  for (int j = 0; j < KH; ++j) {
    const uint8_t *qrow = qcm + (size_t)j * Nfull + n0;
    const __fp16 *srow = sc + (size_t)(j >> 4) * Nfull + n0;   // block j/16
    __fp16 *lo = Wf + (size_t)(2 * j) * nc;
    __fp16 *hi = Wf + (size_t)(2 * j + 1) * nc;
    for (int no = 0; no < nc; no += 128) {
      int16_t128 q = (int16_t128)(*(uint8_t128 *)(qrow + no));
      half128 s = *(half128 *)((__fp16 *)srow + no);
      *(half128 *)(lo + no) = (half128)((q & (int16_t128)15) - (int16_t128)8) * s;
      *(half128 *)(hi + no) = (half128)((q >> (int16_t128)4) - (int16_t128)8) * s;
    }
  }
}

// --------------------------- the matmul op -----------------------------------
static int tl_mm_matches(const struct tl_op_ctx *o) {
  return o->kind == TL_OP_MATMUL && o->weight_type == HTP_TYPE_Q4_0 && !o->has_bias &&
         o->vtcm_base && o->vtcm_rctx && o->m <= 32 && (o->k % 32) == 0 && (o->n % 128) == 0;
}

static int tl_mm_run(const struct tl_op_ctx *o) {
  const int k = o->k, n = o->n, m = o->m;

  uint8_t *qcm; __fp16 *sc;
  if (tl_get_repacked((const uint8_t *)o->weight, k, n, &qcm, &sc) != 0) return -1;

  const size_t budget = (size_t)o->vtcm_size - 8192;
  const size_t af = (size_t)32 * k * 2;
  if (budget / 2 <= af) return -1;
  int NC = (int)(((budget / 2) - af) / ((size_t)k * 2 + 64));
  NC -= (NC % 128);
  if (NC < 128) return -1;
  if (NC > n) NC = n;

  __fp16 *Af = (__fp16 *)((uint8_t *)o->vtcm_base + 2048);
  __fp16 *Wf = Af + (size_t)32 * k;
  __fp16 *Cf = Wf + (size_t)k * NC;

  HAP_compute_res_hmx_lock(o->vtcm_rctx);       // hmx_mm_2d_f32 top: HMX not yet locked
  tl_bridge_enter(o->vtcm_base, o->vtcm_size);

  for (int i = 0; i < 32; ++i)
    for (int j = 0; j < k; ++j)
      Af[(size_t)i * k + j] = (i < m) ? (__fp16)o->act[(size_t)i * o->act_stride + j] : (__fp16)0.0f;
  tl_vtcm_shared_high_water = (unsigned)((uint8_t *)(Cf + (size_t)32 * NC) - (uint8_t *)o->vtcm_base);

  int rc = 0;
  for (int n0 = 0; n0 < n && rc == 0; n0 += NC) {
    const int nc = (n - n0 < NC) ? (n - n0) : NC;
    tl_dequant_wholereg(qcm, sc, Wf, k / 2, n, n0, nc);   // GENERATED whole-register dequant
    rc = tl_hexagon_hmx_gemm(Cf, Af, Wf, 32, nc, k, 0, 0);
    if (rc == 0)
      for (int i = 0; i < m; ++i)
        for (int j = 0; j < nc; ++j)
          o->dst[(size_t)i * o->dst_stride + n0 + j] = (float)Cf[(size_t)i * nc + j];
  }

  tl_bridge_exit();
  HAP_compute_res_hmx_unlock(o->vtcm_rctx);
  return rc == 0 ? 0 : -1;
}

static const struct tl_op_desc tl_mm_desc = { "q4_0_matmul", tl_mm_matches, tl_mm_run };
__attribute__((constructor)) static void tl_mm_register(void) { tl_register_op(&tl_mm_desc); }
