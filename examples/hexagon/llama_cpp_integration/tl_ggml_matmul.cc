// tilelang q4_0 matmul, embedded in ggml-hexagon via the tilelang embedding API.
//
// This TU also hosts the (single, shared) op registry for now.  A tilelang op is
// a `tl_op_desc` (matches + run) that self-registers; the host's stock kernel
// only builds a `tl_op_ctx` and calls tl_dispatch() — it never names this op.
// Adding another kernel = another self-contained .cc with its own descriptor.
// See docs/llama_cpp_integration.md.
#include <tl_templates/hexagon/tl_embed.h>   // C-safe op ctx + registry API
#include <tl_templates/hexagon/tl_bridge.h>  // tl_bridge_enter/exit (+ hmx.h runtime)
#include <HAP_compute_res.h>
#include "htp-ops.h"                          // HTP_TYPE_Q4_0
#include <stdint.h>

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

// --------------------------- q4_0 dequant ------------------------------------
// Dequant repacked q4_0 weight columns [n0, n0+nc) (all K) -> Wf[k, nc] row-major.
// Tile(ct,kt) = weight + (ct*nkt + kt)*576;  byte[cp*32+row] = (q[2cp+1]<<4)|q[2cp];
// fp16 scale[row] at tile+512.  low nibble -> K=2cp, high -> K=2cp+1.
static void tl_dequant_q4_0_chunk(const uint8_t *weight, __fp16 *Wf,
                                  int k, int n0, int nc, int nkt) {
  const int ct0 = n0 / 32;
  const int nct = nc / 32;
  for (int ctl = 0; ctl < nct; ++ctl) {
    const int ct = ct0 + ctl;
    for (int kt = 0; kt < nkt; ++kt) {
      const uint8_t *tile   = weight + (size_t)(ct * nkt + kt) * 576;
      const __fp16  *scales = (const __fp16 *)(tile + 512);
      for (int cp = 0; cp < 16; ++cp) {
        const uint8_t *br     = tile + cp * 32;
        __fp16        *lo_row = Wf + (size_t)(kt * 32 + 2 * cp) * nc + ctl * 32;
        __fp16        *hi_row = Wf + (size_t)(kt * 32 + 2 * cp + 1) * nc + ctl * 32;
        for (int row = 0; row < 32; ++row) {
          const uint8_t b = br[row];
          const float   s = (float)scales[row];
          lo_row[row] = (__fp16)(((float)(b & 0x0F) - 8.0f) * s);
          hi_row[row] = (__fp16)(((float)(b >> 4) - 8.0f) * s);
        }
      }
    }
  }
}

// --------------------------- the matmul op -----------------------------------
static int tl_mm_matches(const struct tl_op_ctx *o) {
  return o->kind == TL_OP_MATMUL && o->weight_type == HTP_TYPE_Q4_0 && !o->has_bias &&
         o->vtcm_base && o->vtcm_rctx && o->m <= 32 && (o->k % 32) == 0 && (o->n % 32) == 0;
}

static int tl_mm_run(const struct tl_op_ctx *o) {
  const int k = o->k, n = o->n, m = o->m, nkt = k / 32;

  const size_t budget = (size_t)o->vtcm_size - 8192;
  const size_t af = (size_t)32 * k * 2;
  if (budget / 2 <= af) return -1;
  int NC = (int)(((budget / 2) - af) / ((size_t)k * 2 + 64));
  NC -= (NC % 32);
  if (NC < 32) return -1;
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
    tl_dequant_q4_0_chunk(o->weight, Wf, k, n0, nc, nkt);
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
