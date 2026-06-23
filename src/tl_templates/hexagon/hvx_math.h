#pragma once
// HVX math primitives for tilelang-generated Hexagon kernels.
//
// These implement the `map` and `reduce` halves of the backend's primitive
// basis {gemm, copy, map, reduce}.  `gemm` runs on HMX (hmx.h) and `copy` is
// the codegen's half8 path; everything pointwise/reductive (softmax, layernorm,
// rmsnorm, gelu/silu, bias, residual) composes from the small set here:
//
//   transcendental lane funcs  : exp2, recip, rsqrt        (the irreducible HW)
//   row reductions             : max, sum                  (over a contiguous row)
//
// All compute happens in fp32 (32 lanes / HVX vector) for accuracy; fp16 rows
// are widened on the way in and narrowed on the way out.  Activations are
// base-2 (`exp2`) so softmax can fold log2e into its score scale and run with no
// per-element base conversion (FlashAttention-2 style).  No VTCM LUT is needed:
// `exp2` is a degree-5 minimax polynomial (≈3e-6 rel err, fp16-exact after
// rounding) plus integer exponent construction.
//
// Pulled in on demand by _fastrpc.py when a kernel references `tl_hvx`.
#include <hexagon_types.h> // HVX_Vector / HVX_VectorPair + Q6_* intrinsics (needs -mhvx)
#include <cmath>           // scalar fallbacks for the ragged tail
#include <stdint.h>

#ifndef TL_DEVICE
#define TL_DEVICE static inline __attribute__((always_inline))
#endif

// HVX vector geometry: 1024-bit register = 32 fp32 lanes = 64 fp16 lanes.
#define TL_HVX_F32_LANES 32
#define TL_HVX_F16_LANES 64

TL_DEVICE int tl_hvx_aligned_128(const void *p) {
  return (((uintptr_t)p) & 127u) == 0;
}

// ---------------------------------------------------------------------------
// fp32 (IEEE "sf") arithmetic.  HVX accumulates in the internal "qf32" format;
// these thin wrappers keep callers in IEEE sf and hide the qf32 round-trip.
// ---------------------------------------------------------------------------
TL_DEVICE HVX_Vector tl_hvx_splat_f(float c) {
  int b;
  __builtin_memcpy(&b, &c, 4);
  return Q6_V_vsplat_R(b);
}
TL_DEVICE HVX_Vector tl_hvx_add_sf(HVX_Vector a, HVX_Vector b) {
  return Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(a, b));
}
TL_DEVICE HVX_Vector tl_hvx_sub_sf(HVX_Vector a, HVX_Vector b) {
  return Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(a, b));
}
TL_DEVICE HVX_Vector tl_hvx_mul_sf(HVX_Vector a, HVX_Vector b) {
  return Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(a, b));
}

// Unaligned 1024-bit load/store.  The shared-memory merge pass packs VTCM tiles
// at element-aligned (not 128-byte) offsets, so the codegen's elementwise
// vectorizer must not assume HVX alignment.  `aligned(1)` makes hexagon-clang
// emit the unaligned vmemu form.
typedef HVX_Vector tl_hvx_uvector __attribute__((aligned(1)));
TL_DEVICE HVX_Vector tl_hvx_loadu(const void *p) {
  return *(const tl_hvx_uvector *)p;
}
TL_DEVICE void tl_hvx_storeu(void *p, HVX_Vector v) {
  *(tl_hvx_uvector *)p = v;
}

// ---------------------------------------------------------------------------
// exp2 over 32 fp32 lanes: 2^x = 2^k · 2^f, k = round(x), f ∈ [-0.5, 0.5].
//   2^f : degree-5 minimax polynomial (Horner, accumulated in qf32)
//   2^k : construct the fp32 bit pattern (k + 127) << 23  (reinterpret is free)
// x is clamped to [-126, 126] so the exponent stays representable (very negative
// inputs flush toward 0, which is what softmax wants).
// ---------------------------------------------------------------------------
TL_DEVICE HVX_Vector tl_hvx_exp2_vsf(HVX_Vector x) {
  x = Q6_Vsf_vmin_VsfVsf(Q6_Vsf_vmax_VsfVsf(x, tl_hvx_splat_f(-126.0f)),
                         tl_hvx_splat_f(126.0f));
  // k = round-to-nearest(x), f = x - k ∈ [-0.5, 0.5].  Q6_Vw_equals_Vsf truncates
  // TOWARD ZERO, so bias by copysign(0.5, x) before the convert — otherwise f
  // lands in (-1, 1) and the degree-5 minimax poly (fit on [-0.5, 0.5]) is
  // evaluated outside its domain (~100x worse: 3e-6 -> ~3e-4).
  HVX_Vector half_signed = Q6_V_vor_VV(
      Q6_V_vand_VV(x, Q6_V_vsplat_R((int)0x80000000)), tl_hvx_splat_f(0.5f));
  HVX_Vector ki = Q6_Vw_equals_Vsf(tl_hvx_add_sf(x, half_signed)); // trunc(x ± 0.5)
  HVX_Vector kf = Q6_Vsf_equals_Vw(ki);    // back to float for the residual
  HVX_Vector f = tl_hvx_sub_sf(x, kf);     // f ∈ [-0.5, 0.5]
  // 2^f via Horner in qf32:  ((((C5 f + C4) f + C3) f + C2) f + C1) f + C0
  HVX_Vector fq = Q6_Vqf32_vadd_VsfVsf(f, Q6_V_vzero()); // f as qf32
  HVX_Vector acc = Q6_Vqf32_vadd_VsfVsf(tl_hvx_splat_f(0.0013333558f), Q6_V_vzero());
  acc = Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_Vqf32Vqf32(acc, fq), tl_hvx_splat_f(0.0096181291f));
  acc = Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_Vqf32Vqf32(acc, fq), tl_hvx_splat_f(0.0555041087f));
  acc = Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_Vqf32Vqf32(acc, fq), tl_hvx_splat_f(0.2402265069f));
  acc = Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_Vqf32Vqf32(acc, fq), tl_hvx_splat_f(0.6931471806f));
  acc = Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_Vqf32Vqf32(acc, fq), tl_hvx_splat_f(1.0000000000f));
  HVX_Vector p = Q6_Vsf_equals_Vqf32(acc); // 2^f
  // 2^k = reinterpret((k + 127) << 23)
  HVX_Vector e = Q6_Vw_vasl_VwR(Q6_Vw_vadd_VwVw(ki, Q6_V_vsplat_R(127)), 23);
  return tl_hvx_mul_sf(p, e);
}

// ---------------------------------------------------------------------------
// fp16 <-> fp32 lane conversions.
//   widen : 64 fp16 -> a pair of 32-lane fp32 vectors (lo = lanes 0..31)
//   narrow: a pair of 32-lane fp32 vectors -> 64 fp16
// ---------------------------------------------------------------------------
TL_DEVICE void tl_hvx_widen_hf(HVX_Vector h, HVX_Vector *lo, HVX_Vector *hi) {
  HVX_VectorPair w = Q6_Wsf_vcvt_Vhf(h);
  *lo = Q6_V_lo_W(w);
  *hi = Q6_V_hi_W(w);
}
TL_DEVICE HVX_Vector tl_hvx_narrow_hf(HVX_Vector lo, HVX_Vector hi) {
  // vcvt.hf.sf / vcvt.sf.hf interleave even/odd lanes; pairing (lo, hi) here is
  // the exact inverse of the Q6_Wsf_vcvt_Vhf split above (device-verified — the
  // swapped order produces adjacent-lane transposition).
  return Q6_Vhf_vcvt_VsfVsf(lo, hi);
}

// ---------------------------------------------------------------------------
// Row map: out[j] = exp2(in[j] - bias), contiguous fp16 row of length n.
// `bias` is the per-row scalar (the running max in softmax; 0 for a plain
// exp2).  Inputs are assumed base-2 already (fold log2e into the score scale).
// The row base is assumed HVX-aligned (VTCM tiles whose width is a multiple of
// 64 fp16 satisfy this); the ragged tail < 64 falls back to scalar.
// ---------------------------------------------------------------------------
TL_DEVICE void tl_hvx_exp2_bias_row(__fp16 *out, const __fp16 *in, float bias, int n) {
  if (!tl_hvx_aligned_128(in) || !tl_hvx_aligned_128(out)) {
    for (int j = 0; j < n; ++j) out[j] = (__fp16)exp2f((float)in[j] - bias);
    return;
  }
  HVX_Vector vb = tl_hvx_splat_f(bias);
  int nv = n / TL_HVX_F16_LANES;
  const HVX_Vector *vin = (const HVX_Vector *)in;
  HVX_Vector *vout = (HVX_Vector *)out;
  for (int v = 0; v < nv; ++v) {
    HVX_Vector lo, hi;
    tl_hvx_widen_hf(vin[v], &lo, &hi);
    lo = tl_hvx_exp2_vsf(tl_hvx_sub_sf(lo, vb));
    hi = tl_hvx_exp2_vsf(tl_hvx_sub_sf(hi, vb));
    vout[v] = tl_hvx_narrow_hf(lo, hi);
  }
  for (int j = nv * TL_HVX_F16_LANES; j < n; ++j)
    out[j] = (__fp16)exp2f((float)in[j] - bias);
}

// ---------------------------------------------------------------------------
// recip / rsqrt over 32 fp32 lanes.  HVX has no hardware seed, so we start from
// the classic bit-pattern approximation and refine with Newton-Raphson (3 steps
// → ~fp32 accuracy, far more than fp16 output needs).
// ---------------------------------------------------------------------------
TL_DEVICE HVX_Vector tl_hvx_recip_vsf(HVX_Vector x) {
  HVX_Vector y = Q6_Vw_vsub_VwVw(Q6_V_vsplat_R(0x7EF127EA), x); // seed ≈ 1/x
  HVX_Vector two = tl_hvx_splat_f(2.0f);
  for (int it = 0; it < 3; ++it) // y = y·(2 − x·y)
    y = tl_hvx_mul_sf(y, tl_hvx_sub_sf(two, tl_hvx_mul_sf(x, y)));
  return y;
}
TL_DEVICE HVX_Vector tl_hvx_rsqrt_vsf(HVX_Vector x) {
  HVX_Vector xhalf = tl_hvx_mul_sf(x, tl_hvx_splat_f(0.5f));
  HVX_Vector y = Q6_Vw_vsub_VwVw(Q6_V_vsplat_R(0x5F3759DF), Q6_Vw_vasr_VwR(x, 1)); // fast inv-sqrt seed
  HVX_Vector c15 = tl_hvx_splat_f(1.5f);
  for (int it = 0; it < 3; ++it) // y = y·(1.5 − x/2·y²)
    y = tl_hvx_mul_sf(y, tl_hvx_sub_sf(c15, tl_hvx_mul_sf(xhalf, tl_hvx_mul_sf(y, y))));
  return y;
}

// Row maps: out[j] = f(in[j]); fp16 row, fp32 internal.  `tl_hvx_rsqrt_eps_row`
// adds eps before the rsqrt (LayerNorm/RMSNorm variance term).
TL_DEVICE void tl_hvx_recip_row(__fp16 *out, const __fp16 *in, int n) {
  if (!tl_hvx_aligned_128(in) || !tl_hvx_aligned_128(out)) {
    for (int j = 0; j < n; ++j) out[j] = (__fp16)(1.0f / (float)in[j]);
    return;
  }
  int nv = n / TL_HVX_F16_LANES;
  const HVX_Vector *vin = (const HVX_Vector *)in;
  HVX_Vector *vout = (HVX_Vector *)out;
  for (int v = 0; v < nv; ++v) {
    HVX_Vector lo, hi;
    tl_hvx_widen_hf(vin[v], &lo, &hi);
    vout[v] = tl_hvx_narrow_hf(tl_hvx_recip_vsf(lo), tl_hvx_recip_vsf(hi));
  }
  for (int j = nv * TL_HVX_F16_LANES; j < n; ++j) out[j] = (__fp16)(1.0f / (float)in[j]);
}
TL_DEVICE void tl_hvx_rsqrt_eps_row(__fp16 *out, const __fp16 *in, float eps, int n) {
  if (!tl_hvx_aligned_128(in) || !tl_hvx_aligned_128(out)) {
    for (int j = 0; j < n; ++j)
      out[j] = (__fp16)(1.0f / sqrtf((float)in[j] + eps));
    return;
  }
  HVX_Vector ve = tl_hvx_splat_f(eps);
  int nv = n / TL_HVX_F16_LANES;
  const HVX_Vector *vin = (const HVX_Vector *)in;
  HVX_Vector *vout = (HVX_Vector *)out;
  for (int v = 0; v < nv; ++v) {
    HVX_Vector lo, hi;
    tl_hvx_widen_hf(vin[v], &lo, &hi);
    lo = tl_hvx_rsqrt_vsf(tl_hvx_add_sf(lo, ve));
    hi = tl_hvx_rsqrt_vsf(tl_hvx_add_sf(hi, ve));
    vout[v] = tl_hvx_narrow_hf(lo, hi);
  }
  for (int j = nv * TL_HVX_F16_LANES; j < n; ++j)
    out[j] = (__fp16)(1.0f / sqrtf((float)in[j] + eps));
}

// ---------------------------------------------------------------------------
// Row reductions over a contiguous fp16 row of length n -> fp32 scalar.
// Accumulate across vectors, then a butterfly rotate-reduce within one vector
// (Q6_V_vror_VR rotates by bytes).  Aligned rows use HVX; short or unaligned
// rows fall back to scalar so arbitrary row widths are correct.
// ---------------------------------------------------------------------------
TL_DEVICE float tl_hvx_lane0_sf(HVX_Vector v) {
  __attribute__((aligned(128))) float buf[TL_HVX_F32_LANES];
  *(HVX_Vector *)buf = v;
  return buf[0];
}
TL_DEVICE float tl_hvx_lane0_hf(HVX_Vector v) {
  __attribute__((aligned(128))) __fp16 buf[TL_HVX_F16_LANES];
  *(HVX_Vector *)buf = v;
  return (float)buf[0];
}
TL_DEVICE float tl_hvx_row_max(const __fp16 *in, int n) {
  if (n <= 0)
    return -3.4028234663852886e38f;
  if (!tl_hvx_aligned_128(in) || n < TL_HVX_F16_LANES) {
    float r = (float)in[0];
    for (int j = 1; j < n; ++j) {
      float x = (float)in[j];
      if (x > r) r = x;
    }
    return r;
  }
  const HVX_Vector *v = (const HVX_Vector *)in;
  int nv = n / TL_HVX_F16_LANES;
  HVX_Vector m = v[0];
  for (int i = 1; i < nv; ++i) m = Q6_Vhf_vmax_VhfVhf(m, v[i]);
  for (int s = 64; s >= 2; s >>= 1) m = Q6_Vhf_vmax_VhfVhf(m, Q6_V_vror_VR(m, s));
  float r = tl_hvx_lane0_hf(m);
  for (int j = nv * TL_HVX_F16_LANES; j < n; ++j) {
    float x = (float)in[j];
    if (x > r) r = x;
  }
  return r;
}
TL_DEVICE float tl_hvx_row_sum(const __fp16 *in, int n) {
  if (n <= 0)
    return 0.0f;
  if (!tl_hvx_aligned_128(in) || n < TL_HVX_F16_LANES) {
    float r = 0.0f;
    for (int j = 0; j < n; ++j) r += (float)in[j];
    return r;
  }
  const HVX_Vector *v = (const HVX_Vector *)in;
  int nv = n / TL_HVX_F16_LANES;
  HVX_Vector acc = Q6_V_vzero();
  for (int i = 0; i < nv; ++i) {
    HVX_Vector lo, hi;
    tl_hvx_widen_hf(v[i], &lo, &hi);
    acc = tl_hvx_add_sf(acc, tl_hvx_add_sf(lo, hi));
  }
  for (int s = 64; s >= 4; s >>= 1) acc = tl_hvx_add_sf(acc, Q6_V_vror_VR(acc, s));
  float r = tl_hvx_lane0_sf(acc);
  for (int j = nv * TL_HVX_F16_LANES; j < n; ++j) r += (float)in[j];
  return r;
}
// Whole-tile reductions: out[i] = reduce(in[i, :]) for i in [0, rows).  Output
// is fp32 (the reduce result feeds the fp32 running-max/sum scalar logic in
// softmax/layernorm).  Rows with 128-byte alignment use HVX; other rows are
// handled by the scalar path in tl_hvx_row_{max,sum}.
TL_DEVICE void tl_hvx_rowmax_mat(float *out, const __fp16 *in, int rows, int n) {
  for (int i = 0; i < rows; ++i) out[i] = tl_hvx_row_max(in + (size_t)i * n, n);
}
TL_DEVICE void tl_hvx_rowsum_mat(float *out, const __fp16 *in, int rows, int n) {
  for (int i = 0; i < rows; ++i) out[i] = tl_hvx_row_sum(in + (size_t)i * n, n);
}
