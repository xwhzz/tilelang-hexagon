#pragma once
// HVX math for tilelang-generated Hexagon kernels.
//
// The codegen prints the fp16/fp32 math builtins (LowerMath) as the
// tl_hvx_<fn> functions here, for scalars and for the codegen's vectors
// (vec_type, common.h), and a float vector division as tl_hvx_div. Each
// transcendental is one 32-lane fp32 register kernel (tl_hvx_<fn>_vsf) built
// from exp2, recip and rsqrt; fp16 computes in fp32 and rounds once. The row
// reductions of the reduce tile op are in hvx_reduce.h.
//
// Pulled in on demand by runtime/fastrpc.py when a kernel references `tl_hvx`.
#include <hexagon_types.h> // HVX_Vector / HVX_VectorPair + Q6_* intrinsics (needs -mhvx)
#include <stdint.h>
#include <tl_templates/hexagon/common.h> // vec_type, the codegen's vectors
#include <tl_templates/hexagon/hvx_lanes.h>
#include <tl_templates/hexagon/hvx_reduce.h>

#ifndef TL_DEVICE
#define TL_DEVICE static inline __attribute__((always_inline))
#endif

// HVX vector geometry: 1024-bit register = 32 fp32 lanes = 64 fp16 lanes.
#define TL_HVX_F32_LANES 32
#define TL_HVX_F16_LANES 64

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

// ---------------------------------------------------------------------------
// exp2 over 32 fp32 lanes: 2^x = 2^k · 2^f, k = round(x), f ∈ [-0.5, 0.5].
//   2^f : degree-5 minimax polynomial (Horner, accumulated in qf32)
//   2^k : construct the fp32 bit pattern (k + 127) << 23  (reinterpret is free)
// x is clamped to [-126, 126] so the exponent stays representable (very
// negative inputs flush toward 0, which is what softmax wants).
// ---------------------------------------------------------------------------
TL_DEVICE HVX_Vector tl_hvx_exp2_vsf(HVX_Vector x) {
  x = Q6_Vsf_vmin_VsfVsf(Q6_Vsf_vmax_VsfVsf(x, tl_hvx_splat_f(-126.0f)),
                         tl_hvx_splat_f(126.0f));
  // k = round-to-nearest(x), f = x - k ∈ [-0.5, 0.5].  Q6_Vw_equals_Vsf
  // truncates TOWARD ZERO, so bias by copysign(0.5, x) before the convert —
  // otherwise f lands in (-1, 1) and the degree-5 minimax poly (fit on [-0.5,
  // 0.5]) is evaluated outside its domain (~100x worse: 3e-6 -> ~3e-4).
  HVX_Vector half_signed = Q6_V_vor_VV(
      Q6_V_vand_VV(x, Q6_V_vsplat_R((int)0x80000000)), tl_hvx_splat_f(0.5f));
  HVX_Vector ki =
      Q6_Vw_equals_Vsf(tl_hvx_add_sf(x, half_signed)); // trunc(x ± 0.5)
  HVX_Vector kf = Q6_Vsf_equals_Vw(ki); // back to float for the residual
  HVX_Vector f = tl_hvx_sub_sf(x, kf);  // f ∈ [-0.5, 0.5]
  // 2^f via Horner in qf32:  ((((C5 f + C4) f + C3) f + C2) f + C1) f + C0
  HVX_Vector fq = Q6_Vqf32_vadd_VsfVsf(f, Q6_V_vzero()); // f as qf32
  HVX_Vector acc =
      Q6_Vqf32_vadd_VsfVsf(tl_hvx_splat_f(0.0013333558f), Q6_V_vzero());
  acc = Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_Vqf32Vqf32(acc, fq),
                               tl_hvx_splat_f(0.0096181291f));
  acc = Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_Vqf32Vqf32(acc, fq),
                               tl_hvx_splat_f(0.0555041087f));
  acc = Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_Vqf32Vqf32(acc, fq),
                               tl_hvx_splat_f(0.2402265069f));
  acc = Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_Vqf32Vqf32(acc, fq),
                               tl_hvx_splat_f(0.6931471806f));
  acc = Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_Vqf32Vqf32(acc, fq),
                               tl_hvx_splat_f(1.0000000000f));
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
  HVX_Vector y = Q6_Vw_vsub_VwVw(Q6_V_vsplat_R(0x5F3759DF),
                                 Q6_Vw_vasr_VwR(x, 1)); // fast inv-sqrt seed
  HVX_Vector c15 = tl_hvx_splat_f(1.5f);
  for (int it = 0; it < 3; ++it) // y = y·(1.5 − x/2·y²)
    y = tl_hvx_mul_sf(
        y, tl_hvx_sub_sf(c15, tl_hvx_mul_sf(xhalf, tl_hvx_mul_sf(y, y))));
  return y;
}

// ---------------------------------------------------------------------------
// Math functions of the codegen's scalars and vectors (LowerMath prints
// tl.hexagon_<fn> as tl_hvx_<fn>). Each has one 32-lane fp32 register kernel,
// tl_hvx_<fn>_vsf, built from the polynomials above; fp16 computes in fp32 and
// rounds once, and a scalar runs the same kernel on a splat, so a result does
// not depend on whether its loop vectorized.
// ---------------------------------------------------------------------------
TL_DEVICE HVX_Vector tl_hvx_bits(int b) { return Q6_V_vsplat_R(b); }

// x where q, else y.
TL_DEVICE HVX_Vector tl_hvx_select(HVX_VectorPred q, HVX_Vector x,
                                   HVX_Vector y) {
  return Q6_V_vmux_QVV(q, x, y);
}

TL_DEVICE HVX_Vector tl_hvx_exp_vsf(HVX_Vector x) {
  return tl_hvx_exp2_vsf(tl_hvx_mul_sf(x, tl_hvx_splat_f(1.4426950408889634f)));
}

// x = m·2^e with m in [sqrt(1/2), sqrt(2)); ln m = 2·atanh(s) with
// s = (m - 1)/(m + 1), |s| <= 0.172, so five series terms are below 1e-9.
TL_DEVICE HVX_Vector tl_hvx_log2_vsf(HVX_Vector x) {
  // Scale subnormals into the normal range first.
  HVX_VectorPred subnormal = Q6_Q_vcmp_gt_VwVw(tl_hvx_bits(0x00800000), x);
  HVX_Vector scaled = tl_hvx_mul_sf(x, tl_hvx_splat_f(8388608.0f)); // 2^23
  HVX_Vector v = tl_hvx_select(subnormal, scaled, x);
  HVX_Vector e = Q6_Vw_vsub_VwVw(
      Q6_V_vand_VV(Q6_Vuw_vlsr_VuwR(v, 23), tl_hvx_bits(0xff)),
      tl_hvx_select(subnormal, tl_hvx_bits(127 + 23), tl_hvx_bits(127)));
  HVX_Vector m = Q6_V_vor_VV(Q6_V_vand_VV(v, tl_hvx_bits(0x007fffff)),
                             tl_hvx_bits(0x3f800000));
  HVX_VectorPred big = Q6_Q_vcmp_gt_VwVw(m, tl_hvx_bits(0x3fb504f3)); // sqrt 2
  m = tl_hvx_select(big, Q6_Vw_vsub_VwVw(m, tl_hvx_bits(0x00800000)), m);
  e = tl_hvx_select(big, Q6_Vw_vadd_VwVw(e, tl_hvx_bits(1)), e);
  HVX_Vector one = tl_hvx_splat_f(1.0f);
  HVX_Vector s = tl_hvx_mul_sf(tl_hvx_sub_sf(m, one),
                               tl_hvx_recip_vsf(tl_hvx_add_sf(m, one)));
  HVX_Vector s2 = tl_hvx_mul_sf(s, s);
  HVX_Vector p = tl_hvx_splat_f(1.0f / 9.0f);
  p = tl_hvx_add_sf(tl_hvx_mul_sf(p, s2), tl_hvx_splat_f(1.0f / 7.0f));
  p = tl_hvx_add_sf(tl_hvx_mul_sf(p, s2), tl_hvx_splat_f(1.0f / 5.0f));
  p = tl_hvx_add_sf(tl_hvx_mul_sf(p, s2), tl_hvx_splat_f(1.0f / 3.0f));
  p = tl_hvx_add_sf(tl_hvx_mul_sf(p, s2), one);
  // log2 x = e + 2·s·p·log2(e)
  HVX_Vector r = tl_hvx_add_sf(
      Q6_Vsf_equals_Vw(e),
      tl_hvx_mul_sf(tl_hvx_mul_sf(s, p), tl_hvx_splat_f(2.8853900817779268f)));
  // 0 -> -inf; negative or NaN -> NaN; +inf -> +inf.
  HVX_Vector magnitude = Q6_V_vand_VV(x, tl_hvx_bits(0x7fffffff));
  r = tl_hvx_select(Q6_Q_vcmp_gt_VwVw(magnitude, tl_hvx_bits(0x7f7fffff)), x,
                    r);
  r = tl_hvx_select(Q6_Q_vcmp_gt_VwVw(Q6_V_vzero(), x), tl_hvx_bits(0x7fc00000),
                    r);
  return tl_hvx_select(Q6_Q_vcmp_eq_VwVw(magnitude, Q6_V_vzero()),
                       tl_hvx_bits((int)0xff800000), r);
}

TL_DEVICE HVX_Vector tl_hvx_log_vsf(HVX_Vector x) {
  return tl_hvx_mul_sf(tl_hvx_log2_vsf(x), tl_hvx_splat_f(0.6931471805599453f));
}

// The Newton-Raphson seed overflows for subnormals, so those are scaled by
// 2^24 first, and the result by 2^12 (rsqrt) or 2^-12 (sqrt).
TL_DEVICE HVX_VectorPred tl_hvx_is_subnormal(HVX_Vector x) {
  HVX_Vector magnitude = Q6_V_vand_VV(x, tl_hvx_bits(0x7fffffff));
  return Q6_Q_and_QQ(Q6_Q_vcmp_gt_VwVw(tl_hvx_bits(0x00800000), magnitude),
                     Q6_Q_vcmp_gt_VwVw(magnitude, Q6_V_vzero()));
}

TL_DEVICE HVX_Vector tl_hvx_rsqrt_scaled_vsf(HVX_Vector x) {
  HVX_VectorPred subnormal = tl_hvx_is_subnormal(x);
  HVX_Vector v = tl_hvx_select(
      subnormal, tl_hvx_mul_sf(x, tl_hvx_splat_f(16777216.0f)), x); // 2^24
  HVX_Vector r = tl_hvx_rsqrt_vsf(v);
  return tl_hvx_select(subnormal, tl_hvx_mul_sf(r, tl_hvx_splat_f(4096.0f)),
                       r); // 2^12
}

// rsqrt with IEEE special values: 0 -> inf, +inf -> 0, negative -> NaN.
TL_DEVICE HVX_Vector tl_hvx_rsqrt_ieee_vsf(HVX_Vector x) {
  HVX_Vector r = tl_hvx_rsqrt_scaled_vsf(x);
  HVX_Vector magnitude = Q6_V_vand_VV(x, tl_hvx_bits(0x7fffffff));
  r = tl_hvx_select(Q6_Q_vcmp_eq_VwVw(x, tl_hvx_bits(0x7f800000)), Q6_V_vzero(),
                    r);
  r = tl_hvx_select(Q6_Q_vcmp_gt_VwVw(Q6_V_vzero(), x), tl_hvx_bits(0x7fc00000),
                    r);
  return tl_hvx_select(
      Q6_Q_vcmp_eq_VwVw(magnitude, Q6_V_vzero()),
      Q6_V_vor_VV(tl_hvx_bits(0x7f800000),
                  Q6_V_vand_VV(x, tl_hvx_bits((int)0x80000000))),
      r);
}

// sqrt(x) = x·rsqrt(x); +-0 and +inf map to themselves, negative to NaN.
TL_DEVICE HVX_Vector tl_hvx_sqrt_vsf(HVX_Vector x) {
  HVX_Vector r = tl_hvx_mul_sf(x, tl_hvx_rsqrt_scaled_vsf(x));
  HVX_Vector magnitude = Q6_V_vand_VV(x, tl_hvx_bits(0x7fffffff));
  r = tl_hvx_select(Q6_Q_vcmp_gt_VwVw(Q6_V_vzero(), x), tl_hvx_bits(0x7fc00000),
                    r);
  HVX_VectorPred same = Q6_Q_vcmp_eq_VwVw(magnitude, Q6_V_vzero());
  same = Q6_Q_or_QQ(same, Q6_Q_vcmp_eq_VwVw(x, tl_hvx_bits(0x7f800000)));
  return tl_hvx_select(same, x, r);
}

// sigmoid(x) = 1/(1 + e^-x); exp2's clamp keeps it finite at both ends.
TL_DEVICE HVX_Vector tl_hvx_sigmoid_vsf(HVX_Vector x) {
  HVX_Vector e = tl_hvx_exp_vsf(Q6_V_vxor_VV(x, tl_hvx_bits((int)0x80000000)));
  return tl_hvx_recip_vsf(tl_hvx_add_sf(tl_hvx_splat_f(1.0f), e));
}

// tanh(x) = 1 - 2/(e^2x + 1), which saturates to +-1. Below |x| = 1/4 that
// form loses digits to cancellation, so an odd Taylor polynomial takes over.
TL_DEVICE HVX_Vector tl_hvx_tanh_vsf(HVX_Vector x) {
  HVX_Vector one = tl_hvx_splat_f(1.0f);
  HVX_Vector e = tl_hvx_exp_vsf(tl_hvx_add_sf(x, x));
  HVX_Vector q = tl_hvx_recip_vsf(tl_hvx_add_sf(e, one));
  HVX_Vector large = tl_hvx_sub_sf(one, tl_hvx_add_sf(q, q));
  HVX_Vector x2 = tl_hvx_mul_sf(x, x);
  HVX_Vector p = tl_hvx_splat_f(-17.0f / 315.0f);
  p = tl_hvx_add_sf(tl_hvx_mul_sf(p, x2), tl_hvx_splat_f(2.0f / 15.0f));
  p = tl_hvx_add_sf(tl_hvx_mul_sf(p, x2), tl_hvx_splat_f(-1.0f / 3.0f));
  HVX_Vector small = tl_hvx_add_sf(x, tl_hvx_mul_sf(tl_hvx_mul_sf(p, x2), x));
  HVX_Vector magnitude = Q6_V_vand_VV(x, tl_hvx_bits(0x7fffffff));
  return tl_hvx_select(Q6_Q_vcmp_gt_VwVw(tl_hvx_bits(0x3e800000), magnitude),
                       small, large);
}

// erf by Abramowitz-Stegun 7.1.26 (absolute error below 1.5e-7):
// erf(x) = sign(x)·(1 - t·(a1 + t·(a2 + ...))·e^(-x^2)), t = 1/(1 + p|x|).
// Below |x| = 1/2 its relative error grows, and the Taylor series
// (2/sqrt(pi))·x·(1 - x^2/3 + x^4/10 - x^6/42 + x^8/216 - x^10/1320) takes
// over.
TL_DEVICE HVX_Vector tl_hvx_erf_vsf(HVX_Vector x) {
  HVX_Vector sign = Q6_V_vand_VV(x, tl_hvx_bits((int)0x80000000));
  HVX_Vector a = Q6_V_vand_VV(x, tl_hvx_bits(0x7fffffff));
  HVX_Vector one = tl_hvx_splat_f(1.0f);
  HVX_Vector t = tl_hvx_recip_vsf(
      tl_hvx_add_sf(one, tl_hvx_mul_sf(a, tl_hvx_splat_f(0.3275911f))));
  HVX_Vector p = tl_hvx_splat_f(1.061405429f);
  p = tl_hvx_add_sf(tl_hvx_mul_sf(p, t), tl_hvx_splat_f(-1.453152027f));
  p = tl_hvx_add_sf(tl_hvx_mul_sf(p, t), tl_hvx_splat_f(1.421413741f));
  p = tl_hvx_add_sf(tl_hvx_mul_sf(p, t), tl_hvx_splat_f(-0.284496736f));
  p = tl_hvx_add_sf(tl_hvx_mul_sf(p, t), tl_hvx_splat_f(0.254829592f));
  p = tl_hvx_mul_sf(p, t);
  HVX_Vector g = tl_hvx_exp_vsf(
      Q6_V_vxor_VV(tl_hvx_mul_sf(a, a), tl_hvx_bits((int)0x80000000)));
  HVX_Vector y = Q6_V_vor_VV(tl_hvx_sub_sf(one, tl_hvx_mul_sf(p, g)), sign);
  HVX_Vector x2 = tl_hvx_mul_sf(x, x);
  HVX_Vector s = tl_hvx_splat_f(-1.0f / 1320.0f);
  s = tl_hvx_add_sf(tl_hvx_mul_sf(s, x2), tl_hvx_splat_f(1.0f / 216.0f));
  s = tl_hvx_add_sf(tl_hvx_mul_sf(s, x2), tl_hvx_splat_f(-1.0f / 42.0f));
  s = tl_hvx_add_sf(tl_hvx_mul_sf(s, x2), tl_hvx_splat_f(1.0f / 10.0f));
  s = tl_hvx_add_sf(tl_hvx_mul_sf(s, x2), tl_hvx_splat_f(-1.0f / 3.0f));
  s = tl_hvx_add_sf(tl_hvx_mul_sf(s, x2), one);
  HVX_Vector small =
      tl_hvx_mul_sf(tl_hvx_mul_sf(s, x), tl_hvx_splat_f(1.1283791670955126f));
  return tl_hvx_select(Q6_Q_vcmp_gt_VwVw(tl_hvx_bits(0x3f000000), a), small, y);
}

// ---------------------------------------------------------------------------
// Fast variants (tl.enable_fast_math): fp16-grade accuracy, about 1e-4
// relative (1e-4 absolute for tanh, sigmoid and erf), for finite inputs in the
// functions' domains. Chains stay in qf32 between operations instead of
// rounding to IEEE after each; Newton-Raphson takes two steps; polynomials
// have lower degrees; special values and subnormals get no fixups. v79 has no
// qf32 negation, nor c − qf32 for an IEEE c, so the steps keep their signs
// with the forms it has: qf32 − c, qf32 + c, and products by a scalar.
// ---------------------------------------------------------------------------
TL_DEVICE HVX_Vector tl_hvx_qf(HVX_Vector sf) {
  return Q6_Vqf32_vadd_VsfVsf(sf, Q6_V_vzero());
}
TL_DEVICE int tl_hvx_float_bits(float c) {
  int b;
  __builtin_memcpy(&b, &c, 4);
  return b;
}

// 1/x: y = y·(2 + (−x)·y) twice from the bit-pattern seed, about 1e-5.
TL_DEVICE HVX_Vector tl_hvx_recip_fast_vsf(HVX_Vector x) {
  HVX_Vector y = Q6_Vw_vsub_VwVw(Q6_V_vsplat_R(0x7EF127EA), x);
  HVX_Vector neg_x = Q6_V_vxor_VV(x, tl_hvx_bits((int)0x80000000));
  HVX_Vector two = tl_hvx_splat_f(2.0f);
  HVX_Vector yq = Q6_Vqf32_vmpy_Vqf32Vqf32(
      tl_hvx_qf(y),
      Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(neg_x, y), two));
  yq = Q6_Vqf32_vmpy_Vqf32Vqf32(
      yq, Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_Vqf32Vqf32(tl_hvx_qf(neg_x), yq),
                                 two));
  return Q6_Vsf_equals_Vqf32(yq);
}

// 1/sqrt(x): w = w·(x/2·w² − 1.5) twice from the bit-pattern seed, about 5e-6.
// Each step is the Newton step y·(1.5 − x/2·y²) negated, so two of them keep
// the sign.
TL_DEVICE HVX_Vector tl_hvx_rsqrt_fast_vsf(HVX_Vector x) {
  HVX_Vector y =
      Q6_Vw_vsub_VwVw(Q6_V_vsplat_R(0x5F3759DF), Q6_Vw_vasr_VwR(x, 1));
  HVX_Vector half_x = Q6_Vqf32_vmpy_VsfRsf(x, tl_hvx_float_bits(0.5f));
  HVX_Vector c15 = tl_hvx_splat_f(1.5f);
  HVX_Vector wq = tl_hvx_qf(y);
#pragma clang loop unroll(full)
  for (int it = 0; it < 2; ++it) {
    HVX_Vector t =
        Q6_Vqf32_vmpy_Vqf32Vqf32(half_x, Q6_Vqf32_vmpy_Vqf32Vqf32(wq, wq));
    wq = Q6_Vqf32_vmpy_Vqf32Vqf32(wq, Q6_Vqf32_vsub_Vqf32Vsf(t, c15));
  }
  return Q6_Vsf_equals_Vqf32(wq);
}

// x·rsqrt(x); 0 stays 0, since the seed of 0 is finite.
TL_DEVICE HVX_Vector tl_hvx_sqrt_fast_vsf(HVX_Vector x) {
  return tl_hvx_mul_sf(x, tl_hvx_rsqrt_fast_vsf(x));
}

// 2^x = 2^k·2^f for x in [-126, 126]: k = round(x) from the truncation of
// x + 128.5, f = x − k in [-0.5, 0.5], 2^f by a degree-3 minimax polynomial
// (7.5e-5 relative), and k added to the exponent field.
TL_DEVICE HVX_Vector tl_hvx_exp2_fast_vsf(HVX_Vector x) {
  x = Q6_Vsf_vmin_VsfVsf(Q6_Vsf_vmax_VsfVsf(x, tl_hvx_splat_f(-126.0f)),
                         tl_hvx_splat_f(126.0f));
  HVX_Vector biased =
      Q6_Vw_equals_Vsf(tl_hvx_add_sf(x, tl_hvx_splat_f(128.5f)));
  HVX_Vector k = Q6_Vw_vsub_VwVw(biased, Q6_V_vsplat_R(128));
  HVX_Vector fq = Q6_Vqf32_vsub_VsfVsf(x, Q6_Vsf_equals_Vw(k));
  HVX_Vector f = Q6_Vsf_equals_Vqf32(fq);
  HVX_Vector p = Q6_Vqf32_vadd_Vqf32Vsf(
      Q6_Vqf32_vmpy_VsfRsf(f, tl_hvx_float_bits(0.0551716045f)),
      tl_hvx_splat_f(0.2426111173f));
  p = Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_Vqf32Vqf32(p, fq),
                             tl_hvx_splat_f(0.6932609974f));
  p = Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_Vqf32Vqf32(p, fq),
                             tl_hvx_splat_f(0.9999280741f));
  return Q6_Vw_vadd_VwVw(Q6_Vsf_equals_Vqf32(p), Q6_Vw_vasl_VwR(k, 23));
}

TL_DEVICE HVX_Vector tl_hvx_exp_fast_vsf(HVX_Vector x) {
  return tl_hvx_exp2_fast_vsf(Q6_Vsf_equals_Vqf32(
      Q6_Vqf32_vmpy_VsfRsf(x, tl_hvx_float_bits(1.4426950408889634f))));
}

// log2 of a positive normal x = m·2^e, m in [sqrt(1/2), sqrt(2)): three terms
// of 2·atanh(s)/ln 2, s = (m − 1)/(m + 1), |s| <= 0.172, about 2e-6.
TL_DEVICE HVX_Vector tl_hvx_log2_fast_vsf(HVX_Vector x) {
  HVX_Vector e =
      Q6_Vw_vsub_VwVw(Q6_V_vand_VV(Q6_Vuw_vlsr_VuwR(x, 23), tl_hvx_bits(0xff)),
                      tl_hvx_bits(127));
  HVX_Vector m = Q6_V_vor_VV(Q6_V_vand_VV(x, tl_hvx_bits(0x007fffff)),
                             tl_hvx_bits(0x3f800000));
  HVX_VectorPred big = Q6_Q_vcmp_gt_VwVw(m, tl_hvx_bits(0x3fb504f3)); // sqrt 2
  m = tl_hvx_select(big, Q6_Vw_vsub_VwVw(m, tl_hvx_bits(0x00800000)), m);
  e = tl_hvx_select(big, Q6_Vw_vadd_VwVw(e, tl_hvx_bits(1)), e);
  HVX_Vector one = tl_hvx_splat_f(1.0f);
  HVX_Vector sq = Q6_Vqf32_vmpy_Vqf32Vqf32(
      Q6_Vqf32_vsub_VsfVsf(m, one),
      tl_hvx_qf(tl_hvx_recip_fast_vsf(tl_hvx_add_sf(m, one))));
  HVX_Vector s2 = Q6_Vqf32_vmpy_Vqf32Vqf32(sq, sq);
  HVX_Vector p = Q6_Vqf32_vadd_Vqf32Vsf(
      Q6_Vqf32_vmpy_VsfRsf(Q6_Vsf_equals_Vqf32(s2), tl_hvx_float_bits(0.2f)),
      tl_hvx_splat_f(1.0f / 3.0f));
  p = Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_Vqf32Vqf32(p, s2), one);
  HVX_Vector r =
      Q6_Vqf32_vmpy_VsfRsf(Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_Vqf32Vqf32(sq, p)),
                           tl_hvx_float_bits(2.8853900817779268f));
  return Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(r, Q6_Vsf_equals_Vw(e)));
}

TL_DEVICE HVX_Vector tl_hvx_log_fast_vsf(HVX_Vector x) {
  return Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfRsf(
      tl_hvx_log2_fast_vsf(x), tl_hvx_float_bits(0.6931471805599453f)));
}

TL_DEVICE HVX_Vector tl_hvx_sigmoid_fast_vsf(HVX_Vector x) {
  HVX_Vector e =
      tl_hvx_exp_fast_vsf(Q6_V_vxor_VV(x, tl_hvx_bits((int)0x80000000)));
  return tl_hvx_recip_fast_vsf(tl_hvx_add_sf(tl_hvx_splat_f(1.0f), e));
}

// 1 − 2/(e^2x + 1), with no small-|x| branch: its absolute error stays small.
TL_DEVICE HVX_Vector tl_hvx_tanh_fast_vsf(HVX_Vector x) {
  HVX_Vector one = tl_hvx_splat_f(1.0f);
  HVX_Vector e = tl_hvx_exp_fast_vsf(tl_hvx_add_sf(x, x));
  HVX_Vector q = tl_hvx_recip_fast_vsf(tl_hvx_add_sf(e, one));
  return Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(
      Q6_Vqf32_vmpy_VsfRsf(q, tl_hvx_float_bits(-2.0f)), one));
}

// Abramowitz-Stegun 7.1.26 alone (absolute error below 1.5e-7 in exact
// arithmetic), with the fast exp and reciprocal.
TL_DEVICE HVX_Vector tl_hvx_erf_fast_vsf(HVX_Vector x) {
  HVX_Vector sign = Q6_V_vand_VV(x, tl_hvx_bits((int)0x80000000));
  HVX_Vector a = Q6_V_vand_VV(x, tl_hvx_bits(0x7fffffff));
  HVX_Vector one = tl_hvx_splat_f(1.0f);
  HVX_Vector t =
      tl_hvx_recip_fast_vsf(Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(
          Q6_Vqf32_vmpy_VsfRsf(a, tl_hvx_float_bits(0.3275911f)), one)));
  HVX_Vector tq = tl_hvx_qf(t);
  HVX_Vector p = Q6_Vqf32_vadd_Vqf32Vsf(
      Q6_Vqf32_vmpy_VsfRsf(t, tl_hvx_float_bits(1.061405429f)),
      tl_hvx_splat_f(-1.453152027f));
  p = Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_Vqf32Vqf32(p, tq),
                             tl_hvx_splat_f(1.421413741f));
  p = Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_Vqf32Vqf32(p, tq),
                             tl_hvx_splat_f(-0.284496736f));
  p = Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_Vqf32Vqf32(p, tq),
                             tl_hvx_splat_f(0.254829592f));
  p = Q6_Vqf32_vmpy_Vqf32Vqf32(p, tq);
  HVX_Vector neg_a = Q6_V_vxor_VV(a, tl_hvx_bits((int)0x80000000));
  HVX_Vector neg_g = Q6_V_vxor_VV(
      tl_hvx_exp_fast_vsf(Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(a, neg_a))),
      tl_hvx_bits((int)0x80000000));
  HVX_Vector y = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(
      Q6_Vqf32_vmpy_Vqf32Vqf32(p, tl_hvx_qf(neg_g)), one));
  return Q6_V_vor_VV(y, sign);
}

// Applies a register kernel to every register of a vector. This loop and the
// division's unroll in full: rolled, they pass every register through the stack
// and run the registers' dependency chains one after another, which made exp
// over 64 fp16 lanes 2.5x slower than two interleaved chains.
template <HVX_Vector (*F)(HVX_Vector), int N>
TL_DEVICE vec_type<float, N> tl_hvx_apply(const vec_type<float, N> &x) {
  static_assert(N % TL_HVX_F32_LANES == 0, "HVX math needs whole registers");
  vec_type<float, N> out;
#pragma clang loop unroll(full)
  for (int i = 0; i < N / TL_HVX_F32_LANES; ++i) {
    HVX_Vector v;
    __builtin_memcpy(&v, (const char *)&x.v + 128 * i, 128);
    v = F(v);
    __builtin_memcpy((char *)&out.v + 128 * i, &v, 128);
  }
  return out;
}

template <HVX_Vector (*F)(HVX_Vector), int N>
TL_DEVICE vec_type<half, N> tl_hvx_apply(const vec_type<half, N> &x) {
  static_assert(N % TL_HVX_F16_LANES == 0, "HVX math needs whole registers");
  vec_type<half, N> out;
#pragma clang loop unroll(full)
  for (int i = 0; i < N / TL_HVX_F16_LANES; ++i) {
    HVX_Vector v, lo, hi;
    __builtin_memcpy(&v, (const char *)&x.v + 128 * i, 128);
    tl_hvx_widen_hf(v, &lo, &hi);
    v = tl_hvx_narrow_hf(F(lo), F(hi));
    __builtin_memcpy((char *)&out.v + 128 * i, &v, 128);
  }
  return out;
}

template <HVX_Vector (*F)(HVX_Vector)> TL_DEVICE float tl_hvx_apply(float x) {
  HVX_Vector v = F(tl_hvx_splat_f(x));
  float r;
  __builtin_memcpy(&r, &v, 4);
  return r;
}

#define TL_HVX_MATH(name, kernel)                                              \
  template <int N>                                                             \
  TL_DEVICE vec_type<float, N> tl_hvx_##name(const vec_type<float, N> &x) {    \
    return tl_hvx_apply<kernel>(x);                                            \
  }                                                                            \
  template <int N>                                                             \
  TL_DEVICE vec_type<half, N> tl_hvx_##name(const vec_type<half, N> &x) {      \
    return tl_hvx_apply<kernel>(x);                                            \
  }                                                                            \
  TL_DEVICE float tl_hvx_##name(float x) { return tl_hvx_apply<kernel>(x); }   \
  /* __fp16 is storage-only: the result is the fp32 value, rounded once by */  \
  /* the store, as scalar fp16 arithmetic is computed in fp32 anyway. */       \
  TL_DEVICE float tl_hvx_##name(const half &x) {                               \
    return tl_hvx_apply<kernel>((float)x);                                     \
  }

TL_HVX_MATH(exp, tl_hvx_exp_vsf)
TL_HVX_MATH(exp2, tl_hvx_exp2_vsf)
TL_HVX_MATH(log, tl_hvx_log_vsf)
TL_HVX_MATH(log2, tl_hvx_log2_vsf)
TL_HVX_MATH(sqrt, tl_hvx_sqrt_vsf)
TL_HVX_MATH(rsqrt, tl_hvx_rsqrt_ieee_vsf)
TL_HVX_MATH(tanh, tl_hvx_tanh_vsf)
TL_HVX_MATH(sigmoid, tl_hvx_sigmoid_vsf)
TL_HVX_MATH(erf, tl_hvx_erf_vsf)
TL_HVX_MATH(exp_fast, tl_hvx_exp_fast_vsf)
TL_HVX_MATH(exp2_fast, tl_hvx_exp2_fast_vsf)
TL_HVX_MATH(log_fast, tl_hvx_log_fast_vsf)
TL_HVX_MATH(log2_fast, tl_hvx_log2_fast_vsf)
TL_HVX_MATH(sqrt_fast, tl_hvx_sqrt_fast_vsf)
TL_HVX_MATH(rsqrt_fast, tl_hvx_rsqrt_fast_vsf)
TL_HVX_MATH(tanh_fast, tl_hvx_tanh_fast_vsf)
TL_HVX_MATH(sigmoid_fast, tl_hvx_sigmoid_fast_vsf)
TL_HVX_MATH(erf_fast, tl_hvx_erf_fast_vsf)
#undef TL_HVX_MATH

// |x| is one vector and that clears the sign bit; clang scalarizes a float
// vector fabs on HVX.
template <int N>
TL_DEVICE vec_type<float, N> tl_hvx_fabs(const vec_type<float, N> &x) {
  typedef int32_t bits_t __attribute__((ext_vector_type(N)));
  return (typename vec_type<float, N>::nat_t)((bits_t)x.v & 0x7fffffff);
}
template <int N>
TL_DEVICE vec_type<half, N> tl_hvx_fabs(const vec_type<half, N> &x) {
  typedef int16_t bits_t __attribute__((ext_vector_type(N)));
  return (typename vec_type<half, N>::nat_t)((bits_t)x.v & (int16_t)0x7fff);
}
TL_DEVICE float tl_hvx_fabs(float x) { return __builtin_fabsf(x); }
TL_DEVICE float tl_hvx_fabs(const half &x) { return __builtin_fabsf((float)x); }

// ---------------------------------------------------------------------------
// Division of the codegen's float vectors (vec_type<float|half, N>, N whole
// registers). HVX has no divide and clang scalarizes a vector `/`, so the
// codegen prints a / b as tl_hvx_div(a, b): a times the reciprocal of b. fp16
// divides in fp32, so the quotient is rounded once.
// ---------------------------------------------------------------------------
template <int N>
TL_DEVICE vec_type<float, N> tl_hvx_div(const vec_type<float, N> &a,
                                        const vec_type<float, N> &b) {
  static_assert(N % TL_HVX_F32_LANES == 0, "tl_hvx_div needs whole registers");
  vec_type<float, N> out;
#pragma clang loop unroll(full)
  for (int i = 0; i < N / TL_HVX_F32_LANES; ++i) {
    HVX_Vector x, y;
    __builtin_memcpy(&x, (const char *)&a.v + 128 * i, 128);
    __builtin_memcpy(&y, (const char *)&b.v + 128 * i, 128);
    HVX_Vector q = tl_hvx_mul_sf(x, tl_hvx_recip_vsf(y));
    __builtin_memcpy((char *)&out.v + 128 * i, &q, 128);
  }
  return out;
}

template <int N>
TL_DEVICE vec_type<half, N> tl_hvx_div(const vec_type<half, N> &a,
                                       const vec_type<half, N> &b) {
  static_assert(N % TL_HVX_F16_LANES == 0, "tl_hvx_div needs whole registers");
  vec_type<half, N> out;
#pragma clang loop unroll(full)
  for (int i = 0; i < N / TL_HVX_F16_LANES; ++i) {
    HVX_Vector x, y, xlo, xhi, ylo, yhi;
    __builtin_memcpy(&x, (const char *)&a.v + 128 * i, 128);
    __builtin_memcpy(&y, (const char *)&b.v + 128 * i, 128);
    tl_hvx_widen_hf(x, &xlo, &xhi);
    tl_hvx_widen_hf(y, &ylo, &yhi);
    HVX_Vector q = tl_hvx_narrow_hf(tl_hvx_mul_sf(xlo, tl_hvx_recip_vsf(ylo)),
                                    tl_hvx_mul_sf(xhi, tl_hvx_recip_vsf(yhi)));
    __builtin_memcpy((char *)&out.v + 128 * i, &q, 128);
  }
  return out;
}

// a / b as a times the fast reciprocal of b (tl.enable_fast_math): LowerMath
// rewrites a float division into hexagon_div_fast, which prints as this for
// scalars and vectors alike.
TL_DEVICE HVX_Vector tl_hvx_div_fast_vsf(HVX_Vector a, HVX_Vector b) {
  return tl_hvx_mul_sf(a, tl_hvx_recip_fast_vsf(b));
}

template <int N>
TL_DEVICE vec_type<float, N> tl_hvx_div_fast(const vec_type<float, N> &a,
                                             const vec_type<float, N> &b) {
  static_assert(N % TL_HVX_F32_LANES == 0, "HVX math needs whole registers");
  vec_type<float, N> out;
#pragma clang loop unroll(full)
  for (int i = 0; i < N / TL_HVX_F32_LANES; ++i) {
    HVX_Vector x, y;
    __builtin_memcpy(&x, (const char *)&a.v + 128 * i, 128);
    __builtin_memcpy(&y, (const char *)&b.v + 128 * i, 128);
    HVX_Vector q = tl_hvx_div_fast_vsf(x, y);
    __builtin_memcpy((char *)&out.v + 128 * i, &q, 128);
  }
  return out;
}

template <int N>
TL_DEVICE vec_type<half, N> tl_hvx_div_fast(const vec_type<half, N> &a,
                                            const vec_type<half, N> &b) {
  static_assert(N % TL_HVX_F16_LANES == 0, "HVX math needs whole registers");
  vec_type<half, N> out;
#pragma clang loop unroll(full)
  for (int i = 0; i < N / TL_HVX_F16_LANES; ++i) {
    HVX_Vector x, y, xlo, xhi, ylo, yhi;
    __builtin_memcpy(&x, (const char *)&a.v + 128 * i, 128);
    __builtin_memcpy(&y, (const char *)&b.v + 128 * i, 128);
    tl_hvx_widen_hf(x, &xlo, &xhi);
    tl_hvx_widen_hf(y, &ylo, &yhi);
    HVX_Vector q = tl_hvx_narrow_hf(tl_hvx_div_fast_vsf(xlo, ylo),
                                    tl_hvx_div_fast_vsf(xhi, yhi));
    __builtin_memcpy((char *)&out.v + 128 * i, &q, 128);
  }
  return out;
}

TL_DEVICE float tl_hvx_div_fast(float a, float b) {
  HVX_Vector v = tl_hvx_div_fast_vsf(tl_hvx_splat_f(a), tl_hvx_splat_f(b));
  float r;
  __builtin_memcpy(&r, &v, 4);
  return r;
}
TL_DEVICE float tl_hvx_div_fast(const half &a, const half &b) {
  return tl_hvx_div_fast((float)a, (float)b);
}

// ---------------------------------------------------------------------------
// q4_0 dequantize -> row-major fp16, vectorized on HVX.  This is the perf-
// critical "codec" primitive.  The packed->fp16 decode is expressed here as
// explicit vector ops and reached from the DSL via T.call_extern (a
// `dequantize` tile op).  Intrinsic sequence
// mirrors ggml-hexagon's dequantize_tiled_weight_to_fp16_task_q4_0:
//   vand/vlsr (nibbles) -> vsub 8 -> vshuff (interleave to K order) ->
//   vunpack b->h (sign-extend) -> hf<-h -> vmpy by per-block scale.
//
//   qs : uint8 [N][K/2]  packed nibbles; byte(n,j): low->W[n][2j],
//   high->W[n][2j+1] sc : fp16  [N][K/32] per-row, per-32-K-block scale W  :
//   fp16  [N][K]    W[n][k] = ((nibble & 0xF) - 8) * sc[n][k/32]
// Requires K % 64 == 0 (LFM2 weights: K in {512,2048,8192} all qualify).  One
// 128-lane op decodes 64 K-values (two scale blocks); the low-32/high-32 lanes
// take the two blocks' scales via a byte predicate mux.  qs is staged through
// an aligned buffer so no over-read past the input; W stored unaligned (no base
// alignment assumption).
TL_DEVICE void tl_hvx_dequant_q4_0(__fp16 *W, const uint8_t *qs,
                                   const __fp16 *sc, int N, int K) {
  const HVX_Vector mask = Q6_Vb_vsplat_R(0x0F);
  const HVX_Vector i8 = Q6_Vb_vsplat_R(8);
  const HVX_VectorPred lo32 =
      Q6_Q_vsetq_R(64); // first 64 bytes = 32 fp16 lanes
  const int kb = K / 32, ng = K / 64;
  for (int n = 0; n < N; ++n) {
    const uint8_t *qrow = qs + (size_t)n * (K / 2);
    const __fp16 *srow = sc + (size_t)n * kb;
    __fp16 *wrow = W + (size_t)n * K;
    for (int g = 0; g < ng; ++g) {
      __attribute__((aligned(128))) uint8_t tmp[128];
      __builtin_memcpy(tmp, qrow + g * 32, 32); // 32 bytes = 64 K-values
      HVX_Vector vq = *(const HVX_Vector *)tmp;
      HVX_Vector lo = Q6_Vb_vsub_VbVb(Q6_V_vand_VV(vq, mask), i8);
      HVX_Vector hi = Q6_Vb_vsub_VbVb(Q6_Vub_vlsr_VubR(vq, 4), i8);
      // interleave lo/hi to K order (byte j -> K 2j, 2j+1), then b->h->hf
      HVX_Vector k64 = Q6_V_lo_W(Q6_W_vshuff_VVR(hi, lo, -1));
      HVX_Vector f16 = Q6_Vhf_equals_Vh(Q6_V_lo_W(Q6_Wh_vunpack_Vb(k64)));
      uint16_t sa, sb;
      __builtin_memcpy(&sa, &srow[2 * g], 2);
      __builtin_memcpy(&sb, &srow[2 * g + 1], 2);
      HVX_Vector vsc =
          Q6_V_vmux_QVV(lo32, Q6_Vh_vsplat_R(sa), Q6_Vh_vsplat_R(sb));
      HVX_Vector res = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(f16, vsc));
      tl_hvx_storeu(wrow + g * 64, res);
    }
  }
}

// Scalar reference for the same contract (A/B perf baseline == the pre-HVX
// path).
TL_DEVICE void tl_hvx_dequant_q4_0_scalar(__fp16 *W, const uint8_t *qs,
                                          const __fp16 *sc, int N, int K) {
  for (int n = 0; n < N; ++n) {
    const uint8_t *qrow = qs + (size_t)n * (K / 2);
    const __fp16 *srow = sc + (size_t)n * (K / 32);
    __fp16 *wrow = W + (size_t)n * K;
    for (int j = 0; j < K / 2; ++j) {
      uint8_t b = qrow[j];
      wrow[2 * j] =
          (__fp16)(((float)(b & 0x0F) - 8.0f) * (float)srow[(2 * j) / 32]);
      wrow[2 * j + 1] =
          (__fp16)(((float)(b >> 4) - 8.0f) * (float)srow[(2 * j + 1) / 32]);
    }
  }
}
