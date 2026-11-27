#pragma once
// Online W8A8 packing for the LFM2 prefill gate/up projection. Both paths
// quantize finite fp16 values to [-127,127] with a single scale per K=2048
// (or a smaller complete K), with ties to even.
#include <tl_templates/hexagon/crouton.h>
#include <tl_templates/hexagon/hvx_math.h>

TL_HEXAGON_IEEE_ARITHMETIC TL_DEVICE HVX_Vector
tl_hexagon_w8a8_mul(HVX_Vector x, HVX_Vector y) {
#if __HVX_ARCH__ == 73
  return Q6_Vsf_vmpy_VsfVsf(x, y);
#else
  return tl_hvx_mul_sf(x, y);
#endif
}

// Truncation of the approximate ratio gives the lower integer, except at an
// integer boundary where either neighbor rounds identically after the
// correction. Classify the midpoint with exact products: a finite fp16
// significand times an integer <= 255 needs at most 19 bits.
TL_HEXAGON_IEEE_ARITHMETIC TL_DEVICE HVX_Vector
tl_hexagon_w8a8_round(HVX_Vector x, HVX_Vector amax, HVX_Vector inv) {
  const HVX_Vector one = Q6_V_vsplat_R(1);
  const HVX_Vector abs = Q6_V_vand_VV(x, Q6_V_vsplat_R(0x7fffffff));
  HVX_Vector q = Q6_Vw_equals_Vsf(tl_hexagon_w8a8_mul(abs, inv));
  HVX_Vector mid = Q6_Vsf_equals_Vw(Q6_Vw_vadd_VwVw(Q6_Vw_vasl_VwR(q, 1), one));
  HVX_Vector lhs = tl_hexagon_w8a8_mul(abs, tl_hvx_splat_f(254.0f));
  HVX_Vector rhs = tl_hexagon_w8a8_mul(amax, mid);
  HVX_VectorPred tie = Q6_Q_vcmp_eq_VwVw(lhs, rhs);
  HVX_VectorPred above = Q6_Q_vcmp_gt_VwVw(lhs, rhs);
  q = Q6_Vw_vadd_VwVw(q,
                      Q6_V_vmux_QVV(tie, Q6_V_vand_VV(q, one),
                                    Q6_V_vmux_QVV(above, one, Q6_V_vzero())));
  return Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VwVw(Q6_V_vzero(), x),
                       Q6_Vw_vsub_VwVw(Q6_V_vzero(), q), q);
}

// src is activation Crouton [R,K]. Each worker packs its own groups of four
// rows to [ceildiv(R,64),K/32,64,32]; four rows occupy one register. For R=32
// the second half of each integer tile is a stored row of zeros (code 128).
TL_HEXAGON_IEEE_ARITHMETIC static __attribute__((noinline)) void
tl_hexagon_quantize_crouton_u8(const half *__restrict__ src,
                               uint8_t *__restrict__ dst,
                               float *__restrict__ scales, int k, int rows,
                               int first, int count) {
  const int kt = k / 32;
  const HVX_Vector abs = Q6_Vh_vsplat_R(0x7fff);
  const HVX_Vector bias = Q6_Vb_vsplat_R(128);
  const auto *a = reinterpret_cast<const HVX_Vector *>(src);
  auto *out = reinterpret_cast<HVX_Vector *>(dst);
  for (int r = first; r < first + count; r += 4) {
    const auto *pair = a + (r / 32) * kt * 16 + (r % 32) / 2;
    HVX_Vector max01 = Q6_V_vzero(), max23 = Q6_V_vzero();
    for (int t = 0; t < kt; ++t) {
      max01 = Q6_Vuh_vmax_VuhVuh(max01, Q6_V_vand_VV(pair[t * 16], abs));
      max23 = Q6_Vuh_vmax_VuhVuh(max23, Q6_V_vand_VV(pair[t * 16 + 1], abs));
    }
    // Reduce over the 32 column lanes, preserving the halfword row parity.
    for (int bytes = 4; bytes <= 64; bytes *= 2) {
      max01 = Q6_Vuh_vmax_VuhVuh(max01, Q6_V_vror_VR(max01, bytes));
      max23 = Q6_Vuh_vmax_VuhVuh(max23, Q6_V_vror_VR(max23, bytes));
    }
    const uint32_t bits01 = Q6_R_vextract_VR(max01, 0);
    const uint32_t bits23 = Q6_R_vextract_VR(max23, 0);
    const uint16_t bits[4] = {
        static_cast<uint16_t>(bits01), static_cast<uint16_t>(bits01 >> 16),
        static_cast<uint16_t>(bits23), static_cast<uint16_t>(bits23 >> 16)};
    HVX_Vector amax[4], inv[4];
#pragma clang loop unroll(full)
    for (int i = 0; i < 4; ++i) {
      half h;
      __builtin_memcpy(&h, &bits[i], sizeof(h));
      float m = h;
      scales[r + i] = m * (1.0f / 127.0f);
      const float safe = m > 0 ? m : 1.0f;
      amax[i] = tl_hvx_splat_f(safe);
      inv[i] = tl_hvx_splat_f(127.0f / safe);
    }
    auto *block = out + (r / 64) * kt * 16 + (r % 64) / 4;
    for (int t = 0; t < kt; ++t) {
      HVX_VectorPair p01 = tl_hvx_convert_hf_to_sf(pair[t * 16]);
      HVX_VectorPair p23 = tl_hvx_convert_hf_to_sf(pair[t * 16 + 1]);
      HVX_Vector q0 = tl_hexagon_w8a8_round(Q6_V_lo_W(p01), amax[0], inv[0]);
      HVX_Vector q1 = tl_hexagon_w8a8_round(Q6_V_hi_W(p01), amax[1], inv[1]);
      HVX_Vector q2 = tl_hexagon_w8a8_round(Q6_V_lo_W(p23), amax[2], inv[2]);
      HVX_Vector q3 = tl_hexagon_w8a8_round(Q6_V_hi_W(p23), amax[3], inv[3]);
      HVX_Vector q01 = Q6_Vh_vpacke_VwVw(q1, q0);
      HVX_Vector q23 = Q6_Vh_vpacke_VwVw(q3, q2);
      block[t * 16] = Q6_V_vxor_VV(Q6_Vb_vpacke_VhVh(q23, q01), bias);
      if (rows == 32)
        block[t * 16 + 8] = bias;
    }
  }
}

// The absolute maximum of the eight signed nibbles in each output channel's
// word, then the maximum across all eight words in the 64-K scale group.
TL_DEVICE HVX_Vector tl_hexagon_panel_group_absmax(const HVX_Vector *panel) {
  const HVX_Vector mask = Q6_Vb_vsplat_R(15);
  const HVX_Vector sixteen = Q6_Vb_vsplat_R(16);
  HVX_Vector maxima = Q6_V_vzero();
  for (int i = 0; i < 8; ++i) {
    const HVX_Vector v = panel[i];
    const HVX_Vector lo = Q6_V_vand_VV(v, mask);
    const HVX_Vector hi = Q6_V_vand_VV(Q6_Vuh_vlsr_VuhR(v, 4), mask);
    const HVX_Vector a = Q6_Vub_vmin_VubVub(lo, Q6_Vb_vsub_VbVb(sixteen, lo));
    const HVX_Vector b = Q6_Vub_vmin_VubVub(hi, Q6_Vb_vsub_VbVb(sixteen, hi));
    maxima = Q6_Vub_vmax_VubVub(maxima, Q6_Vub_vmax_VubVub(a, b));
  }
  maxima = Q6_Vub_vmax_VubVub(maxima, Q6_Vuw_vlsr_VuwR(maxima, 8));
  maxima = Q6_Vub_vmax_VubVub(maxima, Q6_Vuw_vlsr_VuwR(maxima, 16));
  return Q6_V_vand_VV(maxima, Q6_V_vsplat_R(255));
}

// Round |q4|max * |group_scale| to fp16 like the dequantizer, then take
// the channel maxima across groups. Monotonic half rounding means there
// is no need to materialize every value for this first pass.
TL_HEXAGON_IEEE_ARITHMETIC static __attribute__((noinline)) HVX_Vector
tl_hexagon_panel_absmax(const HVX_Vector *panel, const half *sf, int k,
                        int ld) {
  const HVX_Vector abs = Q6_Vh_vsplat_R(0x7fff);
  const HVX_Vector zero = Q6_V_vzero();
  HVX_Vector max = zero;
  for (int g = 0; g < k / 64; ++g) {
    const HVX_Vector codes =
        Q6_Vsf_equals_Vw(tl_hexagon_panel_group_absmax(panel + g * 8));
    const HVX_Vector s = Q6_V_vand_VV(tl_hvx_load_bytes(sf + g * ld, 64), abs);
    const HVX_VectorPair pair = tl_hvx_convert_hf_to_sf(s);
    const HVX_Vector scales =
        Q6_V_lo_W(Q6_W_vshuff_VVR(Q6_V_hi_W(pair), Q6_V_lo_W(pair), -4));
    const HVX_Vector product = tl_hexagon_w8a8_mul(codes, scales);
    const HVX_Vector narrow = tl_crouton_narrow(product, product);
    // Preserve exact zero through the v79 Qfloat multiply and half narrowing.
    const HVX_VectorPred nonzero = Q6_Q_and_QQ(Q6_Q_vcmp_gt_VwVw(codes, zero),
                                               Q6_Q_vcmp_gt_VwVw(scales, zero));
    max = Q6_Vuh_vmax_VuhVuh(max, Q6_V_vmux_QVV(nonzero, narrow, zero));
  }
  return Q6_V_lo_W(tl_hvx_convert_hf_to_sf(max));
}

// One 32-channel panel of signed Q4 with FP16 scales every 64 K values.
// Reuse the W4A16 dequantizer (including FP16 rounding) one scale group
// at a time, then write each group of four K bytes for all 32 channels.
// The sum is kept as 128 * sum(q), the activation zero-point correction.
template <bool Normal>
TL_HEXAGON_IEEE_ARITHMETIC static __attribute__((noinline)) void
tl_hexagon_requantize_panel_s4_s8_impl(const void *__restrict__ panel,
                                       const half *__restrict__ group_scales,
                                       half *__restrict__ tmp,
                                       int8_t *__restrict__ dst,
                                       float *__restrict__ scales,
                                       int32_t *__restrict__ correction, int k,
                                       int scale_ld) {
  const auto *raw = reinterpret_cast<const HVX_Vector *>(panel);
  auto *w = reinterpret_cast<HVX_Vector *>(tmp);
  auto *out = reinterpret_cast<HVX_Vector *>(dst);
  const HVX_Vector max =
      tl_hexagon_panel_absmax(raw, group_scales, k, scale_ld);
  *reinterpret_cast<HVX_Vector *>(scales) =
      tl_hexagon_w8a8_mul(max, tl_hvx_splat_f(1.0f / 127.0f));
  const HVX_Vector safe = Q6_V_vmux_QVV(Q6_Q_vcmp_eq_VwVw(max, Q6_V_vzero()),
                                        tl_hvx_splat_f(1.0f), max);
  const HVX_Vector inv = tl_hvx_div_approx(tl_hvx_splat_f(127.0f), safe);
  const HVX_Vector byte = Q6_V_vsplat_R(255);
  const HVX_Vector one = Q6_Vb_vsplat_R(1);
  HVX_Vector sums = Q6_V_vzero();
  for (int g = 0; g < k / 64; ++g) {
    tl_crouton_dequant4<true, Normal>(
        tmp, raw + g * 8, group_scales + g * scale_ld, scale_ld, 64, 0, 8);
    for (int i = 0; i < 16; ++i) {
      const HVX_VectorPair p01 = tl_hvx_convert_hf_to_sf(w[i * 2]);
      const HVX_VectorPair p23 = tl_hvx_convert_hf_to_sf(w[i * 2 + 1]);
      HVX_Vector q0 = tl_hexagon_w8a8_round(Q6_V_lo_W(p01), safe, inv);
      HVX_Vector q1 = tl_hexagon_w8a8_round(Q6_V_hi_W(p01), safe, inv);
      HVX_Vector q2 = tl_hexagon_w8a8_round(Q6_V_lo_W(p23), safe, inv);
      HVX_Vector q3 = tl_hexagon_w8a8_round(Q6_V_hi_W(p23), safe, inv);
      const HVX_Vector lo = Q6_V_vor_VV(
          Q6_V_vand_VV(q0, byte), Q6_Vw_vasl_VwR(Q6_V_vand_VV(q1, byte), 8));
      const HVX_Vector hi = Q6_V_vor_VV(
          Q6_Vw_vasl_VwR(Q6_V_vand_VV(q2, byte), 16), Q6_Vw_vasl_VwR(q3, 24));
      const HVX_Vector packed = Q6_V_vor_VV(lo, hi);
      out[g * 16 + i] = packed;
      sums = Q6_Vw_vrmpyacc_VwVbVb(sums, packed, one);
    }
  }
  *reinterpret_cast<HVX_Vector *>(correction) = Q6_Vw_vasl_VwR(sums, 7);
}

TL_DEVICE void
tl_hexagon_requantize_panel_s4_s8(const void *panel, const half *group_scales,
                                  half *tmp, int8_t *dst, float *scales,
                                  int32_t *correction, int k, int scale_ld,
                                  bool normal_scales) {
  if (normal_scales)
    tl_hexagon_requantize_panel_s4_s8_impl<true>(
        panel, group_scales, tmp, dst, scales, correction, k, scale_ld);
  else
    tl_hexagon_requantize_panel_s4_s8_impl<false>(
        panel, group_scales, tmp, dst, scales, correction, k, scale_ld);
}
