#pragma once
// Register-granular HVX atoms used by explicit Q4_0 + HMX TileLang kernels.
//
// These helpers intentionally stop below an operator or tile scheduler.  The
// TileLang kernel owns M/N/K traversal, Crouton T.Layout addresses, Q4 group
// selection, and the HMX protocol.  Each helper maps one logical DSL atom to a
// fixed v79 instruction sequence that generic C vectorization cannot recover
// reliably from the interleaved Crouton stores.

#include <tl_templates/hexagon/common.h>
#include <tl_templates/hexagon/hvx_math.h>

#include <stdint.h>

#define TL_Q4_0_QUANT_BYTES 512
#define TL_Q4_0_GROUP_BYTES 128
#define TL_Q4_0_GROUP_OUTPUT_HALF 256

// Convert two contiguous 32-float activation rows into one 64-half HMX
// activation vector: row0[k0], row1[k0], row0[k1], row1[k1], ... .
TL_DEVICE int tl_hexagon_hmx_pack_a_f32_pair_k32(
    __fp16 *__restrict__ dst, const float *__restrict__ row0,
    const float *__restrict__ row1) {
  HVX_Vector v0 = tl_hvx_loadu(row0);
  HVX_Vector v1 = tl_hvx_loadu(row1);
#if __HVX_ARCH__ >= 81
  HVX_Vector q0 = Q6_Vqf32_equals_Vsf(v0);
  HVX_Vector q1 = Q6_Vqf32_equals_Vsf(v1);
#else
  HVX_Vector zero = Q6_V_vzero();
  HVX_Vector q0 = Q6_Vqf32_vadd_VsfVsf(v0, zero);
  HVX_Vector q1 = Q6_Vqf32_vadd_VsfVsf(v1, zero);
#endif
  tl_hvx_storeu(dst, Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(q1, q0)));
  return 0;
}

// Convert one 128-byte HMX output-Crouton row-pair vector into two contiguous
// 32-float output rows. Q6_Wqf32_vmpy_VhfVhf performs the FP16-to-qfloat32
// expansion while preserving the two rows in the low/high vector halves.
TL_DEVICE int tl_hexagon_hmx_unpack_c_f32_pair_n32(
    float *__restrict__ row0, float *__restrict__ row1,
    const __fp16 *__restrict__ src) {
  const HVX_Vector packed = tl_hvx_loadu(src);
  const HVX_Vector one = Q6_Vh_vsplat_R(0x3c00);
  const HVX_VectorPair rows = Q6_Wqf32_vmpy_VhfVhf(packed, one);
  tl_hvx_storeu(row0, Q6_Vsf_equals_Vqf32(Q6_V_lo_W(rows)));
  tl_hvx_storeu(row1, Q6_Vsf_equals_Vqf32(Q6_V_hi_W(rows)));
  return 0;
}

TL_DEVICE HVX_Vector tl_q4_0_scale_i16_to_f16(HVX_Vector values,
                                               HVX_Vector scale) {
  HVX_Vector fp16 = Q6_Vhf_equals_Vh(values);
  return Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(fp16, scale));
}

// Safely read the 64-byte scale tail of one Q4_0 tile. A full unaligned HVX
// load is invalid when this is the final tile in an exact-size RPC buffer.
TL_DEVICE HVX_Vector
tl_q4_0_load_scale_tail(const uint8_t *__restrict__ scales) {
#ifdef TL_Q4_0_PADDED_VTCM_INPUT
  // Embedded runtimes stage a whole Q4 chunk into an aligned VTCM allocation;
  // the 64 bytes following every scale tail remain mapped. Only the low half
  // participates in vshuff, so one full load is both valid and sufficient.
  return tl_hvx_loadu(scales);
#else
  uint64_t raw_words[16] __attribute__((aligned(128)));
  const volatile uint64_t *src_words =
      (const volatile uint64_t *)(const void *)scales;
  // Volatile is deliberate: without it hexagon-clang widens this exact 64B
  // boundary read back into a 128B vmemu and faults on the final RPC tile.
  raw_words[0] = src_words[0];
  raw_words[1] = src_words[1];
  raw_words[2] = src_words[2];
  raw_words[3] = src_words[3];
  raw_words[4] = src_words[4];
  raw_words[5] = src_words[5];
  raw_words[6] = src_words[6];
  raw_words[7] = src_words[7];
  const uint8_t *raw_bytes = (const uint8_t *)(const void *)raw_words;
  return *((const HVX_Vector *)raw_bytes);
#endif
}

// Duplicate each FP16 scale for the low/high nibble pair consumed by a
// weight-Crouton vector.
TL_DEVICE int tl_hexagon_q4_0_prepare_scale_32(
    __fp16 *__restrict__ dst, const uint8_t *__restrict__ scales) {
  const HVX_Vector raw_scale = tl_q4_0_load_scale_tail(scales);
  const HVX_Vector duplicated_scale =
      Q6_V_lo_W(Q6_W_vshuff_VVR(raw_scale, raw_scale, -2));
  tl_hvx_storeu(dst, duplicated_scale);
  return 0;
}

// Expand one 128-byte packed-Q4 register into four consecutive 128-byte FP16
// vectors in HMX weight-Crouton order. `duplicated_scale` is the safe 128-byte
// vector produced once per tile by tl_hexagon_q4_0_prepare_scale_32.
TL_DEVICE int tl_hexagon_q4_0_dequant_group_128(
    __fp16 *__restrict__ dst, const uint8_t *__restrict__ quant,
    const __fp16 *__restrict__ duplicated_scale) {
  const HVX_Vector packed = tl_hvx_loadu(quant);
  const HVX_Vector mask = Q6_Vb_vsplat_R(0x0f);
  const HVX_Vector zero_point = Q6_Vb_vsplat_R(8);

  HVX_Vector low = Q6_Vb_vsub_VbVb(Q6_V_vand_VV(packed, mask), zero_point);
  HVX_Vector high =
      Q6_Vb_vsub_VbVb(Q6_Vub_vlsr_VubR(packed, 4), zero_point);
  HVX_VectorPair interleaved = Q6_W_vshuff_VVR(high, low, -1);
  HVX_VectorPair values01 = Q6_Wh_vunpack_Vb(Q6_V_lo_W(interleaved));
  HVX_VectorPair values23 = Q6_Wh_vunpack_Vb(Q6_V_hi_W(interleaved));

  const HVX_Vector scale = tl_hvx_loadu(duplicated_scale);

  tl_hvx_storeu(dst + 0 * 64,
                tl_q4_0_scale_i16_to_f16(Q6_V_lo_W(values01),
                                          scale));
  tl_hvx_storeu(dst + 1 * 64,
                tl_q4_0_scale_i16_to_f16(Q6_V_hi_W(values01),
                                          scale));
  tl_hvx_storeu(dst + 2 * 64,
                tl_q4_0_scale_i16_to_f16(Q6_V_lo_W(values23),
                                          scale));
  tl_hvx_storeu(dst + 3 * 64,
                tl_q4_0_scale_i16_to_f16(Q6_V_hi_W(values23),
                                          scale));
  return 0;
}

// Expand one complete native 576-byte Q4_0 tile into a 32x32 FP16 weight
// Crouton.  A tile is the useful instruction-scheduling boundary on v79: four
// packed HVX vectors are independent, so keeping all of them live lets clang
// overlap their extract/shuffle/unpack/multiply chains instead of serializing
// four calls to the single-register atom above.
TL_DEVICE int tl_hexagon_q4_0_dequant_tile_32x32(
    __fp16 *__restrict__ dst, const uint8_t *__restrict__ tile) {
  const HVX_Vector mask = Q6_Vb_vsplat_R(0x0f);
  const HVX_Vector zero_point = Q6_Vb_vsplat_R(8);
  const HVX_Vector raw_scale =
      tl_q4_0_load_scale_tail(tile + TL_Q4_0_QUANT_BYTES);
  const HVX_Vector scale =
      Q6_V_lo_W(Q6_W_vshuff_VVR(raw_scale, raw_scale, -2));

  const HVX_Vector q0 = tl_hvx_loadu(tile + 0 * TL_Q4_0_GROUP_BYTES);
  const HVX_Vector q1 = tl_hvx_loadu(tile + 1 * TL_Q4_0_GROUP_BYTES);
  const HVX_Vector q2 = tl_hvx_loadu(tile + 2 * TL_Q4_0_GROUP_BYTES);
  const HVX_Vector q3 = tl_hvx_loadu(tile + 3 * TL_Q4_0_GROUP_BYTES);

  const HVX_Vector lo0 =
      Q6_Vb_vsub_VbVb(Q6_V_vand_VV(q0, mask), zero_point);
  const HVX_Vector lo1 =
      Q6_Vb_vsub_VbVb(Q6_V_vand_VV(q1, mask), zero_point);
  const HVX_Vector lo2 =
      Q6_Vb_vsub_VbVb(Q6_V_vand_VV(q2, mask), zero_point);
  const HVX_Vector lo3 =
      Q6_Vb_vsub_VbVb(Q6_V_vand_VV(q3, mask), zero_point);
  const HVX_Vector hi0 =
      Q6_Vb_vsub_VbVb(Q6_Vub_vlsr_VubR(q0, 4), zero_point);
  const HVX_Vector hi1 =
      Q6_Vb_vsub_VbVb(Q6_Vub_vlsr_VubR(q1, 4), zero_point);
  const HVX_Vector hi2 =
      Q6_Vb_vsub_VbVb(Q6_Vub_vlsr_VubR(q2, 4), zero_point);
  const HVX_Vector hi3 =
      Q6_Vb_vsub_VbVb(Q6_Vub_vlsr_VubR(q3, 4), zero_point);

  const HVX_VectorPair shuf0 = Q6_W_vshuff_VVR(hi0, lo0, -1);
  const HVX_VectorPair shuf1 = Q6_W_vshuff_VVR(hi1, lo1, -1);
  const HVX_VectorPair shuf2 = Q6_W_vshuff_VVR(hi2, lo2, -1);
  const HVX_VectorPair shuf3 = Q6_W_vshuff_VVR(hi3, lo3, -1);

  const HVX_VectorPair v00 = Q6_Wh_vunpack_Vb(Q6_V_lo_W(shuf0));
  const HVX_VectorPair v01 = Q6_Wh_vunpack_Vb(Q6_V_hi_W(shuf0));
  const HVX_VectorPair v10 = Q6_Wh_vunpack_Vb(Q6_V_lo_W(shuf1));
  const HVX_VectorPair v11 = Q6_Wh_vunpack_Vb(Q6_V_hi_W(shuf1));
  const HVX_VectorPair v20 = Q6_Wh_vunpack_Vb(Q6_V_lo_W(shuf2));
  const HVX_VectorPair v21 = Q6_Wh_vunpack_Vb(Q6_V_hi_W(shuf2));
  const HVX_VectorPair v30 = Q6_Wh_vunpack_Vb(Q6_V_lo_W(shuf3));
  const HVX_VectorPair v31 = Q6_Wh_vunpack_Vb(Q6_V_hi_W(shuf3));

#define TL_Q4_0_STORE_PAIR(pair, index)                                        \
  tl_hvx_storeu(dst + ((index) + 0) * 64,                                     \
                tl_q4_0_scale_i16_to_f16(Q6_V_lo_W(pair), scale));             \
  tl_hvx_storeu(dst + ((index) + 1) * 64,                                     \
                tl_q4_0_scale_i16_to_f16(Q6_V_hi_W(pair), scale))

  TL_Q4_0_STORE_PAIR(v00, 0);
  TL_Q4_0_STORE_PAIR(v01, 2);
  TL_Q4_0_STORE_PAIR(v10, 4);
  TL_Q4_0_STORE_PAIR(v11, 6);
  TL_Q4_0_STORE_PAIR(v20, 8);
  TL_Q4_0_STORE_PAIR(v21, 10);
  TL_Q4_0_STORE_PAIR(v30, 12);
  TL_Q4_0_STORE_PAIR(v31, 14);

#undef TL_Q4_0_STORE_PAIR
  return 0;
}
