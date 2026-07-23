#pragma once
// Q8_0 HVX instruction atoms for tilelang-generated Hexagon kernels.
//
// This header deliberately stops at the 32-output dot atom.  DMA descriptor
// ownership, VTCM buffering, activation quantization, and worker scheduling are
// operation/runtime concerns; embedding runtimes can keep those policies while
// reusing this target instruction implementation.

#include <tl_templates/hexagon/hvx_math.h>

#include <stdint.h>

#define TL_Q8_0_BLOCK_K 32
#define TL_Q8_0_BLOCK_N 32
#define TL_Q8_0_STAGED_TILE_BYTES 1152
#define TL_Q8_0_SCALE_OFFSET 1024

// Multiply the lower 32 fp16 lanes and return 32 IEEE fp32 lanes.  The scale
// vectors contain one weight scale per output row and a replicated activation
// scale, so only those lower lanes are part of the Q8_0 contract.
TL_DEVICE HVX_Vector tl_q8_0_mul_scale_lower32(HVX_Vector a, HVX_Vector b) {
#if __HVX_ARCH__ >= 79
  HVX_VectorPair p = Q6_Wsf_vmpy_VhfVhf(a, b);
  return Q6_V_lo_W(Q6_W_vshuff_VVR(Q6_V_hi_W(p), Q6_V_lo_W(p), -4));
#else
  HVX_VectorPair p = Q6_Wqf32_vmpy_VhfVhf(a, b);
  HVX_Vector hi = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(p));
  HVX_Vector lo = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(p));
  return Q6_V_lo_W(Q6_W_vshuff_VVR(hi, lo, -4));
#endif
}

// One K=32 tile.  Each pair of adjacent 128-byte weight vectors is shuffled
// into four consecutive K bytes per output row; vrmpyacc then performs 32
// independent signed int8 dot products in parallel.
TL_DEVICE HVX_Vector tl_q8_0_accum_k32(const uint8_t *weight,
                                       const uint8_t *activation) {
  HVX_Vector sum = Q6_V_vzero();
#pragma unroll
  for (int group = 0; group < 8; ++group) {
    HVX_Vector raw_w = tl_hvx_loadu(weight + group * 128);
    HVX_Vector rotated_w = Q6_V_vror_VR(raw_w, 64);
    HVX_Vector packed_w =
        Q6_V_lo_W(Q6_W_vshuff_VVR(rotated_w, raw_w, -2));
    HVX_Vector replicated_x = tl_hvx_loadu(activation + group * 128);
    sum = Q6_Vw_vrmpyacc_VwVbVb(sum, packed_w, replicated_x);
  }
  return sum;
}

TL_DEVICE int tl_hexagon_q8_0_dot_32x1_impl(
    int k, float *__restrict__ dst, const uint8_t *__restrict__ weight,
    const uint8_t *__restrict__ activation, int valid_rows,
    const float *__restrict__ bias) {
  if (k <= 0 || (k % TL_Q8_0_BLOCK_K) != 0 || valid_rows < 0 ||
      valid_rows > TL_Q8_0_BLOCK_N || dst == nullptr || weight == nullptr ||
      activation == nullptr) {
    return -1;
  }
  if (valid_rows == 0) {
    return 0;
  }

  HVX_Vector accum = Q6_V_vzero();
  const int k_tiles = k / TL_Q8_0_BLOCK_K;
  for (int kt = 0; kt < k_tiles; ++kt) {
    const uint8_t *w = weight + (size_t)kt * TL_Q8_0_STAGED_TILE_BYTES;
    const uint8_t *x = activation + (size_t)kt * TL_Q8_0_STAGED_TILE_BYTES;
    HVX_Vector dot_i32 = tl_q8_0_accum_k32(w, x);
    HVX_Vector dot_f32 = Q6_Vsf_equals_Vw(dot_i32);
    HVX_Vector scale = tl_q8_0_mul_scale_lower32(
        tl_hvx_loadu(w + TL_Q8_0_SCALE_OFFSET),
        tl_hvx_loadu(x + TL_Q8_0_SCALE_OFFSET));
    accum = tl_hvx_add_sf(accum, tl_hvx_mul_sf(dot_f32, scale));
  }

  if (bias != nullptr) {
    if (valid_rows == TL_Q8_0_BLOCK_N) {
      accum = tl_hvx_add_sf(accum, tl_hvx_loadu(bias));
    } else {
      __attribute__((aligned(128))) float padded_bias[TL_Q8_0_BLOCK_N] = {};
      for (int i = 0; i < valid_rows; ++i) {
        padded_bias[i] = bias[i];
      }
      accum = tl_hvx_add_sf(accum, *(const HVX_Vector *)padded_bias);
    }
  }

  if (valid_rows == TL_Q8_0_BLOCK_N) {
    tl_hvx_storeu(dst, accum);
  } else {
    __attribute__((aligned(128))) float padded_out[TL_Q8_0_BLOCK_N];
    *(HVX_Vector *)padded_out = accum;
    for (int i = 0; i < valid_rows; ++i) {
      dst[i] = padded_out[i];
    }
  }
  return 0;
}

TL_DEVICE int tl_hexagon_q8_0_dot_32x1(
    int k, float *__restrict__ dst, const uint8_t *__restrict__ weight,
    const uint8_t *__restrict__ activation, int valid_rows,
    const float *__restrict__ bias) {
  return tl_hexagon_q8_0_dot_32x1_impl(k, dst, weight, activation, valid_rows,
                                      bias);
}

TL_DEVICE int tl_hexagon_q8_0_dot_32x1_nobias(
    int k, float *__restrict__ dst, const uint8_t *__restrict__ weight,
    const uint8_t *__restrict__ activation, int valid_rows) {
  return tl_hexagon_q8_0_dot_32x1_impl(k, dst, weight, activation, valid_rows,
                                      nullptr);
}
