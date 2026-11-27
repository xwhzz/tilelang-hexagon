#pragma once

#include <hvx_hexagon_protos.h>
#include <tl_templates/hexagon/hmx.h>

// One unsigned-byte 64x32 activation tile times one signed-byte 32x32
// weight tile per 32 channels. :cm takes row-major activations. Each weight
// vector stores 32 columns, with four consecutive K bytes per column.
static inline void tl_hexagon_hmx_gemm_u8i8(int k_tiles, const uint8_t *a,
                                            const int8_t *b, int32_t *c,
                                            uint8_t *planes, uint32_t *config) {
  constexpr int activation_bytes = 64 * 32;
  constexpr int weight_bytes = 32 * 32;
  constexpr int config_words = 256;
  constexpr int plane_vectors = activation_bytes / sizeof(HVX_Vector);

  // Each conversion extracts one byte of the accumulator with no saturation.
  // The bank selects a bit shift of 0, 8, 16 or 24 through its column scales.
  memset(config, 0, 4 * config_words * sizeof(*config));
  for (int byte = 0; byte < 3; ++byte) {
    auto *scale = reinterpret_cast<HVX_Vector *>(config + byte * config_words);
    const auto value = Q6_V_vsplat_R((3 - byte) * 0x2000);
    scale[0] = value;
    scale[1] = value;
  }

  tl_hmx_unit_acquire();
  Q6_mxclracc();
  for (int t = 0; t < k_tiles; t += 32) {
    // The deep activation range counts up to 32 tiles, each with 32 channels.
    // The weight range ends at the last 128-byte vector in the same group.
    const int tiles = k_tiles - t < 32 ? k_tiles - t : 32;
    const unsigned act_range = ((tiles - 1) << 11) | 31;
    const unsigned weight_range = tiles * weight_bytes - 128;
    const uint8_t *act = a + t * activation_bytes;
    const int8_t *weight = b + t * weight_bytes;
    asm volatile("{ activation.ub = mxmem(%0, %1):deep:cm\n"
                 "  weight.b = mxmem(%2, %3) }\n" ::"r"(act),
                 "r"(act_range), "r"(weight), "r"(weight_range)
                 : "memory");
  }
  for (int byte = 0; byte < 4; ++byte) {
    Q6_bias_mxmem_A(config + byte * config_words);
    Q6_mxmem_AR_after_retain_cm_ub(planes + byte * activation_bytes, 0);
  }
  tl_hmx_unit_release();

  // The planes have the same row-major order as C. Interleave pairs of
  // bytes, then pairs of halfwords, to preserve every accumulator bit.
  const auto *packed = reinterpret_cast<const HVX_Vector *>(planes);
  auto *output = reinterpret_cast<HVX_Vector *>(c);
  for (int i = 0; i < plane_vectors; ++i) {
    auto low = Q6_W_vshuff_VVR(packed[plane_vectors + i], packed[i], -1);
    auto high = Q6_W_vshuff_VVR(packed[3 * plane_vectors + i],
                                packed[2 * plane_vectors + i], -1);
    auto first = Q6_W_vshuff_VVR(Q6_V_lo_W(high), Q6_V_lo_W(low), -2);
    auto last = Q6_W_vshuff_VVR(Q6_V_hi_W(high), Q6_V_hi_W(low), -2);
    output[4 * i] = Q6_V_lo_W(first);
    output[4 * i + 1] = Q6_V_hi_W(first);
    output[4 * i + 2] = Q6_V_lo_W(last);
    output[4 * i + 3] = Q6_V_hi_W(last);
  }
}
