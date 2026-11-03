// Real Crouton dequantization, exercised by
// test_tilelang_hexagon_dequant_sim.py.
#include <hexagon_standalone.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <tl_templates/hexagon/crouton.h>

constexpr int N = 128;
constexpr int K = 128;
constexpr int GROUP = 64;
constexpr int FINITE_HALVES = 63488;
constexpr int FINITE_ITERATIONS = FINITE_HALVES / (N * (K / GROUP));
constexpr int ITERATIONS = FINITE_ITERATIONS + 32;

template <bool Signed>
static void legacy_row_major(uint16_t *out, const uint16_t *scales) {
  // Independent layout-free evaluation of the expression used before this
  // change. This is diagnostic output only, never the expected RNE result.
  const uint16_t positive[16] = {0x0000, 0x3c00, 0x4000, 0x4200, 0x4400, 0x4500,
                                 0x4600, 0x4700, 0x4800, 0x4880, 0x4900, 0x4980,
                                 0x4a00, 0x4a80, 0x4b00, 0x4b80};
  const uint16_t negative[8] = {0xc800, 0xc700, 0xc600, 0xc500,
                                0xc400, 0xc200, 0xc000, 0xbc00};
  alignas(128) uint16_t code[64];
  for (int row = 0; row < N; ++row) {
    for (int col = 0; col < 64; ++col) {
      int nibble = (row * 3 + col * 5) & 15;
      code[col] =
          Signed && nibble >= 8 ? negative[nibble - 8] : positive[nibble];
    }
    HVX_Vector a = *reinterpret_cast<const HVX_Vector *>(code);
    for (int group = 0; group < K / GROUP; ++group) {
      HVX_Vector b = Q6_Vh_vsplat_R(scales[group * N + row]);
      *reinterpret_cast<HVX_Vector *>(out + row * K + group * GROUP) =
          Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(a, b));
    }
  }
}

template <bool Signed, bool Normal = false>
static void dequantize(uint16_t *out, const uint8_t *panel,
                       const uint16_t *scales) {
  // First panel exercises complete groups/R8. The remaining panels are
  // sliced at non-group boundaries and exercise the R1 remainder path.
  for (int nt = 0; nt < N / 32; ++nt) {
    half *dst = reinterpret_cast<half *>(out + nt * K * 32);
    const void *src = panel + nt * K * 16;
    const half *sfb = reinterpret_cast<const half *>(scales + nt * 32);
    if (nt == 0) {
      tl_crouton_dequant4<Signed, Normal>(dst, src, sfb, N, GROUP, 0, K / 8);
    } else {
      // Includes a call crossing the group boundary at register 8, as well
      // as partial groups on both sides. The calls cover each register once.
      tl_crouton_dequant4<Signed, Normal>(dst, src, sfb, N, GROUP, 0, 3);
      tl_crouton_dequant4<Signed, Normal>(dst, src, sfb, N, GROUP, 3, 6);
      tl_crouton_dequant4<Signed, Normal>(dst, src, sfb, N, GROUP, 9, 7);
    }
  }
}

int main(int argc, char **argv) {
  if (argc != 2)
    return 1;
  SIM_ACQUIRE_HVX;
  SIM_SET_HVX_DOUBLE_MODE;
  FILE *out = fopen(argv[1], "wb");
  if (!out)
    return 2;
  const uint32_t header[] = {0x51444846, __HVX_ARCH__, ITERATIONS, N, K, 4};
  if (fwrite(header, sizeof(header), 1, out) != 1)
    return 3;
  alignas(128) uint8_t panel[N * K / 2] = {};
  alignas(128) uint16_t scales[N * (K / GROUP)];
  alignas(128) uint16_t result[N * K], fast[N * K];
  for (int row = 0; row < N; ++row)
    for (int col = 0; col < K; ++col) {
      int nibble = (row * 3 + col * 5) & 15;
      int offset = (row / 32) * K * 16 + (col / 8) * 128 + (row % 32) * 4 +
                   (col % 8) / 2;
      panel[offset] |= nibble << (4 * (col & 1));
    }
  for (int iteration = 0; iteration < ITERATIONS; ++iteration) {
    for (int group = 0; group < K / GROUP; ++group)
      for (int row = 0; row < N; ++row) {
        int index = iteration * N * (K / GROUP) + group * N + row;
        // All positive finite patterns followed by all negative finite
        // patterns; skip only the two Inf/NaN exponent ranges.
        if (iteration < FINITE_ITERATIONS) {
          scales[group * N + row] = index < 0x7c00 ? index : index + 0x400;
        } else {
          // One exceptional scale at each lane position, surrounded by
          // normal scales, testing mixed groups on the general path.
          int lane = iteration - FINITE_ITERATIONS;
          int panel = lane / 8;
          uint16_t bits = (row + group) & 1 ? 0x3c01 : 0xbc03;
          if (group == 0 && row / 32 == panel && row % 32 == lane)
            bits = lane & 1 ? 0x8001 : 0x0001;
          if (group == 1 && row / 32 == (panel + 1) % 4 &&
              row % 32 == 31 - lane)
            bits = lane & 1 ? 0x8000 : 0x0000;
          scales[group * N + row] = bits;
        }
      }
    memset(result, 0x7e, sizeof(result));
    dequantize<true>(result, panel, scales);
    bool all_normal = true;
    for (uint16_t scale : scales)
      all_normal &= (scale & 0x7c00) != 0 && (scale & 0x7c00) != 0x7c00;
    if (all_normal) {
      dequantize<true, true>(fast, panel, scales);
      if (memcmp(fast, result, sizeof(result)))
        return 7;
    }
    if (fwrite(result, sizeof(result), 1, out) != 1)
      return 4;
    legacy_row_major<true>(result, scales);
    if (fwrite(result, sizeof(result), 1, out) != 1)
      return 4;
    memset(result, 0x7e, sizeof(result));
    dequantize<false>(result, panel, scales);
    if (all_normal) {
      dequantize<false, true>(fast, panel, scales);
      if (memcmp(fast, result, sizeof(result)))
        return 7;
    }
    if (fwrite(result, sizeof(result), 1, out) != 1)
      return 5;
    legacy_row_major<false>(result, scales);
    if (fwrite(result, sizeof(result), 1, out) != 1)
      return 5;
  }
  if (fclose(out))
    return 6;
  puts("HVX_DEQUANT_SIM_OK");
  return 0;
}
