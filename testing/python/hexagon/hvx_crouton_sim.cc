// Exercise unscaled Crouton widening through its public copy/store helpers.
#include <hexagon_standalone.h>
#include <stdint.h>
#include <stdio.h>
#include <tl_templates/hexagon/crouton.h>

int main(int argc, char **argv) {
  if (argc != 2)
    return 1;
  SIM_ACQUIRE_HVX;
  SIM_SET_HVX_DOUBLE_MODE;
  FILE *out = fopen(argv[1], "wb");
  if (!out)
    return 2;
  alignas(128) uint16_t tile[1024];
  alignas(128) float rows[1024];
  for (int base = 0; base < 65536; base += 1024) {
    for (int i = 0; i < 1024; ++i)
      tile[i] = base + i;
    const half *h = reinterpret_cast<const half *>(tile);
    for (int mode = 0; mode < 4; ++mode) {
      if (mode == 0) {
        for (int q = 0; q < 16; ++q)
          tl_crouton_deinterleave<float>(
              reinterpret_cast<const HVX_Vector *>(h)[q], rows + q * 64,
              rows + q * 64 + 32, 32);
      } else if (mode == 1) {
        tl_crouton_copy_tiles<false, false, true>(const_cast<half *>(h), 1, 1,
                                                  0, 0, 0, 32, 32, rows, 32);
      } else if (mode == 2) {
        tl_crouton_tile_store<false, true, true, false>(rows, 32, h);
      } else {
        for (int q = 0; q < 16; ++q) {
          tl_hvx_reg<float> even, odd;
          tl_crouton_rows_of<0, float>(
              reinterpret_cast<const HVX_Vector *>(h)[q], &even, &odd);
          __builtin_memcpy(rows + q * 64, &even, 128);
          __builtin_memcpy(rows + q * 64 + 32, &odd, 128);
        }
      }
      if (fwrite(rows, sizeof(rows), 1, out) != 1)
        return 3;
    }
  }
  return fclose(out) ? 4 : 0;
}
