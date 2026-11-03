// Float-to-half Crouton packing through every vectorized input path.
#include <hexagon_standalone.h>
#include <stdint.h>
#include <stdio.h>
#include <tl_templates/hexagon/crouton.h>

int main(int argc, char **argv) {
  if (argc != 3)
    return 1;
  SIM_ACQUIRE_HVX;
  SIM_SET_HVX_DOUBLE_MODE;
  FILE *in = fopen(argv[1], "rb"), *out = fopen(argv[2], "wb");
  if (!in || !out)
    return 2;
  uint32_t count;
  if (fread(&count, sizeof(count), 1, in) != 1 || count % 1024)
    return 3;
  const uint32_t header[] = {0x4352504b, __HVX_ARCH__, count, 3};
  if (fwrite(header, sizeof(header), 1, out) != 1)
    return 4;
  alignas(128) float rows[1024];
  alignas(128) half tile[1024];
  for (uint32_t base = 0; base < count; base += 1024) {
    if (fread(rows, sizeof(rows), 1, in) != 1)
      return 5;
    for (int mode = 0; mode < 3; ++mode) {
      if (mode == 0) {
        for (int q = 0; q < 16; ++q)
          reinterpret_cast<HVX_Vector *>(tile)[q] =
              tl_crouton_interleave(rows + q * 64, rows + q * 64 + 32, 32);
      } else if (mode == 1) {
        tl_crouton_copy_tiles<true, false, true>(tile, 1, 1, 0, 0, 0, 32, 32,
                                                 rows, 32);
      } else {
        for (int q = 0; q < 16; ++q)
          reinterpret_cast<HVX_Vector *>(tile)[q] =
              tl_crouton_load_column<true>(rows + q * 64);
      }
      if (fwrite(tile, sizeof(tile), 1, out) != 1)
        return 6;
    }
  }
  return fclose(in) || fclose(out) ? 7 : 0;
}
