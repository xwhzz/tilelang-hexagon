// Executed by test_tilelang_hexagon_math_sim.py on the SDK simulator.
#include <hexagon_standalone.h>
#include <stdint.h>
#include <stdio.h>
#include <tl_templates/hexagon/hvx_math.h>

int main(int argc, char **argv) {
  if (argc != 3)
    return 1;
  SIM_ACQUIRE_HVX;
  SIM_SET_HVX_DOUBLE_MODE;
  FILE *in = fopen(argv[1], "rb"), *out = fopen(argv[2], "wb");
  if (!in || !out)
    return 2;
  uint32_t counts[4];
  if (fread(counts, sizeof(counts), 1, in) != 1 || counts[0] % 32 ||
      counts[1] % 64 || counts[2] % 64 || counts[3] % 64)
    return 3;
  uint32_t header[2] = {0x4856584d, __HVX_ARCH__};
  fwrite(header, sizeof(header), 1, out);
  fwrite(counts, sizeof(counts), 1, out);

  for (uint32_t base = 0; base < counts[0]; base += 32) {
    alignas(128) HVX_Vector x;
    if (fread(&x, sizeof(x), 1, in) != 1)
      return 4;
    HVX_Vector result[] = {tl_hvx_log_vsf(x), tl_hvx_log2_vsf(x),
                           tl_hvx_log_fast_vsf(x), tl_hvx_log2_fast_vsf(x)};
    fwrite(result, sizeof(result), 1, out);
  }
  for (int function = 0; function < 3; ++function) {
    for (uint32_t base = 0; base < counts[function + 1]; base += 64) {
      alignas(128) vec_type<half, 64> x;
      if (fread(&x.v, sizeof(x.v), 1, in) != 1)
        return 4;
      auto result = function == 0   ? tl_hvx_exp_fast(x)
                    : function == 1 ? tl_hvx_exp2_fast(x)
                                    : tl_hvx_sigmoid_fast(x);
      fwrite(&result.v, sizeof(result.v), 1, out);
    }
  }
  if (ferror(in) || ferror(out))
    return 5;
  fclose(in);
  if (fclose(out))
    return 6;
  puts("HVX_MATH_SIM_OK");
  return 0;
}
