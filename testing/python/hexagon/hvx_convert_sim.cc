// Executed by test_tilelang_hexagon_convert_sim.py on the real SDK simulator.
#include <hexagon_standalone.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <tl_templates/hexagon/hvx_math.h>

static HVX_Vector load_vector(const void *p) {
  HVX_Vector result;
  memcpy(&result, p, sizeof(result));
  return result;
}

static void write_widen(FILE *out, HVX_VectorPair pair) {
  alignas(128) uint32_t even[32], odd[32], ordered[64];
  HVX_Vector lo = Q6_V_lo_W(pair), hi = Q6_V_hi_W(pair);
  memcpy(even, &lo, 128);
  memcpy(odd, &hi, 128);
  for (int i = 0; i < 32; ++i) {
    ordered[2 * i] = even[i];
    ordered[2 * i + 1] = odd[i];
  }
  fwrite(ordered, sizeof(ordered), 1, out);
}

int main(int argc, char **argv) {
  if (argc != 3)
    return 1;
  SIM_ACQUIRE_HVX;
  SIM_SET_HVX_DOUBLE_MODE;
  FILE *in = fopen(argv[1], "rb"), *out = fopen(argv[2], "wb");
  if (!in || !out)
    return 2;
  uint32_t count;
  if (fread(&count, sizeof(count), 1, in) != 1 || count % 64)
    return 3;
  uint32_t header[3] = {0x48565843, __HVX_ARCH__, count};
  fwrite(header, sizeof(header), 1, out);

  // Every half bit pattern, with distinct neighbouring lanes. The math
  // wrappers are exercised as well as the conversion helpers they call.
  for (uint32_t base = 0; base < 65536; base += 64) {
    alignas(128) uint16_t values[64];
    for (int i = 0; i < 64; ++i)
      values[i] = base + i;
    HVX_Vector h = load_vector(values), lo, hi;
    tl_hvx_widen_hf(h, &lo, &hi);
    write_widen(out, Q6_W_vcombine_VV(hi, lo));
    HVX_Vector back = tl_hvx_narrow_hf(lo, hi);
    fwrite(&back, sizeof(back), 1, out);
#if __HVX_ARCH__ >= 79
    // The exact old v79 operations, including their NaN canonicalization.
    HVX_VectorPair native = Q6_Wsf_vcvt_Vhf(h);
    write_widen(out, native);
    HVX_Vector native_back =
        Q6_Vhf_vcvt_VsfVsf(Q6_V_lo_W(native), Q6_V_hi_W(native));
    fwrite(&native_back, sizeof(native_back), 1, out);
#endif
  }

  for (uint32_t base = 0; base < count; base += 64) {
    alignas(128) uint32_t bits[64], even[32], odd[32];
    if (fread(bits, sizeof(bits), 1, in) != 1)
      return 4;
    for (int i = 0; i < 32; ++i) {
      even[i] = bits[2 * i];
      odd[i] = bits[2 * i + 1];
    }
    HVX_Vector lo = load_vector(even), hi = load_vector(odd);
    HVX_Vector h = tl_hvx_convert_sf_to_hf(lo, hi);
    fwrite(&h, sizeof(h), 1, out);
#if __HVX_ARCH__ >= 79
    HVX_Vector native = Q6_Vhf_vcvt_VsfVsf(lo, hi);
    fwrite(&native, sizeof(native), 1, out);
#endif
  }

  alignas(128) float x[32];
  for (int i = 0; i < 32; ++i)
    x[i] = (i + 1) / 8.0f;
  HVX_Vector v = load_vector(x);
  HVX_Vector math[] = {
      Q6_Vsf_equals_Vqf32(tl_hvx_mul_scalar_qf32(v, tl_hvx_float_bits(0.5f))),
      tl_hvx_rsqrt_fast_vsf(v),
      tl_hvx_exp2_fast_vsf(v),
      tl_hvx_exp_fast_vsf(v),
      tl_hvx_log2_fast_vsf(v),
      tl_hvx_log_fast_vsf(v),
      tl_hvx_tanh_fast_vsf(v),
      tl_hvx_erf_fast_vsf(v)};
  fwrite(math, sizeof(math), 1, out);

  // Include short rows, odd tails, and the four-accumulator qfloat path.
  alignas(128) half row[320];
  for (int i = 0; i < 320; ++i)
    row[i] = (i % 32 + 1) / 8.0f;
  for (int n : {63, 64, 65, 255, 256, 257}) {
    float value;
    tl_hvx_reduce_rows<TL_REDUCE_SUM, float, true>(&value, row, 1, n, n, 1,
                                                   true);
    fwrite(&value, sizeof(value), 1, out);
  }
  if (ferror(in) || ferror(out))
    return 5;
  fclose(in);
  if (fclose(out))
    return 6;
  puts("HVX_CONVERT_SIM_OK");
  return 0;
}
