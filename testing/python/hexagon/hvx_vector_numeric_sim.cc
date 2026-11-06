// Generic generated vector operations, executed by the SDK simulator.
#include <hexagon_standalone.h>
#include <stdint.h>
#include <stdio.h>
#include <tl_templates/hexagon/hvx_math.h>

template <int N> static void write_casts(FILE *out, const uint32_t *bits) {
  alignas(128) uint16_t half_bits[128];
  for (int base = 0; base < 65536; base += N) {
    for (int i = 0; i < N; ++i)
      half_bits[i] = base + i;
    vec_type<half, N> input;
    memcpy(&input, half_bits, sizeof(input));
    vec_type<float, N> result(input);
    fwrite(&result, sizeof(result), 1, out);
  }
  // All widths exercise mixed signs, magnitudes and special values. The full
  // float corpus below additionally covers every half midpoint at width 64.
  for (int base = 0; base < 1024; base += N) {
    vec_type<float, N> input;
    memcpy(&input, bits + base, sizeof(input));
    vec_type<half, N> result(input);
    fwrite(&result, sizeof(result), 1, out);
  }
}

template <typename A, typename D>
static void write_stores(FILE *out, const half *halves) {
  constexpr int lanes = tl_hvx_lanes<A>;
  alignas(128) A input[lanes];
  alignas(128) D dst[128];
  for (int i = 0; i < lanes; ++i)
    input[i] = halves[i];
  tl_hvx_reg<A> value;
  memcpy(&value, input, sizeof(value));
  for (int n : {1, 15, 31, 32, 33, 63, 64}) {
    if (n > lanes)
      continue;
    for (int stride : {1, 2}) {
      for (bool clear : {true, false}) {
        for (int i = 0; i < 128; ++i)
          dst[i] = static_cast<D>(halves[(i + 17) % 64]);
        tl_reduce_store<TL_REDUCE_MAX, A>(dst, stride, value, n, clear);
        fwrite(dst, sizeof(dst), 1, out);
      }
    }
  }
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
  if (fread(&count, sizeof(count), 1, in) != 1 || count % 128 || count < 1024)
    return 3;
  alignas(128) static uint32_t bits[1 << 19];
  if (count > (1 << 19) || fread(bits, 4, count, in) != count)
    return 4;
  write_casts<2>(out, bits);
  write_casts<4>(out, bits);
  write_casts<8>(out, bits);
  write_casts<16>(out, bits);
  write_casts<32>(out, bits);
  write_casts<64>(out, bits);
  write_casts<128>(out, bits);
  for (uint32_t base = 0; base < count; base += 64) {
    float64 input;
    memcpy(&input, bits + base, sizeof(input));
    half64 result(input);
    fwrite(&result, sizeof(result), 1, out);
  }
  alignas(128) half halves[64];
  for (int i = 0; i < 64; ++i) {
    // Exactly representable source values with different adjacent mantissas.
    uint16_t h = (i % 2 ? 0xbc01 : 0x3c03) + 13 * i;
    memcpy(halves + i, &h, sizeof(h));
  }
  write_stores<half, float>(out, halves);
  write_stores<float, half>(out, halves);

  fclose(in);
  return fclose(out);
}
