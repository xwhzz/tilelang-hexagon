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

#if __HVX_ARCH__ == 73 || __HVX_ARCH__ == 75
template <typename To, typename From, int N>
__attribute__((noinline)) static void cast_default(void *dst, const void *src) {
  tl_hvx_convert_vector<To, From, N>(dst, src);
}

template <typename To, typename From, int N>
TL_HEXAGON_IEEE_ARITHMETIC __attribute__((noinline)) static void
cast_ieee(void *dst, const void *src) {
  tl_hvx_convert_vector<To, From, N>(dst, src);
}

template <typename To, typename From, int N>
static void write_cast_guards(FILE *out, const void *input) {
  for (auto cast : {cast_default<To, From, N>, cast_ieee<To, From, N>}) {
    // Unaligned separate, identical, right-overlapping and left-overlapping
    // ranges. Save the entire allocation to check bytes outside the output.
    for (int destination : {1027, 513, 515, 508}) {
      alignas(128) uint8_t bytes[2048];
      memset(bytes, 0xa5, sizeof(bytes));
      memcpy(bytes + 513, input, N * sizeof(From));
      cast(bytes + destination, bytes + 513);
      fwrite(bytes, sizeof(bytes), 1, out);
    }
  }
}

template <int N>
static void write_cast_guards(FILE *out, const uint16_t *halves,
                              const uint32_t *floats) {
  write_cast_guards<float, half, N>(out, halves);
  write_cast_guards<half, float, N>(out, floats);
}

static void write_cast_guards(FILE *out, const uint32_t *bits) {
  const uint16_t special[] = {0,      0x8000, 1,      0x8001, 0x03ff, 0x0400,
                              0x7bff, 0x7c00, 0xfc00, 0x7c01, 0x7e01, 0xfe01};
  for (int round = 0; round < 4; ++round) {
    uint16_t halves[128];
    for (int i = 0; i < 128; ++i)
      halves[i] = i * 257 + round * 8191;
    memcpy(halves, special, sizeof(special));
    const uint32_t *floats = bits + round * 128;
    write_cast_guards<32>(out, halves, floats);
    write_cast_guards<33>(out, halves, floats);
    write_cast_guards<63>(out, halves, floats);
    write_cast_guards<64>(out, halves, floats);
    write_cast_guards<65>(out, halves, floats);
    write_cast_guards<96>(out, halves, floats);
    write_cast_guards<128>(out, halves, floats);
  }
}
#endif

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
#if __HVX_ARCH__ == 73 || __HVX_ARCH__ == 75
  write_cast_guards(out, bits);
#endif

  fclose(in);
  return fclose(out);
}
