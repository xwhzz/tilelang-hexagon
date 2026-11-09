// Exercise conversions inside the same local feature scope as generated code.
#include <hexagon_standalone.h>
#include <stdio.h>
#include <tl_templates/hexagon/common.h>

template <typename T, int N>
TL_HEXAGON_IEEE_ARITHMETIC __attribute__((noinline)) static void
cast_rows(const half *input, void *output, uint32_t count) {
  for (uint32_t base = 0; base < count; base += N) {
    vec_type<half, N> h;
    memcpy(&h, input + base, sizeof(h));
    vec_type<T, N> integer(h);
    memcpy(static_cast<char *>(output) + base * sizeof(T), &integer,
           sizeof(integer));
  }
}

alignas(128) static half input[65536];
alignas(128) static unsigned char output[65536 * 8 + 128];

template <typename T, int N> static int write_rows(FILE *out, uint32_t count) {
  memset(output, 0xa5, sizeof(output));
  cast_rows<T, N>(input, output, count);
  return fwrite(output, count * sizeof(T) + 128, 1, out) == 1 ? 0 : 1;
}

template <typename T> static int check_type(FILE *in, FILE *out) {
  uint32_t count;
  if (fread(&count, 4, 1, in) != 1 || !count || count > 65536 || count % 128 ||
      fread(input, sizeof(half), count, in) != count)
    return 1;
  return write_rows<T, 2>(out, count) || write_rows<T, 4>(out, count) ||
         write_rows<T, 8>(out, count) || write_rows<T, 16>(out, count) ||
         write_rows<T, 32>(out, count) || write_rows<T, 64>(out, count) ||
         write_rows<T, 128>(out, count);
}

template <int N>
TL_HEXAGON_IEEE_ARITHMETIC __attribute__((noinline)) static int
write_bools(FILE *out) {
  for (int base = 0; base < 65536; base += N) {
    uint16_t bits[N];
    for (int i = 0; i < N; ++i)
      bits[i] = base + i;
    vec_type<half, N> h;
    memcpy(&h, bits, sizeof(h));
    vec_type<bool, N> converted(h);
    uint8_t values[N];
    for (int i = 0; i < N; ++i)
      values[i] = converted.v[i];
    if (fwrite(values, sizeof(values), 1, out) != 1)
      return 1;
  }
  return 0;
}

int main(int argc, char **argv) {
  if (argc != 3)
    return 1;
  SIM_ACQUIRE_HVX;
  SIM_SET_HVX_DOUBLE_MODE;
  FILE *in = fopen(argv[1], "rb"), *out = fopen(argv[2], "wb");
  if (!in || !out)
    return 2;
  int error = check_type<int8_t>(in, out) || check_type<uint8_t>(in, out) ||
              check_type<int16_t>(in, out) || check_type<uint16_t>(in, out) ||
              check_type<int32_t>(in, out) || check_type<uint32_t>(in, out) ||
              check_type<int64_t>(in, out) || check_type<uint64_t>(in, out);
  error = error || write_bools<2>(out) || write_bools<4>(out) ||
          write_bools<8>(out) || write_bools<16>(out) || write_bools<32>(out) ||
          write_bools<64>(out) || write_bools<128>(out);
  fclose(in);
  return fclose(out) || error;
}
