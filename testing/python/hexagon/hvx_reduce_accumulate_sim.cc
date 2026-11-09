// Runs generated reductions against their original ordered fp32 arithmetic.
#include <hexagon_standalone.h>
#include <stdint.h>
#include <stdio.h>
#include <tl_templates/hexagon/common.h>

alignas(2048) static uint8_t arena[16384];
static unsigned int tl_vtcm_shared_high_water;
static uint8_t *tl_vtcm_arena() { return arena; }

struct ReductionCase {
  int rows;
  bool maximum;
  bool vectorized;
  int32_t (*kernel)(half *, float *, float *);
};

#include "generated_reductions.inc"

// Widen each input with scalar conversion, independently of the generated
// vector casts. Vector max/min and scalar comparisons have distinct NaN and
// signed-zero semantics, so preserve both original accumulation orders.
static void reference(const half *input, const float *initial, float *output,
                      int rows, bool maximum, bool vectorized) {
  memcpy(output, initial, 76 * sizeof(float));
  typedef float wide_t __attribute__((ext_vector_type(64)));
  wide_t acc;
  memcpy(&acc, initial + 2, sizeof(acc));
  for (int row = 0; vectorized && row < rows; ++row) {
    wide_t value;
#pragma clang loop vectorize(disable)
    for (int lane = 0; lane < 64; ++lane)
      value[lane] = static_cast<float>(input[row * 80 + 3 + lane]);
    acc = maximum ? __builtin_elementwise_max(acc, value)
                  : __builtin_elementwise_min(acc, value);
  }
  memcpy(output + 2, &acc, sizeof(acc));
  for (int lane = vectorized ? 64 : 0; lane < 70; ++lane) {
    float value = initial[lane + 2];
    for (int row = 0; row < rows; ++row) {
      float next = static_cast<float>(input[row * 80 + 3 + lane]);
      value = maximum ? (value > next ? value : next)
                      : (value < next ? value : next);
    }
    output[lane + 2] = value;
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
  uint32_t batches;
  if (fread(&batches, sizeof(batches), 1, in) != 1)
    return 3;
  alignas(128) half input[48 * 80];
  alignas(128) float initial[76], actual[76], expected[76];
  for (const auto &test : reduction_cases) {
    for (uint32_t batch = 0; batch < batches; ++batch) {
      if (fread(input, sizeof(half), test.rows * 80, in) != test.rows * 80 ||
          fread(initial, sizeof(float), 76, in) != 76)
        return 4;
      if (test.kernel(input, initial, actual) != 0 ||
          tl_vtcm_shared_high_water > sizeof(arena))
        return 5;
      reference(input, initial, expected, test.rows, test.maximum,
                test.vectorized);
      if (fwrite(actual, sizeof(actual), 1, out) != 1 ||
          fwrite(expected, sizeof(expected), 1, out) != 1)
        return 6;
    }
  }
  if (fgetc(in) != EOF || ferror(in))
    return 7;
  fclose(in);
  return fclose(out);
}
