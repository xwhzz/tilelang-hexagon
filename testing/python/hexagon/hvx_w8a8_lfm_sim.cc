#include <cstdio>
#include <cstdlib>
#include <tl_templates/hexagon/w8a8.h>

constexpr int K = 2048, Block = 128, MaxRows = 128;
alignas(128) static half a[MaxRows * K], scales[K / 64 * Block], tmp[64 * 32];
alignas(128) static uint8_t aq[MaxRows * K], weights[Block * K / 2];
alignas(128) static int8_t wq[Block / 32][K * 32];
alignas(128) static float sa[MaxRows], sw[Block];
alignas(128) static int32_t correction[Block];

int main(int argc, char **argv) {
  if (argc != 5)
    return 1;
  const int rows = std::atoi(argv[3]);
  const bool normal = std::atoi(argv[4]) != 0;
  if (rows != 32 && rows != MaxRows)
    return 2;

  FILE *in = std::fopen(argv[1], "rb");
  if (!in || std::fread(a, sizeof(half), rows * K, in) != rows * K ||
      std::fread(weights, 1, sizeof(weights), in) != sizeof(weights) ||
      std::fread(scales, 1, sizeof(scales), in) != sizeof(scales))
    return 3;
  std::fclose(in);

  for (int worker = 0; worker < 4; ++worker)
    tl_hexagon_quantize_crouton_u8(a, aq, sa, K, rows, worker * rows / 4,
                                   rows / 4);
  for (int panel = 0; panel < Block / 32; ++panel)
    tl_hexagon_requantize_panel_s4_s8(
        weights + panel * K * 16, scales + panel * 32, tmp, wq[panel],
        sw + panel * 32, correction + panel * 32, K, Block, normal);

  FILE *out = std::fopen(argv[2], "wb");
  if (!out)
    return 4;
  const size_t a_bytes = (rows < 64 ? 64 : rows) * K;
  if (std::fwrite(aq, 1, a_bytes, out) != a_bytes ||
      std::fwrite(wq, 1, sizeof(wq), out) != sizeof(wq) ||
      std::fwrite(sa, sizeof(float), rows, out) != rows ||
      std::fwrite(sw, 1, sizeof(sw), out) != sizeof(sw) ||
      std::fwrite(correction, 1, sizeof(correction), out) !=
          sizeof(correction) ||
      std::fclose(out))
    return 5;
  std::puts("LFM_W8A8_PACK_OK");
  return 0;
}
