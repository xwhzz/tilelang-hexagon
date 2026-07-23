// tilelang Hexagon (cDSP) kernel
#include <tl_templates/hexagon/common.h>

#ifdef __cplusplus
extern "C"
#endif
int32_t qgemv_q8_0_k2048_kernel(uint8_t* weight, uint8_t* activation, float* bias, float* dst) {
  int bx = 0;
  for (bx = 0; bx < 1; ++bx) {
    int tx = 0;
    for (tx = 0; tx < 1; ++tx) {
      int ty = 0;
      for (ty = 0; ty < 1; ++ty) {
        int tz = 0;
        for (tz = 0; tz < 1; ++tz) {
          tl_hexagon_q8_0_dot_32x1(2048, (&(dst[0])), (&(weight[0])), (&(activation[0])), 32, (&(bias[0])));
        }
      }
    }
  }
  return TL_OK;
}

