// tilelang Hexagon (cDSP) kernel
#include <tl_templates/hexagon/common.h>

#ifdef __cplusplus
extern "C"
#endif
int32_t matmul_kernel(half* A, half* B, half* C) {
  int bx = 0;
  for (bx = 0; bx < 4; ++bx) {
    uint8_t* buf_dyn_shmem = (uint8_t*)((char*)tl_vtcm_base() + 2048);
    tl_vtcm_shared_high_water = 75776u;
    void* A_sh = ((void*)((char*)buf_dyn_shmem + 0));
    void* B_sh = ((void*)((char*)buf_dyn_shmem + 32768));
    void* C_sh = ((void*)((char*)buf_dyn_shmem + 65536));
    int by = 0;
    for (by = 0; by < 4; ++by) {
      int tx = 0;
      for (tx = 0; tx < 1; ++tx) {
        int ty = 0;
        for (ty = 0; ty < 1; ++ty) {
          int tz = 0;
          for (tz = 0; tz < 1; ++tz) {
            for (int32_t i = 0; i < 2048; ++i) {
              *(half8*)(((half*)A_sh) + (i * 8)) = *(half8*)(A + ((by * 16384) + (i * 8)));
            }
            for (int32_t i_1 = 0; i_1 < 2048; ++i_1) {
              *(half8*)(((half*)B_sh) + (i_1 * 8)) = *(half8*)(B + ((((i_1 >> 3) * 256) + (bx * 64)) + ((i_1 & 7) * 8)));
            }
            tl_hexagon_hmx_gemm((&(((half*)C_sh)[0])), (&(((half*)A_sh)[0])), (&(((half*)B_sh)[0])), 64, 64, 256, 0, 0);
            for (int32_t i_2 = 0; i_2 < 512; ++i_2) {
              *(half8*)(C + ((((by * 16384) + ((i_2 >> 3) * 256)) + (bx * 64)) + ((i_2 & 7) * 8))) = *(half8*)(((half*)C_sh) + (i_2 * 8));
            }
          }
        }
      }
    }
  }
  return TL_OK;
}

