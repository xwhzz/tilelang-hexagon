// tilelang Hexagon (cDSP) kernel
#include <tl_templates/hexagon/common.h>

#ifdef __cplusplus
extern "C"
#endif
int32_t qmatmul_compact_kernel(half* A, uint8_t* qcm, half* sc, half* C) {
  int bx = 0;
  for (bx = 0; bx < 1; ++bx) {
    uint8_t* buf_dyn_shmem = (uint8_t*)((char*)tl_vtcm_base() + 2048);
    tl_vtcm_shared_high_water = 665600u;
    void* A_sh = ((void*)((char*)buf_dyn_shmem + 0));
    void* B_sh = ((void*)((char*)buf_dyn_shmem + 131072));
    void* C_sh = ((void*)((char*)buf_dyn_shmem + 655360));
    int tx = 0;
    for (tx = 0; tx < 1; ++tx) {
      int ty = 0;
      for (ty = 0; ty < 1; ++ty) {
        int tz = 0;
        for (tz = 0; tz < 1; ++tz) {
          for (int32_t i = 0; i < 1024; ++i) {
            *(half64*)(((half*)A_sh) + (i * 64)) = *(half64*)(A + (i * 64));
          }
          for (int32_t j = 0; j < 1024; ++j) {
            int16_t128 q = ((int16_t128)*(uint8_t128*)(qcm + (j * 128)));
            int16_t broadcast_var = (int16_t)15;
            int16_t broadcast_var_1 = (int16_t)8;
            *(half128*)(((half*)B_sh) + (j * 256)) = (((half128)((q  &  ((int16_t128)(broadcast_var))) - ((int16_t128)(broadcast_var_1)))) * *(half128*)(sc + ((j >> 4) * 128)));
            int16_t broadcast_var_2 = (int16_t)4;
            int16_t broadcast_var_3 = (int16_t)8;
            *(half128*)(((half*)B_sh) + ((j * 256) + 128)) = (((half128)((q  >>  ((int16_t128)(broadcast_var_2))) - ((int16_t128)(broadcast_var_3)))) * *(half128*)(sc + ((j >> 4) * 128)));
          }
          tl_hexagon_hmx_gemm((&(((half*)C_sh)[0])), (&(((half*)A_sh)[0])), (&(((half*)B_sh)[0])), 32, 128, 2048, 0, 0);
          for (int32_t i_1 = 0; i_1 < 64; ++i_1) {
            *(half64*)(C + (i_1 * 64)) = *(half64*)(((half*)C_sh) + (i_1 * 64));
          }
        }
      }
    }
  }
  return TL_OK;
}

