// tilelang Hexagon (cDSP) kernel
#include <tl_templates/hexagon/common.h>

#ifdef __cplusplus
extern "C"
#endif
int32_t matmul_kernel(half* A, half* B, half* C) {
  int bx = 0;
  for (bx = 0; bx < 4; ++bx) {
    uint8_t* buf_dyn_shmem = (uint8_t*)((char*)tl_vtcm_base() + 2048);
    tl_vtcm_shared_high_water = 77824u;
    void* A_hmx = ((void*)((char*)buf_dyn_shmem + 0));
    void* B_hmx = ((void*)((char*)buf_dyn_shmem + 32768));
    void* bias_vtcm = ((void*)((char*)buf_dyn_shmem + 65536));
    void* C_hmx = ((void*)((char*)buf_dyn_shmem + 67584));
    int by = 0;
    for (by = 0; by < 4; ++by) {
      int tx = 0;
      for (tx = 0; tx < 1; ++tx) {
        int ty = 0;
        for (ty = 0; ty < 1; ++ty) {
          int tz = 0;
          for (tz = 0; tz < 1; ++tz) {
            tl_hexagon_hmx_pack_crouton((&(((half*)A_hmx)[0])), (&(A[(by * 16384)])), 64, 256, 256, 1, 0);
            tl_hexagon_hmx_pack_crouton((&(((half*)B_hmx)[0])), (&(B[(bx * 64)])), 256, 64, 256, 1, 1);
            uint8_t acc[1];
            uint8_t bias[1];
            uint8_t cvt[1];
            for (int32_t i = 0; i < 32; ++i) {
              ((uint32_t*)bias_vtcm)[i] = (uint32_t)15360;
              ((uint32_t*)bias_vtcm)[(i + 32)] = (uint32_t)0;
            }
            tl_hexagon_hmx_acc_acquire((&(acc[0])));
            for (int32_t inst_m_idx = 0; inst_m_idx < 2; ++inst_m_idx) {
              for (int32_t inst_n_idx = 0; inst_n_idx < 2; ++inst_n_idx) {
                tl_hexagon_hmx_clear_acc((&(acc[0])));
                tl_hexagon_hmx_load_bias((&(bias[0])), (&(((uint32_t*)bias_vtcm)[0])));
                for (int32_t k_inner = 0; k_inner < 8; ++k_inner) {
                  tl_hexagon_hmx_mma_atom((&(acc[0])), (&(((half*)A_hmx)[((inst_m_idx * 8192) + (k_inner * 1024))])), (&(((half*)B_hmx)[((inst_n_idx * 8192) + (k_inner * 1024))])));
                }
                tl_hexagon_hmx_convert_acc((&(cvt[0])), (&(acc[0])), (&(bias[0])), (&(((uint32_t*)bias_vtcm)[0])), 2);
                tl_hexagon_hmx_store_cvt_state((&(cvt[0])), (&(((half*)C_hmx)[((inst_m_idx * 2048) + (inst_n_idx * 1024))])), (&(acc[0])), (&(bias[0])), (&(((uint32_t*)bias_vtcm)[0])), (&(((half*)A_hmx)[0])), (&(((half*)B_hmx)[0])));
              }
            }
            tl_hexagon_hmx_acc_release((&(acc[0])));
            tl_hexagon_hmx_unpack_crouton((&(C[((by * 16384) + (bx * 64))])), (&(((half*)C_hmx)[0])), 64, 64, 256, 1, 0);
          }
        }
      }
    }
  }
  return TL_OK;
}

