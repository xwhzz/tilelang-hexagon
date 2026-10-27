// tilelang Hexagon (cDSP) kernel
#include <tl_templates/hexagon/common.h>

#ifdef __cplusplus
extern "C"
#endif
    int32_t matmul_kernel(half *A, half *B, half *C) {
  for (int32_t bx = 0; bx < 4; ++bx) {
    for (int32_t by = 0; by < 4; ++by) {
      uint8_t *buf_dyn_shmem = (uint8_t *)tl_vtcm_arena();
      tl_vtcm_shared_high_water = 75776u;
      void *A_hmx = tl_vtcm_exclusive(((void *)((char *)buf_dyn_shmem + 0)));
      void *B_hmx =
          tl_vtcm_exclusive(((void *)((char *)buf_dyn_shmem + 32768)));
      void *bias_vtcm =
          tl_vtcm_exclusive(((void *)((char *)buf_dyn_shmem + 65536)));
      void *C_hmx =
          tl_vtcm_exclusive(((void *)((char *)buf_dyn_shmem + 67584)));
      tl_crouton_copy<true, false>((&(((half *)A_hmx)[0])), 2, 8, 0, 0, 64, 256,
                                   (&(A[(by * 16384)])), 256, 1);
      tl_crouton_copy<true, true>((&(((half *)B_hmx)[0])), 8, 2, 0, 0, 256, 64,
                                  (&(B[(bx * 64)])), 256, 1);
      uint8_t acc[1] __attribute__((aligned(128)));
      uint8_t bias[1] __attribute__((aligned(128)));
      uint8_t cvt[1] __attribute__((aligned(128)));
      for (int32_t i = 0; i < 32; ++i) {
        ((uint32_t *)bias_vtcm)[i] = (uint32_t)15360;
        ((uint32_t *)bias_vtcm)[(i + 32)] = (uint32_t)0;
      }
      tl_hexagon_hmx_acc_acquire((&(acc[0])));
      tl_hexagon_hmx_load_bias((&(bias[0])), (&(((uint32_t *)bias_vtcm)[0])));
      for (int32_t inst_m_idx = 0; inst_m_idx < 2; ++inst_m_idx) {
        for (int32_t inst_n_idx = 0; inst_n_idx < 2; ++inst_n_idx) {
          tl_hexagon_hmx_clear_acc((&(acc[0])));
          tl_hexagon_hmx_mma_atom((&(acc[0])),
                                  (&(((half *)A_hmx)[(inst_m_idx * 8192)])),
                                  (&(((half *)B_hmx)[(inst_n_idx * 8192)])), 8);
          tl_hexagon_hmx_convert_acc((&(cvt[0])), (&(acc[0])), (&(bias[0])),
                                     (&(((uint32_t *)bias_vtcm)[0])), 2);
          tl_hexagon_hmx_store_cvt_state(
              (&(cvt[0])),
              (&(((half *)C_hmx)[((inst_m_idx * 2048) + (inst_n_idx * 1024))])),
              (&(acc[0])), (&(bias[0])), (&(((uint32_t *)bias_vtcm)[0])));
        }
      }
      tl_hexagon_hmx_acc_release((&(acc[0])));
      tl_crouton_copy<false, false>((&(((half *)C_hmx)[0])), 2, 2, 0, 0, 64, 64,
                                    (&(C[((by * 16384) + (bx * 64))])), 256, 1);
    }
  }
  return TL_OK;
}
