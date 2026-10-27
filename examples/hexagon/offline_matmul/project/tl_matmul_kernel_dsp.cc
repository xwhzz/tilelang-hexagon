// Auto-generated FastRPC skel impl for a tilelang Hexagon kernel.
#include "tl_matmul_kernel.h" // QAIC-generated handler prototypes
#include <AEEStdErr.h>
#include <type_traits>

#include <tl_templates/hexagon/crouton.h>
#include <tl_templates/hexagon/hmx.h>
#include <tl_templates/hexagon/vtcm.h>
// ---- the tilelang-generated device kernel ----
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

template <typename Fn>
static AEEResult tl_hexagon_forward_kernel_status(Fn &&fn) {
  if constexpr (std::is_void_v<decltype(fn())>) {
    fn();
    return AEE_SUCCESS;
  } else {
    int tl_rc = (int)fn();
    return tl_rc == 0 ? AEE_SUCCESS : AEE_EFAILED;
  }
}

#ifdef __cplusplus
extern "C" {
#endif

AEEResult tl_matmul_kernel_open(const char *uri, remote_handle64 *h) {
  (void)uri;
  *h = 0;
  tl_hmx_session_init();
  if (!tl_hmx_session_ok())
    return AEE_EFAILED;
  return AEE_SUCCESS;
}
AEEResult tl_matmul_kernel_close(remote_handle64 h) {
  (void)h;
  tl_hmx_session_deinit();
  return AEE_SUCCESS;
}

AEEResult tl_matmul_kernel_run(remote_handle64 _h, const unsigned char *p0,
                               int p0Len, const unsigned char *p1, int p1Len,
                               unsigned char *p2, int p2Len) {
  return tl_hexagon_forward_kernel_status([&]() {
    return matmul_kernel((__fp16 *)p0, (__fp16 *)p1, (__fp16 *)p2);
  });
}

#ifdef __cplusplus
}
#endif
