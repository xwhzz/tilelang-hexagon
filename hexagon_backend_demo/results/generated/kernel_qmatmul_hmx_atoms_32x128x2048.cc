// Generated TileLang kernel plus its same-TU llama.cpp runtime bridge.
#include <tl_templates/hexagon/tl_bridge.h>
#define TL_Q4_0_PADDED_VTCM_INPUT 1
#include <tl_templates/hexagon/qmatmul.h>

// tilelang Hexagon (cDSP) kernel
#include <tl_templates/hexagon/common.h>

#ifdef __cplusplus
extern "C"
#endif
int32_t tl_q4_hmx_m32_n128_k2048_kernel(float* A, uint8_t* W, float* C) {
  int bx = 0;
  for (bx = 0; bx < 1; ++bx) {
    uint8_t* buf_dyn_shmem = (uint8_t*)((char*)tl_vtcm_base() + 2048);
    tl_vtcm_shared_high_water = 268288u;
    void* bias_vtcm = ((void*)((char*)buf_dyn_shmem + 0));
    void* A_hmx = ((void*)((char*)buf_dyn_shmem + 2048));
    void* B_hmx = ((void*)((char*)buf_dyn_shmem + 133120));
    void* C_hmx = ((void*)((char*)buf_dyn_shmem + 264192));
    uint8_t acc[1];
    uint8_t bias[1];
    uint8_t cvt[1];
    int tx = 0;
    for (tx = 0; tx < 1; ++tx) {
      int ty = 0;
      for (ty = 0; ty < 1; ++ty) {
        int tz = 0;
        for (tz = 0; tz < 1; ++tz) {
          for (int32_t i = 0; i < 32; ++i) {
            ((uint32_t*)bias_vtcm)[i] = (uint32_t)15360;
            ((uint32_t*)bias_vtcm)[(i + 32)] = (uint32_t)0;
          }
          for (int32_t kt = 0; kt < 64; ++kt) {
            for (int32_t mpair = 0; mpair < 16; ++mpair) {
              tl_hexagon_hmx_pack_a_f32_pair_k32((&(((half*)A_hmx)[((kt * 1024) + (mpair * 64))])), (&(A[((mpair * 4096) + (kt * 32))])), (&(A[(((mpair * 4096) + (kt * 32)) + 2048)])));
            }
          }
          tl_hexagon_hmx_acc_acquire((&(acc[0])));
          for (int32_t nt = 0; nt < 4; ++nt) {
            for (int32_t kt_1 = 0; kt_1 < 64; ++kt_1) {
              tl_hexagon_q4_0_dequant_tile_32x32((&(((half*)B_hmx)[(kt_1 * 1024)])), (&(W[((nt * 36864) + (kt_1 * 576))])));
            }
            tl_hexagon_hmx_clear_acc((&(acc[0])));
            tl_hexagon_hmx_load_bias((&(bias[0])), (&(((uint32_t*)bias_vtcm)[0])));
            for (int32_t kt_2 = 0; kt_2 < 64; ++kt_2) {
              tl_hexagon_hmx_mma_atom((&(acc[0])), (&(((half*)A_hmx)[(kt_2 * 1024)])), (&(((half*)B_hmx)[(kt_2 * 1024)])));
            }
            tl_hexagon_hmx_convert_acc((&(cvt[0])), (&(acc[0])), (&(bias[0])), (&(((uint32_t*)bias_vtcm)[0])), 2);
            tl_hexagon_hmx_store_cvt_state((&(cvt[0])), (&(((half*)C_hmx)[0])), (&(acc[0])), (&(bias[0])), (&(((uint32_t*)bias_vtcm)[0])), (&(((half*)A_hmx)[0])), (&(((half*)B_hmx)[0])));
            for (int32_t mpair_1 = 0; mpair_1 < 16; ++mpair_1) {
              tl_hexagon_hmx_unpack_c_f32_pair_n32((&(C[((mpair_1 * 256) + (nt * 32))])), (&(C[(((mpair_1 * 256) + (nt * 32)) + 128)])), (&(((half*)C_hmx)[(mpair_1 * 64)])));
            }
          }
          tl_hexagon_hmx_acc_release((&(acc[0])));
        }
      }
    }
  }
  return TL_OK;
}


#ifdef __cplusplus
extern "C"
#endif
int32_t tl_q4_hmx_m32_n128_k2048_embedded(void* vtcm_base, unsigned int vtcm_size, float* activation, uint8_t* weight, float* output) {
  if (tl_bridge_enter(vtcm_base, vtcm_size) != 0) return -1;
  int32_t rc = tl_q4_hmx_m32_n128_k2048_kernel(activation, weight, output);
  tl_bridge_exit();
  return rc;
}
