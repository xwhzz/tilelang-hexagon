// Generated TileLang stages for llama.cpp-owned DMA and workers.
#include <tl_templates/hexagon/tl_bridge.h>
#define TL_Q4_0_PADDED_VTCM_INPUT 1
#include <tl_templates/hexagon/qmatmul.h>

// tilelang Hexagon (cDSP) kernel
#include <tl_templates/hexagon/common.h>

#ifdef __cplusplus
extern "C"
#endif
int32_t tl_q4_hmx_staged_m32_n256_k8192_pack_kernel(float* A, half* A_hmx, uint32_t* bias_vtcm) {
  int bx = 0;
  for (bx = 0; bx < 1; ++bx) {
    int tx = 0;
    for (tx = 0; tx < 1; ++tx) {
      int ty = 0;
      for (ty = 0; ty < 1; ++ty) {
        int tz = 0;
        for (tz = 0; tz < 1; ++tz) {
          for (int32_t i = 0; i < 32; ++i) {
            bias_vtcm[i] = (uint32_t)15360;
            bias_vtcm[(i + 32)] = (uint32_t)0;
          }
          for (int32_t kt = 0; kt < 256; ++kt) {
            for (int32_t mpair = 0; mpair < 16; ++mpair) {
              tl_hexagon_hmx_pack_a_f32_pair_k32((&(A_hmx[((kt * 1024) + (mpair * 64))])), (&(A[((mpair * 16384) + (kt * 32))])), (&(A[(((mpair * 16384) + (kt * 32)) + 8192)])));
            }
          }
        }
      }
    }
  }
  return TL_OK;
}


// tilelang Hexagon (cDSP) kernel
#include <tl_templates/hexagon/common.h>

#ifdef __cplusplus
extern "C"
#endif
int32_t tl_q4_hmx_staged_m32_n256_k8192_dequant_kernel(uint8_t* W, half* B_hmx, int32_t tile_begin, int32_t tile_end) {
  int bx = 0;
  for (bx = 0; bx < 1; ++bx) {
    int tx = 0;
    for (tx = 0; tx < 1; ++tx) {
      int ty = 0;
      for (ty = 0; ty < 1; ++ty) {
        int tz = 0;
        for (tz = 0; tz < 1; ++tz) {
          for (int32_t tile = tile_begin; tile < tile_end; ++tile) {
            if (0 <= tile) {
              if (tile < 2048) {
                tl_hexagon_q4_0_dequant_tile_32x32((&(B_hmx[(((int64_t)tile) * (int64_t)1024)])), (&(W[(((int64_t)tile) * (int64_t)640)])));
              }
            }
          }
        }
      }
    }
  }
  return TL_OK;
}


// tilelang Hexagon (cDSP) kernel
#include <tl_templates/hexagon/common.h>

#ifdef __cplusplus
extern "C"
#endif
int32_t tl_q4_hmx_staged_m32_n256_k8192_compute_kernel(half* A_hmx, half* B_hmx, uint32_t* bias_vtcm, half* C_hmx, float* C, int32_t dst_stride) {
  int bx = 0;
  for (bx = 0; bx < 1; ++bx) {
    uint8_t acc[1];
    uint8_t bias[1];
    uint8_t cvt[1];
    int tx = 0;
    for (tx = 0; tx < 1; ++tx) {
      int ty = 0;
      for (ty = 0; ty < 1; ++ty) {
        int tz = 0;
        for (tz = 0; tz < 1; ++tz) {
          tl_hexagon_hmx_acc_acquire((&(acc[0])));
          for (int32_t nt = 0; nt < 8; ++nt) {
            tl_hexagon_hmx_clear_acc((&(acc[0])));
            tl_hexagon_hmx_load_bias((&(bias[0])), (&(bias_vtcm[0])));
            for (int32_t kt = 0; kt < 256; ++kt) {
              tl_hexagon_hmx_mma_atom((&(acc[0])), (&(A_hmx[(kt * 1024)])), (&(B_hmx[((nt * 262144) + (kt * 1024))])));
            }
            tl_hexagon_hmx_convert_acc((&(cvt[0])), (&(acc[0])), (&(bias[0])), (&(bias_vtcm[0])), 2);
            tl_hexagon_hmx_store_cvt_state((&(cvt[0])), (&(C_hmx[0])), (&(acc[0])), (&(bias[0])), (&(bias_vtcm[0])), (&(A_hmx[0])), (&(B_hmx[0])));
            if ((nt * 32) < dst_stride) {
              for (int32_t mpair = 0; mpair < 16; ++mpair) {
                tl_hexagon_hmx_unpack_c_f32_pair_n32((&(C[((((int64_t)nt) * (int64_t)32) + ((((int64_t)mpair) * ((int64_t)dst_stride)) * (int64_t)2))])), (&(C[((((int64_t)nt) * (int64_t)32) + (((((int64_t)mpair) * (int64_t)2) + (int64_t)1) * ((int64_t)dst_stride)))])), (&(C_hmx[(mpair * 64)])));
              }
            }
          }
          tl_hexagon_hmx_acc_release((&(acc[0])));
        }
      }
    }
  }
  return TL_OK;
}

