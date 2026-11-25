// tilelang Hexagon (cDSP) kernel
#define TL_HEXAGON_USES_HMX 1
#define TL_HEXAGON_USES_VTCM 1
#include <tl_templates/hexagon/common.h>
#include <tl_templates/hexagon/crouton.h>
#include <tl_templates/hexagon/dma.h>
#include <tl_templates/hexagon/hmx.h>
#include <tl_templates/hexagon/hmx_gemm.h>
#include <tl_templates/hexagon/hvx_lanes.h>
#include <tl_templates/hexagon/hvx_math.h>
#include <tl_templates/hexagon/hvx_reduce.h>
#include <tl_templates/hexagon/vtcm.h>

alignas(128) static const int32_t tl_prefill_diag_offsets[32] = {0, 132, 264, 396, 528, 660, 792, 924, 1056, 1188, 1320, 1452, 1584, 1716, 1848, 1980, 2112, 2244, 2376, 2508, 2640, 2772, 2904, 3036, 3168, 3300, 3432, 3564, 3696, 3828, 3960, 4092};

#ifdef __cplusplus
extern "C"
#endif
int32_t prefill_kernel(half* in_w, half* qkv_w, half* out_w, half* w13_w, half* w2_w, half* head_rows, float* op_norm, float* ffn_norm, float* post_op_norm, float* post_ffn_norm, float* q_norm, float* k_norm, float* conv_w, float* rope_cos, float* rope_sin, half* kv_k, half* kv_v, float* conv_state, int32_t* kinds, float* x, half* qs, int32_t* tokens, int32_t pos0, int32_t n, int32_t chunk) {
  float* cos_rows = (float*)tl_exclusive(tl_vtcm_arena() + 0);
  float* sin_rows = (float*)tl_exclusive(tl_vtcm_arena() + 65536);
  half* a = (half*)tl_exclusive(tl_vtcm_arena() + 131072);
  half* hid = (half*)tl_exclusive(tl_vtcm_arena() + 655360);
  float* f = (float*)tl_exclusive(tl_vtcm_arena() + 1179648);
  float* psum = (float*)tl_exclusive(tl_vtcm_arena() + 2228224);
  {
    int32_t tl_status = tl_parallel_for<tl_hmx_worker_scope, true>(4, 4, tl_vtcm_worker_arenas(768000u, 2230272u), [&](int32_t kvh, uint8_t *tl_arena) TL_HEXAGON_IEEE_ARITHMETIC -> int32_t {
      half* a_w = (half*)tl_exclusive(a);
      tl_escape(a_w);
      float* cos_rows_w = (float*)tl_exclusive(cos_rows);
      tl_escape(cos_rows_w);
      float* sin_rows_w = (float*)tl_exclusive(sin_rows);
      tl_escape(sin_rows_w);
      float* rope_cos_w = (float*)tl_exclusive(rope_cos);
      tl_escape(rope_cos_w);
      int32_t const chunk_w = chunk;
      int32_t const pos0_w = pos0;
      float* rope_sin_w = (float*)tl_exclusive(rope_sin);
      tl_escape(rope_sin_w);
      float* op_norm_w = (float*)tl_exclusive(op_norm);
      tl_escape(op_norm_w);
      int32_t* tokens_w = (int32_t*)tl_exclusive(tokens);
      tl_escape(tokens_w);
      int32_t const n_w = n;
      half* head_rows_w = (half*)tl_exclusive(head_rows);
      tl_escape(head_rows_w);
      float* x_w = (float*)tl_exclusive(x);
      tl_escape(x_w);
      float* f_w = (float*)tl_exclusive(f);
      tl_escape(f_w);
      float* psum_w = (float*)tl_exclusive(psum);
      tl_escape(psum_w);
      int32_t* kinds_w = (int32_t*)tl_exclusive(kinds);
      tl_escape(kinds_w);
      half* qkv_w_w = (half*)tl_exclusive(qkv_w);
      tl_escape(qkv_w_w);
      float* q_norm_w = (float*)tl_exclusive(q_norm);
      tl_escape(q_norm_w);
      half* qs_w = (half*)tl_exclusive(qs);
      tl_escape(qs_w);
      float* k_norm_w = (float*)tl_exclusive(k_norm);
      tl_escape(k_norm_w);
      half* kv_k_w = (half*)tl_exclusive(kv_k);
      tl_escape(kv_k_w);
      half* kv_v_w = (half*)tl_exclusive(kv_v);
      tl_escape(kv_v_w);
      half* hid_w = (half*)tl_exclusive(hid);
      tl_escape(hid_w);
      half* in_w_w = (half*)tl_exclusive(in_w);
      tl_escape(in_w_w);
      float* conv_w_w = (float*)tl_exclusive(conv_w);
      tl_escape(conv_w_w);
      float* conv_state_w = (float*)tl_exclusive(conv_state);
      tl_escape(conv_state_w);
      half* out_w_w = (half*)tl_exclusive(out_w);
      tl_escape(out_w_w);
      half* w13_w_w = (half*)tl_exclusive(w13_w);
      tl_escape(w13_w_w);
      float* post_op_norm_w = (float*)tl_exclusive(post_op_norm);
      tl_escape(post_op_norm_w);
      float* ffn_norm_w = (float*)tl_exclusive(ffn_norm);
      tl_escape(ffn_norm_w);
      half* w2_w_w = (half*)tl_exclusive(w2_w);
      tl_escape(w2_w_w);
      float* post_ffn_norm_w = (float*)tl_exclusive(post_ffn_norm);
      tl_escape(post_ffn_norm_w);
      uint8_t* buf_dyn_shmem = (uint8_t*)tl_arena;
      void* sum_ones = tl_exclusive(((void*)((char*)buf_dyn_shmem + 0)));
      void* nw = tl_exclusive(((void*)((char*)buf_dyn_shmem + 8192)));
      void* xr = tl_exclusive(((void*)((char*)buf_dyn_shmem + 16384)));
      void* sq = tl_exclusive(((void*)((char*)buf_dyn_shmem + 81920)));
      void* erow = ((void*)((char*)buf_dyn_shmem + 114688));
      void* gates = ((void*)((char*)buf_dyn_shmem + 114688));
      void* alpha = ((void*)((char*)buf_dyn_shmem + 116736));
      void* ls = ((void*)((char*)buf_dyn_shmem + 116992));
      void* m = ((void*)((char*)buf_dyn_shmem + 117248));
      void* rmax = ((void*)((char*)buf_dyn_shmem + 117504));
      void* rsum = ((void*)((char*)buf_dyn_shmem + 117760));
      void* mh = ((void*)((char*)buf_dyn_shmem + 118016));
      void* rsum16 = ((void*)((char*)buf_dyn_shmem + 118144));
      void* sw = ((void*)((char*)buf_dyn_shmem + 118272));
      void* config = ((void*)((char*)buf_dyn_shmem + 118400));
      void* config_1 = ((void*)((char*)buf_dyn_shmem + 118400));
      void* config_2 = ((void*)((char*)buf_dyn_shmem + 118400));
      void* config_3 = ((void*)((char*)buf_dyn_shmem + 118400));
      void* config_4 = ((void*)((char*)buf_dyn_shmem + 118400));
      void* config_5 = ((void*)((char*)buf_dyn_shmem + 118400));
      void* config_6 = ((void*)((char*)buf_dyn_shmem + 118400));
      void* config_7 = ((void*)((char*)buf_dyn_shmem + 118400));
      void* config_8 = ((void*)((char*)buf_dyn_shmem + 118400));
      void* config_9 = ((void*)((char*)buf_dyn_shmem + 118400));
      void* config_10 = ((void*)((char*)buf_dyn_shmem + 118400));
      void* config_11 = ((void*)((char*)buf_dyn_shmem + 118400));
      void* config_12 = ((void*)((char*)buf_dyn_shmem + 118400));
      void* config_13 = ((void*)((char*)buf_dyn_shmem + 118400));
      void* config_14 = ((void*)((char*)buf_dyn_shmem + 118400));
      void* config_15 = ((void*)((char*)buf_dyn_shmem + 118400));
      void* config_16 = ((void*)((char*)buf_dyn_shmem + 118400));
      void* config_17 = ((void*)((char*)buf_dyn_shmem + 118400));
      void* ss = tl_exclusive(((void*)((char*)buf_dyn_shmem + 118784)));
      void* v = tl_exclusive(((void*)((char*)buf_dyn_shmem + 118912)));
      void* st = tl_exclusive(((void*)((char*)buf_dyn_shmem + 381056)));
      void* hs32 = tl_exclusive(((void*)((char*)buf_dyn_shmem + 381824)));
      void* hss = tl_exclusive(((void*)((char*)buf_dyn_shmem + 382336)));
      void* hd = tl_exclusive(((void*)((char*)buf_dyn_shmem + 382848)));
      void* kblk = tl_exclusive(((void*)((char*)buf_dyn_shmem + 448384)));
      void* vblk = tl_exclusive(((void*)((char*)buf_dyn_shmem + 513920)));
      void* acc_o = tl_exclusive(((void*)((char*)buf_dyn_shmem + 579456)));
      void* cc = tl_exclusive(((void*)((char*)buf_dyn_shmem + 612224)));
      void* rot = tl_exclusive(((void*)((char*)buf_dyn_shmem + 644992)));
      void* p = tl_exclusive(((void*)((char*)buf_dyn_shmem + 677760)));
      void* qb = tl_exclusive(((void*)((char*)buf_dyn_shmem + 694144)));
      void* s = tl_exclusive(((void*)((char*)buf_dyn_shmem + 710528)));
      void* sq32 = tl_exclusive(((void*)((char*)buf_dyn_shmem + 726912)));
      void* gl = tl_exclusive(((void*)((char*)buf_dyn_shmem + 743296)));
      void* pw = tl_exclusive(((void*)((char*)buf_dyn_shmem + 751488)));
      void* sum_hmx = tl_exclusive(((void*)((char*)buf_dyn_shmem + 759680)));
      void* stage = ((void*)((char*)buf_dyn_shmem + 763776));
      void* stage_1 = ((void*)((char*)buf_dyn_shmem + 763776));
      void* stage_2 = ((void*)((char*)buf_dyn_shmem + 763776));
      void* stage_3 = ((void*)((char*)buf_dyn_shmem + 763776));
      void* stage_4 = ((void*)((char*)buf_dyn_shmem + 763776));
      void* stage_5 = ((void*)((char*)buf_dyn_shmem + 763776));
      void* stage_6 = ((void*)((char*)buf_dyn_shmem + 763776));
      void* stage_7 = ((void*)((char*)buf_dyn_shmem + 763776));
      void* stage_8 = ((void*)((char*)buf_dyn_shmem + 763776));
      void* stage_9 = ((void*)((char*)buf_dyn_shmem + 763776));
      void* stage_10 = ((void*)((char*)buf_dyn_shmem + 763776));
      void* stage_11 = ((void*)((char*)buf_dyn_shmem + 763776));
      void* stage_12 = ((void*)((char*)buf_dyn_shmem + 763776));
      void* stage_13 = ((void*)((char*)buf_dyn_shmem + 763776));
      void* stage_14 = ((void*)((char*)buf_dyn_shmem + 763776));
      void* stage_15 = ((void*)((char*)buf_dyn_shmem + 763776));
      void* stage_16 = ((void*)((char*)buf_dyn_shmem + 763776));
      void* stage_17 = ((void*)((char*)buf_dyn_shmem + 763776));
      tl_hexagon_dma_async_context<10, 10> tl_dma_async;
      if (tl_dma_async.status != TL_OK) return TL_ERR_DMA;
      for (int32_t i_ones = 0; i_ones < 4; ++i_ones) {
        half* sum_ones_row = ((half*)sum_ones) + (i_ones * 1024);
        for (int32_t j_ones = 0; j_ones < 16; ++j_ones) {
          half broadcast_var_ones = (half)1.000000e+00f;
          *(half64*)(sum_ones_row + (j_ones * 64)) = ((half64)(broadcast_var_ones));
        }
      }
      for (int32_t i = 0; i < 128; ++i) {
        float broadcast_var = 0.000000e+00f;
        int64_t v_ = (((((int64_t)chunk_w) * (int64_t)128) + (((int64_t)kvh) * (int64_t)32)) + ((((int64_t)i) & (int64_t)63) >> (int64_t)1)) + ((int64_t)pos0_w);
        int64_t v__1 = (int64_t)1023;
        float32 guarded = *(float32*)(rope_cos_w + ((0 <= ((((chunk_w * 128) + (kvh * 32)) + ((i & 63) >> 1)) + pos0_w)) ? ((((((int64_t)i) >> (int64_t)6) * (int64_t)65536) + (((v_) < (v__1) ? (v_) : (v__1)) * (int64_t)64)) + ((((int64_t)i) & (int64_t)1) * (int64_t)32)) : (int64_t)0));
        *(float32*)(cos_rows_w + ((((i >> 6) * 8192) + (kvh * 2048)) + ((i & 63) * 32))) = ((0 <= ((((chunk_w * 128) + (kvh * 32)) + ((i & 63) >> 1)) + pos0_w)) ? guarded : ((float32)(broadcast_var)));
        float broadcast_var_1 = 0.000000e+00f;
        float32 guarded_1 = *(float32*)(rope_sin_w + ((0 <= ((((chunk_w * 128) + (kvh * 32)) + ((i & 63) >> 1)) + pos0_w)) ? ((((((int64_t)i) >> (int64_t)6) * (int64_t)65536) + (((v_) < (v__1) ? (v_) : (v__1)) * (int64_t)64)) + ((((int64_t)i) & (int64_t)1) * (int64_t)32)) : (int64_t)0));
        *(float32*)(sin_rows_w + ((((i >> 6) * 8192) + (kvh * 2048)) + ((i & 63) * 32))) = ((0 <= ((((chunk_w * 128) + (kvh * 32)) + ((i & 63) >> 1)) + pos0_w)) ? guarded_1 : ((float32)(broadcast_var_1)));
      }
      for (int32_t i_1 = 0; i_1 < 64; ++i_1) {
        *(float32*)(((float*)nw) + (i_1 * 32)) = *(float32*)(op_norm_w + (i_1 * 32));
      }
      for (int32_t rb = 0; rb < 8; ++rb) {
        for (int32_t i_2 = 0; i_2 < 4; ++i_2) {
          int64_t v__2 = (((((int64_t)chunk_w) * (int64_t)128) + (((int64_t)kvh) * (int64_t)32)) + (((int64_t)rb) * (int64_t)4)) + ((int64_t)i_2);
          int64_t v__3 = ((int64_t)n_w) - (int64_t)1;
          int32_t guarded_2 = tokens_w[(((((((((chunk_w * 128) + (kvh * 32)) + (rb * 4)) + i_2) < n_w) && ((chunk_w < 8) || (n_w < 1025))) && (0 <= chunk_w)) && (1 <= n_w)) ? ((v__2) < (v__3) ? (v__2) : (v__3)) : (int64_t)0)];
          int32_t v__4 = ((((((((chunk_w * 128) + (kvh * 32)) + (rb * 4)) + i_2) < n_w) && ((chunk_w < 8) || (n_w < 1025))) && (0 <= chunk_w)) && (1 <= n_w)) ? guarded_2 : 0;
          int32_t v__5 = (v__4) < (4095) ? (v__4) : (4095);
          int32_t token = ((v__5) > (0) ? (v__5) : (0));
          if (int32_t tl_status = tl_hexagon_dma_async_submit(tl_dma_async, (&(((half*)erow)[0])), (&(head_rows_w[(((((int64_t)token) >> (int64_t)5) * (int64_t)65536) + ((((int64_t)token) & (int64_t)31) * (int64_t)2))])), (uint32_t)4, (uint32_t)128, (uint32_t)4, (uint32_t)1024, 3)) return tl_status;
          if (int32_t tl_status = tl_hexagon_dma_async_commit(tl_dma_async)) return tl_status;
          if (int32_t tl_status = tl_hexagon_dma_async_wait(tl_dma_async, 0)) return tl_status;
          for (int32_t i_3 = 0; i_3 < 32; ++i_3) {
            half erow_local_cast_1[64] __attribute__((aligned(128)));
            float xr_local_cast[64] __attribute__((aligned(128)));
            *(half64*)(erow_local_cast_1 + 0) = *(half64*)(((half*)erow) + (i_3 * 64));
            *(float64*)(xr_local_cast + 0) = ((float64)*(half64*)(erow_local_cast_1 + 0));
            float* xr_row = ((float*)xr) + ((i_2 * 2048) + (i_3 * 64));
            for (int32_t vec_copy = 0; vec_copy < 2; ++vec_copy) {
              *(float32*)(xr_row + (vec_copy * 32)) = *(float32*)(xr_local_cast + (vec_copy * 32));
            }
          }
        }
        if (int32_t tl_status = tl_hexagon_dma_async_submit(tl_dma_async, (&(x_w[((kvh * 65536) + (rb * 8192))])), (&(((float*)xr)[0])), (uint32_t)32768, (uint32_t)32768, (uint32_t)32768, (uint32_t)1, 4)) return tl_status;
        if (int32_t tl_status = tl_hexagon_dma_async_commit(tl_dma_async)) return tl_status;
        for (int32_t i_4 = 0; i_4 < 256; ++i_4) {
          *(float32*)(((float*)sq) + (i_4 * 32)) = (*(float32*)(((float*)xr) + (i_4 * 32)) * *(float32*)(((float*)xr) + (i_4 * 32)));
        }
        tl_hvx_reduce_rows<TL_REDUCE_SUM, float, true>((&(((float*)ss)[0])), (&(((float*)sq)[0])), 4, 2048, 2048, 1, (bool)1);
        for (int32_t i_5 = 0; i_5 < 4; ++i_5) {
          ((float*)ss)[i_5] = tl_hvx_rsqrt_fast(((((float*)ss)[i_5] * 4.882812e-04f) + 1.000000e-05f));
        }
        for (int32_t j = 0; j < 2; ++j) {
          for (int32_t i_6 = 0; i_6 < 64; ++i_6) {
            *(half64*)(a_w + ((((kvh * 65536) + (i_6 * 1024)) + (rb * 128)) + (j * 64))) = half64(__builtin_shufflevector((((half64)float64(__builtin_shufflevector((((*(float32*)(((float*)xr) + ((j * 4096) + (i_6 * 32))) * ((float32)(((float*)ss)[(j * 2)]))) * *(float32*)(((float*)nw) + (i_6 * 32)))).v, (((*(float32*)(((float*)xr) + (((j * 4096) + (i_6 * 32)) + 2048)) * ((float32)(((float*)ss)[((j * 2) + 1)]))) * *(float32*)(((float*)nw) + (i_6 * 32)))).v, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63)))).v, (half64{}).v, 0, 32, 1, 33, 2, 34, 3, 35, 4, 36, 5, 37, 6, 38, 7, 39, 8, 40, 9, 41, 10, 42, 11, 43, 12, 44, 13, 45, 14, 46, 15, 47, 16, 48, 17, 49, 18, 50, 19, 51, 20, 52, 21, 53, 22, 54, 23, 55, 24, 56, 25, 57, 26, 58, 27, 59, 28, 60, 29, 61, 30, 62, 31, 63));
          }
        }
      }
      if (int32_t tl_status = tl_hexagon_dma_async_wait(tl_dma_async, 0)) return tl_status;
      if (int32_t tl_status = tl_grid_sync(kvh)) return tl_status;
      for (int32_t layer = 0; layer < 4; ++layer) {
        int32_t v__6 = kinds_w[((layer * 5) + 1)];
        int32_t v__7 = (v__6) < (1) ? (v__6) : (1);
        int32_t ci = ((v__7) > (0) ? (v__7) : (0));
        int32_t ai = ((v__7) > (0) ? (v__7) : (0));
        int32_t v__8 = kinds_w[((layer * 5) + 2)];
        int32_t v__9 = (v__8) < (1) ? (v__8) : (1);
        int32_t table = ((v__9) > (0) ? (v__9) : (0));
        int32_t window = kinds_w[((layer * 5) + 3)];
        int32_t kv0 = kinds_w[((layer * 5) + 4)];
        if (kinds_w[(layer * 5)] == 1) {
          if (int32_t tl_status = tl_hexagon_dma_async_submit(tl_dma_async, (&(((half*)v)[0])), (&(qkv_w_w[((((int64_t)ai) * (int64_t)6553600) + (((int64_t)kvh) * (int64_t)1638400))])), (uint32_t)131072, (uint32_t)131072, (uint32_t)131072, (uint32_t)1, 3)) return tl_status;
          if (int32_t tl_status = tl_hexagon_dma_async_commit(tl_dma_async)) return tl_status;
          if (int32_t tl_status = tl_hexagon_dma_async_submit(tl_dma_async, (&(((half*)v)[65536])), (&(qkv_w_w[(((((int64_t)ai) * (int64_t)6553600) + (((int64_t)kvh) * (int64_t)1638400)) + (int64_t)65536)])), (uint32_t)131072, (uint32_t)131072, (uint32_t)131072, (uint32_t)1, 3)) return tl_status;
          if (int32_t tl_status = tl_hexagon_dma_async_commit(tl_dma_async)) return tl_status;
          for (int32_t b = 0; b < 23; ++b) {
            if (int32_t tl_status = tl_hexagon_dma_async_wait(tl_dma_async, 1)) return tl_status;
            tl_hexagon_hmx_gemm<true, true, true, false>(128, 32, 2048, (&(((float*)hd)[((b & 3) * 32)])), 128, (&(a_w[0])), 65536, (&(((half*)v)[((b & 1) * 65536)])), 65536, (bool)1, (&(((half*)stage)[0])), (&(((uint32_t*)config)[0])));
            if (b <= 19) {
              for (int32_t i_7 = 0; i_7 < 128; ++i_7) {
                *(float32*)(((float*)sq32) + (i_7 * 32)) = (*(float32*)(((float*)hd) + ((i_7 * 128) + ((b & 3) * 32))) * *(float32*)(((float*)hd) + ((i_7 * 128) + ((b & 3) * 32))));
              }
              tl_hvx_reduce_rows<TL_REDUCE_SUM, float, true>((&(((float*)hs32)[0])), (&(((float*)sq32)[0])), 128, 32, 32, 1, (bool)1);
              for (int32_t i_8 = 0; i_8 < 4; ++i_8) {
                *(float32*)(((float*)hss) + (i_8 * 32)) = (((b % 4) == 0) ? *(float32*)(((float*)hs32) + (i_8 * 32)) : (*(float32*)(((float*)hss) + (i_8 * 32)) + *(float32*)(((float*)hs32) + (i_8 * 32))));
              }
            }
            if (((b & 3) * 32) == 96) {
              if (b < 16) {
                #pragma clang loop unroll_count(2)
                for (int32_t i_9 = 0; i_9 < 4; ++i_9) {
                  float broadcast_var_2 = 7.812500e-03f;
                  float broadcast_var_3 = 1.000000e-05f;
                  float broadcast_var_4 = 1.275174e-01f;
                  *(float32*)(((float*)hss) + (i_9 * 32)) = (tl_hvx_rsqrt_fast(((*(float32*)(((float*)hss) + (i_9 * 32)) * ((float32)(broadcast_var_2))) + ((float32)(broadcast_var_3)))) * ((float32)(broadcast_var_4)));
                }
                for (int32_t i_strip = 0; i_strip < 4; ++i_strip) {
                  float32 hss_lanes = tl_hvx_load_lanes<32>((&(((float*)hss)[(i_strip * 32)])), 32);
                  for (int32_t i_lane = 0; i_lane < 32; ++i_lane) {
                    float hd_local_cast_3[64] __attribute__((aligned(128)));
                    float q_norm_local_cast_4[64] __attribute__((aligned(128)));
                    float hd_local_cast_5[64] __attribute__((aligned(128)));
                    float q_norm_local_cast_6[64] __attribute__((aligned(128)));
                    float hss_local_cast_7[64] __attribute__((aligned(128)));
                    half rot_local_cast_2[64] __attribute__((aligned(128)));
                    float* hd_row = ((float*)hd) + ((i_strip * 4096) + (i_lane * 128));
                    for (int32_t vec_copy_1 = 0; vec_copy_1 < 2; ++vec_copy_1) {
                      *(float32*)(hd_local_cast_3 + (vec_copy_1 * 32)) = *(float32*)(hd_row + (vec_copy_1 * 32));
                    }
                    for (int32_t vec_copy_2 = 0; vec_copy_2 < 2; ++vec_copy_2) {
                      float broadcast_var_5 = 0.000000e+00f;
                      float32 guarded_3 = *(float32*)(q_norm_w + (((ai < 2) && (0 <= ai)) ? ((((int64_t)ai) * (int64_t)128) + (((int64_t)vec_copy_2) * (int64_t)32)) : (int64_t)0));
                      *(float32*)(q_norm_local_cast_4 + (vec_copy_2 * 32)) = (((ai < 2) && (0 <= ai)) ? guarded_3 : ((float32)(broadcast_var_5)));
                    }
                    float* hd_row_1 = ((float*)hd) + (((i_strip * 4096) + (i_lane * 128)) + 64);
                    for (int32_t vec_copy_3 = 0; vec_copy_3 < 2; ++vec_copy_3) {
                      *(float32*)(hd_local_cast_5 + (vec_copy_3 * 32)) = *(float32*)(hd_row_1 + (vec_copy_3 * 32));
                    }
                    for (int32_t vec_copy_4 = 0; vec_copy_4 < 2; ++vec_copy_4) {
                      float broadcast_var_6 = 0.000000e+00f;
                      float32 guarded_4 = *(float32*)(q_norm_w + (((ai < 2) && (0 <= ai)) ? (((((int64_t)ai) * (int64_t)128) + (((int64_t)vec_copy_4) * (int64_t)32)) + (int64_t)64) : (int64_t)0));
                      *(float32*)(q_norm_local_cast_6 + (vec_copy_4 * 32)) = (((ai < 2) && (0 <= ai)) ? guarded_4 : ((float32)(broadcast_var_6)));
                    }
                    for (int32_t vec_copy_5 = 0; vec_copy_5 < 2; ++vec_copy_5) {
                      *(float32*)(hss_local_cast_7 + (vec_copy_5 * 32)) = tl_hvx_broadcast_lane<32, 1>(hss_lanes, i_lane);
                    }
                    float broadcast_var_7 = -1.000000e+00f;
                    *(half64*)(rot_local_cast_2 + 0) = ((half64)((((*(float64*)(hd_local_cast_3 + 0) * *(float64*)(q_norm_local_cast_4 + 0)) * *(float64*)(cos_rows_w + (((((int64_t)table) * (int64_t)8192) + (((int64_t)i_strip) * (int64_t)2048)) + (((int64_t)i_lane) * (int64_t)64)))) + (((((float64)(broadcast_var_7)) * *(float64*)(hd_local_cast_5 + 0)) * *(float64*)(q_norm_local_cast_6 + 0)) * *(float64*)(sin_rows_w + (((((int64_t)table) * (int64_t)8192) + (((int64_t)i_strip) * (int64_t)2048)) + (((int64_t)i_lane) * (int64_t)64))))) * *(float64*)(hss_local_cast_7 + 0)));
                    *(half64*)(((half*)rot) + ((i_strip * 4096) + (i_lane * 128))) = *(half64*)(rot_local_cast_2 + 0);
                    float* hd_row_2 = ((float*)hd) + (((i_strip * 4096) + (i_lane * 128)) + 64);
                    for (int32_t vec_copy_6 = 0; vec_copy_6 < 2; ++vec_copy_6) {
                      *(float32*)(hd_local_cast_3 + (vec_copy_6 * 32)) = *(float32*)(hd_row_2 + (vec_copy_6 * 32));
                    }
                    for (int32_t vec_copy_7 = 0; vec_copy_7 < 2; ++vec_copy_7) {
                      float broadcast_var_8 = 0.000000e+00f;
                      float32 guarded_5 = *(float32*)(q_norm_w + (((ai < 2) && (0 <= ai)) ? (((((int64_t)ai) * (int64_t)128) + (((int64_t)vec_copy_7) * (int64_t)32)) + (int64_t)64) : (int64_t)0));
                      *(float32*)(q_norm_local_cast_4 + (vec_copy_7 * 32)) = (((ai < 2) && (0 <= ai)) ? guarded_5 : ((float32)(broadcast_var_8)));
                    }
                    float* hd_row_3 = ((float*)hd) + ((i_strip * 4096) + (i_lane * 128));
                    for (int32_t vec_copy_8 = 0; vec_copy_8 < 2; ++vec_copy_8) {
                      *(float32*)(hd_local_cast_5 + (vec_copy_8 * 32)) = *(float32*)(hd_row_3 + (vec_copy_8 * 32));
                    }
                    for (int32_t vec_copy_9 = 0; vec_copy_9 < 2; ++vec_copy_9) {
                      float broadcast_var_9 = 0.000000e+00f;
                      float32 guarded_6 = *(float32*)(q_norm_w + (((ai < 2) && (0 <= ai)) ? ((((int64_t)ai) * (int64_t)128) + (((int64_t)vec_copy_9) * (int64_t)32)) : (int64_t)0));
                      *(float32*)(q_norm_local_cast_6 + (vec_copy_9 * 32)) = (((ai < 2) && (0 <= ai)) ? guarded_6 : ((float32)(broadcast_var_9)));
                    }
                    for (int32_t vec_copy_10 = 0; vec_copy_10 < 2; ++vec_copy_10) {
                      *(float32*)(hss_local_cast_7 + (vec_copy_10 * 32)) = tl_hvx_broadcast_lane<32, 1>(hss_lanes, i_lane);
                    }
                    float broadcast_var_10 = 1.000000e+00f;
                    *(half64*)(rot_local_cast_2 + 0) = ((half64)((((*(float64*)(hd_local_cast_3 + 0) * *(float64*)(q_norm_local_cast_4 + 0)) * *(float64*)(cos_rows_w + (((((int64_t)table) * (int64_t)8192) + (((int64_t)i_strip) * (int64_t)2048)) + (((int64_t)i_lane) * (int64_t)64)))) + (((((float64)(broadcast_var_10)) * *(float64*)(hd_local_cast_5 + 0)) * *(float64*)(q_norm_local_cast_6 + 0)) * *(float64*)(sin_rows_w + (((((int64_t)table) * (int64_t)8192) + (((int64_t)i_strip) * (int64_t)2048)) + (((int64_t)i_lane) * (int64_t)64))))) * *(float64*)(hss_local_cast_7 + 0)));
                    *(half64*)(((half*)rot) + (((i_strip * 4096) + (i_lane * 128)) + 64)) = *(half64*)(rot_local_cast_2 + 0);
                  }
                }
                tl_crouton_copy<true, false>((&(qs_w[((kvh * 65536) + ((b >> 2) * 16384))])), 4, 4, 0, 0, 128, 128, (&(((half*)rot)[0])), 128, 1);
                if (int32_t tl_status = tl_hexagon_cache_writeback((&(qs_w[((kvh * 65536) + ((b >> 2) * 16384))])), (uint32_t)8192, (uint32_t)4, (uint32_t)8192)) return tl_status;
              } else {
                #pragma clang loop unroll_count(2)
                for (int32_t i_10 = 0; i_10 < 4; ++i_10) {
                  float broadcast_var_11 = 7.812500e-03f;
                  float broadcast_var_12 = 1.000000e-05f;
                  *(float32*)(((float*)hss) + (i_10 * 32)) = tl_hvx_rsqrt_fast(((*(float32*)(((float*)hss) + (i_10 * 32)) * ((float32)(broadcast_var_11))) + ((float32)(broadcast_var_12))));
                }
                for (int32_t i_strip_1 = 0; i_strip_1 < 4; ++i_strip_1) {
                  float32 hss_lanes_1 = tl_hvx_load_lanes<32>((&(((float*)hss)[(i_strip_1 * 32)])), 32);
                  for (int32_t i_lane_1 = 0; i_lane_1 < 32; ++i_lane_1) {
                    float hd_local_cast_9[64] __attribute__((aligned(128)));
                    float k_norm_local_cast_10[64] __attribute__((aligned(128)));
                    float hd_local_cast_11[64] __attribute__((aligned(128)));
                    float k_norm_local_cast_12[64] __attribute__((aligned(128)));
                    float hss_local_cast_13[64] __attribute__((aligned(128)));
                    half rot_local_cast_8[64] __attribute__((aligned(128)));
                    float* hd_row_4 = ((float*)hd) + ((i_strip_1 * 4096) + (i_lane_1 * 128));
                    for (int32_t vec_copy_11 = 0; vec_copy_11 < 2; ++vec_copy_11) {
                      *(float32*)(hd_local_cast_9 + (vec_copy_11 * 32)) = *(float32*)(hd_row_4 + (vec_copy_11 * 32));
                    }
                    for (int32_t vec_copy_12 = 0; vec_copy_12 < 2; ++vec_copy_12) {
                      float broadcast_var_13 = 0.000000e+00f;
                      float32 guarded_7 = *(float32*)(k_norm_w + (((ai < 2) && (0 <= ai)) ? ((((int64_t)ai) * (int64_t)128) + (((int64_t)vec_copy_12) * (int64_t)32)) : (int64_t)0));
                      *(float32*)(k_norm_local_cast_10 + (vec_copy_12 * 32)) = (((ai < 2) && (0 <= ai)) ? guarded_7 : ((float32)(broadcast_var_13)));
                    }
                    float* hd_row_5 = ((float*)hd) + (((i_strip_1 * 4096) + (i_lane_1 * 128)) + 64);
                    for (int32_t vec_copy_13 = 0; vec_copy_13 < 2; ++vec_copy_13) {
                      *(float32*)(hd_local_cast_11 + (vec_copy_13 * 32)) = *(float32*)(hd_row_5 + (vec_copy_13 * 32));
                    }
                    for (int32_t vec_copy_14 = 0; vec_copy_14 < 2; ++vec_copy_14) {
                      float broadcast_var_14 = 0.000000e+00f;
                      float32 guarded_8 = *(float32*)(k_norm_w + (((ai < 2) && (0 <= ai)) ? (((((int64_t)ai) * (int64_t)128) + (((int64_t)vec_copy_14) * (int64_t)32)) + (int64_t)64) : (int64_t)0));
                      *(float32*)(k_norm_local_cast_12 + (vec_copy_14 * 32)) = (((ai < 2) && (0 <= ai)) ? guarded_8 : ((float32)(broadcast_var_14)));
                    }
                    for (int32_t vec_copy_15 = 0; vec_copy_15 < 2; ++vec_copy_15) {
                      *(float32*)(hss_local_cast_13 + (vec_copy_15 * 32)) = tl_hvx_broadcast_lane<32, 1>(hss_lanes_1, i_lane_1);
                    }
                    float broadcast_var_15 = -1.000000e+00f;
                    *(half64*)(rot_local_cast_8 + 0) = ((half64)((((*(float64*)(hd_local_cast_9 + 0) * *(float64*)(k_norm_local_cast_10 + 0)) * *(float64*)(cos_rows_w + (((((int64_t)table) * (int64_t)8192) + (((int64_t)i_strip_1) * (int64_t)2048)) + (((int64_t)i_lane_1) * (int64_t)64)))) + (((((float64)(broadcast_var_15)) * *(float64*)(hd_local_cast_11 + 0)) * *(float64*)(k_norm_local_cast_12 + 0)) * *(float64*)(sin_rows_w + (((((int64_t)table) * (int64_t)8192) + (((int64_t)i_strip_1) * (int64_t)2048)) + (((int64_t)i_lane_1) * (int64_t)64))))) * *(float64*)(hss_local_cast_13 + 0)));
                    *(half64*)(((half*)rot) + ((i_strip_1 * 4096) + (i_lane_1 * 128))) = *(half64*)(rot_local_cast_8 + 0);
                    float* hd_row_6 = ((float*)hd) + (((i_strip_1 * 4096) + (i_lane_1 * 128)) + 64);
                    for (int32_t vec_copy_16 = 0; vec_copy_16 < 2; ++vec_copy_16) {
                      *(float32*)(hd_local_cast_9 + (vec_copy_16 * 32)) = *(float32*)(hd_row_6 + (vec_copy_16 * 32));
                    }
                    for (int32_t vec_copy_17 = 0; vec_copy_17 < 2; ++vec_copy_17) {
                      float broadcast_var_16 = 0.000000e+00f;
                      float32 guarded_9 = *(float32*)(k_norm_w + (((ai < 2) && (0 <= ai)) ? (((((int64_t)ai) * (int64_t)128) + (((int64_t)vec_copy_17) * (int64_t)32)) + (int64_t)64) : (int64_t)0));
                      *(float32*)(k_norm_local_cast_10 + (vec_copy_17 * 32)) = (((ai < 2) && (0 <= ai)) ? guarded_9 : ((float32)(broadcast_var_16)));
                    }
                    float* hd_row_7 = ((float*)hd) + ((i_strip_1 * 4096) + (i_lane_1 * 128));
                    for (int32_t vec_copy_18 = 0; vec_copy_18 < 2; ++vec_copy_18) {
                      *(float32*)(hd_local_cast_11 + (vec_copy_18 * 32)) = *(float32*)(hd_row_7 + (vec_copy_18 * 32));
                    }
                    for (int32_t vec_copy_19 = 0; vec_copy_19 < 2; ++vec_copy_19) {
                      float broadcast_var_17 = 0.000000e+00f;
                      float32 guarded_10 = *(float32*)(k_norm_w + (((ai < 2) && (0 <= ai)) ? ((((int64_t)ai) * (int64_t)128) + (((int64_t)vec_copy_19) * (int64_t)32)) : (int64_t)0));
                      *(float32*)(k_norm_local_cast_12 + (vec_copy_19 * 32)) = (((ai < 2) && (0 <= ai)) ? guarded_10 : ((float32)(broadcast_var_17)));
                    }
                    for (int32_t vec_copy_20 = 0; vec_copy_20 < 2; ++vec_copy_20) {
                      *(float32*)(hss_local_cast_13 + (vec_copy_20 * 32)) = tl_hvx_broadcast_lane<32, 1>(hss_lanes_1, i_lane_1);
                    }
                    float broadcast_var_18 = 1.000000e+00f;
                    *(half64*)(rot_local_cast_8 + 0) = ((half64)((((*(float64*)(hd_local_cast_9 + 0) * *(float64*)(k_norm_local_cast_10 + 0)) * *(float64*)(cos_rows_w + (((((int64_t)table) * (int64_t)8192) + (((int64_t)i_strip_1) * (int64_t)2048)) + (((int64_t)i_lane_1) * (int64_t)64)))) + (((((float64)(broadcast_var_18)) * *(float64*)(hd_local_cast_11 + 0)) * *(float64*)(k_norm_local_cast_12 + 0)) * *(float64*)(sin_rows_w + (((((int64_t)table) * (int64_t)8192) + (((int64_t)i_strip_1) * (int64_t)2048)) + (((int64_t)i_lane_1) * (int64_t)64))))) * *(float64*)(hss_local_cast_13 + 0)));
                    *(half64*)(((half*)rot) + (((i_strip_1 * 4096) + (i_lane_1 * 128)) + 64)) = *(half64*)(rot_local_cast_8 + 0);
                  }
                }
                int32_t condval;
                if ((0 < window)) {
                  condval = ((((chunk_w * 128) + pos0_w) & 1023) + 512);
                } else {
                  condval = ((chunk_w * 128) + pos0_w);
                }
                tl_crouton_copy<true, true>((&(kv_k_w[(kvh * 393216)])), 4, 96, 0, ((kv0 * 128) + condval), 128, 128, (&(((half*)rot)[0])), 1, 128);
                int32_t condval_1;
                if ((0 < window)) {
                  condval_1 = ((((chunk_w * 128) + pos0_w) & 1023) + 512);
                } else {
                  condval_1 = ((chunk_w * 128) + pos0_w);
                }
                int32_t condval_2;
                if ((0 < window)) {
                  condval_2 = ((((chunk_w * 128) + pos0_w) & 1023) + 512);
                } else {
                  condval_2 = ((chunk_w * 128) + pos0_w);
                }
                int32_t condval_3;
                if ((0 < window)) {
                  condval_3 = ((((chunk_w * 128) + pos0_w) & 1023) + 512);
                } else {
                  condval_3 = ((chunk_w * 128) + pos0_w);
                }
                int32_t condval_4;
                if ((0 < window)) {
                  condval_4 = ((((chunk_w * 128) + pos0_w) & 1023) + 512);
                } else {
                  condval_4 = ((chunk_w * 128) + pos0_w);
                }
                int32_t condval_5;
                if ((0 < window)) {
                  condval_5 = ((((chunk_w * 128) + pos0_w) & 1023) + 512);
                } else {
                  condval_5 = ((chunk_w * 128) + pos0_w);
                }
                if (int32_t tl_status = tl_hexagon_cache_writeback((&(kv_k_w[(((((((((condval_1 >> 7) + kv0) / 24) + ((((condval_2 >> 7) + kv0) % 24) >> 31)) + kvh) & 3) * 393216) + ((96 & ((((kv0 * 4) + (condval_3 >> 5)) % 96) >> 31)) * 4096)) + ((((kv0 * 4) + (condval_4 >> 5)) % 96) * 4096))])), (uint32_t)8192, ((uint32_t)(((condval_5 & 31) + 159) >> 5)), (uint32_t)8192)) return tl_status;
                if ((0 < window) && (896 < (((chunk_w * 128) + pos0_w) & 1023))) {
                  tl_crouton_copy<true, true>((&(kv_k_w[(kvh * 393216)])), 4, 96, 0, (((kv0 * 128) + (((chunk_w * 128) + pos0_w) & 1023)) - 512), 128, 128, (&(((half*)rot)[0])), 1, 128);
                  if (int32_t tl_status = tl_hexagon_cache_writeback((&(kv_k_w[((((((((((kv0 * 128) + ((((chunk_w * 4) + (pos0_w >> 5)) & 31) * 32)) - 512) / 3072) + (((((kv0 * 128) + ((((chunk_w * 4) + (pos0_w >> 5)) & 31) * 32)) - 512) % 3072) >> 31)) + kvh) & 3) * 393216) + ((96 & (((((kv0 * 4) + (((chunk_w * 4) + (pos0_w >> 5)) & 31)) + 80) % 96) >> 31)) * 4096)) + (((((kv0 * 4) + (((chunk_w * 4) + (pos0_w >> 5)) & 31)) + 80) % 96) * 4096))])), (uint32_t)8192, ((uint32_t)((((pos0_w & 31) - 353) >> 5) + 16)), (uint32_t)8192)) return tl_status;
                }
              }
            }
            if (int32_t tl_status = tl_hexagon_dma_async_submit(tl_dma_async, (&(((half*)v)[((b & 1) * 65536)])), (&(qkv_w_w[((((((int64_t)ai) * (int64_t)6553600) + (((int64_t)kvh) * (int64_t)1638400)) + (((int64_t)b) * (int64_t)65536)) + (int64_t)131072)])), (uint32_t)131072, (uint32_t)131072, (uint32_t)131072, (uint32_t)1, 3)) return tl_status;
            if (int32_t tl_status = tl_hexagon_dma_async_commit(tl_dma_async)) return tl_status;
          }
          if (int32_t tl_status = tl_hexagon_dma_async_wait(tl_dma_async, 1)) return tl_status;
          tl_hexagon_hmx_gemm<true, true, true, false>(128, 32, 2048, (&(((float*)hd)[96])), 128, (&(a_w[0])), 65536, (&(((half*)v)[65536])), 65536, (bool)1, (&(((half*)stage_1)[0])), (&(((uint32_t*)config_1)[0])));
          int32_t condval_6;
          if ((0 < window)) {
            condval_6 = ((((chunk_w * 128) + pos0_w) & 1023) + 512);
          } else {
            condval_6 = ((chunk_w * 128) + pos0_w);
          }
          tl_crouton_copy<true, true>((&(kv_v_w[(kvh * 393216)])), 96, 4, ((kv0 * 128) + condval_6), 0, 128, 128, (&(((float*)hd)[0])), 128, 1);
          int32_t condval_7;
          if ((0 < window)) {
            condval_7 = ((((chunk_w * 128) + pos0_w) & 1023) + 512);
          } else {
            condval_7 = ((chunk_w * 128) + pos0_w);
          }
          int32_t condval_8;
          if ((0 < window)) {
            condval_8 = ((((chunk_w * 128) + pos0_w) & 1023) + 512);
          } else {
            condval_8 = ((chunk_w * 128) + pos0_w);
          }
          int32_t condval_9;
          if ((0 < window)) {
            condval_9 = ((((chunk_w * 128) + pos0_w) & 1023) + 512);
          } else {
            condval_9 = ((chunk_w * 128) + pos0_w);
          }
          int32_t condval_10;
          if ((0 < window)) {
            condval_10 = ((((chunk_w * 128) + pos0_w) & 1023) + 512);
          } else {
            condval_10 = ((chunk_w * 128) + pos0_w);
          }
          int32_t condval_11;
          if ((0 < window)) {
            condval_11 = ((((chunk_w * 128) + pos0_w) & 1023) + 512);
          } else {
            condval_11 = ((chunk_w * 128) + pos0_w);
          }
          if (int32_t tl_status = tl_hexagon_cache_writeback((&(kv_v_w[(((((((((condval_7 >> 7) + kv0) / 24) + ((((condval_8 >> 7) + kv0) % 24) >> 31)) + kvh) & 3) * 393216) + ((96 & ((((kv0 * 4) + (condval_9 >> 5)) % 96) >> 31)) * 1024)) + ((((kv0 * 4) + (condval_10 >> 5)) % 96) * 1024))])), ((uint32_t)((((condval_11 & 31) + 159) >> 5) * 2048)), (uint32_t)4, (uint32_t)196608)) return tl_status;
          if ((0 < window) && (896 < (((chunk_w * 128) + pos0_w) & 1023))) {
            tl_crouton_copy<true, true>((&(kv_v_w[(kvh * 393216)])), 96, 4, (((kv0 * 128) + (((chunk_w * 128) + pos0_w) & 1023)) - 512), 0, 128, 128, (&(((float*)hd)[0])), 128, 1);
            if (int32_t tl_status = tl_hexagon_cache_writeback((&(kv_v_w[((((((((((kv0 * 128) + ((((chunk_w * 4) + (pos0_w >> 5)) & 31) * 32)) - 512) / 3072) + (((((kv0 * 128) + ((((chunk_w * 4) + (pos0_w >> 5)) & 31) * 32)) - 512) % 3072) >> 31)) + kvh) & 3) * 393216) + ((96 & (((((kv0 * 4) + (((chunk_w * 4) + (pos0_w >> 5)) & 31)) + 80) % 96) >> 31)) * 1024)) + (((((kv0 * 4) + (((chunk_w * 4) + (pos0_w >> 5)) & 31)) + 80) % 96) * 1024))])), ((uint32_t)(((((pos0_w & 31) - 353) >> 5) * 2048) + 32768)), (uint32_t)4, (uint32_t)196608)) return tl_status;
          }
          if (int32_t tl_status = tl_hexagon_dma_async_wait(tl_dma_async, 0)) return tl_status;
          tl_hexagon_hmx_gemm<true, true, true, false>(128, 32, 2048, (&(((float*)cc)[0])), 32, (&(a_w[0])), 65536, (&(((half*)v)[0])), 65536, (bool)1, (&(((half*)stage_2)[0])), (&(((uint32_t*)config_2)[0])));
          for (int32_t i_11 = 0; i_11 < 512; ++i_11) {
            ((float*)gates)[(((i_11 & 3) * 128) + (i_11 >> 2))] = tl_hvx_sigmoid_fast(((float*)cc)[(((i_11 >> 2) * 32) + (i_11 & 3))]);
          }
          for (int32_t hq = 0; hq < 4; ++hq) {
            for (int32_t rb_1 = 0; rb_1 < 2; ++rb_1) {
              if (int32_t tl_status = tl_hexagon_dma_async_submit(tl_dma_async, (&(((half*)qb)[0])), (&(qs_w[(((kvh * 65536) + (hq * 16384)) + (rb_1 * 8192))])), (uint32_t)16384, (uint32_t)16384, (uint32_t)16384, (uint32_t)1, 3)) return tl_status;
              if (int32_t tl_status = tl_hexagon_dma_async_commit(tl_dma_async)) return tl_status;
              for (int32_t i_12 = 0; i_12 < 256; ++i_12) {
                float broadcast_var_19 = 0.000000e+00f;
                *(float32*)(((float*)acc_o) + (i_12 * 32)) = ((float32)(broadcast_var_19));
              }
              for (int32_t i_13 = 0; i_13 < 2; ++i_13) {
                float broadcast_var_20 = -3.000000e+38f;
                *(float32*)(((float*)m) + (i_13 * 32)) = ((float32)(broadcast_var_20));
                float broadcast_var_21 = 0.000000e+00f;
                *(float32*)(((float*)ls) + (i_13 * 32)) = ((float32)(broadcast_var_21));
              }
              int32_t condval_12;
              if ((0 < window)) {
                int32_t v__10 = ((((chunk_w * 128) + (rb_1 * 64)) + pos0_w) + 1) - window;
                condval_12 = (((((v__10) > (0) ? (v__10) : (0)) & 1023) >> 7) + 4);
              } else {
                int32_t v__11 = (((chunk_w * 128) + (rb_1 * 64)) + pos0_w) - 1151;
                int32_t v__12 = ((v__11) > (0) ? (v__11) : (0)) >> 7;
                condval_12 = ((v__12) < (7) ? (v__12) : (7));
              }
              int32_t v__13 = kv0 + condval_12;
              int32_t v__14 = (v__13) < (23) ? (v__13) : (23);
              if (int32_t tl_status = tl_hexagon_dma_async_submit(tl_dma_async, (&(((half*)kblk)[0])), (&(kv_k_w[((kvh * 393216) + (((v__14) > (0) ? (v__14) : (0)) * 16384))])), (uint32_t)32768, (uint32_t)32768, (uint32_t)32768, (uint32_t)1, 3)) return tl_status;
              if (int32_t tl_status = tl_hexagon_dma_async_commit(tl_dma_async)) return tl_status;
              int32_t condval_13;
              if ((0 < window)) {
                int32_t v__15 = ((((chunk_w * 128) + (rb_1 * 64)) + pos0_w) + 1) - window;
                condval_13 = (((((v__15) > (0) ? (v__15) : (0)) & 1023) >> 7) + 4);
              } else {
                int32_t v__16 = (((chunk_w * 128) + (rb_1 * 64)) + pos0_w) - 1151;
                int32_t v__17 = ((v__16) > (0) ? (v__16) : (0)) >> 7;
                condval_13 = ((v__17) < (7) ? (v__17) : (7));
              }
              int32_t v__18 = kv0 + condval_13;
              int32_t v__19 = (v__18) < (23) ? (v__18) : (23);
              if (int32_t tl_status = tl_hexagon_dma_async_submit(tl_dma_async, (&(((half*)vblk)[0])), (&(kv_v_w[((kvh * 393216) + (((v__19) > (0) ? (v__19) : (0)) * 4096))])), (uint32_t)8192, (uint32_t)196608, (uint32_t)8192, (uint32_t)4, 3)) return tl_status;
              if (int32_t tl_status = tl_hexagon_dma_async_commit(tl_dma_async)) return tl_status;
              int32_t condval_14;
              if ((0 < window)) {
                condval_14 = window;
              } else {
                condval_14 = 1152;
              }
              int32_t v__20 = ((((chunk_w * 128) + (rb_1 * 64)) + pos0_w) + 1) - condval_14;
              for (int32_t kb = (((v__20) > (0) ? (v__20) : (0)) >> 7); kb < (((((pos0_w + 191) >> 6) + rb_1) >> 1) + chunk_w); ++kb) {
                if ((kb + 1) < (((((pos0_w + 191) >> 6) + rb_1) >> 1) + chunk_w)) {
                  int32_t condval_15;
                  if ((0 < window)) {
                    condval_15 = window;
                  } else {
                    condval_15 = 1152;
                  }
                  int32_t v__21 = ((((chunk_w * 128) + (rb_1 * 64)) + pos0_w) + 1) - condval_15;
                  int32_t condval_16;
                  if ((0 < window)) {
                    condval_16 = (((kb + 1) & 7) + 4);
                  } else {
                    int32_t v__22 = kb + 1;
                    condval_16 = ((v__22) < (7) ? (v__22) : (7));
                  }
                  int32_t v__23 = kv0 + condval_16;
                  int32_t v__24 = (v__23) < (23) ? (v__23) : (23);
                  if (int32_t tl_status = tl_hexagon_dma_async_submit(tl_dma_async, (&(((half*)kblk)[(16384 - (((kb - (((v__21) > (0) ? (v__21) : (0)) >> 7)) & 1) * 16384))])), (&(kv_k_w[((kvh * 393216) + (((v__24) > (0) ? (v__24) : (0)) * 16384))])), (uint32_t)32768, (uint32_t)32768, (uint32_t)32768, (uint32_t)1, 3)) return tl_status;
                  if (int32_t tl_status = tl_hexagon_dma_async_commit(tl_dma_async)) return tl_status;
                  int32_t condval_17;
                  if ((0 < window)) {
                    condval_17 = window;
                  } else {
                    condval_17 = 1152;
                  }
                  int32_t v__25 = ((((chunk_w * 128) + (rb_1 * 64)) + pos0_w) + 1) - condval_17;
                  int32_t condval_18;
                  if ((0 < window)) {
                    condval_18 = (((kb + 1) & 7) + 4);
                  } else {
                    int32_t v__26 = kb + 1;
                    condval_18 = ((v__26) < (7) ? (v__26) : (7));
                  }
                  int32_t v__27 = kv0 + condval_18;
                  int32_t v__28 = (v__27) < (23) ? (v__27) : (23);
                  if (int32_t tl_status = tl_hexagon_dma_async_submit(tl_dma_async, (&(((half*)vblk)[(16384 - (((kb - (((v__25) > (0) ? (v__25) : (0)) >> 7)) & 1) * 16384))])), (&(kv_v_w[((kvh * 393216) + (((v__28) > (0) ? (v__28) : (0)) * 4096))])), (uint32_t)8192, (uint32_t)196608, (uint32_t)8192, (uint32_t)4, 3)) return tl_status;
                  if (int32_t tl_status = tl_hexagon_dma_async_commit(tl_dma_async)) return tl_status;
                  if (int32_t tl_status = tl_hexagon_dma_async_wait(tl_dma_async, 2)) return tl_status;
                } else {
                  if (int32_t tl_status = tl_hexagon_dma_async_wait(tl_dma_async, 0)) return tl_status;
                }
                int32_t condval_19;
                if ((0 < window)) {
                  condval_19 = window;
                } else {
                  condval_19 = 1152;
                }
                int32_t v__29 = ((((chunk_w * 128) + (rb_1 * 64)) + pos0_w) + 1) - condval_19;
                tl_hexagon_hmx_gemm<false, true, true, true>(64, 128, 128, (&(((half*)s)[0])), 4, (&(((half*)qb)[0])), 4096, (&(((half*)kblk)[(((kb - (((v__29) > (0) ? (v__29) : (0)) >> 7)) & 1) * 16384)])), 4096, (bool)1, (&(((half*)stage_3)[0])), (&(((uint32_t*)config_3)[0])));
                int32_t condval_20;
                if ((0 < window)) {
                  condval_20 = window;
                } else {
                  condval_20 = 1152;
                }
                if (((((chunk_w * 128) + (rb_1 * 64)) + pos0_w) < ((kb * 128) + 127)) || ((kb * 128) <= (((((chunk_w * 128) + (rb_1 * 64)) + pos0_w) + 63) - condval_20))) {
                  for (int32_t i_14 = 0; i_14 < 8; ++i_14) {
                    for (int32_t j_1 = 0; j_1 < 16; ++j_1) {
                      int32_t broadcast_var_22 = 2;
                      int32_t broadcast_var_23 = 2;
                      half broadcast_var_24 = (half)-6.550400e+04f;
                      half64 causal = tl_vec_select(((((int32_t64)(((kb * 128) + ((i_14 & 3) * 32)))) + (tl_vec_ramp<int32_t, 64>(0, 1) / ((int32_t64)(broadcast_var_22)))) <= (((int32_t64)((((((chunk_w * 128) + (rb_1 * 64)) + ((i_14 >> 2) * 32)) + (j_1 * 2)) + pos0_w))) + (tl_vec_ramp<int32_t, 64>(0, 1) % ((int32_t64)(broadcast_var_23))))), *(half64*)(((half*)s) + ((i_14 * 1024) + (j_1 * 64))), ((half64)(broadcast_var_24)));
                      int32_t broadcast_var_25 = 2;
                      int32_t broadcast_var_26 = 2;
                      half broadcast_var_27 = (half)-6.550400e+04f;
                      *(half64*)(((half*)s) + ((i_14 * 1024) + (j_1 * 64))) = tl_vec_select((((((int32_t64)((((((chunk_w * 128) + (rb_1 * 64)) + ((i_14 >> 2) * 32)) + (j_1 * 2)) + pos0_w))) + (tl_vec_ramp<int32_t, 64>(0, 1) % ((int32_t64)(broadcast_var_25)))) - ((int32_t64)(((0 < window) ? window : 1152)))) < (((int32_t64)(((kb * 128) + ((i_14 & 3) * 32)))) + (tl_vec_ramp<int32_t, 64>(0, 1) / ((int32_t64)(broadcast_var_26))))), causal, ((half64)(broadcast_var_27)));
                    }
                  }
                }
                tl_crouton_reduce<TL_REDUCE_MAX, half, true, false>((&(((float*)rmax)[0])), (&(((half*)s)[0])), 2, 4, 1, (bool)1);
                *(float64*)(((float*)rmax) + 0) = tl_vec_max(*(float64*)(((float*)m) + 0), *(float64*)(((float*)rmax) + 0));
                *(float64*)(((float*)alpha) + 0) = tl_hvx_exp2_fast((*(float64*)(((float*)m) + 0) - *(float64*)(((float*)rmax) + 0)));
                *(float64*)(((float*)m) + 0) = *(float64*)(((float*)rmax) + 0);
                *(half64*)(((half*)mh) + 0) = ((half64)*(float64*)(((float*)rmax) + 0));
                for (int32_t i_15 = 0; i_15 < 8; ++i_15) {
                  for (int32_t j_strip = 0; j_strip < 1; ++j_strip) {
                    half64 mh_lanes = tl_hvx_load_lanes<64>((&(((half*)mh)[((i_15 >> 2) * 32)])), 32);
                    half* s_row = ((half*)s) + (i_15 * 1024);
                    half* p_row = ((half*)p) + (i_15 * 1024);
                    #pragma clang loop unroll_count(2)
                    for (int32_t j_lane = 0; j_lane < 16; ++j_lane) {
                      *(half64*)(p_row + (j_lane * 64)) = tl_hvx_exp2_fast((*(half64*)(s_row + (j_lane * 64)) - tl_hvx_broadcast_lane<64, 2>(mh_lanes, j_lane)));
                    }
                  }
                }
                tl_hexagon_hmx_gemm<false, true, true, true>(64, 32, 128, (&(((half*)sum_hmx)[0])), 1, (&(((half*)p)[0])), 4096, (&(((half*)sum_ones)[0])), 4096, (bool)1, (&(((half*)stage_4)[0])), (&(((uint32_t*)config_4)[0])));
                Q6_vgather_ARMVw((HVX_Vector*)rsum16, (size_t)sum_hmx, 4095, *reinterpret_cast<const HVX_Vector*>(tl_prefill_diag_offsets));
                asm volatile("" ::: "memory");
                *(float64*)(((float*)rsum) + 0) = (float64)(*(half64*)(((half*)rsum16) + 0));
                for (int32_t i_16 = 0; i_16 < 2; ++i_16) {
                  *(float32*)(((float*)ls) + (i_16 * 32)) = ((*(float32*)(((float*)ls) + (i_16 * 32)) * *(float32*)(((float*)alpha) + (i_16 * 32))) + *(float32*)(((float*)rsum) + (i_16 * 32)));
                }
                for (int32_t i_strip_2 = 0; i_strip_2 < 2; ++i_strip_2) {
                  float32 alpha_lanes = tl_hvx_load_lanes<32>((&(((float*)alpha)[(i_strip_2 * 32)])), 32);
                  for (int32_t i_lane_2 = 0; i_lane_2 < 32; ++i_lane_2) {
                    *(float32*)(((float*)acc_o) + ((i_strip_2 * 4096) + (i_lane_2 * 128))) = (*(float32*)(((float*)acc_o) + ((i_strip_2 * 4096) + (i_lane_2 * 128))) * tl_hvx_broadcast_lane<32, 1>(alpha_lanes, i_lane_2));
                    *(float32*)(((float*)acc_o) + (((i_strip_2 * 4096) + (i_lane_2 * 128)) + 32)) = (*(float32*)(((float*)acc_o) + (((i_strip_2 * 4096) + (i_lane_2 * 128)) + 32)) * tl_hvx_broadcast_lane<32, 1>(alpha_lanes, i_lane_2));
                    *(float32*)(((float*)acc_o) + (((i_strip_2 * 4096) + (i_lane_2 * 128)) + 64)) = (*(float32*)(((float*)acc_o) + (((i_strip_2 * 4096) + (i_lane_2 * 128)) + 64)) * tl_hvx_broadcast_lane<32, 1>(alpha_lanes, i_lane_2));
                    *(float32*)(((float*)acc_o) + (((i_strip_2 * 4096) + (i_lane_2 * 128)) + 96)) = (*(float32*)(((float*)acc_o) + (((i_strip_2 * 4096) + (i_lane_2 * 128)) + 96)) * tl_hvx_broadcast_lane<32, 1>(alpha_lanes, i_lane_2));
                  }
                }
                int32_t condval_21;
                if ((0 < window)) {
                  condval_21 = window;
                } else {
                  condval_21 = 1152;
                }
                int32_t v__30 = ((((chunk_w * 128) + (rb_1 * 64)) + pos0_w) + 1) - condval_21;
                tl_hexagon_hmx_gemm<true, true, true, false>(64, 128, 128, (&(((float*)acc_o)[0])), 128, (&(((half*)p)[0])), 4096, (&(((half*)vblk)[(((kb - (((v__30) > (0) ? (v__30) : (0)) >> 7)) & 1) * 16384)])), 4096, (bool)0, (&(((half*)stage_5)[0])), (&(((uint32_t*)config_5)[0])));
              }
              if (int32_t tl_status = tl_hexagon_dma_async_wait(tl_dma_async, 0)) return tl_status;
              #pragma clang loop unroll_count(2)
              for (int32_t i_17 = 0; i_17 < 2; ++i_17) {
                *(float32*)(((float*)alpha) + (i_17 * 32)) = tl_hvx_div_fast(*(float32*)(((float*)gates) + (((hq * 128) + (rb_1 * 64)) + (i_17 * 32))), *(float32*)(((float*)ls) + (i_17 * 32)));
              }
              for (int32_t i_strip_3 = 0; i_strip_3 < 2; ++i_strip_3) {
                float32 alpha_lanes_1 = tl_hvx_load_lanes<32>((&(((float*)alpha)[(i_strip_3 * 32)])), 32);
                for (int32_t i_lane_3 = 0; i_lane_3 < 32; ++i_lane_3) {
                  *(float32*)(((float*)acc_o) + ((i_strip_3 * 4096) + (i_lane_3 * 128))) = (*(float32*)(((float*)acc_o) + ((i_strip_3 * 4096) + (i_lane_3 * 128))) * tl_hvx_broadcast_lane<32, 1>(alpha_lanes_1, i_lane_3));
                  *(float32*)(((float*)acc_o) + (((i_strip_3 * 4096) + (i_lane_3 * 128)) + 32)) = (*(float32*)(((float*)acc_o) + (((i_strip_3 * 4096) + (i_lane_3 * 128)) + 32)) * tl_hvx_broadcast_lane<32, 1>(alpha_lanes_1, i_lane_3));
                  *(float32*)(((float*)acc_o) + (((i_strip_3 * 4096) + (i_lane_3 * 128)) + 64)) = (*(float32*)(((float*)acc_o) + (((i_strip_3 * 4096) + (i_lane_3 * 128)) + 64)) * tl_hvx_broadcast_lane<32, 1>(alpha_lanes_1, i_lane_3));
                  *(float32*)(((float*)acc_o) + (((i_strip_3 * 4096) + (i_lane_3 * 128)) + 96)) = (*(float32*)(((float*)acc_o) + (((i_strip_3 * 4096) + (i_lane_3 * 128)) + 96)) * tl_hvx_broadcast_lane<32, 1>(alpha_lanes_1, i_lane_3));
                }
              }
              tl_crouton_copy<true, false>((&(hid_w[0])), 4, 64, (rb_1 * 64), ((kvh * 512) + (hq * 128)), 64, 128, (&(((float*)acc_o)[0])), 128, 1);
            }
          }
        } else {
          if (int32_t tl_status = tl_hexagon_dma_async_submit(tl_dma_async, (&(((half*)v)[0])), (&(in_w_w[((((int64_t)ci) * (int64_t)12582912) + (((int64_t)kvh) * (int64_t)3145728))])), (uint32_t)131072, (uint32_t)131072, (uint32_t)131072, (uint32_t)1, 3)) return tl_status;
          if (int32_t tl_status = tl_hexagon_dma_async_commit(tl_dma_async)) return tl_status;
          if (int32_t tl_status = tl_hexagon_dma_async_submit(tl_dma_async, (&(((half*)v)[65536])), (&(in_w_w[(((((int64_t)ci) * (int64_t)12582912) + (((int64_t)kvh) * (int64_t)3145728)) + (int64_t)131072)])), (uint32_t)131072, (uint32_t)131072, (uint32_t)131072, (uint32_t)1, 3)) return tl_status;
          if (int32_t tl_status = tl_hexagon_dma_async_commit(tl_dma_async)) return tl_status;
          for (int32_t b_1 = 0; b_1 < 46; ++b_1) {
            if (int32_t tl_status = tl_hexagon_dma_async_wait(tl_dma_async, 1)) return tl_status;
            tl_hexagon_hmx_gemm<true, true, true, false>(128, 32, 2048, (&(((float*)cc)[(((b_1 % 3) & 1) * 4096)])), 32, (&(a_w[0])), 65536, (&(((half*)v)[((b_1 & 1) * 65536)])), 65536, (bool)1, (&(((half*)stage_6)[0])), (&(((uint32_t*)config_6)[0])));
            if ((b_1 % 3) == 1) {
              for (int32_t i_18 = 0; i_18 < 128; ++i_18) {
                *(float32*)(((float*)cc) + (i_18 * 32)) = (*(float32*)(((float*)cc) + (i_18 * 32)) * *(float32*)(((float*)cc) + ((i_18 * 32) + 4096)));
              }
              for (int32_t i_19 = 0; i_19 < 128; ++i_19) {
                float broadcast_var_28 = 0.000000e+00f;
                float32 guarded_11 = *(float32*)(conv_w_w + (((ci < 2) && (0 <= ci)) ? ((((((int64_t)ci) * (int64_t)14336) + (((int64_t)kvh) * (int64_t)512)) + ((((int64_t)b_1) / (int64_t)3) * (int64_t)32)) + (int64_t)12288) : (int64_t)0));
                float32 y = ((((ci < 2) && (0 <= ci)) ? guarded_11 : ((float32)(broadcast_var_28))) * *(float32*)(((float*)cc) + (i_19 * 32)));
                float broadcast_var_29 = 0.000000e+00f;
                float32 condval_22;
                if ((6 <= i_19)) {
                  condval_22 = *(float32*)(((float*)cc) + ((i_19 * 32) - 192));
                } else {
                  float32 guarded_12 = *(float32*)(conv_state_w + (((ci < 2) && (0 <= ci)) ? ((((((int64_t)ci) * (int64_t)12288) + (((int64_t)i_19) * (int64_t)2048)) + (((int64_t)kvh) * (int64_t)512)) + ((((int64_t)b_1) / (int64_t)3) * (int64_t)32)) : (int64_t)0));
                  condval_22 = (((ci < 2) && (0 <= ci)) ? guarded_12 : ((float32)(broadcast_var_29)));
                }
                float32 prev = condval_22;
                float broadcast_var_30 = 0.000000e+00f;
                float32 guarded_13 = *(float32*)(conv_w_w + (((ci < 2) && (0 <= ci)) ? (((((int64_t)ci) * (int64_t)14336) + (((int64_t)kvh) * (int64_t)512)) + ((((int64_t)b_1) / (int64_t)3) * (int64_t)32)) : (int64_t)0));
                float32 y_1 = (y + ((((ci < 2) && (0 <= ci)) ? guarded_13 : ((float32)(broadcast_var_30))) * prev));
                float broadcast_var_31 = 0.000000e+00f;
                float32 condval_23;
                if ((5 <= i_19)) {
                  condval_23 = *(float32*)(((float*)cc) + ((i_19 * 32) - 160));
                } else {
                  float32 guarded_14 = *(float32*)(conv_state_w + (((ci < 2) && (0 <= ci)) ? (((((((int64_t)ci) * (int64_t)12288) + (((int64_t)i_19) * (int64_t)2048)) + (((int64_t)kvh) * (int64_t)512)) + ((((int64_t)b_1) / (int64_t)3) * (int64_t)32)) + (int64_t)2048) : (int64_t)0));
                  condval_23 = (((ci < 2) && (0 <= ci)) ? guarded_14 : ((float32)(broadcast_var_31)));
                }
                float32 prev_1 = condval_23;
                float broadcast_var_32 = 0.000000e+00f;
                float32 guarded_15 = *(float32*)(conv_w_w + (((ci < 2) && (0 <= ci)) ? ((((((int64_t)ci) * (int64_t)14336) + (((int64_t)kvh) * (int64_t)512)) + ((((int64_t)b_1) / (int64_t)3) * (int64_t)32)) + (int64_t)2048) : (int64_t)0));
                float32 y_2 = (y_1 + ((((ci < 2) && (0 <= ci)) ? guarded_15 : ((float32)(broadcast_var_32))) * prev_1));
                float broadcast_var_33 = 0.000000e+00f;
                float32 condval_24;
                if ((4 <= i_19)) {
                  condval_24 = *(float32*)(((float*)cc) + ((i_19 * 32) - 128));
                } else {
                  float32 guarded_16 = *(float32*)(conv_state_w + (((ci < 2) && (0 <= ci)) ? (((((((int64_t)ci) * (int64_t)12288) + (((int64_t)i_19) * (int64_t)2048)) + (((int64_t)kvh) * (int64_t)512)) + ((((int64_t)b_1) / (int64_t)3) * (int64_t)32)) + (int64_t)4096) : (int64_t)0));
                  condval_24 = (((ci < 2) && (0 <= ci)) ? guarded_16 : ((float32)(broadcast_var_33)));
                }
                float32 prev_2 = condval_24;
                float broadcast_var_34 = 0.000000e+00f;
                float32 guarded_17 = *(float32*)(conv_w_w + (((ci < 2) && (0 <= ci)) ? ((((((int64_t)ci) * (int64_t)14336) + (((int64_t)kvh) * (int64_t)512)) + ((((int64_t)b_1) / (int64_t)3) * (int64_t)32)) + (int64_t)4096) : (int64_t)0));
                float32 y_3 = (y_2 + ((((ci < 2) && (0 <= ci)) ? guarded_17 : ((float32)(broadcast_var_34))) * prev_2));
                float broadcast_var_35 = 0.000000e+00f;
                float32 condval_25;
                if ((3 <= i_19)) {
                  condval_25 = *(float32*)(((float*)cc) + ((i_19 * 32) - 96));
                } else {
                  float32 guarded_18 = *(float32*)(conv_state_w + (((ci < 2) && (0 <= ci)) ? (((((((int64_t)ci) * (int64_t)12288) + (((int64_t)i_19) * (int64_t)2048)) + (((int64_t)kvh) * (int64_t)512)) + ((((int64_t)b_1) / (int64_t)3) * (int64_t)32)) + (int64_t)6144) : (int64_t)0));
                  condval_25 = (((ci < 2) && (0 <= ci)) ? guarded_18 : ((float32)(broadcast_var_35)));
                }
                float32 prev_3 = condval_25;
                float broadcast_var_36 = 0.000000e+00f;
                float32 guarded_19 = *(float32*)(conv_w_w + (((ci < 2) && (0 <= ci)) ? ((((((int64_t)ci) * (int64_t)14336) + (((int64_t)kvh) * (int64_t)512)) + ((((int64_t)b_1) / (int64_t)3) * (int64_t)32)) + (int64_t)6144) : (int64_t)0));
                float32 y_4 = (y_3 + ((((ci < 2) && (0 <= ci)) ? guarded_19 : ((float32)(broadcast_var_36))) * prev_3));
                float broadcast_var_37 = 0.000000e+00f;
                float32 condval_26;
                if ((2 <= i_19)) {
                  condval_26 = *(float32*)(((float*)cc) + ((i_19 * 32) - 64));
                } else {
                  float32 guarded_20 = *(float32*)(conv_state_w + (((ci < 2) && (0 <= ci)) ? (((((((int64_t)ci) * (int64_t)12288) + (((int64_t)i_19) * (int64_t)2048)) + (((int64_t)kvh) * (int64_t)512)) + ((((int64_t)b_1) / (int64_t)3) * (int64_t)32)) + (int64_t)8192) : (int64_t)0));
                  condval_26 = (((ci < 2) && (0 <= ci)) ? guarded_20 : ((float32)(broadcast_var_37)));
                }
                float32 prev_4 = condval_26;
                float broadcast_var_38 = 0.000000e+00f;
                float32 guarded_21 = *(float32*)(conv_w_w + (((ci < 2) && (0 <= ci)) ? ((((((int64_t)ci) * (int64_t)14336) + (((int64_t)kvh) * (int64_t)512)) + ((((int64_t)b_1) / (int64_t)3) * (int64_t)32)) + (int64_t)8192) : (int64_t)0));
                float32 y_5 = (y_4 + ((((ci < 2) && (0 <= ci)) ? guarded_21 : ((float32)(broadcast_var_38))) * prev_4));
                float broadcast_var_39 = 0.000000e+00f;
                float32 condval_27;
                if ((1 <= i_19)) {
                  condval_27 = *(float32*)(((float*)cc) + ((i_19 * 32) - 32));
                } else {
                  float32 guarded_22 = *(float32*)(conv_state_w + (((ci < 2) && (0 <= ci)) ? (((((((int64_t)ci) * (int64_t)12288) + (((int64_t)i_19) * (int64_t)2048)) + (((int64_t)kvh) * (int64_t)512)) + ((((int64_t)b_1) / (int64_t)3) * (int64_t)32)) + (int64_t)10240) : (int64_t)0));
                  condval_27 = (((ci < 2) && (0 <= ci)) ? guarded_22 : ((float32)(broadcast_var_39)));
                }
                float32 prev_5 = condval_27;
                float broadcast_var_40 = 0.000000e+00f;
                float32 guarded_23 = *(float32*)(conv_w_w + (((ci < 2) && (0 <= ci)) ? ((((((int64_t)ci) * (int64_t)14336) + (((int64_t)kvh) * (int64_t)512)) + ((((int64_t)b_1) / (int64_t)3) * (int64_t)32)) + (int64_t)10240) : (int64_t)0));
                float32 y_6 = (y_5 + ((((ci < 2) && (0 <= ci)) ? guarded_23 : ((float32)(broadcast_var_40))) * prev_5));
                *(float32*)(((float*)cc) + ((i_19 * 32) + 4096)) = y_6;
              }
              for (int32_t i_20 = 0; i_20 < 6; ++i_20) {
                float broadcast_var_41 = 0.000000e+00f;
                int32_t v__31 = n_w - (chunk_w * 128);
                float32 condval_28;
                if ((6 <= (((128) < (v__31) ? (128) : (v__31)) + i_20))) {
                  int64_t v__32 = (int64_t)128;
                  int64_t v__33 = ((int64_t)n_w) - (((int64_t)chunk_w) * (int64_t)128);
                  condval_28 = *(float32*)(((float*)cc) + (((((v__32) < (v__33) ? (v__32) : (v__33)) * (int64_t)32) + (((int64_t)i_20) * (int64_t)32)) - (int64_t)192));
                } else {
                  int64_t v__34 = (int64_t)128;
                  int64_t v__35 = ((int64_t)n_w) - (((int64_t)chunk_w) * (int64_t)128);
                  float32 guarded_24 = *(float32*)(conv_state_w + ((((0 <= (((128) < (v__31) ? (128) : (v__31)) + i_20)) && (0 <= ci)) && (ci < 2)) ? (((((((int64_t)ci) * (int64_t)12288) + (((v__34) < (v__35) ? (v__34) : (v__35)) * (int64_t)2048)) + (((int64_t)i_20) * (int64_t)2048)) + (((int64_t)kvh) * (int64_t)512)) + ((((int64_t)b_1) / (int64_t)3) * (int64_t)32)) : (int64_t)0));
                  condval_28 = ((((0 <= (((128) < (v__31) ? (128) : (v__31)) + i_20)) && (0 <= ci)) && (ci < 2)) ? guarded_24 : ((float32)(broadcast_var_41)));
                }
                *(float32*)(((float*)st) + (i_20 * 32)) = condval_28;
              }
              if ((ci < 2) && (0 <= ci)) {
                for (int32_t i_21 = 0; i_21 < 6; ++i_21) {
                  *(float32*)(conv_state_w + ((((ci * 12288) + (i_21 * 2048)) + (kvh * 512)) + ((b_1 / 3) * 32))) = *(float32*)(((float*)st) + (i_21 * 32));
                }
              }
            }
            if ((b_1 % 3) == 2) {
              for (int32_t i_22 = 0; i_22 < 128; ++i_22) {
                *(float32*)(((float*)cc) + (i_22 * 32)) = (*(float32*)(((float*)cc) + (i_22 * 32)) * *(float32*)(((float*)cc) + ((i_22 * 32) + 4096)));
              }
              tl_crouton_copy<true, false>((&(hid_w[0])), 4, 64, 0, ((kvh * 512) + ((b_1 / 3) * 32)), 128, 32, (&(((float*)cc)[0])), 32, 1);
            }
            if (int32_t tl_status = tl_hexagon_dma_async_submit(tl_dma_async, (&(((half*)v)[((b_1 & 1) * 65536)])), (&(in_w_w[((((((int64_t)ci) * (int64_t)12582912) + (((int64_t)kvh) * (int64_t)3145728)) + (((((int64_t)b_1) + (int64_t)2) / (int64_t)3) * (int64_t)196608)) + (((((((int64_t)b_1) + (int64_t)2) % (int64_t)3) * (int64_t)2) % (int64_t)3) * (int64_t)65536))])), (uint32_t)131072, (uint32_t)131072, (uint32_t)131072, (uint32_t)1, 3)) return tl_status;
            if (int32_t tl_status = tl_hexagon_dma_async_commit(tl_dma_async)) return tl_status;
          }
          if (int32_t tl_status = tl_hexagon_dma_async_wait(tl_dma_async, 1)) return tl_status;
          tl_hexagon_hmx_gemm<true, true, true, false>(128, 32, 2048, (&(((float*)cc)[4096])), 32, (&(a_w[0])), 65536, (&(((half*)v)[0])), 65536, (bool)1, (&(((half*)stage_7)[0])), (&(((uint32_t*)config_7)[0])));
          for (int32_t i_23 = 0; i_23 < 128; ++i_23) {
            *(float32*)(((float*)cc) + (i_23 * 32)) = (*(float32*)(((float*)cc) + (i_23 * 32)) * *(float32*)(((float*)cc) + ((i_23 * 32) + 4096)));
          }
          for (int32_t i_24 = 0; i_24 < 128; ++i_24) {
            float broadcast_var_42 = 0.000000e+00f;
            float32 guarded_25 = *(float32*)(conv_w_w + (((ci < 2) && (0 <= ci)) ? (((((int64_t)ci) * (int64_t)14336) + (((int64_t)kvh) * (int64_t)512)) + (int64_t)12768) : (int64_t)0));
            float32 y_7 = ((((ci < 2) && (0 <= ci)) ? guarded_25 : ((float32)(broadcast_var_42))) * *(float32*)(((float*)cc) + (i_24 * 32)));
            float broadcast_var_43 = 0.000000e+00f;
            float32 condval_29;
            if ((6 <= i_24)) {
              condval_29 = *(float32*)(((float*)cc) + ((i_24 * 32) - 192));
            } else {
              float32 guarded_26 = *(float32*)(conv_state_w + (((ci < 2) && (0 <= ci)) ? ((((((int64_t)ci) * (int64_t)12288) + (((int64_t)i_24) * (int64_t)2048)) + (((int64_t)kvh) * (int64_t)512)) + (int64_t)480) : (int64_t)0));
              condval_29 = (((ci < 2) && (0 <= ci)) ? guarded_26 : ((float32)(broadcast_var_43)));
            }
            float32 prev_6 = condval_29;
            float broadcast_var_44 = 0.000000e+00f;
            float32 guarded_27 = *(float32*)(conv_w_w + (((ci < 2) && (0 <= ci)) ? (((((int64_t)ci) * (int64_t)14336) + (((int64_t)kvh) * (int64_t)512)) + (int64_t)480) : (int64_t)0));
            float32 y_8 = (y_7 + ((((ci < 2) && (0 <= ci)) ? guarded_27 : ((float32)(broadcast_var_44))) * prev_6));
            float broadcast_var_45 = 0.000000e+00f;
            float32 condval_30;
            if ((5 <= i_24)) {
              condval_30 = *(float32*)(((float*)cc) + ((i_24 * 32) - 160));
            } else {
              float32 guarded_28 = *(float32*)(conv_state_w + (((ci < 2) && (0 <= ci)) ? ((((((int64_t)ci) * (int64_t)12288) + (((int64_t)i_24) * (int64_t)2048)) + (((int64_t)kvh) * (int64_t)512)) + (int64_t)2528) : (int64_t)0));
              condval_30 = (((ci < 2) && (0 <= ci)) ? guarded_28 : ((float32)(broadcast_var_45)));
            }
            float32 prev_7 = condval_30;
            float broadcast_var_46 = 0.000000e+00f;
            float32 guarded_29 = *(float32*)(conv_w_w + (((ci < 2) && (0 <= ci)) ? (((((int64_t)ci) * (int64_t)14336) + (((int64_t)kvh) * (int64_t)512)) + (int64_t)2528) : (int64_t)0));
            float32 y_9 = (y_8 + ((((ci < 2) && (0 <= ci)) ? guarded_29 : ((float32)(broadcast_var_46))) * prev_7));
            float broadcast_var_47 = 0.000000e+00f;
            float32 condval_31;
            if ((4 <= i_24)) {
              condval_31 = *(float32*)(((float*)cc) + ((i_24 * 32) - 128));
            } else {
              float32 guarded_30 = *(float32*)(conv_state_w + (((ci < 2) && (0 <= ci)) ? ((((((int64_t)ci) * (int64_t)12288) + (((int64_t)i_24) * (int64_t)2048)) + (((int64_t)kvh) * (int64_t)512)) + (int64_t)4576) : (int64_t)0));
              condval_31 = (((ci < 2) && (0 <= ci)) ? guarded_30 : ((float32)(broadcast_var_47)));
            }
            float32 prev_8 = condval_31;
            float broadcast_var_48 = 0.000000e+00f;
            float32 guarded_31 = *(float32*)(conv_w_w + (((ci < 2) && (0 <= ci)) ? (((((int64_t)ci) * (int64_t)14336) + (((int64_t)kvh) * (int64_t)512)) + (int64_t)4576) : (int64_t)0));
            float32 y_10 = (y_9 + ((((ci < 2) && (0 <= ci)) ? guarded_31 : ((float32)(broadcast_var_48))) * prev_8));
            float broadcast_var_49 = 0.000000e+00f;
            float32 condval_32;
            if ((3 <= i_24)) {
              condval_32 = *(float32*)(((float*)cc) + ((i_24 * 32) - 96));
            } else {
              float32 guarded_32 = *(float32*)(conv_state_w + (((ci < 2) && (0 <= ci)) ? ((((((int64_t)ci) * (int64_t)12288) + (((int64_t)i_24) * (int64_t)2048)) + (((int64_t)kvh) * (int64_t)512)) + (int64_t)6624) : (int64_t)0));
              condval_32 = (((ci < 2) && (0 <= ci)) ? guarded_32 : ((float32)(broadcast_var_49)));
            }
            float32 prev_9 = condval_32;
            float broadcast_var_50 = 0.000000e+00f;
            float32 guarded_33 = *(float32*)(conv_w_w + (((ci < 2) && (0 <= ci)) ? (((((int64_t)ci) * (int64_t)14336) + (((int64_t)kvh) * (int64_t)512)) + (int64_t)6624) : (int64_t)0));
            float32 y_11 = (y_10 + ((((ci < 2) && (0 <= ci)) ? guarded_33 : ((float32)(broadcast_var_50))) * prev_9));
            float broadcast_var_51 = 0.000000e+00f;
            float32 condval_33;
            if ((2 <= i_24)) {
              condval_33 = *(float32*)(((float*)cc) + ((i_24 * 32) - 64));
            } else {
              float32 guarded_34 = *(float32*)(conv_state_w + (((ci < 2) && (0 <= ci)) ? ((((((int64_t)ci) * (int64_t)12288) + (((int64_t)i_24) * (int64_t)2048)) + (((int64_t)kvh) * (int64_t)512)) + (int64_t)8672) : (int64_t)0));
              condval_33 = (((ci < 2) && (0 <= ci)) ? guarded_34 : ((float32)(broadcast_var_51)));
            }
            float32 prev_10 = condval_33;
            float broadcast_var_52 = 0.000000e+00f;
            float32 guarded_35 = *(float32*)(conv_w_w + (((ci < 2) && (0 <= ci)) ? (((((int64_t)ci) * (int64_t)14336) + (((int64_t)kvh) * (int64_t)512)) + (int64_t)8672) : (int64_t)0));
            float32 y_12 = (y_11 + ((((ci < 2) && (0 <= ci)) ? guarded_35 : ((float32)(broadcast_var_52))) * prev_10));
            float broadcast_var_53 = 0.000000e+00f;
            float32 condval_34;
            if ((1 <= i_24)) {
              condval_34 = *(float32*)(((float*)cc) + ((i_24 * 32) - 32));
            } else {
              float32 guarded_36 = *(float32*)(conv_state_w + (((ci < 2) && (0 <= ci)) ? ((((((int64_t)ci) * (int64_t)12288) + (((int64_t)i_24) * (int64_t)2048)) + (((int64_t)kvh) * (int64_t)512)) + (int64_t)10720) : (int64_t)0));
              condval_34 = (((ci < 2) && (0 <= ci)) ? guarded_36 : ((float32)(broadcast_var_53)));
            }
            float32 prev_11 = condval_34;
            float broadcast_var_54 = 0.000000e+00f;
            float32 guarded_37 = *(float32*)(conv_w_w + (((ci < 2) && (0 <= ci)) ? (((((int64_t)ci) * (int64_t)14336) + (((int64_t)kvh) * (int64_t)512)) + (int64_t)10720) : (int64_t)0));
            float32 y_13 = (y_12 + ((((ci < 2) && (0 <= ci)) ? guarded_37 : ((float32)(broadcast_var_54))) * prev_11));
            *(float32*)(((float*)cc) + ((i_24 * 32) + 4096)) = y_13;
          }
          for (int32_t i_25 = 0; i_25 < 6; ++i_25) {
            float broadcast_var_55 = 0.000000e+00f;
            int32_t v__36 = n_w - (chunk_w * 128);
            float32 condval_35;
            if ((6 <= (((128) < (v__36) ? (128) : (v__36)) + i_25))) {
              int64_t v__37 = (int64_t)128;
              int64_t v__38 = ((int64_t)n_w) - (((int64_t)chunk_w) * (int64_t)128);
              condval_35 = *(float32*)(((float*)cc) + (((((v__37) < (v__38) ? (v__37) : (v__38)) * (int64_t)32) + (((int64_t)i_25) * (int64_t)32)) - (int64_t)192));
            } else {
              int64_t v__39 = (int64_t)128;
              int64_t v__40 = ((int64_t)n_w) - (((int64_t)chunk_w) * (int64_t)128);
              float32 guarded_38 = *(float32*)(conv_state_w + ((((0 <= (((128) < (v__36) ? (128) : (v__36)) + i_25)) && (0 <= ci)) && (ci < 2)) ? (((((((int64_t)ci) * (int64_t)12288) + (((v__39) < (v__40) ? (v__39) : (v__40)) * (int64_t)2048)) + (((int64_t)i_25) * (int64_t)2048)) + (((int64_t)kvh) * (int64_t)512)) + (int64_t)480) : (int64_t)0));
              condval_35 = ((((0 <= (((128) < (v__36) ? (128) : (v__36)) + i_25)) && (0 <= ci)) && (ci < 2)) ? guarded_38 : ((float32)(broadcast_var_55)));
            }
            *(float32*)(((float*)st) + (i_25 * 32)) = condval_35;
          }
          if ((ci < 2) && (0 <= ci)) {
            for (int32_t i_26 = 0; i_26 < 6; ++i_26) {
              *(float32*)(conv_state_w + ((((ci * 12288) + (i_26 * 2048)) + (kvh * 512)) + 480)) = *(float32*)(((float*)st) + (i_26 * 32));
            }
          }
          if (int32_t tl_status = tl_hexagon_dma_async_wait(tl_dma_async, 0)) return tl_status;
          tl_hexagon_hmx_gemm<true, true, true, false>(128, 32, 2048, (&(((float*)cc)[0])), 32, (&(a_w[0])), 65536, (&(((half*)v)[65536])), 65536, (bool)1, (&(((half*)stage_8)[0])), (&(((uint32_t*)config_8)[0])));
          for (int32_t i_27 = 0; i_27 < 128; ++i_27) {
            *(float32*)(((float*)cc) + (i_27 * 32)) = (*(float32*)(((float*)cc) + (i_27 * 32)) * *(float32*)(((float*)cc) + ((i_27 * 32) + 4096)));
          }
          tl_crouton_copy<true, false>((&(hid_w[0])), 4, 64, 0, ((kvh * 512) + 480), 128, 32, (&(((float*)cc)[0])), 32, 1);
        }
        if (int32_t tl_status = tl_grid_sync(kvh)) return tl_status;
        if (layer < 3) {
          if (int32_t tl_status = tl_hexagon_dma_async_submit(tl_dma_async, (&(((half*)v)[0])), (&(out_w_w[((layer * 4194304) + (kvh * 1048576))])), (uint32_t)131072, (uint32_t)131072, (uint32_t)131072, (uint32_t)1, 3)) return tl_status;
          if (int32_t tl_status = tl_hexagon_dma_async_commit(tl_dma_async)) return tl_status;
          if (int32_t tl_status = tl_hexagon_dma_async_submit(tl_dma_async, (&(((half*)v)[65536])), (&(out_w_w[(((layer * 4194304) + (kvh * 1048576)) + 65536)])), (uint32_t)131072, (uint32_t)131072, (uint32_t)131072, (uint32_t)1, 3)) return tl_status;
          if (int32_t tl_status = tl_hexagon_dma_async_commit(tl_dma_async)) return tl_status;
          for (int32_t b_2 = 0; b_2 < 14; ++b_2) {
            if (int32_t tl_status = tl_hexagon_dma_async_wait(tl_dma_async, 1)) return tl_status;
            tl_hexagon_hmx_gemm<true, true, true, false>(128, 32, 2048, (&(f_w[((kvh * 512) + (b_2 * 32))])), 2048, (&(hid_w[0])), 65536, (&(((half*)v)[((b_2 & 1) * 65536)])), 65536, (bool)1, (&(((half*)stage_9)[0])), (&(((uint32_t*)config_9)[0])));
            for (int32_t i_28 = 0; i_28 < 128; ++i_28) {
              *(float32*)(((float*)sq32) + (i_28 * 32)) = (*(float32*)(f_w + (((i_28 * 2048) + (kvh * 512)) + (b_2 * 32))) * *(float32*)(f_w + (((i_28 * 2048) + (kvh * 512)) + (b_2 * 32))));
            }
            tl_hvx_reduce_rows<TL_REDUCE_SUM, float, true>((&(((float*)hs32)[0])), (&(((float*)sq32)[0])), 128, 32, 32, 1, (bool)1);
            for (int32_t i_29 = 0; i_29 < 4; ++i_29) {
              *(float32*)(psum_w + ((kvh * 128) + (i_29 * 32))) = ((b_2 == 0) ? *(float32*)(((float*)hs32) + (i_29 * 32)) : (*(float32*)(psum_w + ((kvh * 128) + (i_29 * 32))) + *(float32*)(((float*)hs32) + (i_29 * 32))));
            }
            if (int32_t tl_status = tl_hexagon_dma_async_submit(tl_dma_async, (&(((half*)v)[((b_2 & 1) * 65536)])), (&(out_w_w[((((layer * 4194304) + (kvh * 1048576)) + (b_2 * 65536)) + 131072)])), (uint32_t)131072, (uint32_t)131072, (uint32_t)131072, (uint32_t)1, 3)) return tl_status;
            if (int32_t tl_status = tl_hexagon_dma_async_commit(tl_dma_async)) return tl_status;
          }
          if (int32_t tl_status = tl_hexagon_dma_async_wait(tl_dma_async, 1)) return tl_status;
          tl_hexagon_hmx_gemm<true, true, true, false>(128, 32, 2048, (&(f_w[((kvh * 512) + 448)])), 2048, (&(hid_w[0])), 65536, (&(((half*)v)[0])), 65536, (bool)1, (&(((half*)stage_10)[0])), (&(((uint32_t*)config_10)[0])));
          for (int32_t i_30 = 0; i_30 < 128; ++i_30) {
            *(float32*)(((float*)sq32) + (i_30 * 32)) = (*(float32*)(f_w + (((i_30 * 2048) + (kvh * 512)) + 448)) * *(float32*)(f_w + (((i_30 * 2048) + (kvh * 512)) + 448)));
          }
          tl_hvx_reduce_rows<TL_REDUCE_SUM, float, true>((&(((float*)hs32)[0])), (&(((float*)sq32)[0])), 128, 32, 32, 1, (bool)1);
          for (int32_t i_31 = 0; i_31 < 4; ++i_31) {
            *(float32*)(psum_w + ((kvh * 128) + (i_31 * 32))) = (*(float32*)(psum_w + ((kvh * 128) + (i_31 * 32))) + *(float32*)(((float*)hs32) + (i_31 * 32)));
          }
          if (int32_t tl_status = tl_hexagon_dma_async_submit(tl_dma_async, (&(((half*)v)[0])), (&(w13_w_w[((layer * 4194304) + (kvh * 1048576))])), (uint32_t)131072, (uint32_t)131072, (uint32_t)131072, (uint32_t)1, 3)) return tl_status;
          if (int32_t tl_status = tl_hexagon_dma_async_commit(tl_dma_async)) return tl_status;
          if (int32_t tl_status = tl_hexagon_dma_async_wait(tl_dma_async, 1)) return tl_status;
          tl_hexagon_hmx_gemm<true, true, true, false>(128, 32, 2048, (&(f_w[((kvh * 512) + 480)])), 2048, (&(hid_w[0])), 65536, (&(((half*)v)[65536])), 65536, (bool)1, (&(((half*)stage_11)[0])), (&(((uint32_t*)config_11)[0])));
          for (int32_t i_32 = 0; i_32 < 128; ++i_32) {
            *(float32*)(((float*)sq32) + (i_32 * 32)) = (*(float32*)(f_w + (((i_32 * 2048) + (kvh * 512)) + 480)) * *(float32*)(f_w + (((i_32 * 2048) + (kvh * 512)) + 480)));
          }
          tl_hvx_reduce_rows<TL_REDUCE_SUM, float, true>((&(((float*)hs32)[0])), (&(((float*)sq32)[0])), 128, 32, 32, 1, (bool)1);
          for (int32_t i_33 = 0; i_33 < 4; ++i_33) {
            *(float32*)(psum_w + ((kvh * 128) + (i_33 * 32))) = (*(float32*)(psum_w + ((kvh * 128) + (i_33 * 32))) + *(float32*)(((float*)hs32) + (i_33 * 32)));
          }
          if (int32_t tl_status = tl_hexagon_dma_async_submit(tl_dma_async, (&(((half*)v)[65536])), (&(w13_w_w[(((layer * 4194304) + (kvh * 1048576)) + 65536)])), (uint32_t)131072, (uint32_t)131072, (uint32_t)131072, (uint32_t)1, 3)) return tl_status;
          if (int32_t tl_status = tl_hexagon_dma_async_commit(tl_dma_async)) return tl_status;
          if (int32_t tl_status = tl_grid_sync(kvh)) return tl_status;
          for (int32_t i_34 = 0; i_34 < 64; ++i_34) {
            *(float32*)(((float*)pw) + (i_34 * 32)) = *(float32*)(post_op_norm_w + ((layer * 2048) + (i_34 * 32)));
          }
          float32 total = *(float32*)(psum_w + (kvh * 32));
          float32 total_1 = (total + *(float32*)(psum_w + ((kvh * 32) + 128)));
          float32 total_2 = (total_1 + *(float32*)(psum_w + ((kvh * 32) + 256)));
          float32 total_3 = (total_2 + *(float32*)(psum_w + ((kvh * 32) + 384)));
          float broadcast_var_56 = 4.882812e-04f;
          float broadcast_var_57 = 1.000000e-05f;
          *(float32*)(((float*)sw) + 0) = tl_hvx_rsqrt_fast(((total_3 * ((float32)(broadcast_var_56))) + ((float32)(broadcast_var_57))));
          for (int32_t i_35 = 0; i_35 < 64; ++i_35) {
            *(float32*)(((float*)nw) + (i_35 * 32)) = *(float32*)(ffn_norm_w + ((layer * 2048) + (i_35 * 32)));
          }
          if (int32_t tl_status = tl_hexagon_dma_async_submit(tl_dma_async, (&(((float*)xr)[0])), (&(x_w[(kvh * 65536)])), (uint32_t)32768, (uint32_t)32768, (uint32_t)32768, (uint32_t)1, 3)) return tl_status;
          if (int32_t tl_status = tl_hexagon_dma_async_commit(tl_dma_async)) return tl_status;
          for (int32_t pair = 0; pair < 4; ++pair) {
            if (int32_t tl_status = tl_hexagon_dma_async_wait(tl_dma_async, 0)) return tl_status;
            if (int32_t tl_status = tl_hexagon_dma_async_submit(tl_dma_async, (&(((float*)xr)[8192])), (&(x_w[(((kvh * 65536) + (pair * 16384)) + 8192)])), (uint32_t)32768, (uint32_t)32768, (uint32_t)32768, (uint32_t)1, 3)) return tl_status;
            if (int32_t tl_status = tl_hexagon_dma_async_commit(tl_dma_async)) return tl_status;
            for (int32_t i_strip_4 = 0; i_strip_4 < 1; ++i_strip_4) {
              float32 sw_lanes = tl_hvx_load_lanes<32>((&(((float*)sw)[(pair * 8)])), 4);
              for (int32_t i_lane_4 = 0; i_lane_4 < 4; ++i_lane_4) {
                float* xr_row_1 = ((float*)xr) + (i_lane_4 * 2048);
                float* f_row = ((float*)f_w) + (((kvh * 65536) + (pair * 16384)) + (i_lane_4 * 2048));
                float* sq_row = ((float*)sq) + (i_lane_4 * 2048);
                for (int32_t i_col = 0; i_col < 64; ++i_col) {
                  float32 v_1 = (*(float32*)(xr_row_1 + (i_col * 32)) + ((*(float32*)(f_row + (i_col * 32)) * tl_hvx_broadcast_lane<32, 1>(sw_lanes, i_lane_4)) * *(float32*)(((float*)pw) + (i_col * 32))));
                  *(float32*)(xr_row_1 + (i_col * 32)) = v_1;
                  *(float32*)(sq_row + (i_col * 32)) = (v_1 * v_1);
                }
              }
            }
            if (int32_t tl_status = tl_hexagon_dma_async_submit(tl_dma_async, (&(x_w[((kvh * 65536) + (pair * 16384))])), (&(((float*)xr)[0])), (uint32_t)32768, (uint32_t)32768, (uint32_t)32768, (uint32_t)1, 4)) return tl_status;
            if (int32_t tl_status = tl_hexagon_dma_async_commit(tl_dma_async)) return tl_status;
            tl_hvx_reduce_rows<TL_REDUCE_SUM, float, true>((&(((float*)ss)[0])), (&(((float*)sq)[0])), 4, 2048, 2048, 1, (bool)1);
            for (int32_t i_36 = 0; i_36 < 4; ++i_36) {
              ((float*)ss)[i_36] = tl_hvx_rsqrt_fast(((((float*)ss)[i_36] * 4.882812e-04f) + 1.000000e-05f));
            }
            for (int32_t j_2 = 0; j_2 < 2; ++j_2) {
              for (int32_t i_37 = 0; i_37 < 64; ++i_37) {
                *(half64*)(a_w + ((((kvh * 65536) + (i_37 * 1024)) + (pair * 256)) + (j_2 * 64))) = half64(__builtin_shufflevector((((half64)float64(__builtin_shufflevector((((*(float32*)(((float*)xr) + ((j_2 * 4096) + (i_37 * 32))) * ((float32)(((float*)ss)[(j_2 * 2)]))) * *(float32*)(((float*)nw) + (i_37 * 32)))).v, (((*(float32*)(((float*)xr) + (((j_2 * 4096) + (i_37 * 32)) + 2048)) * ((float32)(((float*)ss)[((j_2 * 2) + 1)]))) * *(float32*)(((float*)nw) + (i_37 * 32)))).v, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63)))).v, (half64{}).v, 0, 32, 1, 33, 2, 34, 3, 35, 4, 36, 5, 37, 6, 38, 7, 39, 8, 40, 9, 41, 10, 42, 11, 43, 12, 44, 13, 45, 14, 46, 15, 47, 16, 48, 17, 49, 18, 50, 19, 51, 20, 52, 21, 53, 22, 54, 23, 55, 24, 56, 25, 57, 26, 58, 27, 59, 28, 60, 29, 61, 30, 62, 31, 63));
              }
            }
            if (int32_t tl_status = tl_hexagon_dma_async_wait(tl_dma_async, 0)) return tl_status;
            if (pair < 3) {
              if (int32_t tl_status = tl_hexagon_dma_async_submit(tl_dma_async, (&(((float*)xr)[0])), (&(x_w[(((kvh * 65536) + (pair * 16384)) + 16384)])), (uint32_t)32768, (uint32_t)32768, (uint32_t)32768, (uint32_t)1, 3)) return tl_status;
              if (int32_t tl_status = tl_hexagon_dma_async_commit(tl_dma_async)) return tl_status;
            }
            for (int32_t i_strip_5 = 0; i_strip_5 < 1; ++i_strip_5) {
              float32 sw_lanes_1 = tl_hvx_load_lanes<32>((&(((float*)sw)[((pair * 8) + 4)])), 4);
              for (int32_t i_lane_5 = 0; i_lane_5 < 4; ++i_lane_5) {
                float* xr_row_2 = ((float*)xr) + ((i_lane_5 * 2048) + 8192);
                float* f_row_1 = ((float*)f_w) + ((((kvh * 65536) + (pair * 16384)) + (i_lane_5 * 2048)) + 8192);
                float* sq_row_1 = ((float*)sq) + (i_lane_5 * 2048);
                for (int32_t i_col_1 = 0; i_col_1 < 64; ++i_col_1) {
                  float32 v_2 = (*(float32*)(xr_row_2 + (i_col_1 * 32)) + ((*(float32*)(f_row_1 + (i_col_1 * 32)) * tl_hvx_broadcast_lane<32, 1>(sw_lanes_1, i_lane_5)) * *(float32*)(((float*)pw) + (i_col_1 * 32))));
                  *(float32*)(xr_row_2 + (i_col_1 * 32)) = v_2;
                  *(float32*)(sq_row_1 + (i_col_1 * 32)) = (v_2 * v_2);
                }
              }
            }
            if (int32_t tl_status = tl_hexagon_dma_async_submit(tl_dma_async, (&(x_w[(((kvh * 65536) + (pair * 16384)) + 8192)])), (&(((float*)xr)[8192])), (uint32_t)32768, (uint32_t)32768, (uint32_t)32768, (uint32_t)1, 4)) return tl_status;
            if (int32_t tl_status = tl_hexagon_dma_async_commit(tl_dma_async)) return tl_status;
            tl_hvx_reduce_rows<TL_REDUCE_SUM, float, true>((&(((float*)ss)[0])), (&(((float*)sq)[0])), 4, 2048, 2048, 1, (bool)1);
            for (int32_t i_38 = 0; i_38 < 4; ++i_38) {
              ((float*)ss)[i_38] = tl_hvx_rsqrt_fast(((((float*)ss)[i_38] * 4.882812e-04f) + 1.000000e-05f));
            }
            for (int32_t j_3 = 0; j_3 < 2; ++j_3) {
              for (int32_t i_39 = 0; i_39 < 64; ++i_39) {
                *(half64*)(a_w + (((((kvh * 65536) + (i_39 * 1024)) + (pair * 256)) + (j_3 * 64)) + 128)) = half64(__builtin_shufflevector((((half64)float64(__builtin_shufflevector((((*(float32*)(((float*)xr) + (((j_3 * 4096) + (i_39 * 32)) + 8192)) * ((float32)(((float*)ss)[(j_3 * 2)]))) * *(float32*)(((float*)nw) + (i_39 * 32)))).v, (((*(float32*)(((float*)xr) + (((j_3 * 4096) + (i_39 * 32)) + 10240)) * ((float32)(((float*)ss)[((j_3 * 2) + 1)]))) * *(float32*)(((float*)nw) + (i_39 * 32)))).v, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63)))).v, (half64{}).v, 0, 32, 1, 33, 2, 34, 3, 35, 4, 36, 5, 37, 6, 38, 7, 39, 8, 40, 9, 41, 10, 42, 11, 43, 12, 44, 13, 45, 14, 46, 15, 47, 16, 48, 17, 49, 18, 50, 19, 51, 20, 52, 21, 53, 22, 54, 23, 55, 24, 56, 25, 57, 26, 58, 27, 59, 28, 60, 29, 61, 30, 62, 31, 63));
              }
            }
          }
          if (int32_t tl_status = tl_hexagon_dma_async_wait(tl_dma_async, 0)) return tl_status;
          if (int32_t tl_status = tl_grid_sync(kvh)) return tl_status;
          for (int32_t b_3 = 0; b_3 < 14; ++b_3) {
            if (int32_t tl_status = tl_hexagon_dma_async_wait(tl_dma_async, 1)) return tl_status;
            tl_hexagon_hmx_gemm<true, true, true, false>(128, 32, 2048, (&(((float*)cc)[((b_3 & 1) * 4096)])), 32, (&(a_w[0])), 65536, (&(((half*)v)[((b_3 & 1) * 65536)])), 65536, (bool)1, (&(((half*)stage_12)[0])), (&(((uint32_t*)config_12)[0])));
            if ((b_3 & 1) == 1) {
              for (int32_t i_40 = 0; i_40 < 64; ++i_40) {
                float cc_local_cast_15[64] __attribute__((aligned(128)));
                float cc_local_cast_16[64] __attribute__((aligned(128)));
                half gl_local_cast_14[64] __attribute__((aligned(128)));
                float* cc_row = ((float*)cc) + (i_40 * 64);
                for (int32_t vec_copy_21 = 0; vec_copy_21 < 2; ++vec_copy_21) {
                  *(float32*)(cc_local_cast_15 + (vec_copy_21 * 32)) = *(float32*)(cc_row + (vec_copy_21 * 32));
                }
                float* cc_row_1 = ((float*)cc) + ((i_40 * 64) + 4096);
                for (int32_t vec_copy_22 = 0; vec_copy_22 < 2; ++vec_copy_22) {
                  *(float32*)(cc_local_cast_16 + (vec_copy_22 * 32)) = *(float32*)(cc_row_1 + (vec_copy_22 * 32));
                }
                *(half64*)(gl_local_cast_14 + 0) = ((((half64)*(float64*)(cc_local_cast_15 + 0)) * tl_hvx_sigmoid_fast(((half64)*(float64*)(cc_local_cast_15 + 0)))) * ((half64)*(float64*)(cc_local_cast_16 + 0)));
                *(half64*)(((half*)gl) + (i_40 * 64)) = *(half64*)(gl_local_cast_14 + 0);
              }
              tl_crouton_copy<true, false>((&(hid_w[0])), 4, 64, 0, ((kvh * 256) + ((b_3 >> 1) * 32)), 128, 32, (&(((half*)gl)[0])), 32, 1);
            }
            if (int32_t tl_status = tl_hexagon_dma_async_submit(tl_dma_async, (&(((half*)v)[((b_3 & 1) * 65536)])), (&(w13_w_w[((((layer * 4194304) + (kvh * 1048576)) + (b_3 * 65536)) + 131072)])), (uint32_t)131072, (uint32_t)131072, (uint32_t)131072, (uint32_t)1, 3)) return tl_status;
            if (int32_t tl_status = tl_hexagon_dma_async_commit(tl_dma_async)) return tl_status;
          }
          if (int32_t tl_status = tl_hexagon_dma_async_wait(tl_dma_async, 1)) return tl_status;
          tl_hexagon_hmx_gemm<true, true, true, false>(128, 32, 2048, (&(((float*)cc)[0])), 32, (&(a_w[0])), 65536, (&(((half*)v)[0])), 65536, (bool)1, (&(((half*)stage_13)[0])), (&(((uint32_t*)config_13)[0])));
          if (int32_t tl_status = tl_hexagon_dma_async_submit(tl_dma_async, (&(((half*)v)[0])), (&(w2_w_w[((layer * 2097152) + (kvh * 524288))])), (uint32_t)65536, (uint32_t)65536, (uint32_t)65536, (uint32_t)1, 3)) return tl_status;
          if (int32_t tl_status = tl_hexagon_dma_async_commit(tl_dma_async)) return tl_status;
          if (int32_t tl_status = tl_hexagon_dma_async_wait(tl_dma_async, 1)) return tl_status;
          tl_hexagon_hmx_gemm<true, true, true, false>(128, 32, 2048, (&(((float*)cc)[4096])), 32, (&(a_w[0])), 65536, (&(((half*)v)[65536])), 65536, (bool)1, (&(((half*)stage_14)[0])), (&(((uint32_t*)config_14)[0])));
          for (int32_t i_41 = 0; i_41 < 64; ++i_41) {
            float cc_local_cast_18[64] __attribute__((aligned(128)));
            float cc_local_cast_19[64] __attribute__((aligned(128)));
            half gl_local_cast_17[64] __attribute__((aligned(128)));
            float* cc_row_2 = ((float*)cc) + (i_41 * 64);
            for (int32_t vec_copy_23 = 0; vec_copy_23 < 2; ++vec_copy_23) {
              *(float32*)(cc_local_cast_18 + (vec_copy_23 * 32)) = *(float32*)(cc_row_2 + (vec_copy_23 * 32));
            }
            float* cc_row_3 = ((float*)cc) + ((i_41 * 64) + 4096);
            for (int32_t vec_copy_24 = 0; vec_copy_24 < 2; ++vec_copy_24) {
              *(float32*)(cc_local_cast_19 + (vec_copy_24 * 32)) = *(float32*)(cc_row_3 + (vec_copy_24 * 32));
            }
            *(half64*)(gl_local_cast_17 + 0) = ((((half64)*(float64*)(cc_local_cast_18 + 0)) * tl_hvx_sigmoid_fast(((half64)*(float64*)(cc_local_cast_18 + 0)))) * ((half64)*(float64*)(cc_local_cast_19 + 0)));
            *(half64*)(((half*)gl) + (i_41 * 64)) = *(half64*)(gl_local_cast_17 + 0);
          }
          tl_crouton_copy<true, false>((&(hid_w[0])), 4, 64, 0, ((kvh * 256) + 224), 128, 32, (&(((half*)gl)[0])), 32, 1);
          if (int32_t tl_status = tl_hexagon_dma_async_submit(tl_dma_async, (&(((half*)v)[65536])), (&(w2_w_w[(((layer * 2097152) + (kvh * 524288)) + 32768)])), (uint32_t)65536, (uint32_t)65536, (uint32_t)65536, (uint32_t)1, 3)) return tl_status;
          if (int32_t tl_status = tl_hexagon_dma_async_commit(tl_dma_async)) return tl_status;
          if (int32_t tl_status = tl_grid_sync(kvh)) return tl_status;
          for (int32_t b_4 = 0; b_4 < 14; ++b_4) {
            if (int32_t tl_status = tl_hexagon_dma_async_wait(tl_dma_async, 1)) return tl_status;
            tl_hexagon_hmx_gemm<true, true, true, false>(128, 32, 1024, (&(f_w[((kvh * 512) + (b_4 * 32))])), 2048, (&(hid_w[0])), 65536, (&(((half*)v)[((b_4 & 1) * 65536)])), 65536, (bool)1, (&(((half*)stage_15)[0])), (&(((uint32_t*)config_15)[0])));
            for (int32_t i_42 = 0; i_42 < 128; ++i_42) {
              *(float32*)(((float*)sq32) + (i_42 * 32)) = (*(float32*)(f_w + (((i_42 * 2048) + (kvh * 512)) + (b_4 * 32))) * *(float32*)(f_w + (((i_42 * 2048) + (kvh * 512)) + (b_4 * 32))));
            }
            tl_hvx_reduce_rows<TL_REDUCE_SUM, float, true>((&(((float*)hs32)[0])), (&(((float*)sq32)[0])), 128, 32, 32, 1, (bool)1);
            for (int32_t i_43 = 0; i_43 < 4; ++i_43) {
              *(float32*)(psum_w + ((kvh * 128) + (i_43 * 32))) = ((b_4 == 0) ? *(float32*)(((float*)hs32) + (i_43 * 32)) : (*(float32*)(psum_w + ((kvh * 128) + (i_43 * 32))) + *(float32*)(((float*)hs32) + (i_43 * 32))));
            }
            if (int32_t tl_status = tl_hexagon_dma_async_submit(tl_dma_async, (&(((half*)v)[((b_4 & 1) * 65536)])), (&(w2_w_w[((((layer * 2097152) + (kvh * 524288)) + (b_4 * 32768)) + 65536)])), (uint32_t)65536, (uint32_t)65536, (uint32_t)65536, (uint32_t)1, 3)) return tl_status;
            if (int32_t tl_status = tl_hexagon_dma_async_commit(tl_dma_async)) return tl_status;
          }
          if (int32_t tl_status = tl_hexagon_dma_async_wait(tl_dma_async, 1)) return tl_status;
          tl_hexagon_hmx_gemm<true, true, true, false>(128, 32, 1024, (&(f_w[((kvh * 512) + 448)])), 2048, (&(hid_w[0])), 65536, (&(((half*)v)[0])), 65536, (bool)1, (&(((half*)stage_16)[0])), (&(((uint32_t*)config_16)[0])));
          for (int32_t i_44 = 0; i_44 < 128; ++i_44) {
            *(float32*)(((float*)sq32) + (i_44 * 32)) = (*(float32*)(f_w + (((i_44 * 2048) + (kvh * 512)) + 448)) * *(float32*)(f_w + (((i_44 * 2048) + (kvh * 512)) + 448)));
          }
          tl_hvx_reduce_rows<TL_REDUCE_SUM, float, true>((&(((float*)hs32)[0])), (&(((float*)sq32)[0])), 128, 32, 32, 1, (bool)1);
          for (int32_t i_45 = 0; i_45 < 4; ++i_45) {
            *(float32*)(psum_w + ((kvh * 128) + (i_45 * 32))) = (*(float32*)(psum_w + ((kvh * 128) + (i_45 * 32))) + *(float32*)(((float*)hs32) + (i_45 * 32)));
          }
          if (int32_t tl_status = tl_hexagon_dma_async_wait(tl_dma_async, 0)) return tl_status;
          tl_hexagon_hmx_gemm<true, true, true, false>(128, 32, 1024, (&(f_w[((kvh * 512) + 480)])), 2048, (&(hid_w[0])), 65536, (&(((half*)v)[65536])), 65536, (bool)1, (&(((half*)stage_17)[0])), (&(((uint32_t*)config_17)[0])));
          for (int32_t i_46 = 0; i_46 < 128; ++i_46) {
            *(float32*)(((float*)sq32) + (i_46 * 32)) = (*(float32*)(f_w + (((i_46 * 2048) + (kvh * 512)) + 480)) * *(float32*)(f_w + (((i_46 * 2048) + (kvh * 512)) + 480)));
          }
          tl_hvx_reduce_rows<TL_REDUCE_SUM, float, true>((&(((float*)hs32)[0])), (&(((float*)sq32)[0])), 128, 32, 32, 1, (bool)1);
          for (int32_t i_47 = 0; i_47 < 4; ++i_47) {
            *(float32*)(psum_w + ((kvh * 128) + (i_47 * 32))) = (*(float32*)(psum_w + ((kvh * 128) + (i_47 * 32))) + *(float32*)(((float*)hs32) + (i_47 * 32)));
          }
          if (int32_t tl_status = tl_grid_sync(kvh)) return tl_status;
          for (int32_t i_48 = 0; i_48 < 64; ++i_48) {
            *(float32*)(((float*)pw) + (i_48 * 32)) = *(float32*)(post_ffn_norm_w + ((layer * 2048) + (i_48 * 32)));
          }
          float32 total_4 = *(float32*)(psum_w + (kvh * 32));
          float32 total_5 = (total_4 + *(float32*)(psum_w + ((kvh * 32) + 128)));
          float32 total_6 = (total_5 + *(float32*)(psum_w + ((kvh * 32) + 256)));
          float32 total_7 = (total_6 + *(float32*)(psum_w + ((kvh * 32) + 384)));
          float broadcast_var_58 = 4.882812e-04f;
          float broadcast_var_59 = 1.000000e-05f;
          *(float32*)(((float*)sw) + 0) = tl_hvx_rsqrt_fast(((total_7 * ((float32)(broadcast_var_58))) + ((float32)(broadcast_var_59))));
          for (int32_t i_49 = 0; i_49 < 64; ++i_49) {
            *(float32*)(((float*)nw) + (i_49 * 32)) = *(float32*)(op_norm_w + (((layer * 2048) + (i_49 * 32)) + 2048));
          }
          if (int32_t tl_status = tl_hexagon_dma_async_submit(tl_dma_async, (&(((float*)xr)[0])), (&(x_w[(kvh * 65536)])), (uint32_t)32768, (uint32_t)32768, (uint32_t)32768, (uint32_t)1, 3)) return tl_status;
          if (int32_t tl_status = tl_hexagon_dma_async_commit(tl_dma_async)) return tl_status;
          for (int32_t pair_1 = 0; pair_1 < 4; ++pair_1) {
            if (int32_t tl_status = tl_hexagon_dma_async_wait(tl_dma_async, 0)) return tl_status;
            if (int32_t tl_status = tl_hexagon_dma_async_submit(tl_dma_async, (&(((float*)xr)[8192])), (&(x_w[(((kvh * 65536) + (pair_1 * 16384)) + 8192)])), (uint32_t)32768, (uint32_t)32768, (uint32_t)32768, (uint32_t)1, 3)) return tl_status;
            if (int32_t tl_status = tl_hexagon_dma_async_commit(tl_dma_async)) return tl_status;
            for (int32_t i_strip_6 = 0; i_strip_6 < 1; ++i_strip_6) {
              float32 sw_lanes_2 = tl_hvx_load_lanes<32>((&(((float*)sw)[(pair_1 * 8)])), 4);
              for (int32_t i_lane_6 = 0; i_lane_6 < 4; ++i_lane_6) {
                float* xr_row_3 = ((float*)xr) + (i_lane_6 * 2048);
                float* f_row_2 = ((float*)f_w) + (((kvh * 65536) + (pair_1 * 16384)) + (i_lane_6 * 2048));
                float* sq_row_2 = ((float*)sq) + (i_lane_6 * 2048);
                for (int32_t i_col_2 = 0; i_col_2 < 64; ++i_col_2) {
                  float32 v_3 = (*(float32*)(xr_row_3 + (i_col_2 * 32)) + ((*(float32*)(f_row_2 + (i_col_2 * 32)) * tl_hvx_broadcast_lane<32, 1>(sw_lanes_2, i_lane_6)) * *(float32*)(((float*)pw) + (i_col_2 * 32))));
                  *(float32*)(xr_row_3 + (i_col_2 * 32)) = v_3;
                  *(float32*)(sq_row_2 + (i_col_2 * 32)) = (v_3 * v_3);
                }
              }
            }
            if (int32_t tl_status = tl_hexagon_dma_async_submit(tl_dma_async, (&(x_w[((kvh * 65536) + (pair_1 * 16384))])), (&(((float*)xr)[0])), (uint32_t)32768, (uint32_t)32768, (uint32_t)32768, (uint32_t)1, 4)) return tl_status;
            if (int32_t tl_status = tl_hexagon_dma_async_commit(tl_dma_async)) return tl_status;
            tl_hvx_reduce_rows<TL_REDUCE_SUM, float, true>((&(((float*)ss)[0])), (&(((float*)sq)[0])), 4, 2048, 2048, 1, (bool)1);
            for (int32_t i_50 = 0; i_50 < 4; ++i_50) {
              ((float*)ss)[i_50] = tl_hvx_rsqrt_fast(((((float*)ss)[i_50] * 4.882812e-04f) + 1.000000e-05f));
            }
            for (int32_t j_4 = 0; j_4 < 2; ++j_4) {
              for (int32_t i_51 = 0; i_51 < 64; ++i_51) {
                *(half64*)(a_w + ((((kvh * 65536) + (i_51 * 1024)) + (pair_1 * 256)) + (j_4 * 64))) = half64(__builtin_shufflevector((((half64)float64(__builtin_shufflevector((((*(float32*)(((float*)xr) + ((j_4 * 4096) + (i_51 * 32))) * ((float32)(((float*)ss)[(j_4 * 2)]))) * *(float32*)(((float*)nw) + (i_51 * 32)))).v, (((*(float32*)(((float*)xr) + (((j_4 * 4096) + (i_51 * 32)) + 2048)) * ((float32)(((float*)ss)[((j_4 * 2) + 1)]))) * *(float32*)(((float*)nw) + (i_51 * 32)))).v, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63)))).v, (half64{}).v, 0, 32, 1, 33, 2, 34, 3, 35, 4, 36, 5, 37, 6, 38, 7, 39, 8, 40, 9, 41, 10, 42, 11, 43, 12, 44, 13, 45, 14, 46, 15, 47, 16, 48, 17, 49, 18, 50, 19, 51, 20, 52, 21, 53, 22, 54, 23, 55, 24, 56, 25, 57, 26, 58, 27, 59, 28, 60, 29, 61, 30, 62, 31, 63));
              }
            }
            if (int32_t tl_status = tl_hexagon_dma_async_wait(tl_dma_async, 0)) return tl_status;
            if (pair_1 < 3) {
              if (int32_t tl_status = tl_hexagon_dma_async_submit(tl_dma_async, (&(((float*)xr)[0])), (&(x_w[(((kvh * 65536) + (pair_1 * 16384)) + 16384)])), (uint32_t)32768, (uint32_t)32768, (uint32_t)32768, (uint32_t)1, 3)) return tl_status;
              if (int32_t tl_status = tl_hexagon_dma_async_commit(tl_dma_async)) return tl_status;
            }
            for (int32_t i_strip_7 = 0; i_strip_7 < 1; ++i_strip_7) {
              float32 sw_lanes_3 = tl_hvx_load_lanes<32>((&(((float*)sw)[((pair_1 * 8) + 4)])), 4);
              for (int32_t i_lane_7 = 0; i_lane_7 < 4; ++i_lane_7) {
                float* xr_row_4 = ((float*)xr) + ((i_lane_7 * 2048) + 8192);
                float* f_row_3 = ((float*)f_w) + ((((kvh * 65536) + (pair_1 * 16384)) + (i_lane_7 * 2048)) + 8192);
                float* sq_row_3 = ((float*)sq) + (i_lane_7 * 2048);
                for (int32_t i_col_3 = 0; i_col_3 < 64; ++i_col_3) {
                  float32 v_4 = (*(float32*)(xr_row_4 + (i_col_3 * 32)) + ((*(float32*)(f_row_3 + (i_col_3 * 32)) * tl_hvx_broadcast_lane<32, 1>(sw_lanes_3, i_lane_7)) * *(float32*)(((float*)pw) + (i_col_3 * 32))));
                  *(float32*)(xr_row_4 + (i_col_3 * 32)) = v_4;
                  *(float32*)(sq_row_3 + (i_col_3 * 32)) = (v_4 * v_4);
                }
              }
            }
            if (int32_t tl_status = tl_hexagon_dma_async_submit(tl_dma_async, (&(x_w[(((kvh * 65536) + (pair_1 * 16384)) + 8192)])), (&(((float*)xr)[8192])), (uint32_t)32768, (uint32_t)32768, (uint32_t)32768, (uint32_t)1, 4)) return tl_status;
            if (int32_t tl_status = tl_hexagon_dma_async_commit(tl_dma_async)) return tl_status;
            tl_hvx_reduce_rows<TL_REDUCE_SUM, float, true>((&(((float*)ss)[0])), (&(((float*)sq)[0])), 4, 2048, 2048, 1, (bool)1);
            for (int32_t i_52 = 0; i_52 < 4; ++i_52) {
              ((float*)ss)[i_52] = tl_hvx_rsqrt_fast(((((float*)ss)[i_52] * 4.882812e-04f) + 1.000000e-05f));
            }
            for (int32_t j_5 = 0; j_5 < 2; ++j_5) {
              for (int32_t i_53 = 0; i_53 < 64; ++i_53) {
                *(half64*)(a_w + (((((kvh * 65536) + (i_53 * 1024)) + (pair_1 * 256)) + (j_5 * 64)) + 128)) = half64(__builtin_shufflevector((((half64)float64(__builtin_shufflevector((((*(float32*)(((float*)xr) + (((j_5 * 4096) + (i_53 * 32)) + 8192)) * ((float32)(((float*)ss)[(j_5 * 2)]))) * *(float32*)(((float*)nw) + (i_53 * 32)))).v, (((*(float32*)(((float*)xr) + (((j_5 * 4096) + (i_53 * 32)) + 10240)) * ((float32)(((float*)ss)[((j_5 * 2) + 1)]))) * *(float32*)(((float*)nw) + (i_53 * 32)))).v, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63)))).v, (half64{}).v, 0, 32, 1, 33, 2, 34, 3, 35, 4, 36, 5, 37, 6, 38, 7, 39, 8, 40, 9, 41, 10, 42, 11, 43, 12, 44, 13, 45, 14, 46, 15, 47, 16, 48, 17, 49, 18, 50, 19, 51, 20, 52, 21, 53, 22, 54, 23, 55, 24, 56, 25, 57, 26, 58, 27, 59, 28, 60, 29, 61, 30, 62, 31, 63));
              }
            }
          }
          if (int32_t tl_status = tl_hexagon_dma_async_wait(tl_dma_async, 0)) return tl_status;
          if (int32_t tl_status = tl_grid_sync(kvh)) return tl_status;
        }
      }
      if (int32_t tl_status = tl_hexagon_dma_async_wait(tl_dma_async, 0)) return tl_status;
      return TL_OK;
    });
    if (tl_status)
      return tl_status;
  }
  return TL_OK;
}
