#pragma once
// ---------------------------------------------------------------------------
// tilelang embedding API — run a tilelang kernel as ONE op inside a host
// runtime's DSP skel (e.g. llama.cpp ggml-hexagon), sharing the host's VTCM +
// HMX rather than opening a standalone FastRPC session.  See
// docs/llama_cpp_integration.md.
//
// This header is intentionally C-safe and free of <hmx.h>: the host's stock op
// function (often C) includes it to build a `tl_op_ctx` and call `tl_dispatch`.
// The kernel implementations that actually touch HMX include <tl_bridge.h>.
// ---------------------------------------------------------------------------
#include <stdint.h>

// Kind of op being offered to the registry (extend as kernels are added).
enum tl_op_kind {
  TL_OP_NONE       = 0,
  TL_OP_MATMUL     = 1,
  TL_OP_QGEMV_DOT  = 2,
};

// Minimal ordered DMA channel borrowed from an embedding runtime.  The queue,
// descriptor format, engine acquisition, and cache policy remain private to the
// runtime.  TileLang ops may only enqueue a contiguous transfer and pop the
// oldest completion; this is sufficient to express double-buffered tile flow.
struct tl_dma_channel {
  void *queue;
  int (*push_1d)(void *queue, void *dst, const void *src, uint32_t bytes);
  int (*push_2d)(void *queue, void *dst, const void *src,
                 uint32_t dst_stride, uint32_t src_stride,
                 uint32_t width, uint32_t height);
  void *(*pop)(void *queue);
};

// Borrowed synchronous parallel dispatch. The callback ABI deliberately
// matches llama.cpp's worker pool: one invocation per worker index, and run()
// returns only after every callback has completed. TileLang never creates or
// owns threads in an embedded op.
typedef void (*tl_parallel_task)(unsigned int workers, unsigned int worker,
                                 void *data);
struct tl_parallel_channel {
  void *pool;
  int (*run)(void *pool, tl_parallel_task task, void *data,
             unsigned int workers);
  unsigned int max_workers;
};

// The host resources an embedded op RIDES (never acquires) plus the op's tensors.
// Preconditions the host guarantees for a registered op it dispatches:
//   * `vtcm_base`/`vtcm_size` is a valid VTCM region the op owns for its duration;
//   * `vtcm_rctx` is a compute-res handle that has HMX (lock it, do not acquire).
struct tl_op_ctx {
  void *   vtcm_base;
  unsigned vtcm_size;
  uint32_t vtcm_rctx;
  struct tl_dma_channel dma;  // caller-owned; embedded ops never acquire DMA
  struct tl_parallel_channel parallel;  // caller-owned; run() is synchronous

  int      kind;         // enum tl_op_kind
  int      weight_type;  // host dtype id (opaque here; the op's matches() knows it)
  int      has_bias;     // a fused bias/residual is present -> most ops decline

  // matmul tensors:  dst[m,n] f32 = act[m,k] f32 @ dequant(weight)[n,k]^T
  float *         dst;
  const float *   act;
  const uint8_t * weight;
  int m, k, n, act_stride, dst_stride;

  // Q8_0 decode instruction atom.  The host has already DMA-staged one
  // [32,K] weight tile and quantized the activation into the matching
  // 1152-byte-per-K-block VTCM layout.  This boundary intentionally rides the
  // host's streaming pipeline rather than duplicating its DMA/worker policy.
  const uint8_t * tiled_act;
  const float *   bias;
  int             valid_rows;
};

// A registered tilelang op = a predicate + a runner.  Adding a kernel is: write
// these two and register the descriptor (no edits to host stock functions).
struct tl_op_desc {
  const char *name;
  int (*matches)(const struct tl_op_ctx *);  // is this op mine to run?
  int (*run)(const struct tl_op_ctx *);        // 0 = handled, -1 = decline (fall back)
};

#ifdef __cplusplus
extern "C" {
#endif

extern int tl_mm_enabled;                        // master A/B toggle (0 = all stock)
void       tl_register_op(const struct tl_op_desc *d);
int        tl_dispatch(const struct tl_op_ctx *octx);  // 0 if some op handled it

#ifdef __cplusplus
}
#endif
