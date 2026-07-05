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
  TL_OP_NONE   = 0,
  TL_OP_MATMUL = 1,
};

// The host resources an embedded op RIDES (never acquires) plus the op's tensors.
// Preconditions the host guarantees for a registered op it dispatches:
//   * `vtcm_base`/`vtcm_size` is a valid VTCM region the op owns for its duration;
//   * `vtcm_rctx` is a compute-res handle that has HMX (lock it, do not acquire).
struct tl_op_ctx {
  void *   vtcm_base;
  unsigned vtcm_size;
  uint32_t vtcm_rctx;

  int      kind;         // enum tl_op_kind
  int      weight_type;  // host dtype id (opaque here; the op's matches() knows it)
  int      has_bias;     // a fused bias/residual is present -> most ops decline

  // matmul tensors:  dst[m,n] f32 = act[m,k] f32 @ dequant(weight)[n,k]^T
  float *         dst;
  const float *   act;
  const uint8_t * weight;
  int m, k, n, act_stride, dst_stride;
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
