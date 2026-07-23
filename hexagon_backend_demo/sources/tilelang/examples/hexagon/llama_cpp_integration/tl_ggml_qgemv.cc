// TileLang Q8_0 decode atom embedded in ggml-hexagon.
//
// ggml keeps ownership of the six-worker loop, activation quantization, 2D DMA
// queue, and VTCM buffers.  The dispatch seam is after dma_queue_pop(), where a
// complete [32,K] staged weight tile and matching activation tile are ready.
// TileLang owns only the signed-int8 HVX dot instruction atom.

#include <tl_templates/hexagon/qgemv.h>
#include <tl_templates/hexagon/tl_embed.h>

#include "htp-ops.h"

#include <stdint.h>

// These bodies are emitted by emit_qgemv_q8_0.py.  They are ordinary TileLang
// Hexagon kernels (no FastRPC wrapper) and call the qgemv instruction atom above.
#include "kernel_qgemv_q8_0_k2048.c"
#include "kernel_qgemv_q8_0_k2048_nobias.c"
#include "kernel_qgemv_q8_0_k8192.c"
#include "kernel_qgemv_q8_0_k8192_nobias.c"

#ifndef TL_QGEMV_ENABLED
#define TL_QGEMV_ENABLED 1
#endif

// Keep the existing embedding ABI name so this file can replace the older q4
// proof-of-concept TU without changing tl_embed.h or the host patch.
int tl_mm_enabled = TL_QGEMV_ENABLED;

#define TL_MAX_OPS 8
static const struct tl_op_desc *g_tl_ops[TL_MAX_OPS];
static int g_tl_nops = 0;

extern "C" void tl_register_op(const struct tl_op_desc *desc) {
  if (g_tl_nops < TL_MAX_OPS) {
    g_tl_ops[g_tl_nops++] = desc;
  }
}

extern "C" int tl_dispatch(const struct tl_op_ctx *octx) {
  if (!tl_mm_enabled) {
    return -1;
  }
  for (int i = 0; i < g_tl_nops; ++i) {
    if (g_tl_ops[i]->matches(octx) && g_tl_ops[i]->run(octx) == 0) {
      return 0;
    }
  }
  return -1;
}

static int tl_qgemv_matches(const struct tl_op_ctx *o) {
  return o->kind == TL_OP_QGEMV_DOT && o->weight_type == HTP_TYPE_Q8_0 &&
         o->dst != nullptr && o->weight != nullptr && o->tiled_act != nullptr &&
         (o->k == 2048 || o->k == 8192) && o->valid_rows == 32;
}

static int tl_qgemv_run(const struct tl_op_ctx *o) {
  int rc;
  if (o->k == 2048) {
    if (o->bias != nullptr) {
      rc = qgemv_q8_0_k2048_kernel(
          const_cast<uint8_t *>(o->weight),
          const_cast<uint8_t *>(o->tiled_act), const_cast<float *>(o->bias),
          o->dst);
    } else {
      rc = qgemv_q8_0_k2048_nobias_kernel(
          const_cast<uint8_t *>(o->weight),
          const_cast<uint8_t *>(o->tiled_act), o->dst);
    }
  } else if (o->k == 8192) {
    if (o->bias != nullptr) {
      rc = qgemv_q8_0_k8192_kernel(
          const_cast<uint8_t *>(o->weight),
          const_cast<uint8_t *>(o->tiled_act), const_cast<float *>(o->bias),
          o->dst);
    } else {
      rc = qgemv_q8_0_k8192_nobias_kernel(
          const_cast<uint8_t *>(o->weight),
          const_cast<uint8_t *>(o->tiled_act), o->dst);
    }
  } else {
    return -1;
  }
  return rc == 0 ? 0 : -1;
}

static const struct tl_op_desc tl_qgemv_desc = {
    "q8_0_dot_32x1", tl_qgemv_matches, tl_qgemv_run};

__attribute__((constructor)) static void tl_qgemv_register(void) {
  tl_register_op(&tl_qgemv_desc);
}
