// Minimal, real llama.cpp adapter for the walkthrough's single generated shape.
//
// This file is compiled into the existing ggml Hexagon DSP skel. llama.cpp owns
// VTCM, DMA, workers, and the HMX compute resource; generated TileLang stages
// receive borrowed buffers/callbacks through tl_op_ctx.
#include <tl_templates/hexagon/tl_embed.h>

#include <HAP_compute_res.h>
#include "htp-ops.h"

#include <stddef.h>
#include <stdint.h>

extern "C" int32_t tl_walkthrough_q4_m32_n256_k2048_pack_kernel(
    float *, void *, uint32_t *);
extern "C" int32_t tl_walkthrough_q4_m32_n256_k2048_dequant_kernel(
    uint8_t *, void *, int32_t, int32_t);
extern "C" int32_t tl_walkthrough_q4_m32_n256_k2048_compute_kernel(
    void *, void *, uint32_t *, void *, float *, int32_t);

#ifndef TL_Q4_HMX_ENABLED
#define TL_Q4_HMX_ENABLED 0
#endif

int tl_mm_enabled = TL_Q4_HMX_ENABLED;

// One shared registry TU is sufficient for this minimal example.
#define TL_MAX_OPS 8
static const struct tl_op_desc *g_ops[TL_MAX_OPS];
static int g_num_ops;

extern "C" void tl_register_op(const struct tl_op_desc *desc) {
  if (g_num_ops < TL_MAX_OPS) {
    g_ops[g_num_ops++] = desc;
  }
}

extern "C" int tl_dispatch(const struct tl_op_ctx *octx) {
  if (!tl_mm_enabled) {
    return -1;
  }
  for (int i = 0; i < g_num_ops; ++i) {
    if (g_ops[i]->matches(octx) && g_ops[i]->run(octx) == 0) {
      return 0;
    }
  }
  return -1;
}

static uintptr_t align_up(uintptr_t value, uintptr_t alignment) {
  return (value + alignment - 1u) & ~(alignment - 1u);
}

static uintptr_t align_down(uintptr_t value, uintptr_t alignment) {
  return value & ~(alignment - 1u);
}

#define MAX_WORKERS 16
struct dequant_work {
  uint8_t *weight;
  void *weight_hmx;
  int tiles;
  int rc[MAX_WORKERS];
};

static void dequant_worker(
    unsigned int workers, unsigned int worker, void *opaque) {
  struct dequant_work *work = (struct dequant_work *)opaque;
  const int begin = (int)(((uint64_t)work->tiles * worker) / workers);
  const int end = (int)(((uint64_t)work->tiles * (worker + 1u)) / workers);
  work->rc[worker] =
      tl_walkthrough_q4_m32_n256_k2048_dequant_kernel(
          work->weight, work->weight_hmx, begin, end);
}

static int matches(const struct tl_op_ctx *o) {
  return o->kind == TL_OP_MATMUL && o->weight_type == HTP_TYPE_Q4_0 &&
         !o->has_bias && o->m == 32 && o->n > 0 && (o->n % 256) == 0 &&
         o->k == 2048 && o->act_stride == o->k && o->dst_stride >= o->n &&
         o->dst && o->act && o->weight && o->vtcm_base && o->vtcm_rctx &&
         o->dma.queue && o->dma.push_2d && o->dma.pop && o->parallel.pool &&
         o->parallel.run && o->parallel.max_workers > 0;
}

static int run(const struct tl_op_ctx *o) {
  constexpr int kChunkN = 256;
  constexpr int kTiles = (kChunkN / 32) * (2048 / 32);
  constexpr size_t kActivationBytes = 32u * 2048u * sizeof(uint16_t);
  constexpr size_t kDequantBytes = 256u * 2048u * sizeof(uint16_t);
  constexpr size_t kOutputCroutonBytes = 32u * 32u * sizeof(uint16_t);
  constexpr size_t kWeightSourceBytes = (size_t)kTiles * 576u;
  constexpr size_t kWeightStageBytes = (size_t)kTiles * 640u;
  if ((size_t)o->vtcm_size < 2u * kWeightStageBytes + 2u * 127u) {
    return -1;
  }

  const uintptr_t vtcm_begin = (uintptr_t)o->vtcm_base;
  const uintptr_t vtcm_end = vtcm_begin + (size_t)o->vtcm_size;
  uintptr_t cursor = align_up(vtcm_begin, 256u);

  uint32_t *bias_vtcm = (uint32_t *)cursor;
  cursor += 64u * sizeof(uint32_t);
  cursor = align_up(cursor, 2048u);
  void *activation_hmx = (void *)cursor;
  cursor += kActivationBytes;
  cursor = align_up(cursor, 2048u);
  void *weight_hmx = (void *)cursor;
  cursor += kDequantBytes;
  cursor = align_up(cursor, 2048u);
  void *output_hmx = (void *)cursor;
  cursor += kOutputCroutonBytes;

  const uintptr_t stage1_addr =
      align_down(vtcm_end - kWeightStageBytes, 128u);
  const uintptr_t stage0_addr =
      align_down(stage1_addr - kWeightStageBytes, 128u);
  if (stage0_addr < cursor) {
    return -1;
  }
  uint8_t *weight_stage[2] = {
      (uint8_t *)stage0_addr,
      (uint8_t *)stage1_addr,
  };

  const uint8_t *weight = o->weight;
  const int chunks = o->n / kChunkN;
  int pending_dma = 0;
  auto push_weight = [&](int chunk) {
    const bool pushed =
        o->dma.push_2d(
            o->dma.queue,
            weight_stage[chunk & 1],
            weight + (size_t)chunk * kWeightSourceBytes,
            640u,
            576u,
            576u,
            (uint32_t)kTiles) == 0;
    pending_dma += pushed ? 1 : 0;
    return pushed;
  };
  auto pop_weight = [&]() {
    if (o->dma.pop(o->dma.queue)) {
      --pending_dma;
      return true;
    }
    return false;
  };

  if (!push_weight(0) || (chunks > 1 && !push_weight(1))) {
    while (pending_dma > 0) {
      pop_weight();
    }
    return -1;
  }

  int rc = tl_walkthrough_q4_m32_n256_k2048_pack_kernel(
      const_cast<float *>(o->act), activation_hmx, bias_vtcm);
  if (rc != 0 || !pop_weight()) {
    while (pending_dma > 0) {
      pop_weight();
    }
    return -1;
  }

  unsigned int workers = o->parallel.max_workers;
  if (workers > (unsigned int)kTiles) {
    workers = (unsigned int)kTiles;
  }
  if (workers > MAX_WORKERS) {
    workers = MAX_WORKERS;
  }

  HAP_compute_res_hmx_lock(o->vtcm_rctx);
  for (int chunk = 0; chunk < chunks && rc == 0; ++chunk) {
    struct dequant_work work = {};
    work.weight = weight_stage[chunk & 1];
    work.weight_hmx = weight_hmx;
    work.tiles = kTiles;
    rc = o->parallel.run(
        o->parallel.pool, dequant_worker, &work, workers);
    for (unsigned int worker = 0; worker < workers && rc == 0; ++worker) {
      rc = work.rc[worker];
    }

    if (rc == 0) {
      rc = tl_walkthrough_q4_m32_n256_k2048_compute_kernel(
          activation_hmx,
          weight_hmx,
          bias_vtcm,
          output_hmx,
          o->dst + (size_t)chunk * kChunkN,
          o->dst_stride);
    }
    if (rc == 0 && chunk + 2 < chunks && !push_weight(chunk + 2)) {
      rc = -1;
    }
    if (rc == 0 && chunk + 1 < chunks && !pop_weight()) {
      rc = -1;
    }
  }
  HAP_compute_res_hmx_unlock(o->vtcm_rctx);

  while (pending_dma > 0) {
    pop_weight();
  }
  return rc == 0 ? 0 : -1;
}

static const struct tl_op_desc kWalkthroughOp = {
    "walkthrough_q4_m32_n256_k2048", matches, run};

__attribute__((constructor)) static void register_walkthrough_op(void) {
  tl_register_op(&kWalkthroughOp);
}
