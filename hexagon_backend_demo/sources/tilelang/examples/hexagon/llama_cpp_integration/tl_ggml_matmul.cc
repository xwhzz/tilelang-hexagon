// TileLang Q4_0 prefill matmul embedded in ggml-hexagon.
//
// The generated kernels consume ggml's native 576-byte Q4_0 tiles and lower
// dequantization directly into the HMX weight Crouton layout.  There is no
// persistent weight repack cache, row-major FP16 weight matrix, T.gemm, or
// monolithic tl_hexagon_hmx_gemm call in this path.
#include <tl_templates/hexagon/tl_embed.h>

#include <HAP_compute_res.h>
#include "htp-ops.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

extern "C" int32_t tl_q4_hmx_m32_n128_k2048_embedded(
    void *, unsigned int, float *, uint8_t *, float *);
extern "C" int32_t tl_q4_hmx_m32_n128_k8192_embedded(
    void *, unsigned int, float *, uint8_t *, float *);
extern "C" int32_t tl_q4_hmx_m32_n512_k2048_embedded(
    void *, unsigned int, float *, uint8_t *, float *);
extern "C" int32_t tl_q4_hmx_m32_n512_k8192_embedded(
    void *, unsigned int, float *, uint8_t *, float *);
extern "C" int32_t tl_q4_hmx_staged_m32_n256_k2048_pack_kernel(
    float *, void *, uint32_t *);
extern "C" int32_t tl_q4_hmx_staged_m32_n256_k2048_dequant_kernel(
    uint8_t *, void *, int32_t, int32_t);
extern "C" int32_t tl_q4_hmx_staged_m32_n256_k2048_compute_kernel(
    void *, void *, uint32_t *, void *, float *, int32_t);
extern "C" int32_t tl_q4_hmx_staged_m32_n256_k8192_pack_kernel(
    float *, void *, uint32_t *);
extern "C" int32_t tl_q4_hmx_staged_m32_n256_k8192_dequant_kernel(
    uint8_t *, void *, int32_t, int32_t);
extern "C" int32_t tl_q4_hmx_staged_m32_n256_k8192_compute_kernel(
    void *, void *, uint32_t *, void *, float *, int32_t);

#ifndef TL_Q4_HMX_ENABLED
#define TL_Q4_HMX_ENABLED 0
#endif

int tl_mm_enabled = TL_Q4_HMX_ENABLED;

// --------------------------- op registry -----------------------------------
#define TL_MAX_OPS 16
static const struct tl_op_desc *g_tl_ops[TL_MAX_OPS];
static int g_tl_nops;

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

// --------------------------- generated kernel dispatch ---------------------
typedef int32_t (*tl_q4_hmx_kernel)(
    void *, unsigned int, float *, uint8_t *, float *);

struct tl_q4_hmx_shape {
  int k;
  int n;
  unsigned int vtcm_high_water;
  tl_q4_hmx_kernel kernel;
};

static const struct tl_q4_hmx_shape TL_Q4_HMX_SHAPES[] = {
    {2048, 512, 268288u, tl_q4_hmx_m32_n512_k2048_embedded},
    {8192, 512, 1054720u, tl_q4_hmx_m32_n512_k8192_embedded},
    {2048, 128, 268288u, tl_q4_hmx_m32_n128_k2048_embedded},
    {8192, 128, 1054720u, tl_q4_hmx_m32_n128_k8192_embedded},
};

static const struct tl_q4_hmx_shape *tl_q4_hmx_find_shape(int k, int n) {
  const size_t count = sizeof(TL_Q4_HMX_SHAPES) / sizeof(TL_Q4_HMX_SHAPES[0]);
  for (size_t i = 0; i < count; ++i) {
    if (TL_Q4_HMX_SHAPES[i].k == k && (n % TL_Q4_HMX_SHAPES[i].n) == 0) {
      return &TL_Q4_HMX_SHAPES[i];
    }
  }
  return NULL;
}

typedef int32_t (*tl_q4_pack_stage)(float *, void *, uint32_t *);
typedef int32_t (*tl_q4_dequant_stage)(uint8_t *, void *, int32_t, int32_t);
typedef int32_t (*tl_q4_compute_stage)(
    void *, void *, uint32_t *, void *, float *, int32_t);

struct tl_q4_staged_shape {
  int k;
  int n;
  tl_q4_pack_stage pack;
  tl_q4_dequant_stage dequant;
  tl_q4_compute_stage compute;
};

static const struct tl_q4_staged_shape TL_Q4_STAGED_SHAPES[] = {
    {2048, 256, tl_q4_hmx_staged_m32_n256_k2048_pack_kernel,
     tl_q4_hmx_staged_m32_n256_k2048_dequant_kernel,
     tl_q4_hmx_staged_m32_n256_k2048_compute_kernel},
    {8192, 256, tl_q4_hmx_staged_m32_n256_k8192_pack_kernel,
     tl_q4_hmx_staged_m32_n256_k8192_dequant_kernel,
     tl_q4_hmx_staged_m32_n256_k8192_compute_kernel},
};

static const struct tl_q4_staged_shape *tl_q4_staged_find_shape(int k, int n) {
  const size_t count =
      sizeof(TL_Q4_STAGED_SHAPES) / sizeof(TL_Q4_STAGED_SHAPES[0]);
  for (size_t i = 0; i < count; ++i) {
    if (TL_Q4_STAGED_SHAPES[i].k == k &&
        (n % TL_Q4_STAGED_SHAPES[i].n) == 0) {
      return &TL_Q4_STAGED_SHAPES[i];
    }
  }
  return NULL;
}

static uintptr_t tl_align_down(uintptr_t value, uintptr_t alignment) {
  return value & ~(alignment - 1u);
}

static uintptr_t tl_align_up(uintptr_t value, uintptr_t alignment) {
  return (value + alignment - 1u) & ~(alignment - 1u);
}

#define TL_Q4_MAX_PARALLEL_WORKERS 16
struct tl_q4_dequant_work {
  tl_q4_dequant_stage dequant;
  uint8_t *weight;
  void *output;
  int tiles;
  int rc[TL_Q4_MAX_PARALLEL_WORKERS];
};

static void tl_q4_dequant_worker(
    unsigned int workers, unsigned int worker, void *opaque) {
  struct tl_q4_dequant_work *work = (struct tl_q4_dequant_work *)opaque;
  const int begin = (int)(((uint64_t)work->tiles * worker) / workers);
  const int end = (int)(((uint64_t)work->tiles * (worker + 1u)) / workers);
  work->rc[worker] = work->dequant(
      work->weight, work->output, begin, end);
}

// Prefer 512 output channels per kernel to amortize activation packing and HMX
// setup; retain the 128-channel family for smaller shapes. The runtime DMA queue
// double-buffers native weight slices at the top of VTCM while generated HMX
// allocations grow bottom-up.  TileLang never acquires or resets the DMA engine.
static int tl_q4_hmx_run_legacy(const struct tl_op_ctx *o) {
  const struct tl_q4_hmx_shape *shape = tl_q4_hmx_find_shape(o->k, o->n);
  if (!shape || !o->dma.queue || !o->dma.push_1d || !o->dma.pop) {
    return -1;
  }

  const size_t output_stage_bytes = 32u * (size_t)shape->n * sizeof(float);
  const size_t weight_stage_bytes =
      (size_t)(shape->n / 32) * (size_t)(o->k / 32) * 576u;
  const size_t top_stage_bytes =
      output_stage_bytes + 2u * weight_stage_bytes + 3u * 127u;
  if ((size_t)o->vtcm_size < top_stage_bytes) {
    return -1;
  }
  const uintptr_t vtcm_begin = (uintptr_t)o->vtcm_base;
  const uintptr_t vtcm_end = vtcm_begin + (size_t)o->vtcm_size;
  const uintptr_t kernel_begin = tl_align_up(vtcm_begin, 2048u);
  if (kernel_begin >= vtcm_end) {
    return -1;
  }
  const unsigned int kernel_size =
      (unsigned int)(vtcm_end - kernel_begin);
  const uintptr_t output_stage_addr =
      tl_align_down(vtcm_end - output_stage_bytes, 128u);
  const uintptr_t weight_stage_1_addr =
      tl_align_down(output_stage_addr - weight_stage_bytes, 128u);
  const uintptr_t weight_stage_0_addr =
      tl_align_down(weight_stage_1_addr - weight_stage_bytes, 128u);
  if (weight_stage_0_addr < kernel_begin + shape->vtcm_high_water) {
    return -1;
  }

  float *output_stage = (float *)output_stage_addr;
  uint8_t *weight_stage[2] = {
      (uint8_t *)weight_stage_0_addr,
      (uint8_t *)weight_stage_1_addr,
  };
  float *activation = const_cast<float *>(o->act);
  const uint8_t *weight = (const uint8_t *)o->weight;
  const int n_chunks = o->n / shape->n;
  int pending_dma = 0;

  auto push_weight = [&](int chunk) {
    bool pushed =
        o->dma.push_1d(
            o->dma.queue,
            weight_stage[chunk & 1],
            weight + (size_t)chunk * weight_stage_bytes,
            (uint32_t)weight_stage_bytes) == 0;
    pending_dma += pushed ? 1 : 0;
    return pushed;
  };
  auto pop_weight = [&]() {
    void *completed = o->dma.pop(o->dma.queue);
    if (completed) {
      --pending_dma;
      return true;
    }
    return false;
  };

  if (!push_weight(0) || (n_chunks > 1 && !push_weight(1)) || !pop_weight()) {
    while (pending_dma > 0) {
      pop_weight();
    }
    return -1;
  }

  HAP_compute_res_hmx_lock(o->vtcm_rctx);

  int rc = 0;
  for (int chunk = 0; chunk < n_chunks && rc == 0; ++chunk) {
    const int n0 = chunk * shape->n;
    for (int m0 = 0; m0 < o->m && rc == 0; m0 += 32) {
      rc = shape->kernel(
          (void *)kernel_begin,
          kernel_size,
          activation + (size_t)m0 * o->act_stride,
          weight_stage[chunk & 1],
          output_stage);
      if (rc == 0) {
        for (int m = 0; m < 32; ++m) {
          memcpy(
              o->dst + (size_t)(m0 + m) * o->dst_stride + n0,
              output_stage + (size_t)m * shape->n,
              (size_t)shape->n * sizeof(float));
        }
      }
    }
    if (rc == 0 && chunk + 2 < n_chunks && !push_weight(chunk + 2)) {
      rc = -1;
    }
    if (rc == 0 && chunk + 1 < n_chunks && !pop_weight()) {
      rc = -1;
    }
  }
  while (pending_dma > 0) {
    pop_weight();
  }

  HAP_compute_res_hmx_unlock(o->vtcm_rctx);
  return rc == 0 ? 0 : -1;
}

// The staged schedule keeps all instruction-level work in generated TileLang
// functions, but lends the dequant tile range to the host's existing workers.
// Raw 576-byte tiles are asynchronously DMA-padded to 640 bytes so every HVX
// load is aligned and the final 64-byte scale tail is safely readable.
static int tl_q4_hmx_run_staged(
    const struct tl_op_ctx *o, const struct tl_q4_staged_shape *shape) {
  if (!shape || !o->dma.queue || !o->dma.push_2d || !o->dma.pop ||
      !o->parallel.pool || !o->parallel.run ||
      o->parallel.max_workers == 0) {
    return -1;
  }

  const int nt = shape->n / 32;
  const int kt = o->k / 32;
  const int tiles = nt * kt;
  const size_t activation_bytes = 32u * (size_t)o->k * sizeof(uint16_t);
  const size_t dequant_bytes =
      (size_t)shape->n * (size_t)o->k * sizeof(uint16_t);
  const size_t output_crouton_bytes = 32u * 32u * sizeof(uint16_t);
  const size_t weight_stage_bytes = (size_t)tiles * 640u;
  const size_t weight_source_bytes = (size_t)tiles * 576u;
  if ((size_t)o->vtcm_size < 2u * weight_stage_bytes + 2u * 127u) {
    return -1;
  }

  const uintptr_t vtcm_begin = (uintptr_t)o->vtcm_base;
  const uintptr_t vtcm_end = vtcm_begin + (size_t)o->vtcm_size;
  uintptr_t cursor = tl_align_up(vtcm_begin, 256u);
  uint32_t *bias_vtcm = (uint32_t *)cursor;
  cursor += 64u * sizeof(uint32_t);
  cursor = tl_align_up(cursor, 2048u);
  void *activation_hmx = (void *)cursor;
  cursor += activation_bytes;
  cursor = tl_align_up(cursor, 2048u);
  void *dequant_hmx = (void *)cursor;
  cursor += dequant_bytes;
  cursor = tl_align_up(cursor, 2048u);
  void *output_hmx = (void *)cursor;
  cursor += output_crouton_bytes;

  const uintptr_t weight_stage_1_addr =
      tl_align_down(vtcm_end - weight_stage_bytes, 128u);
  const uintptr_t weight_stage_0_addr =
      tl_align_down(weight_stage_1_addr - weight_stage_bytes, 128u);
  if (weight_stage_0_addr < cursor) {
    return -1;
  }
  uint8_t *weight_stage[2] = {
      (uint8_t *)weight_stage_0_addr,
      (uint8_t *)weight_stage_1_addr,
  };

  const uint8_t *weight = (const uint8_t *)o->weight;
  const int n_chunks = o->n / shape->n;
  int pending_dma = 0;
  auto push_weight = [&](int chunk) {
    const bool pushed =
        o->dma.push_2d(
            o->dma.queue,
            weight_stage[chunk & 1],
            weight + (size_t)chunk * weight_source_bytes,
            640u,
            576u,
            576u,
            (uint32_t)tiles) == 0;
    pending_dma += pushed ? 1 : 0;
    return pushed;
  };
  auto pop_weight = [&]() {
    void *completed = o->dma.pop(o->dma.queue);
    if (completed) {
      --pending_dma;
      return true;
    }
    return false;
  };

  if (!push_weight(0) || (n_chunks > 1 && !push_weight(1))) {
    while (pending_dma > 0) {
      pop_weight();
    }
    return -1;
  }
  int rc = shape->pack(
      const_cast<float *>(o->act), activation_hmx, bias_vtcm);
  if (rc != 0 || !pop_weight()) {
    while (pending_dma > 0) {
      pop_weight();
    }
    return -1;
  }

  unsigned int workers = o->parallel.max_workers;
  if (workers > (unsigned int)tiles) {
    workers = (unsigned int)tiles;
  }
  if (workers > TL_Q4_MAX_PARALLEL_WORKERS) {
    workers = TL_Q4_MAX_PARALLEL_WORKERS;
  }

  HAP_compute_res_hmx_lock(o->vtcm_rctx);
  for (int chunk = 0; chunk < n_chunks && rc == 0; ++chunk) {
    struct tl_q4_dequant_work work = {};
    work.dequant = shape->dequant;
    work.weight = weight_stage[chunk & 1];
    work.output = dequant_hmx;
    work.tiles = tiles;
    rc = o->parallel.run(
        o->parallel.pool, tl_q4_dequant_worker, &work, workers);
    for (unsigned int worker = 0; worker < workers && rc == 0; ++worker) {
      rc = work.rc[worker];
    }
    if (rc == 0) {
      rc = shape->compute(
          activation_hmx,
          dequant_hmx,
          bias_vtcm,
          output_hmx,
          o->dst + (size_t)chunk * shape->n,
          o->dst_stride);
    }
    if (rc == 0 && chunk + 2 < n_chunks && !push_weight(chunk + 2)) {
      rc = -1;
    }
    if (rc == 0 && chunk + 1 < n_chunks && !pop_weight()) {
      rc = -1;
    }
  }
  HAP_compute_res_hmx_unlock(o->vtcm_rctx);

  while (pending_dma > 0) {
    pop_weight();
  }
  return rc == 0 ? 0 : -1;
}

static int tl_q4_hmx_run(const struct tl_op_ctx *o) {
  const struct tl_q4_staged_shape *staged =
      tl_q4_staged_find_shape(o->k, o->n);
  if (o->m == 32 && staged && tl_q4_hmx_run_staged(o, staged) == 0) {
    return 0;
  }
  return tl_q4_hmx_run_legacy(o);
}

static int tl_q4_hmx_matches(const struct tl_op_ctx *o) {
  return o->kind == TL_OP_MATMUL && o->weight_type == HTP_TYPE_Q4_0 &&
         !o->has_bias && o->vtcm_base && o->vtcm_rctx && o->dma.queue &&
         o->dma.push_1d && o->dma.pop && o->m > 0 && (o->m % 32) == 0 &&
         o->n > 0 && (o->n % 128) == 0 && o->act_stride == o->k &&
         o->dst_stride >= o->n && tl_q4_hmx_find_shape(o->k, o->n) != NULL;
}

static const struct tl_op_desc TL_Q4_HMX_DESC = {
    "q4_0_hmx_atoms", tl_q4_hmx_matches, tl_q4_hmx_run};

__attribute__((constructor)) static void tl_q4_hmx_register(void) {
  tl_register_op(&TL_Q4_HMX_DESC);
}
