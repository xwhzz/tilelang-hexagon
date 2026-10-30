#pragma once
// VTCM arena for tilelang Hexagon kernels.
//
// Buffers with storage scope `shared` (alloc_shared) are placed in VTCM rather
// than on the stack, because the HMX matrix engine reads its operands from VTCM
// (`mxmem`) and HVX wants the tightly-coupled scratch.  The compiler lays out
// the shared buffers of a tile program in one arena, which the codegen binds
// to `tl_vtcm_arena()`; a worker loop splits the region into per-worker arenas
// instead (tl_vtcm_worker_arenas in worker.h).
//
// Acquisition is idempotent until release, and may happen lazily on first use.
// The caller chooses its lifetime: the resident runtime holds it for a session
// by default, or for each invocation with resource_lifetime=call. This header
// only owns acquire/release; the compiler supplies the arena layout.
//
// Compiled into the FastRPC skel (.cc) with the Hexagon SDK on the include
// path; it pulls in HAP headers, so it is only included when a kernel uses
// shared mem.
#include <HAP_compute_res.h>
#include <stddef.h>
#include <stdint.h>

// Some BSP loaders reject the SDK's optional weak v2 import. Opt in only for
// firmware known to need the legacy API (validated on SA8650/v73). ISA version
// alone says nothing about firmware API support. With -Bsymbolic this fallback
// binds locally, so it must remain disabled for ordinary Android/Linux builds.
#if defined(TL_HEXAGON_LEGACY_COMPUTE_RES) && TL_HEXAGON_LEGACY_COMPUTE_RES
#include <string.h>
#ifdef __cplusplus
extern "C" {
#endif
int __attribute__((weak))
compute_resource_attr_init_v2(compute_res_attr_t *attr, unsigned int size,
                              unsigned int version) {
  if (attr)
    memset(attr, 0, size);
  (void)version;
  return -1;
}
#ifdef __cplusplus
}
#endif
#endif

#define TL_VTCM_INLINE static inline __attribute__((unused))

static uint8_t *tl_vtcm_base_ptr;  // base of the acquired VTCM region
static unsigned int tl_vtcm_total; // bytes acquired (capacity guard)
static int tl_vtcm_ctx;            // HAP_compute_res handle

// The VTCM arena of the current serial kernel, in bytes from
// tl_vtcm_base_ptr. Codegen publishes this for embedding/export capacity
// checks. Worker loops size their per-worker arenas at run time instead and do
// not write it.
static unsigned int tl_vtcm_shared_high_water;

// HMX mxmem addresses are Crouton-tile addresses: the low 11 bits are not a
// byte offset.  Align the arena itself, not only each compiler-assigned offset.
// HAP and embedding runtimes do not promise a 2 KiB-aligned region base.
#define TL_VTCM_HMX_ALIGNMENT 2048u

TL_VTCM_INLINE int tl_vtcm_bind_region(void *base, unsigned int size) {
  uintptr_t raw = (uintptr_t)base;
  uintptr_t aligned = (raw + TL_VTCM_HMX_ALIGNMENT - 1u) &
                      ~(uintptr_t)(TL_VTCM_HMX_ALIGNMENT - 1u);
  size_t padding = (size_t)(aligned - raw);
  if (!base || padding >= (size_t)size) {
    tl_vtcm_base_ptr = 0;
    tl_vtcm_total = 0;
    return -1;
  }
  tl_vtcm_base_ptr = (uint8_t *)aligned;
  tl_vtcm_total = size - (unsigned int)padding;
  return 0;
}

// Acquire the full VTCM region. A second call is a no-op while the region is
// held, so alloc_shared kernels and HMX share one arena. After release, the
// next acquisition obtains a new grant; its address need not be the same.
TL_VTCM_INLINE void tl_vtcm_acquire(void) {
  if (tl_vtcm_base_ptr)
    return;
  unsigned int avail, total;
  compute_res_vtcm_page_t avail_pg, total_pg;
  if (HAP_compute_res_query_VTCM(0, &total, &total_pg, &avail, &avail_pg))
    return;
  compute_res_attr_t req;
  HAP_compute_res_attr_init(&req);
  HAP_compute_res_attr_set_vtcm_param(&req, total, 1);
  tl_vtcm_ctx = HAP_compute_res_acquire(&req, 10000);
  if (!tl_vtcm_ctx)
    return;
  void *raw_base = HAP_compute_res_attr_get_vtcm_ptr(&req);
  if (tl_vtcm_bind_region(raw_base, total) != 0) {
    HAP_compute_res_release(tl_vtcm_ctx);
    tl_vtcm_ctx = 0;
  }
}

TL_VTCM_INLINE void tl_vtcm_release(void) {
  if (tl_vtcm_ctx)
    HAP_compute_res_release(tl_vtcm_ctx);
  tl_vtcm_ctx = 0;
  tl_vtcm_base_ptr = 0;
  tl_vtcm_total = 0;
}

// Base of the VTCM arena (lazily acquires on first use).  Returns NULL if VTCM
// could not be acquired — generated code that dereferences it would fault,
// which is the intended loud failure (a kernel that needs VTCM and can't get
// it).
TL_VTCM_INLINE void *tl_vtcm_base(void) {
  tl_vtcm_acquire();
  return tl_vtcm_base_ptr;
}

// Base of the kernel arena, the whole region (lazily acquires), or NULL
// without VTCM.
TL_VTCM_INLINE uint8_t *tl_vtcm_arena(void) {
  return (uint8_t *)tl_vtcm_base();
}

// Bytes of the kernel arena (0 without VTCM).
TL_VTCM_INLINE unsigned int tl_vtcm_arena_bytes(void) {
  tl_vtcm_acquire();
  return tl_vtcm_total;
}
