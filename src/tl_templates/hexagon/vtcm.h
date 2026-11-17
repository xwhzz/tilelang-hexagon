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
// The region belongs to the DSP process's compute-resource context: a cached
// HAP_compute_res context, which the skel opens (tl_vtcm_open). A kernel's run
// takes it (tl_vtcm_take) and gives it back when it ends (tl_vtcm_give), so
// that another DSP process of the session, or another client, can use VTCM in
// between; the region keeps its address throughout. The layout comes from the
// compiler, so this header only owns the region itself.
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

static uint8_t *tl_vtcm_base_ptr;  // base of the region the current run holds
static unsigned int tl_vtcm_total; // its bytes (capacity guard)
static int tl_vtcm_ctx;            // the process's HAP_compute_res context
static void *tl_vtcm_region;       // the context's region, as HAP gives it
static unsigned int tl_vtcm_region_bytes;
// An unsuccessful release does not prove that the resource was relinquished.
// Keep ownership poisoned until the process exits; never pretend cleanup ran.
static int tl_vtcm_held;
static int tl_vtcm_poisoned;
static int tl_vtcm_hmx_requested;
static int tl_vtcm_cooperative;

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

// How long a run waits for VTCM that another client holds.
#define TL_VTCM_TAKE_TIMEOUT_US 1000000u

// Open the process's context: the whole VTCM and, for kernels that issue HMX,
// the HMX unit (hmx.h). Returns 0, or nonzero when there is no context to be
// had; a run takes it.
TL_VTCM_INLINE int
tl_vtcm_open_mode(int hmx, int (*release_callback)(unsigned int, void *),
                  void *opaque) {
  tl_vtcm_hmx_requested = hmx;
  if (__atomic_load_n(&tl_vtcm_poisoned, __ATOMIC_ACQUIRE))
    return -1;
  if (tl_vtcm_ctx)
    return tl_vtcm_cooperative == (release_callback != nullptr) ? 0 : -1;
  unsigned int avail, total;
  compute_res_vtcm_page_t avail_pg, total_pg;
  if (HAP_compute_res_query_VTCM(0, &total, &total_pg, &avail, &avail_pg) ||
      !total)
    return -1;
  compute_res_attr_t req;
  if (HAP_compute_res_attr_init(&req) ||
      HAP_compute_res_attr_set_vtcm_param(&req, total, 1) ||
      (hmx && HAP_compute_res_attr_set_hmx_param(&req, 1)) ||
      HAP_compute_res_attr_set_cache_mode(&req, 1) ||
      (release_callback && HAP_compute_res_attr_set_release_callback(
                               &req, release_callback, opaque)))
    return -1;
  tl_vtcm_ctx = HAP_compute_res_acquire(&req, 10000);
  if (!tl_vtcm_ctx)
    return -1;
  tl_vtcm_region = HAP_compute_res_attr_get_vtcm_ptr(&req);
  // v1 requests an absolute size (unlike the optional minimum-size v2 API).
  // Cached acquire preserves both this address and this allocation size.
  tl_vtcm_region_bytes = total;
  tl_vtcm_cooperative = release_callback != nullptr;
  return 0;
}

TL_VTCM_INLINE int tl_vtcm_open(int hmx) {
  return tl_vtcm_open_mode(hmx, nullptr, nullptr);
}

// A run takes the region: 0, or nonzero when it cannot be had in time.
TL_VTCM_INLINE int tl_vtcm_take(void) {
  if (__atomic_load_n(&tl_vtcm_poisoned, __ATOMIC_ACQUIRE))
    return -1;
  if (tl_vtcm_held)
    return tl_vtcm_base_ptr ? 0 : -1;
  if (!tl_vtcm_ctx ||
      HAP_compute_res_acquire_cached(tl_vtcm_ctx, TL_VTCM_TAKE_TIMEOUT_US))
    return -1;
  tl_vtcm_held = 1;
  if (tl_vtcm_bind_region(tl_vtcm_region, tl_vtcm_region_bytes) != 0) {
    if (HAP_compute_res_release_cached(tl_vtcm_ctx))
      __atomic_store_n(&tl_vtcm_poisoned, 1, __ATOMIC_RELEASE);
    else
      tl_vtcm_held = 0;
    return -1;
  }
  return 0;
}

// Return checked status even when legacy callers do not consume it.
TL_VTCM_INLINE int tl_vtcm_give(void) {
  if (__atomic_load_n(&tl_vtcm_poisoned, __ATOMIC_ACQUIRE))
    return -1;
  if (!tl_vtcm_held)
    return 0;
  if (HAP_compute_res_release_cached(tl_vtcm_ctx)) {
    __atomic_store_n(&tl_vtcm_poisoned, 1, __ATOMIC_RELEASE);
    return -1;
  }
  tl_vtcm_held = 0;
  tl_vtcm_base_ptr = 0;
  tl_vtcm_total = 0;
  return 0;
}

TL_VTCM_INLINE int tl_vtcm_close(void) {
  if (tl_vtcm_give())
    return -1;
  if (tl_vtcm_ctx && HAP_compute_res_release(tl_vtcm_ctx)) {
    __atomic_store_n(&tl_vtcm_poisoned, 1, __ATOMIC_RELEASE);
    return -1;
  }
  tl_vtcm_ctx = 0;
  tl_vtcm_region = nullptr;
  tl_vtcm_region_bytes = 0;
  tl_vtcm_cooperative = 0;
  return 0;
}

// Base of the VTCM arena: the region the current run holds, NULL outside a
// run. Generated code that dereferences NULL faults, which is the intended
// loud failure.
TL_VTCM_INLINE void *tl_vtcm_base(void) { return tl_vtcm_base_ptr; }

// Base of the kernel arena, the whole region, or NULL outside a run.
TL_VTCM_INLINE uint8_t *tl_vtcm_arena(void) { return tl_vtcm_base_ptr; }

// Bytes of the kernel arena (0 outside a run).
TL_VTCM_INLINE unsigned int tl_vtcm_arena_bytes(void) { return tl_vtcm_total; }
