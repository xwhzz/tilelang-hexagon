#pragma once
// VTCM arena for tilelang Hexagon kernels.
//
// Buffers with storage scope `shared` (alloc_shared) are placed in VTCM rather
// than on the stack, because the HMX matrix engine reads its operands from VTCM
// (`mxmem`) and HVX wants the tightly-coupled scratch.  The codegen assigns each
// shared buffer a compile-time byte offset into one session-acquired VTCM region
// and emits `(T*)((char*)tl_vtcm_base() + offset)`.
//
// One acquisition is held for the whole FastRPC session (idempotent, lazy on
// first use); the bump offsets are emitted by the codegen, so this header only
// owns the acquire/release of the region itself.
//
// Compiled into the FastRPC skel (.cc) with the Hexagon SDK on the include path;
// it pulls in HAP headers, so it is only included when a kernel uses shared mem.
#include <HAP_compute_res.h>
#include <stdint.h>

#define TL_VTCM_INLINE static inline __attribute__((unused))

static uint8_t *tl_vtcm_base_ptr;  // base of the acquired VTCM region
static unsigned int tl_vtcm_total; // bytes acquired (capacity guard)
static int tl_vtcm_ctx;            // HAP_compute_res handle

// Bottom-up high-water mark of alloc_shared VTCM usage for the CURRENT kernel, in
// bytes from tl_vtcm_base_ptr.  The codegen publishes it (one monotonic assignment
// per shared buffer, all emitted before the compute that uses them), so a runtime
// that carves scratch TOP-DOWN from the arena end (the HMX gemm) knows how far the
// live shared tiles reach and can refuse rather than overlap them.
static unsigned int tl_vtcm_shared_high_water;

// Acquire the full VTCM region once.  Idempotent: a second call is a no-op while
// a region is held, so alloc_shared kernels and (later) the HMX path can share
// the same arena without double-acquiring.
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
  tl_vtcm_base_ptr = (uint8_t *)HAP_compute_res_attr_get_vtcm_ptr(&req);
  tl_vtcm_total = total;
}

TL_VTCM_INLINE void tl_vtcm_release(void) {
  if (tl_vtcm_ctx)
    HAP_compute_res_release(tl_vtcm_ctx);
  tl_vtcm_ctx = 0;
  tl_vtcm_base_ptr = 0;
  tl_vtcm_total = 0;
}

// Base of the VTCM arena (lazily acquires on first use).  Returns NULL if VTCM
// could not be acquired — generated code that dereferences it would fault, which
// is the intended loud failure (a kernel that needs VTCM and can't get it).
TL_VTCM_INLINE void *tl_vtcm_base(void) {
  tl_vtcm_acquire();
  return tl_vtcm_base_ptr;
}
