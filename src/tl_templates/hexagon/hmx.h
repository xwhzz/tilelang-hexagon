#pragma once
// HMX FP16 matmul runtime for tilelang-generated Hexagon kernels. All the
// HMX/HVX/ VTCM/Crouton hardware intimacy lives here; the codegen stays a thin
// C emitter that just calls into it. Distilled from the verified mini-htp /
// htp-ops-lib recipes.
//
// The GEMMs on HMX (hmx_gemm.h) drive it through these instructions; the
// runtime owns the session.
//
// Compiled into the FastRPC skel (.cc) with the Hexagon SDK on the include path
// and -mhmx; it pulls in HAP headers, so it is only included by kernels that
// use HMX.
#include <HAP_compute_res.h>
#include <HAP_perf.h> // Generated DSP timing wrappers also use the qtimer API.
#include <hexagon_types.h> // HVX_Vector + Q6_* intrinsics (needs -mhvx)
#include <hmx_hexagon_protos.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <tl_templates/hexagon/power.h> // the session's power requests
#include <tl_templates/hexagon/vtcm.h> // unified VTCM arena (shared with alloc_shared)
#include <tl_templates/hexagon/worker.h> // HW-thread worker pool (multithreaded HMX/HVX)

#define TL_HMX_INLINE static inline __attribute__((unused, always_inline))

// ------------------------------ HMX lock ----------------------------------
// HMX is part of the process's compute-resource context (vtcm.h), with the
// VTCM its operands live in: tl_vtcm_open(1) asks for both.
static int tl_hmx_spin;
// Per-thread HMX SHARED enable: each thread that issues HMX MACs must call this
// once its run holds the context. Returns 0 if this thread now holds the lock,
// non-zero on failure (incl. no context) so callers can refuse rather than run
// MACs with HMX unlocked.
TL_HMX_INLINE int tl_hmx_enable(void) {
  if (!tl_vtcm_ctx)
    return -1;
  return HAP_compute_res_hmx_lock2(tl_vtcm_ctx, HAP_COMPUTE_RES_HMX_SHARED);
}
TL_HMX_INLINE void tl_hmx_disable(void) {
  if (tl_vtcm_ctx)
    HAP_compute_res_hmx_unlock2(tl_vtcm_ctx, HAP_COMPUTE_RES_HMX_SHARED);
}
TL_HMX_INLINE void tl_hmx_unit_acquire(void) {
  int *lp = &tl_hmx_spin;
  asm volatile("1:  r0 = memw_locked(%0)     \n"
               "    p0 = cmp.eq(r0, #0)      \n"
               "    if (!p0) jump 2f         \n"
               "    memw_locked(%0, p0) = %0 \n"
               "    if (p0) jump 3f          \n"
               "2:  pause(#8)                \n"
               "    jump 1b                  \n"
               "3:"
               : "+r"(lp)::"p0", "r0");
}
// Release with a plain volatile store (matching htp-ops-lib's hmx_mgr). No
// extra barrier is needed: the only state the lock guards is the HMX
// accumulator, which the next holder unconditionally re-clears (mxclracc)
// before use, and every worker reads its accumulator out to its OWN VTCM region
// — so no worker depends on another's in-flight accumulator/VTCM writes being
// visible at release time.
TL_HMX_INLINE void tl_hmx_unit_release(void) {
  *(volatile int *)&tl_hmx_spin = 0;
}

// ------------------------- the HMX matmul ---------------------------------
#define TL_HMX_T 32
#define TL_HMX_TILE_ELMS 1024
#define TL_HMX_TILE_BYTES 2048
#define TL_HMX_SPATIAL_MASK 0x1cu
#define TL_HMX_SPATIAL_RT                                                      \
  ((((TL_HMX_SPATIAL_MASK) >> 1) << 7) | (((TL_HMX_SPATIAL_MASK) & 1u) << 1))
#define TL_HMX_ACTIVATION_RT (TL_HMX_SPATIAL_RT | (31u << 2))

TL_HMX_INLINE void tl_hmx_clear_acc(void) {
  asm volatile("mxclracc.hf" ::: "memory");
}
TL_HMX_INLINE void tl_hmx_set_scales(const void *s) {
  asm volatile("bias = mxmem2(%0)" ::"r"(s) : "memory");
}
TL_HMX_INLINE void tl_hmx_mac_tiles(const __fp16 *a, const __fp16 *w,
                                    size_t n) {
  // Caller guarantees 1 <= n <= 32 and contiguous native tiles.
  const uint32_t act_rt = TL_HMX_ACTIVATION_RT | ((n - 1) << 11);
  const uint32_t weight_rt = n * TL_HMX_TILE_BYTES - 1;
  asm volatile("{ activation.hf = mxmem(%0, %1):deep\n"
               "  weight.hf = mxmem(%2, %3) }\n" ::"r"(a),
               "r"(act_rt), "r"(w), "r"(weight_rt)
               : "memory");
}
TL_HMX_INLINE void tl_hmx_convert_acc(int mode) {
  asm volatile("cvt.hf = acc(%0)" ::"r"(mode) : "memory");
}
TL_HMX_INLINE void tl_hmx_store_cvt(__fp16 *o) {
  asm volatile("mxmem(%0, %1) = cvt" ::"r"(o), "r"(TL_HMX_SPATIAL_RT)
               : "memory");
}
// The process's resources: HMX's power and the compute-resource context, its
// VTCM and HMX. The generated FastRPC _open handler calls init and _close
// deinit; each run takes the context (tl_hmx_take) and gives it back
// (tl_hmx_give), so that another DSP process of the session can run in
// between. NOTE: the check-then-set of these globals is NOT internally
// synchronized; it is safe because every caller invokes it single-threaded
// BEFORE tl_parallel spawns workers.
static int tl_hmx_inited;
static qurt_thread_t
    tl_hmx_run_tid; // the run's thread, whose HMX enable take holds
static void tl_hmx_session_init(void) {
  if (tl_hmx_inited)
    return;
  tl_power_hmx(true);
  if (tl_vtcm_open(1) == 0)
    tl_hmx_inited = 1;
  else
    tl_power_hmx(false); // a failed init must not leave HMX powered
}
static void tl_hmx_session_deinit(void) {
  if (!tl_hmx_inited)
    return;
  tl_vtcm_close();
  tl_power_hmx(false);
  tl_hmx_inited = 0;
}

// Let the FastRPC runtime reject execution if the context could not be had.
static int tl_hmx_session_ok(void) { return tl_hmx_inited; }

// A run takes the context, VTCM and HMX, and enables HMX for its own thread;
// pool threads enable it for themselves (tl_hmx_worker_scope). Returns TL_OK,
// or the TL_ERR_* of what could not be had.
static int tl_hmx_take(void) {
  if (tl_vtcm_take() != 0)
    return TL_ERR_VTCM;
  if (tl_hmx_enable() != 0) {
    tl_vtcm_give();
    return TL_ERR_HMX;
  }
  tl_hmx_run_tid = qurt_thread_get_id();
  return TL_OK;
}
static void tl_hmx_give(void) {
  tl_hmx_disable();
  tl_hmx_run_tid = 0;
  tl_vtcm_give();
}

// tl_parallel_for worker scope for tile programs that issue HMX: a pool thread
// enables HMX for itself around its programs. The run's thread already holds
// the enable that tl_hmx_take took.
struct tl_hmx_worker_scope {
  int status = TL_OK;
  bool enabled = false;

  tl_hmx_worker_scope() {
    if (qurt_thread_get_id() == tl_hmx_run_tid)
      return;
    if (tl_hmx_enable() != 0)
      status = TL_ERR_HMX; // refuse rather than run MACs with HMX unlocked
    else
      enabled = true;
  }
  ~tl_hmx_worker_scope() {
    if (enabled)
      tl_hmx_disable();
  }
  tl_hmx_worker_scope(const tl_hmx_worker_scope &) = delete;
  tl_hmx_worker_scope &operator=(const tl_hmx_worker_scope &) = delete;
};
