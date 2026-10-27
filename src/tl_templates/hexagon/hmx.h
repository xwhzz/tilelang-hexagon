#pragma once
// HMX FP16 matmul runtime for tilelang-generated Hexagon kernels. All the
// HMX/HVX/ VTCM/Crouton hardware intimacy lives here; the codegen stays a thin
// C emitter that just calls into it. Distilled from the verified mini-htp /
// htp-ops-lib recipes.
//
// T.gemm and explicit DSL kernels use stateful HMX instruction atoms.
// Layout-copy helpers transform caller-owned buffers; the runtime owns
// sessions.
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
#include <tl_templates/hexagon/power.h>   // the session's power requests
#include <tl_templates/hexagon/qmatmul.h> // reusable FP32/Crouton row-pair atoms
#include <tl_templates/hexagon/vtcm.h> // unified VTCM arena (shared with alloc_shared)
#include <tl_templates/hexagon/worker.h> // HW-thread worker pool (multithreaded HMX/HVX)

#define TL_HMX_INLINE static inline __attribute__((unused, always_inline))

// ------------------------------ VTCM --------------------------------------
// Delegated to the shared tl_vtcm arena (vtcm.h) so HMX scratch and
// alloc_shared operands draw from one acquisition — no double-acquire when a
// T.gemm kernel has its operands in VTCM and also drives HMX.
TL_HMX_INLINE void tl_hmx_vtcm_setup(void) { tl_vtcm_acquire(); }
TL_HMX_INLINE void tl_hmx_vtcm_reset(void) { tl_vtcm_release(); }

// ------------------------------ HMX lock ----------------------------------
static int tl_hmx_ctx;
static int tl_hmx_spin;
TL_HMX_INLINE void tl_hmx_setup(void) {
  compute_res_attr_t req;
  HAP_compute_res_attr_init(&req);
  HAP_compute_res_attr_set_hmx_param(&req, 1);
  tl_hmx_ctx = HAP_compute_res_acquire(&req, 10000);
}
TL_HMX_INLINE void tl_hmx_reset(void) {
  if (tl_hmx_ctx)
    HAP_compute_res_release(tl_hmx_ctx);
  tl_hmx_ctx = 0;
}
// Per-thread HMX SHARED enable: each thread that issues HMX MACs must call this
// once. Returns 0 if this thread now holds the lock, non-zero on failure
// (incl. no ctx) so callers can refuse rather than run MACs with HMX unlocked.
TL_HMX_INLINE int tl_hmx_enable(void) {
  if (!tl_hmx_ctx)
    return -1;
  return HAP_compute_res_hmx_lock2(tl_hmx_ctx, HAP_COMPUTE_RES_HMX_SHARED);
}
TL_HMX_INLINE void tl_hmx_disable(void) {
  if (tl_hmx_ctx)
    HAP_compute_res_hmx_unlock2(tl_hmx_ctx, HAP_COMPUTE_RES_HMX_SHARED);
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
// Persistent session resources: power/VTCM/HMX are acquired once and held
// across many matmuls. The generated FastRPC _open handler calls init; _close
// calls deinit. The one-shot runtime also initializes resources before kernel
// execution. NOTE: the check-then-set of these globals is NOT internally
// synchronized; it is safe because every caller invokes it single-threaded
// BEFORE tl_parallel spawns workers. A future persistent/cross-kernel pool that
// could init from two threads must add a lock here.
static int tl_hmx_inited;
static qurt_thread_t
    tl_hmx_session_tid; // thread whose HMX SHARED enable session_init holds
static void tl_hmx_session_init(void) {
  if (tl_hmx_inited)
    return;
  tl_power_hmx(true);
  tl_hmx_vtcm_setup();
  tl_hmx_setup();
  // Require BOTH the VTCM grant AND a successful per-thread HMX enable (which
  // needs the ctx from tl_hmx_setup): otherwise tear everything down so
  // tl_hmx_session_ok() reports failure loudly, rather than marking the session
  // ready while enable was a silent no-op (ctx==0) and every later MAC runs
  // with HMX never locked.
  if (tl_vtcm_base_ptr && tl_hmx_enable() == 0) {
    tl_hmx_session_tid =
        qurt_thread_get_id(); // this enable is bound to this thread
    tl_hmx_inited = 1;
  } else {
    // VTCM acquire failed: release the HMX ctx and power HMX down so a failed
    // init doesn't leak the unit for the agent's lifetime.
    tl_hmx_reset();
    tl_hmx_vtcm_reset();
    tl_power_hmx(false);
  }
}
static void tl_hmx_session_deinit(void) {
  if (!tl_hmx_inited)
    return;
  tl_hmx_disable();
  tl_hmx_reset();
  tl_hmx_vtcm_reset();
  tl_power_hmx(false);
  tl_hmx_inited = 0;
}

// Let the FastRPC runtime reject execution if VTCM/HMX acquisition failed.
static int tl_hmx_session_ok(void) { return tl_hmx_inited; }

// tl_parallel_for worker scope for tile programs that issue HMX: a pool thread
// enables HMX for itself around its programs. The session thread already holds
// the enable that tl_hmx_session_init took.
struct tl_hmx_worker_scope {
  int status = TL_OK;
  bool enabled = false;

  tl_hmx_worker_scope() {
    if (qurt_thread_get_id() == tl_hmx_session_tid)
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

// ---- explicit HMX atoms: the DSL owns the K-loop and Crouton storage
// --------- These thin wrappers expose the accumulator protocol one instruction
// at a time. A DSL kernel decides where dequantization sits relative to
// clear/MAC/convert/ store and owns every Crouton buffer through alloc_shared.
// TILE = 32; one tile is 1024 fp16 = 2 KB. On one Q6 thread, a source tile is
// reusable after its final multiply packet; accumulator convert and output
// store do not access A/W. `tl_hexagon_hmx` prefix so runtime/fastrpc.py pulls
// in this header.
//
// The atom wrappers retain the dependency-only TIR state tokens as C
// parameters. Hardware state remains implicit; the tokens only expose ordering
// and ownership to compiler analyses.
TL_HMX_INLINE void tl_hexagon_hmx_acc_acquire(void *acc_state) {
  (void)acc_state;
  tl_hmx_unit_acquire();
}
TL_HMX_INLINE void tl_hexagon_hmx_acc_release(void *acc_state) {
  (void)acc_state;
  tl_hmx_unit_release();
}
TL_HMX_INLINE void tl_hexagon_hmx_load_bias(void *bias_state,
                                            const void *bias_vtcm) {
  (void)bias_state;
  tl_hmx_set_scales(bias_vtcm);
}
TL_HMX_INLINE void tl_hexagon_hmx_clear_acc(void *acc_state) {
  (void)acc_state;
  tl_hmx_clear_acc();
}
TL_HMX_INLINE void tl_hexagon_hmx_mma_atom(void *acc_state,
                                           const __fp16 *activation,
                                           const __fp16 *weight, size_t n = 1) {
  (void)acc_state;
  tl_hmx_mac_tiles(activation, weight, n);
}
TL_HMX_INLINE void tl_hexagon_hmx_convert_acc(void *cvt_state, void *acc_state,
                                              void *bias_state,
                                              const void *bias_source,
                                              int mode) {
  (void)cvt_state;
  (void)acc_state;
  (void)bias_state;
  // The source address is carried through TIR until conversion so the shared
  // memory planner cannot recycle it while mxmem2 is still in flight.
  (void)bias_source;
  tl_hmx_convert_acc(mode);
}
TL_HMX_INLINE void
tl_hexagon_hmx_store_cvt_state(void *cvt_state, __fp16 *output, void *acc_state,
                               void *bias_state, const void *bias_source) {
  (void)cvt_state;
  (void)acc_state;
  (void)bias_state;
  (void)bias_source;
  // Conversion is a separate TileLang atom. This instruction only transfers
  // the explicit HMX convert-state token to its Crouton VTCM destination.
  tl_hmx_store_cvt(output);
}
