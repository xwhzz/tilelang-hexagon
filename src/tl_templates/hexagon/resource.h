#pragma once
// Checked compute-resource ownership for cooperative kernel Runs. This header
// deliberately contains no vector operations: preparation and the SDK callback
// execute before the caller owns an HVX/HMX worker context.
#include <HAP_perf.h>
#include <qurt.h>
#include <stdint.h>
#include <stdlib.h>
#include <tl_templates/hexagon/common.h>
#include <tl_templates/hexagon/vtcm.h>

struct tl_resource_worker {
  qurt_thread_t tid;
  unsigned entered;
  unsigned held; // 0 unheld, 1 held, 2 unknown
  unsigned terminal;
};
struct tl_cooperative_run {
  unsigned participants;
  int uses_hmx;
  int first_error;
  int unsafe;
  uint8_t *region;
  uint8_t *shadow;
  unsigned bytes;
  tl_resource_worker worker[QURT_MAX_HTHREAD_LIMIT];

  // Session lifetime fields. They are never memset at a Run boundary.
  void *shadow_allocation;
  unsigned busy;
  unsigned cooperative;
  unsigned gate; // epoch:23 | open:1 | admitted callbacks:8
  unsigned inflight;
  unsigned request_seq;
  unsigned served_seq;
  qurt_thread_t owner;
};
static tl_cooperative_run tl_resource_state;

static inline int tl_resource_error(tl_cooperative_run *run) {
  return __atomic_load_n(&run->first_error, __ATOMIC_ACQUIRE);
}
static inline int tl_resource_fail(tl_cooperative_run *run, int status,
                                   bool unsafe = false) {
  if (!status)
    status = TL_ERR_RESOURCE;
  int expected = TL_OK;
  __atomic_compare_exchange_n(&run->first_error, &expected, status, false,
                              __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE);
  if (unsafe) {
    __atomic_store_n(&run->unsafe, 1, __ATOMIC_RELEASE);
    __atomic_store_n(&tl_vtcm_poisoned, 1, __ATOMIC_RELEASE);
  }
  return tl_resource_error(run);
}
static inline tl_cooperative_run *tl_resource_current() {
  return __atomic_load_n(&tl_resource_state.busy, __ATOMIC_ACQUIRE) &&
                 tl_resource_state.cooperative
             ? &tl_resource_state
             : nullptr;
}

static int tl_resource_release_callback(unsigned context, void *opaque) {
  auto *run = static_cast<tl_cooperative_run *>(opaque);
  if (run != &tl_resource_state)
    return 0;
  __atomic_add_fetch(&run->inflight, 1u, __ATOMIC_ACQ_REL);
  unsigned gate = __atomic_load_n(&run->gate, __ATOMIC_ACQUIRE);
  const unsigned epoch = gate >> 9;
  bool admitted = false;
  while ((gate & 0x100u) && (gate >> 9) == epoch) {
    if ((gate & 0xffu) == 0xffu) {
      tl_resource_fail(run, TL_ERR_RESOURCE);
      break;
    }
    if (__atomic_compare_exchange_n(&run->gate, &gate, gate + 1u, false,
                                    __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) {
      admitted = true;
      break;
    }
  }
  if (admitted) {
    if (context == static_cast<unsigned>(tl_vtcm_ctx)) {
      unsigned seq = __atomic_load_n(&run->request_seq, __ATOMIC_ACQUIRE);
      for (;;) {
        if (seq == UINT32_MAX) {
          tl_resource_fail(run, TL_ERR_RESOURCE);
          break;
        }
        if (__atomic_compare_exchange_n(&run->request_seq, &seq, seq + 1u,
                                        false, __ATOMIC_ACQ_REL,
                                        __ATOMIC_ACQUIRE))
          break;
      }
    }
    __atomic_sub_fetch(&run->gate, 1u, __ATOMIC_RELEASE);
  }
  __atomic_sub_fetch(&run->inflight, 1u, __ATOMIC_RELEASE);
  return 0;
}

static inline int tl_resource_close_admission(tl_cooperative_run *run) {
  __atomic_fetch_and(&run->gate, ~0x100u, __ATOMIC_ACQ_REL);
  const uint64_t start = HAP_perf_get_qtimer_count();
  while ((__atomic_load_n(&run->gate, __ATOMIC_ACQUIRE) & 0xffu) ||
         __atomic_load_n(&run->inflight, __ATOMIC_ACQUIRE)) {
    // This is a diagnostic failure bound, never a callback-side wait or a
    // cancellation of an SDK acquire. Keep the session unusable on failure.
    if (HAP_perf_get_qtimer_count() - start > 384000u)
      return tl_resource_fail(run, TL_ERR_RESOURCE, true);
  }
  return TL_OK;
}
static inline int tl_resource_open_admission(tl_cooperative_run *run) {
  unsigned gate = __atomic_load_n(&run->gate, __ATOMIC_ACQUIRE);
  if ((gate & 0x1ffu) || (gate >> 9) == 0x7fffffu)
    return tl_resource_fail(run, TL_ERR_RESOURCE, true);
  __atomic_store_n(&run->gate, (((gate >> 9) + 1u) << 9) | 0x100u,
                   __ATOMIC_RELEASE);
  return TL_OK;
}
static inline unsigned tl_resource_request_snapshot(tl_cooperative_run *run) {
  return __atomic_load_n(&run->request_seq, __ATOMIC_ACQUIRE);
}
static inline bool tl_resource_request_pending(tl_cooperative_run *run) {
  return tl_resource_request_snapshot(run) != run->served_seq;
}

static inline int tl_resource_group_admit(tl_cooperative_run *run, unsigned nw,
                                          bool uses_hmx) {
  if (!run || run != tl_resource_current() || !nw ||
      nw > QURT_MAX_HTHREAD_LIMIT || !tl_vtcm_held ||
      (uses_hmx && !tl_vtcm_hmx_requested))
    return tl_resource_fail(&tl_resource_state, TL_ERR_RESOURCE);
  unsigned expected = 0;
  if (!__atomic_compare_exchange_n(&run->participants, &expected, nw, false,
                                   __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE))
    return tl_resource_fail(run, TL_ERR_RESOURCE);
  run->uses_hmx = uses_hmx;
  return tl_resource_error(run);
}
static inline int tl_resource_worker_relock(tl_cooperative_run *run,
                                            unsigned wid) {
  if (wid >= run->participants || !run->worker[wid].entered ||
      run->worker[wid].tid != qurt_thread_get_id() || run->worker[wid].held ||
      run->worker[wid].terminal || !tl_vtcm_held ||
      __atomic_load_n(&run->unsafe, __ATOMIC_ACQUIRE))
    return tl_resource_fail(run, TL_ERR_RESOURCE, true);
  if (!run->uses_hmx)
    return TL_OK;
  if (HAP_compute_res_hmx_lock2(tl_vtcm_ctx, HAP_COMPUTE_RES_HMX_SHARED)) {
    run->worker[wid].held = 2;
    return tl_resource_fail(run, TL_ERR_HMX, true);
  }
  run->worker[wid].held = 1;
  return TL_OK;
}
static inline int tl_resource_worker_enter(tl_cooperative_run *run,
                                           unsigned wid) {
  if (wid >= run->participants)
    return tl_resource_fail(run, TL_ERR_RESOURCE);
  auto &worker = run->worker[wid];
  unsigned expected = 0;
  if (!__atomic_compare_exchange_n(&worker.entered, &expected, 1u, false,
                                   __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE))
    return tl_resource_fail(run, TL_ERR_RESOURCE, true);
  worker.tid = qurt_thread_get_id();
  if (tl_resource_error(run))
    return tl_resource_error(run);
  return tl_resource_worker_relock(run, wid);
}
static inline int tl_resource_worker_unlock(tl_cooperative_run *run,
                                            unsigned wid) {
  if (wid >= run->participants || !run->worker[wid].entered ||
      run->worker[wid].tid != qurt_thread_get_id() ||
      __atomic_load_n(&run->unsafe, __ATOMIC_ACQUIRE) ||
      run->worker[wid].held == 2)
    return tl_resource_fail(run, TL_ERR_RESOURCE, true);
  if (!run->worker[wid].held)
    return TL_OK;
  if (HAP_compute_res_hmx_unlock2(tl_vtcm_ctx, HAP_COMPUTE_RES_HMX_SHARED)) {
    run->worker[wid].held = 2;
    return tl_resource_fail(run, TL_ERR_HMX, true);
  }
  run->worker[wid].held = 0;
  return TL_OK;
}
static inline int tl_resource_worker_terminal(tl_cooperative_run *run,
                                              unsigned wid) {
  if (wid >= run->participants || !run->worker[wid].entered ||
      run->worker[wid].tid != qurt_thread_get_id() || run->worker[wid].held)
    return tl_resource_fail(run, TL_ERR_RESOURCE, true);
  unsigned expected = 0;
  if (!__atomic_compare_exchange_n(&run->worker[wid].terminal, &expected, 1u,
                                   false, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE))
    return tl_resource_fail(run, TL_ERR_RESOURCE, true);
  return tl_resource_error(run);
}
static inline int tl_resource_owner_yield(tl_cooperative_run *run,
                                          unsigned snapshot) {
  if (qurt_thread_get_id() != run->owner || tl_resource_error(run) ||
      __atomic_load_n(&run->unsafe, __ATOMIC_ACQUIRE) || !tl_vtcm_held ||
      snapshot < run->served_seq ||
      snapshot > tl_resource_request_snapshot(run))
    return tl_resource_fail(run, TL_ERR_RESOURCE);
  for (unsigned w = 0; w < run->participants; ++w)
    if (!run->worker[w].entered || run->worker[w].held)
      return tl_resource_fail(run, TL_ERR_RESOURCE, true);
  if (tl_vtcm_give())
    return tl_resource_fail(run, TL_ERR_VTCM, true);
  // Publish only the request snapshot selected before the collective. A later
  // callback remains pending even if this physical release also satisfied it.
  run->served_seq = snapshot;
  if (tl_vtcm_take())
    return tl_resource_fail(
        run, TL_ERR_VTCM,
        __atomic_load_n(&tl_vtcm_poisoned, __ATOMIC_ACQUIRE) != 0);
  if (tl_vtcm_base_ptr != run->region || tl_vtcm_total != run->bytes)
    return tl_resource_fail(run, TL_ERR_RESOURCE, true);
  return TL_OK;
}

static inline int tl_resource_close() {
  auto *run = &tl_resource_state;
  if (__atomic_load_n(&run->busy, __ATOMIC_ACQUIRE))
    return TL_ERR_RESOURCE;
  if (tl_resource_close_admission(run) ||
      __atomic_load_n(&run->unsafe, __ATOMIC_ACQUIRE) ||
      __atomic_load_n(&tl_vtcm_poisoned, __ATOMIC_ACQUIRE))
    return tl_resource_fail(run, TL_ERR_RESOURCE, true);
  if (tl_vtcm_close())
    return tl_resource_fail(run, TL_ERR_VTCM, true);
  free(run->shadow_allocation);
  run->shadow_allocation = nullptr;
  run->shadow = nullptr;
  run->region = nullptr;
  run->bytes = 0;
  return TL_OK;
}

// Called before the ordinary take/give path or instead of it for cooperative
// Runs. A successful prepare is paired with finish even when the body fails.
static inline int tl_resource_prepare(bool cooperative) {
  auto *run = &tl_resource_state;
  unsigned expected = 0;
  if (!__atomic_compare_exchange_n(&run->busy, &expected, 1u, false,
                                   __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE))
    return TL_ERR_RESOURCE;
  if (__atomic_load_n(&run->unsafe, __ATOMIC_ACQUIRE) ||
      __atomic_load_n(&tl_vtcm_poisoned, __ATOMIC_ACQUIRE) || tl_vtcm_held) {
    __atomic_store_n(&run->busy, 0u, __ATOMIC_RELEASE);
    return TL_ERR_RESOURCE;
  }
  run->first_error = TL_OK;
  run->cooperative = cooperative;
  run->owner = qurt_thread_get_id();
  run->participants = 0;
  run->uses_hmx = 0;
  run->request_seq = run->served_seq = 0;
  // Volatile scalar stores prevent a pre-take auto-vectorized memset.
  for (unsigned w = 0; w < QURT_MAX_HTHREAD_LIMIT; ++w) {
    volatile auto &worker = run->worker[w];
    worker.tid = 0;
    worker.entered = worker.held = worker.terminal = 0;
  }
  int rc = TL_OK;
  if (tl_vtcm_ctx && tl_vtcm_cooperative != cooperative) {
    if (tl_vtcm_close())
      rc = tl_resource_fail(run, TL_ERR_VTCM, true);
    else {
      free(run->shadow_allocation);
      run->shadow_allocation = nullptr;
      run->shadow = nullptr;
    }
  }
  if (!rc &&
      tl_vtcm_open_mode(tl_vtcm_hmx_requested,
                        cooperative ? tl_resource_release_callback : nullptr,
                        cooperative ? run : nullptr))
    rc = tl_resource_fail(
        run, TL_ERR_VTCM,
        __atomic_load_n(&tl_vtcm_poisoned, __ATOMIC_ACQUIRE) != 0);
  if (!rc && cooperative) {
    uintptr_t raw = reinterpret_cast<uintptr_t>(tl_vtcm_region);
    uintptr_t aligned = (raw + TL_VTCM_HMX_ALIGNMENT - 1u) &
                        ~(uintptr_t)(TL_VTCM_HMX_ALIGNMENT - 1u);
    unsigned padding = static_cast<unsigned>(aligned - raw);
    if (!raw || padding >= tl_vtcm_region_bytes)
      rc = tl_resource_fail(run, TL_ERR_VTCM);
    else {
      run->region = reinterpret_cast<uint8_t *>(aligned);
      run->bytes = tl_vtcm_region_bytes - padding;
      if (!run->shadow_allocation && run->bytes > SIZE_MAX - 127u)
        rc = tl_resource_fail(run, TL_ERR_VTCM);
      if (!rc && !run->shadow_allocation) {
        run->shadow_allocation = malloc(static_cast<size_t>(run->bytes) + 127u);
        if (!run->shadow_allocation)
          rc = tl_resource_fail(run, TL_ERR_VTCM);
        else
          run->shadow = reinterpret_cast<uint8_t *>(
              (reinterpret_cast<uintptr_t>(run->shadow_allocation) + 127u) &
              ~(uintptr_t)127u);
      }
    }
    if (!rc)
      rc = tl_resource_open_admission(run);
    if (!rc && tl_vtcm_take())
      rc = tl_resource_fail(
          run, TL_ERR_VTCM,
          __atomic_load_n(&tl_vtcm_poisoned, __ATOMIC_ACQUIRE) != 0);
  }
  if (rc) {
    tl_resource_close_admission(run);
    __atomic_store_n(&run->busy, 0u, __ATOMIC_RELEASE);
  }
  return rc;
}

static inline int tl_resource_finish(int status) {
  auto *run = &tl_resource_state;
  if (!__atomic_load_n(&run->busy, __ATOMIC_ACQUIRE) ||
      run->owner != qurt_thread_get_id())
    return TL_ERR_RESOURCE;
  if (status)
    tl_resource_fail(run, status);
  if (run->cooperative) {
    for (unsigned w = 0; w < run->participants; ++w)
      if (!run->worker[w].terminal || run->worker[w].held)
        tl_resource_fail(run, TL_ERR_RESOURCE, true);
    // Keep callback admission open through the final physical release.
    if (!__atomic_load_n(&run->unsafe, __ATOMIC_ACQUIRE) && tl_vtcm_give())
      tl_resource_fail(run, TL_ERR_VTCM, true);
  }
  tl_resource_close_admission(run);
  if (__atomic_load_n(&tl_vtcm_poisoned, __ATOMIC_ACQUIRE) || tl_vtcm_held)
    tl_resource_fail(run, TL_ERR_RESOURCE, true);
  // An ordinary worker scope may report a cleanup error while unwinding a
  // nonzero body return, before that body status reaches the wrapper. Preserve
  // the kernel's returned status; unsafe ownership remains independently
  // sticky. Cooperative jobs already publish and return their first observed
  // error.
  int result = !run->cooperative && status ? status : tl_resource_error(run);
  __atomic_store_n(&run->busy, 0u, __ATOMIC_RELEASE);
  return result;
}
