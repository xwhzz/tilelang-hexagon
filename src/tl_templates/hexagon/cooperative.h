#pragma once
// Explicit, whole-group resource checkpoints. Only kernels opted into this
// protocol call these helpers; ordinary worker grids retain their own barrier.
#include <hexagon_types.h>
#include <tl_templates/hexagon/dma.h>
#include <tl_templates/hexagon/resource.h>
#include <tl_templates/hexagon/session_state.h>
#include <tl_templates/hexagon/worker.h>

enum tl_cooperative_decision { TL_CP_WAIT, TL_CP_COMMITTED, TL_CP_ABORTED };
struct tl_cooperative_slot {
  unsigned epoch, state, yield, requests;
};
struct tl_cooperative_group {
  tl_cooperative_run *run;
  unsigned nw, arrived, phase_arrived, phase;
  unsigned passed[TL_MAX_WORKERS];
  tl_cooperative_slot slots[2];
};
// Dispatch is serial. This stack-owned group remains live until the persistent
// pool has joined every worker, including workers returning an error.
TL_HEXAGON_SESSION_STATE tl_cooperative_group *tl_cooperative_active;

static inline void tl_cooperative_pause() { asm volatile("pause(#8)"); }
static inline bool tl_cooperative_safe(tl_cooperative_run *run) {
  return !__atomic_load_n(&run->unsafe, __ATOMIC_ACQUIRE);
}
static inline void tl_cooperative_barrier(tl_cooperative_group *group) {
  unsigned phase = __atomic_load_n(&group->phase, __ATOMIC_ACQUIRE);
  __atomic_thread_fence(__ATOMIC_SEQ_CST);
  if (__atomic_add_fetch(&group->phase_arrived, 1u, __ATOMIC_ACQ_REL) ==
      group->nw) {
    __atomic_store_n(&group->phase_arrived, 0u, __ATOMIC_RELAXED);
    __atomic_store_n(&group->phase, phase + 1u, __ATOMIC_RELEASE);
  } else {
    while (__atomic_load_n(&group->phase, __ATOMIC_ACQUIRE) == phase)
      tl_cooperative_pause();
  }
}

// Divide complete HVX vectors, not arbitrary bytes: both endpoints of every
// stripe remain aligned. The final worker also copies the sub-vector tail.
// All live VTCM pointers escape in codegen before a checkpoint can run.
static inline void tl_cooperative_copy_stripe(tl_cooperative_group *group,
                                              unsigned wid, bool restore) {
  auto *run = group->run;
  size_t vectors = run->bytes / 128u;
  size_t begin = vectors * wid / group->nw;
  size_t end = vectors * (wid + 1u) / group->nw;
  uint8_t *dst = restore ? run->region : run->shadow;
  const uint8_t *src = restore ? run->shadow : run->region;
  asm volatile("" : : "r"(dst), "r"(src) : "memory");
  for (size_t i = begin; i < end; ++i) {
#ifdef __hexagon__
    reinterpret_cast<HVX_Vector *>(dst)[i] =
        reinterpret_cast<const HVX_Vector *>(src)[i];
#else
    for (size_t j = 0; j < 128u; ++j)
      dst[i * 128u + j] = src[i * 128u + j];
#endif
  }
  if (wid + 1u == group->nw)
    for (size_t i = vectors * 128u; i < run->bytes; ++i)
      dst[i] = src[i];
  __atomic_thread_fence(__ATOMIC_SEQ_CST);
  asm volatile("" : : "r"(dst), "r"(src) : "memory");
}

struct tl_cooperative_no_dma {
  int wait(unsigned) { return TL_HEXAGON_DMA_OK; }
  bool quiescent() const { return tl_hexagon_dma_engine_idle(); }
};

template <typename DMA>
static inline int tl_cooperative_yield(tl_cooperative_group *group,
                                       unsigned wid, unsigned requests,
                                       DMA &dma) {
  auto *run = group->run;
  // All COMMITTED members execute this same barrier sequence, even after a
  // concurrent error. No partial group may leave and strand a peer here.
  if (dma.wait(0) != TL_HEXAGON_DMA_OK || !dma.quiescent())
    tl_resource_fail(run, TL_ERR_DMA, true);
  tl_cooperative_barrier(group);
  if (!tl_resource_error(run))
    tl_cooperative_copy_stripe(group, wid, false);
  tl_cooperative_barrier(group);
  for (unsigned w = 0; w < group->nw; ++w) {
    if (wid == w && tl_cooperative_safe(run))
      tl_resource_worker_unlock(run, wid);
    tl_cooperative_barrier(group);
  }
  if (!wid && !tl_resource_error(run))
    tl_resource_owner_yield(run, requests);
  tl_cooperative_barrier(group);
  if (!tl_resource_error(run))
    tl_cooperative_copy_stripe(group, wid, true);
  tl_cooperative_barrier(group);
  for (unsigned w = 0; w < group->nw; ++w) {
    if (wid == w && !tl_resource_error(run))
      tl_resource_worker_relock(run, wid);
    tl_cooperative_barrier(group);
  }
  return tl_resource_error(run);
}

template <typename DMA>
static inline int tl_cooperative_checkpoint(int32_t worker, DMA &dma) {
  auto *group = tl_cooperative_active;
  auto *run = tl_resource_current();
  if (!group || group->run != run || worker < 0 ||
      static_cast<unsigned>(worker) >= group->nw)
    return run ? tl_resource_fail(run, TL_ERR_GRID) : TL_ERR_GRID;
  unsigned wid = static_cast<unsigned>(worker);
  // Refuse generation wrap before it could alias a live slot.
  if (group->passed[wid] >= UINT32_MAX - 1u)
    return tl_resource_fail(run, TL_ERR_GRID);
  unsigned epoch = ++group->passed[wid];
  auto &slot = group->slots[epoch & 1u];
  if (slot.epoch != epoch)
    return tl_resource_fail(run, TL_ERR_GRID);
  __atomic_thread_fence(__ATOMIC_SEQ_CST);
  if (__atomic_add_fetch(&group->arrived, 1u, __ATOMIC_ACQ_REL) == group->nw) {
    __atomic_store_n(&group->arrived, 0u, __ATOMIC_RELAXED);
    // Arrival at g proves every member has consumed g-1. Initialize g+1's
    // opposite slot before publishing this decision, so even a slow reader
    // cannot race a slot initializer or be overtaken by two generations.
    group->slots[(epoch + 1u) & 1u] = {epoch + 1u, TL_CP_WAIT, 0u, 0u};
    slot.requests = tl_resource_request_snapshot(run);
    slot.yield = tl_resource_request_pending(run);
    unsigned expected = TL_CP_WAIT;
    __atomic_compare_exchange_n(&slot.state, &expected,
                                tl_resource_error(run) ? TL_CP_ABORTED
                                                       : TL_CP_COMMITTED,
                                false, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE);
  }
  unsigned state;
  while ((state = __atomic_load_n(&slot.state, __ATOMIC_ACQUIRE)) ==
         TL_CP_WAIT) {
    if (tl_resource_error(run)) {
      unsigned expected = TL_CP_WAIT;
      __atomic_compare_exchange_n(&slot.state, &expected, TL_CP_ABORTED, false,
                                  __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE);
    }
    tl_cooperative_pause();
  }
  if (state != TL_CP_COMMITTED)
    return tl_resource_error(run) ? tl_resource_error(run)
                                  : tl_resource_fail(run, TL_ERR_GRID);
  // Once COMMITTED, an error cannot change participation in a positive yield.
  if (slot.yield)
    return tl_cooperative_yield(group, wid, slot.requests, dma);
  return tl_resource_error(run);
}

static inline int tl_cooperative_checkpoint(int32_t worker) {
  tl_cooperative_no_dma dma;
  return tl_cooperative_checkpoint(worker, dma);
}

template <typename DMA> struct tl_cooperative_dma_scope {
  DMA &dma;
  tl_cooperative_run *run;
  explicit tl_cooperative_dma_scope(DMA &context, int32_t)
      : dma(context), run(tl_resource_current()) {}
  ~tl_cooperative_dma_scope() {
    // This precedes the legacy DMA destructor. An eventual destructor reset
    // must never turn a failed checked drain into permission to release VTCM.
    if (run && tl_cooperative_safe(run) &&
        (dma.finish() != TL_HEXAGON_DMA_OK || !dma.quiescent()))
      tl_resource_fail(run, TL_ERR_DMA, true);
  }
  tl_cooperative_dma_scope(const tl_cooperative_dma_scope &) = delete;
  tl_cooperative_dma_scope &
  operator=(const tl_cooperative_dma_scope &) = delete;
};

template <typename Body> struct tl_cooperative_job {
  Body *body;
  tl_worker_arenas arenas;
  tl_cooperative_group *group;
  static int run(void *opaque, int wid, int) {
    auto *job = static_cast<tl_cooperative_job *>(opaque);
    auto *group = job->group;
    auto *resource = group->run;
    for (unsigned w = 0; w < group->nw; ++w) {
      if (static_cast<unsigned>(wid) == w)
        tl_resource_worker_enter(resource, wid);
      tl_cooperative_barrier(group);
    }
    if (!tl_hexagon_dma_engine_idle())
      tl_resource_fail(resource, TL_ERR_DMA, true);
    tl_cooperative_barrier(group);
    if (!tl_resource_error(resource)) {
      uint8_t *arena = job->arenas.base
                           ? job->arenas.base + size_t(wid) * job->arenas.bytes
                           : nullptr;
      int rc = (*job->body)(wid, arena);
      if (rc)
        tl_resource_fail(resource, rc);
    }
    if (!tl_hexagon_dma_engine_idle())
      tl_resource_fail(resource, TL_ERR_DMA, true);
    // Every body and its checked DMA scope have returned before any terminal
    // unlock. Peers still in a body barrier observe the sticky resource error.
    tl_cooperative_barrier(group);
    for (unsigned w = 0; w < group->nw; ++w) {
      if (static_cast<unsigned>(wid) == w && tl_cooperative_safe(resource))
        tl_resource_worker_unlock(resource, wid);
      tl_cooperative_barrier(group);
    }
    tl_resource_worker_terminal(resource, wid);
    tl_cooperative_barrier(group);
    return tl_resource_error(resource);
  }
};

template <bool UsesHMX, typename Body>
static inline int tl_cooperative_parallel_for(int32_t extent, int max_workers,
                                              tl_worker_arenas arenas,
                                              Body &&body) {
  auto *run = tl_resource_current();
  if (!run)
    return TL_ERR_RESOURCE;
  int rc = tl_worker_require_coresident(extent, max_workers, arenas);
  if (rc)
    return tl_resource_fail(run, rc);
  uintptr_t region = reinterpret_cast<uintptr_t>(run->region);
  uintptr_t base = reinterpret_cast<uintptr_t>(arenas.base);
  if (tl_cooperative_active || !region || !run->shadow || !run->bytes ||
      ((region | reinterpret_cast<uintptr_t>(run->shadow)) & 127u) ||
      (base &&
       (base < region || base - region > run->bytes ||
        uint64_t(arenas.bytes) * extent > run->bytes - (base - region))) ||
      (!base && arenas.bytes))
    return tl_resource_fail(run, TL_ERR_VTCM);
  rc = tl_resource_group_admit(run, static_cast<unsigned>(extent), UsesHMX);
  if (rc)
    return rc;
  tl_cooperative_group group{};
  group.run = run;
  group.nw = static_cast<unsigned>(extent);
  group.slots[1] = {1u, TL_CP_WAIT, 0u, 0u};
  group.slots[0] = {2u, TL_CP_WAIT, 0u, 0u};
  tl_cooperative_active = &group;
  using BodyType = typename std::remove_reference<Body>::type;
  tl_cooperative_job<BodyType> job{&body, arenas, &group};
  rc = tl_parallel(&tl_cooperative_job<BodyType>::run, &job, extent, true);
  tl_cooperative_active = nullptr;
  return rc;
}

template <bool UsesHMX, typename Body>
static inline int tl_cooperative_parallel_for(int32_t extent, int max_workers,
                                              Body &&body) {
  auto with_arena = [&](int32_t wid, uint8_t *) { return body(wid); };
  return tl_cooperative_parallel_for<UsesHMX>(
      extent, max_workers, {nullptr, 0u, TL_MAX_WORKERS}, with_arena);
}
