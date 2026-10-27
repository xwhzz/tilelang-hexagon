#pragma once
// HW-thread worker pool for tilelang Hexagon kernels.
//
// The Hexagon DSP has a few HW threads, each able to issue HVX (and,
// time-shared, HMX) instructions.  tl_parallel runs a callback on `nw` workers:
// worker 0 on the CALLING thread, 1..nw-1 on a PERSISTENT pool of qurt threads.
// Following the htp-ops-lib pattern we NEVER qurt_hvx_lock — we never spawn
// more HVX-using workers than there are HVX contexts (= HW threads), so QuRT
// assigns each an HVX context implicitly and they're never oversubscribed.
//
// The pool is spawned ONCE (lazily, on the first dispatch — FastRPC _run is
// serial, so no init race) and the workers BLOCK on a per-worker semaphore when
// idle (no CPU, no HVX held), so the per-dispatch cost is a semaphore up/down,
// not a thread spawn+join (which dominated tiny ops).  Big buffers live in
// VTCM, so the per-worker stack stays small.  The pool lives for the FastRPC
// session and is reclaimed when the PD unloads.
//
// Generated kernels reach the pool through tl_parallel_for, the form of a
// parallel grid loop: one call runs every tile program of the grid.
//
// Compiled into the FastRPC skel with qurt.h on the include path.
#include <qurt.h>
#include <stdint.h>
#include <stdlib.h>
#include <tl_templates/hexagon/common.h>
#include <tl_templates/hexagon/vtcm.h>
#include <type_traits>

#define TL_MAX_WORKERS 6
#define TL_WORKER_STACK_SZ (2 * 16384) // 32 KiB/worker (htp-ops-lib default)

// Worker callback: (ctx, worker id in [0,nw), total worker count nw).  Returns
// a status code (0 = ok); tl_parallel returns the code of the lowest-numbered
// failing worker, so a per-worker device failure (e.g. HMX enable) propagates
// to the kernel entry and the host.
typedef int (*tl_worker_fn)(void *ctx, int wid, int nw);

// Number of HW threads available, clamped to [1, TL_MAX_WORKERS].  This is the
// worker count: == HVX context count on the target, so HVX is never
// oversubscribed.
static inline int tl_num_workers(void) {
  qurt_sysenv_max_hthreads_t h;
  if (qurt_sysenv_get_max_hw_threads(&h) != QURT_EOK)
    return 1;
  int n = (int)h.max_hthreads;
  if (n > TL_MAX_WORKERS)
    n = TL_MAX_WORKERS;
  if (n < 1)
    n = 1;
  return n;
}

// Persistent pool: workers 1..TL_MAX_WORKERS-1 are spawned once; each blocks on
// its `go` semaphore until the dispatcher releases it for a job, runs the
// shared (fn,ctx,nw), then `up`s `done`.  The dispatcher is the ONLY writer of
// fn/ctx/nw and writes them only between a worker's previous completion and its
// next release, so no extra lock is needed: qurt_sem_up/down are out-of-line
// futex calls (opaque compiler barriers + kernel-side ordering), so the writes
// happen-before the worker's reads. (Do NOT replace them with an inlined
// lockless path without an explicit barrier.)
static struct {
  qurt_sem_t
      go[TL_MAX_WORKERS]; // dispatcher ups go[w]; worker w blocks on down
  qurt_sem_t done; // worker ups; dispatcher downs once per released worker
  tl_worker_fn fn;
  void *ctx;
  int nw;
  int spawned; // highest worker id spawned (0 = none)
  int tried;   // spawn attempted (so a failed spawn isn't retried/leaked)
  qurt_thread_t th[TL_MAX_WORKERS];
  void *blob;                 // one stack blob for the spawned workers
  int status[TL_MAX_WORKERS]; // worker w's return; dispatcher reads it after
                              // the join
} tl_pool;

static void tl_pool_worker(void *arg) {
  int wid = (int)(intptr_t)arg; // 1..spawned, fixed for this thread's life
  for (;;) {
    qurt_sem_down(&tl_pool.go[wid]); // BLOCK until released for a job
    // Publish status[wid] BEFORE the done-up: the dispatcher reads it only
    // after the matching done-down, so the up/down pair orders this write to
    // its read.
    tl_pool.status[wid] = tl_pool.fn(tl_pool.ctx, wid, tl_pool.nw);
    qurt_sem_up(&tl_pool.done);
  }
}

// Spawn the pool once.  Single-threaded (FastRPC _run is serial).  On a spawn
// failure we stop; tl_parallel runs the un-spawned workers' slices inline.
static void tl_pool_spawn(void) {
  tl_pool.tried =
      1; // attempt once; a failed spawn must not be retried (would leak)
  int n = TL_MAX_WORKERS - 1; // worker 0 is always the caller
  tl_pool.blob = malloc((size_t)TL_WORKER_STACK_SZ * n);
  if (!tl_pool.blob)
    return; // spawned stays 0 -> tl_parallel falls back to serial inline
  qurt_sem_init_val(&tl_pool.done, 0);
  for (int w = 1; w <= n; ++w)
    qurt_sem_init_val(&tl_pool.go[w],
                      0); // init ALL go[] before any create, so a
                          // partial-spawn failure leaves none undefined
  int prio = qurt_thread_get_priority(qurt_thread_get_id());
  for (int w = 1; w <= n; ++w) {
    qurt_thread_attr_t attr;
    qurt_thread_attr_init(&attr);
    qurt_thread_attr_set_stack_addr(
        &attr, (char *)tl_pool.blob + (size_t)TL_WORKER_STACK_SZ * (w - 1));
    qurt_thread_attr_set_stack_size(&attr, TL_WORKER_STACK_SZ);
    qurt_thread_attr_set_priority(&attr, prio);
    if (qurt_thread_create(&tl_pool.th[w], &attr, tl_pool_worker,
                           (void *)(intptr_t)w) != QURT_EOK)
      break; // spawned stays at the last success
    tl_pool.spawned = w;
  }
  if (tl_pool.spawned ==
      0) { // not one worker started -> free the blob, run serial
    free(tl_pool.blob);
    tl_pool.blob = 0;
  }
}

// Run `fn` on `nw` workers and block until all finish.  Worker 0 runs inline on
// the calling thread; 1..nw-1 go to the persistent pool (any beyond the spawned
// count run inline too).  Falls back to fully serial if the pool couldn't be
// created.
static inline int tl_parallel(tl_worker_fn fn, void *ctx, int nw) {
  if (nw < 1)
    nw = 1;
  if (nw > TL_MAX_WORKERS)
    nw = TL_MAX_WORKERS;
  if (nw == 1)
    return fn(ctx, 0, 1);
  if (!tl_pool.tried)
    tl_pool_spawn(); // lazy, once per session (tried set even on failure, no
                     // retry)
  int pooled = nw - 1;
  if (pooled > tl_pool.spawned)
    pooled = tl_pool.spawned;
  tl_pool.fn = fn;
  tl_pool.ctx = ctx;
  tl_pool.nw = nw;
  for (int w = 1; w <= pooled; ++w)
    qurt_sem_up(&tl_pool.go[w]); // release the pooled workers
  int status[TL_MAX_WORKERS];
  status[0] = fn(ctx, 0, nw); // worker 0 on the calling thread
  for (int w = pooled + 1; w < nw; ++w)
    status[w] = fn(ctx, w, nw); // overflow (fewer spawned than requested)
  for (int w = 1; w <= pooled; ++w)
    qurt_sem_down(&tl_pool.done); // wait for the pooled workers
  for (int w = 1; w <= pooled; ++w)
    status[w] = tl_pool.status[w]; // the join established happens-before
  for (int w = 0; w < nw; ++w) {
    if (status[w] != 0)
      return status[w];
  }
  return 0;
}

// --- parallel grid loops ---------------------------------------------------
// tl_parallel_for(extent, max_workers, body) runs body(i) for every tile
// program i in [0, extent) on up to min(max_workers, tl_num_workers(), extent)
// workers. Worker w runs i = w, w + nw, ... in order and stops at its first
// nonzero status. Given per-worker arenas, body(i, arena) also receives its
// worker's private VTCM arena, and no more workers run than arenas fit.
//
// `Scope` holds per-worker resources: every worker constructs one on its own
// thread before its first program and destroys it after its last. A scope
// whose status is nonzero after construction fails the worker.
struct tl_worker_scope {
  int status = TL_OK;
};

// The kernel arena split into `count` per-worker arenas of `bytes` each.
struct tl_worker_arenas {
  uint8_t *base;
  unsigned int bytes;
  unsigned int count;
};

// Per-worker arenas for a tile program whose arena takes `bytes`. Each starts
// on a whole HMX tile, like the region itself, so that it keeps the alignment
// the program's buffers were laid out with.
static inline tl_worker_arenas tl_vtcm_worker_arenas(unsigned int bytes) {
  bytes = (bytes + TL_VTCM_HMX_ALIGNMENT - 1u) & ~(TL_VTCM_HMX_ALIGNMENT - 1u);
  uint8_t *base = tl_vtcm_arena();
  unsigned int count = base && bytes ? tl_vtcm_arena_bytes() / bytes : 0u;
  return {base, bytes, count};
}

template <typename Scope, typename Body> struct tl_parallel_for_job {
  Body *body;
  int32_t extent;
  tl_worker_arenas arenas;

  static int run(void *ctx, int wid, int nw) {
    tl_parallel_for_job *job = static_cast<tl_parallel_for_job *>(ctx);
    Scope scope;
    if (scope.status != TL_OK)
      return scope.status;
    uint8_t *arena = job->arenas.base + (size_t)wid * job->arenas.bytes;
    for (int32_t i = wid; i < job->extent; i += nw) {
      if (int status = (*job->body)(i, arena))
        return status;
    }
    return TL_OK;
  }
};

template <typename Scope = tl_worker_scope, typename Body>
static inline int tl_parallel_for(int32_t extent, int max_workers,
                                  tl_worker_arenas arenas, Body &&body) {
  if (extent <= 0)
    return TL_OK;
  if (arenas.count < 1)
    return TL_ERR_VTCM; // not even one worker's arena fits the VTCM grant
  int nw = tl_num_workers();
  if (nw > max_workers)
    nw = max_workers;
  if ((unsigned int)nw > arenas.count)
    nw = (int)arenas.count;
  if (nw > extent)
    nw = (int)extent;
  tl_parallel_for_job<Scope, typename std::remove_reference<Body>::type> job{
      &body, extent, arenas};
  return tl_parallel(job.run, &job, nw);
}

template <typename Scope = tl_worker_scope, typename Body>
static inline int tl_parallel_for(int32_t extent, int max_workers,
                                  Body &&body) {
  auto program = [&body](int32_t i, uint8_t *) { return body(i); };
  return tl_parallel_for<Scope>(extent, max_workers,
                                tl_worker_arenas{nullptr, 0u, TL_MAX_WORKERS},
                                program);
}

// --- threading selftest: proves dispatch + worker count run on-device, in
// isolation from VTCM/HMX.  Each worker stamps its own output slot; the caller
// also writes the worker count to the last slot.  out must hold >= nw+1 floats.
typedef struct {
  float *out;
  int n;
} tl_worker_selftest_ctx_t;
static int tl_worker_selftest_body(void *ctx, int wid, int nw) {
  tl_worker_selftest_ctx_t *c = (tl_worker_selftest_ctx_t *)ctx;
  if (wid < c->n)
    c->out[wid] = (float)((wid + 1) * 100 + nw); // distinct slot per worker
  return 0;
}
// out[0..nw-1] = (wid+1)*100+nw (each by a distinct worker); out[n-1] = nw; the
// remaining slots stay -1, so the host can verify exactly nw workers ran.
static inline int tl_hexagon_worker_selftest(float *out, int n) {
  for (int i = 0; i < n; ++i)
    out[i] = -1.0f;
  int nw = tl_num_workers();
  tl_worker_selftest_ctx_t c = {out, n};
  tl_parallel(tl_worker_selftest_body, &c, nw);
  if (n > 0)
    out[n - 1] = (float)nw;
  return 0;
}
