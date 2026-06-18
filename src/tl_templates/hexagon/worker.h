#pragma once
// HW-thread worker pool for tilelang Hexagon kernels.
//
// The Hexagon DSP has a few HW threads, each able to issue HVX (and, time-shared,
// HMX) instructions.  tl_parallel runs a callback on `nw` workers: worker 0 on the
// CALLING thread, 1..nw-1 on a PERSISTENT pool of qurt threads.  Following the
// htp-ops-lib pattern we NEVER qurt_hvx_lock — we never spawn more HVX-using workers
// than there are HVX contexts (= HW threads), so QuRT assigns each an HVX context
// implicitly and they're never oversubscribed.
//
// The pool is spawned ONCE (lazily, on the first dispatch — FastRPC _run is serial,
// so no init race) and the workers BLOCK on a per-worker semaphore when idle (no CPU,
// no HVX held), so the per-dispatch cost is a semaphore up/down, not a thread
// spawn+join (which dominated tiny ops).  Big buffers live in VTCM, so the per-worker
// stack stays small.  The pool lives for the FastRPC session and is reclaimed when
// the PD unloads.
//
// Compiled into the FastRPC skel with qurt.h on the include path.
#include <qurt.h>
#include <stdint.h>
#include <stdlib.h>

#define TL_MAX_WORKERS 6
#define TL_WORKER_STACK_SZ (2 * 16384) // 32 KiB/worker (htp-ops-lib default)

// Worker callback: (ctx, worker id in [0,nw), total worker count nw).
typedef void (*tl_worker_fn)(void *ctx, int wid, int nw);

// Number of HW threads available, clamped to [1, TL_MAX_WORKERS].  This is the
// worker count: == HVX context count on the target, so HVX is never oversubscribed.
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

// Persistent pool: workers 1..TL_MAX_WORKERS-1 are spawned once; each blocks on its
// `go` semaphore until the dispatcher releases it for a job, runs the shared
// (fn,ctx,nw), then `up`s `done`.  The dispatcher is the ONLY writer of fn/ctx/nw and
// writes them only between a worker's previous completion and its next release, so no
// extra lock is needed; the semaphores carry the memory ordering.
static struct {
  qurt_sem_t go[TL_MAX_WORKERS]; // dispatcher ups go[w]; worker w blocks on down
  qurt_sem_t done;               // worker ups; dispatcher downs once per released worker
  tl_worker_fn fn;
  void *ctx;
  int nw;
  int spawned;                   // highest worker id spawned (0 = pool not yet created)
  qurt_thread_t th[TL_MAX_WORKERS];
  void *blob;                    // one stack blob for the spawned workers
} tl_pool;

static void tl_pool_worker(void *arg) {
  int wid = (int)(intptr_t)arg; // 1..spawned, fixed for this thread's life
  for (;;) {
    qurt_sem_down(&tl_pool.go[wid]); // BLOCK until released for a job
    tl_pool.fn(tl_pool.ctx, wid, tl_pool.nw);
    qurt_sem_up(&tl_pool.done);
  }
}

// Spawn the pool once.  Single-threaded (FastRPC _run is serial).  On a spawn
// failure we stop; tl_parallel runs the un-spawned workers' slices inline.
static void tl_pool_spawn(void) {
  int n = TL_MAX_WORKERS - 1; // worker 0 is always the caller
  tl_pool.blob = malloc((size_t)TL_WORKER_STACK_SZ * n);
  if (!tl_pool.blob)
    return; // spawned stays 0 -> tl_parallel falls back to serial inline
  qurt_sem_init_val(&tl_pool.done, 0);
  int prio = qurt_thread_get_priority(qurt_thread_get_id());
  for (int w = 1; w <= n; ++w) {
    qurt_sem_init_val(&tl_pool.go[w], 0);
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
}

// Run `fn` on `nw` workers and block until all finish.  Worker 0 runs inline on the
// calling thread; 1..nw-1 go to the persistent pool (any beyond the spawned count
// run inline too).  Falls back to fully serial if the pool couldn't be created.
static inline int tl_parallel(tl_worker_fn fn, void *ctx, int nw) {
  if (nw < 1)
    nw = 1;
  if (nw > TL_MAX_WORKERS)
    nw = TL_MAX_WORKERS;
  if (nw == 1) {
    fn(ctx, 0, 1);
    return 0;
  }
  if (tl_pool.spawned == 0)
    tl_pool_spawn(); // lazy, once per session
  int pooled = nw - 1;
  if (pooled > tl_pool.spawned)
    pooled = tl_pool.spawned;
  tl_pool.fn = fn;
  tl_pool.ctx = ctx;
  tl_pool.nw = nw;
  for (int w = 1; w <= pooled; ++w)
    qurt_sem_up(&tl_pool.go[w]); // release the pooled workers
  fn(ctx, 0, nw);                // worker 0 on the calling thread
  for (int w = pooled + 1; w < nw; ++w)
    fn(ctx, w, nw); // overflow (fewer spawned than requested) runs inline
  for (int w = 1; w <= pooled; ++w)
    qurt_sem_down(&tl_pool.done); // wait for the pooled workers
  return 0;
}

// --- threading selftest: proves dispatch + worker count run on-device, in
// isolation from VTCM/HMX.  Each worker stamps its own output slot; the caller
// also writes the worker count to the last slot.  out must hold >= nw+1 floats.
typedef struct {
  float *out;
  int n;
} tl_worker_selftest_ctx_t;
static void tl_worker_selftest_body(void *ctx, int wid, int nw) {
  tl_worker_selftest_ctx_t *c = (tl_worker_selftest_ctx_t *)ctx;
  if (wid < c->n)
    c->out[wid] = (float)((wid + 1) * 100 + nw); // distinct slot per worker
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
