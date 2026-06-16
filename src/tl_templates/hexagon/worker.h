#pragma once
// Minimal HW-thread worker pool for tilelang Hexagon kernels.
//
// The Hexagon DSP has a few HW threads, each able to issue HVX (and, time-shared,
// HMX) instructions.  This spawns up to TL_MAX_WORKERS qurt threads to run a
// callback in parallel (worker 0 on the *calling* thread, 1..nw-1 spawned), then
// joins.  Following the htp-ops-lib pattern: we NEVER qurt_hvx_lock — we simply
// never spawn more HVX-using workers than there are HVX contexts (= HW threads),
// so QuRT assigns each thread an HVX context implicitly and they are never
// oversubscribed.  Per-call spawn/join (no persistent pool) — simple; the spawn
// cost is tiny next to the compute and is excluded from the timed region in
// benchmarks.  Big buffers live in VTCM, so the per-worker stack stays small.
//
// Compiled into the FastRPC skel with qurt.h on the include path.
#include <qurt.h>
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

typedef struct {
  tl_worker_fn fn;
  void *ctx;
  int wid;
  int nw;
} tl_worker_arg_t;

static void tl_worker_entry(void *p) {
  tl_worker_arg_t *a = (tl_worker_arg_t *)p;
  a->fn(a->ctx, a->wid, a->nw);
}

// Run `fn` on `nw` workers and block until all finish.  Worker 0 runs inline on
// the calling thread; 1..nw-1 are spawned qurt threads sharing one stack blob.
// Falls back to a serial loop if nw<=1 or the stack allocation fails.
static inline int tl_parallel(tl_worker_fn fn, void *ctx, int nw) {
  if (nw < 1)
    nw = 1;
  if (nw > TL_MAX_WORKERS)
    nw = TL_MAX_WORKERS;
  if (nw == 1) {
    fn(ctx, 0, 1);
    return 0;
  }
  char *blob = (char *)malloc((size_t)TL_WORKER_STACK_SZ * (nw - 1));
  if (!blob) {
    for (int w = 0; w < nw; ++w)
      fn(ctx, w, nw);
    return 0;
  }
  qurt_thread_t th[TL_MAX_WORKERS];
  tl_worker_arg_t args[TL_MAX_WORKERS];
  int prio = qurt_thread_get_priority(qurt_thread_get_id());
  for (int w = 1; w < nw; ++w) {
    args[w].fn = fn;
    args[w].ctx = ctx;
    args[w].wid = w;
    args[w].nw = nw;
    qurt_thread_attr_t attr;
    qurt_thread_attr_init(&attr);
    qurt_thread_attr_set_stack_addr(&attr, blob + (size_t)TL_WORKER_STACK_SZ * (w - 1));
    qurt_thread_attr_set_stack_size(&attr, TL_WORKER_STACK_SZ);
    qurt_thread_attr_set_priority(&attr, prio);
    qurt_thread_create(&th[w], &attr, tl_worker_entry, &args[w]);
  }
  fn(ctx, 0, nw); // worker 0 on the calling thread
  for (int w = 1; w < nw; ++w) {
    int status;
    qurt_thread_join(th[w], &status);
  }
  free(blob);
  return 0;
}

// --- threading selftest: proves spawn/join + worker count run on-device, in
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
