// Actual resource/vtcm headers; only SDK/thread/time providers are simulated.
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

enum {
  TL_OK = 0,
  TL_ERR_VTCM = 1,
  TL_ERR_HMX = 2,
  TL_ERR_DMA = 3,
  TL_ERR_GRID = 4,
  TL_ERR_RESOURCE = 5,
  QURT_MAX_HTHREAD_LIMIT = 8,
  HAP_COMPUTE_RES_HMX_SHARED = 1
};
using qurt_thread_t = unsigned;
static thread_local unsigned tid = 1;
static unsigned qurt_thread_get_id() { return tid; }
static bool fixed_clock;
static std::atomic<unsigned> hook_phase{0}, hook_reached{0};
static std::atomic<bool> hook_release{false};
static void resource_test_hook(unsigned phase) {
  if (tid == 2 && hook_phase.load() == phase) {
    hook_reached.store(phase);
    while (!hook_release.load())
      std::this_thread::yield();
  }
}
static uint64_t HAP_perf_get_qtimer_count() {
  if (fixed_clock)
    return 100;
  return std::chrono::steady_clock::now().time_since_epoch().count() * 192 /
         10000;
}
struct compute_res_attr_t {
  int (*callback)(unsigned, void *) = nullptr;
  void *opaque = nullptr;
};
using compute_res_vtcm_page_t = unsigned;
alignas(2048) static unsigned char arena[16384];
static unsigned granted = 8192, offset = 0, ctx = 17;
static unsigned opens, closes, takes, gives;
static std::atomic<unsigned> locks{0}, unlocks{0};
static bool live, held;
static int (*callback)(unsigned, void *);
static void *opaque;
static std::string failure;
static bool callback_on_release, callback_on_acquire;
static int HAP_compute_res_query_VTCM(int, unsigned *total, unsigned *,
                                      unsigned *, unsigned *) {
  *total = granted;
  return failure == "query" ? -1 : 0;
}
static int HAP_compute_res_attr_init(compute_res_attr_t *a) {
  *a = {};
  return 0;
}
static int HAP_compute_res_attr_set_vtcm_param(compute_res_attr_t *, unsigned n,
                                               int) {
  assert(n == granted);
  return failure == "attribute" ? -1 : 0;
}
static int HAP_compute_res_attr_set_hmx_param(compute_res_attr_t *, int) {
  return 0;
}
static int HAP_compute_res_attr_set_cache_mode(compute_res_attr_t *, int) {
  return 0;
}
static int HAP_compute_res_attr_set_release_callback(
    compute_res_attr_t *a, int (*cb)(unsigned, void *), void *p) {
  a->callback = cb;
  a->opaque = p;
  return failure == "callback-attribute" ? -1 : 0;
}
static unsigned HAP_compute_res_acquire(compute_res_attr_t *a, unsigned) {
  assert(!live && !held);
  if (failure == "open")
    return 0;
  callback = a->callback;
  opaque = a->opaque;
  live = true;
  ++opens;
  return ++ctx;
}
static void *HAP_compute_res_attr_get_vtcm_ptr(compute_res_attr_t *) {
  return failure == "bind" ? nullptr : arena + offset;
}
static int HAP_compute_res_acquire_cached(unsigned c, unsigned) {
  assert(live && c == ctx && !held);
  if (failure == "take")
    return -1;
  held = true;
  ++takes;
  if (callback_on_acquire && callback)
    callback(ctx, opaque);
  return 0;
}
static int HAP_compute_res_release_cached(unsigned c) {
  assert(live && c == ctx && held);
  if (failure == "give")
    return -1;
  if (callback_on_release && callback)
    callback(ctx, opaque);
  held = false;
  ++gives;
  return 0;
}
static int HAP_compute_res_release(unsigned c) {
  assert(live && c == ctx && !held);
  if (failure == "close")
    return -1;
  live = false;
  ++closes;
  callback = nullptr;
  return 0;
}
static int HAP_compute_res_hmx_lock2(unsigned c, int) {
  assert(live && held && c == ctx);
  if (failure == "lock")
    return -1;
  ++locks;
  return 0;
}
static int HAP_compute_res_hmx_unlock2(unsigned c, int) {
  assert(live && held && c == ctx);
  if (failure == "unlock")
    return -1;
  ++unlocks;
  return 0;
}
#include <tl_templates/hexagon/resource.h>
static bool hmx_power;
static void tl_power_hmx(bool enabled) { hmx_power = enabled; }
#include <tl_templates/hexagon/hmx_lifecycle.h>

static tl_cooperative_run *begin(unsigned nw = 1, bool hmx = true) {
  assert(tl_vtcm_open(hmx) == 0);
  assert(tl_resource_prepare(true) == 0);
  auto *r = tl_resource_current();
  assert(r && held);
  assert(tl_resource_group_admit(r, nw, hmx) == 0);
  return r;
}
static void end(tl_cooperative_run *r, int rc = 0) {
  assert(tl_resource_worker_unlock(r, 0) == 0);
  assert(tl_resource_worker_terminal(r, 0) == rc);
  assert(tl_resource_finish(rc) == rc);
  assert(!held);
  assert(tl_resource_close() == 0);
  assert(!live && opens == closes && takes == gives);
}
int main(int argc, char **argv) {
  assert(argc == 2);
  std::string mode = argv[1];
  if (mode == "dynamic-size") {
    granted = 12289;
    offset = 1;
  }
  if (mode == "ordinary-hmx" || mode == "ordinary-hmx-unlock-fail" ||
      mode == "ordinary-hmx-body-pool-fail" ||
      mode == "ordinary-hmx-body-give-fail") {
    tl_hmx_session_init();
    assert(tl_hmx_session_ok() && hmx_power);
    assert(tl_resource_prepare(false) == 0);
    assert(tl_hmx_take() == 0);
    {
      tl_hmx_worker_scope caller;
      assert(caller.status == 0);
      assert(locks == 1);
    }
    tid = 2;
    {
      tl_hmx_worker_scope worker;
      assert(worker.status == 0);
      assert(locks == 2);
      if (mode == "ordinary-hmx-unlock-fail" ||
          mode == "ordinary-hmx-body-pool-fail")
        failure = "unlock";
    }
    tid = 1;
    if (mode == "ordinary-hmx-body-give-fail") {
      tl_resource_fail(&tl_resource_state,
                       37); // Wrapper publishes before give.
      failure = "unlock";
    }
    if (mode != "ordinary-hmx") {
      int body_status = mode == "ordinary-hmx-unlock-fail" ? TL_OK : 37;
      assert(tl_hmx_give() == TL_ERR_HMX);
      assert(tl_resource_finish(body_status) ==
             (body_status ? body_status : TL_ERR_HMX));
      assert(tl_hmx_session_deinit() != 0);
      assert(hmx_power && held && live);
      return 0;
    }
    assert(tl_hmx_give() == 0);
    assert(locks == unlocks);
    assert(tl_resource_finish(0) == 0);
    assert(tl_hmx_session_deinit() == 0);
    assert(!hmx_power && !live);
    return 0;
  }
  if (mode == "four-runs") {
    assert(tl_vtcm_open(1) == 0);
    uint8_t *shadow = nullptr;
    unsigned previous_epoch = 0;
    for (unsigned call = 0; call < 4; ++call) {
      assert(tl_resource_prepare(true) == 0);
      auto *r = tl_resource_current();
      if (shadow)
        assert(shadow == r->shadow);
      shadow = r->shadow;
      assert(r->request_seq == 0 && r->served_seq == 0);
      assert((r->gate >> 9) > previous_epoch);
      previous_epoch = r->gate >> 9;
      assert(tl_resource_group_admit(r, 1, true) == 0);
      assert(tl_resource_worker_enter(r, 0) == 0);
      callback(ctx, opaque);
      assert(r->request_seq == 1);
      assert(tl_resource_worker_unlock(r, 0) == 0);
      assert(tl_resource_worker_terminal(r, 0) == 0);
      assert(tl_resource_finish(0) == 0);
    }
    assert(opens == 2 && closes == 1 && takes == 4 && gives == 4);
    assert(tl_resource_close() == 0);
    return 0;
  }
  if (mode == "ordinary-switch") {
    assert(tl_vtcm_open(1) == 0);
    for (bool coop : {false, false, true, true, false}) {
      assert(tl_resource_prepare(coop) == 0);
      if (coop) {
        auto *r = tl_resource_current();
        assert(tl_resource_group_admit(r, 1, false) == 0);
        assert(tl_resource_worker_enter(r, 0) == 0);
        assert(tl_resource_worker_terminal(r, 0) == 0);
      } else {
        assert(!callback);
        assert(tl_vtcm_take() == 0);
        assert(tl_vtcm_give() == 0);
      }
      assert(tl_resource_finish(0) == 0);
    }
    assert(opens == 3 && closes == 2);
    assert(tl_resource_close() == 0);
    return 0;
  }
  if (mode == "query" || mode == "attribute" || mode == "open") {
    failure = mode;
    assert(tl_resource_prepare(true) == TL_ERR_VTCM);
    assert(!held && !tl_resource_state.busy);
    assert(tl_resource_close() == 0);
    return 0;
  }
  if (mode == "take" || mode == "bind" || mode == "callback-attribute") {
    failure = mode;
    assert(tl_resource_prepare(true) == TL_ERR_VTCM);
    assert(!held && !tl_resource_state.busy);
    failure = "";
    assert(tl_resource_close() == 0);
    return 0;
  }
  auto *r = begin(mode == "parallel-workers" ? 8 : 1, mode != "no-hmx");
  if (mode == "dynamic-size") {
    assert(r->region == arena + 2048);
    assert(r->bytes == 10242);
    assert((uintptr_t(r->shadow) & 127) == 0);
  }
  if (mode == "parallel-workers") {
    std::vector<std::thread> t;
    for (unsigned w = 0; w < 8; ++w)
      t.emplace_back([=] {
        tid = w + 1;
        assert(tl_resource_worker_enter(r, w) == 0);
        assert(tl_resource_worker_unlock(r, w) == 0);
        assert(tl_resource_worker_terminal(r, w) == 0);
      });
    for (auto &x : t)
      x.join();
    assert(tl_resource_finish(0) == 0);
    assert(tl_resource_close() == 0);
    return 0;
  }
  if (mode == "lock") {
    failure = mode;
    assert(tl_resource_worker_enter(r, 0) == TL_ERR_HMX);
    assert(tl_resource_finish(TL_ERR_HMX) == TL_ERR_HMX);
    assert(tl_resource_close() != 0);
    assert(held && gives == 0);
    return 0;
  }
  assert(tl_resource_worker_enter(r, 0) == 0);
  if (mode == "callback-before-cas-close" ||
      mode == "callback-admitted-close") {
    fixed_clock = true;
    hook_phase = mode == "callback-before-cas-close" ? 1 : 2;
    std::thread publisher([] {
      tid = 2;
      callback(ctx, opaque);
    });
    while (!hook_reached.load())
      std::this_thread::yield();
    const unsigned old_epoch = r->gate >> 9;
    std::atomic<bool> closed{false};
    std::thread closer([&] {
      assert(tl_resource_close_admission(r) == 0);
      closed = true;
    });
    while (__atomic_load_n(&r->gate, __ATOMIC_ACQUIRE) & 0x100u)
      std::this_thread::yield();
    assert(!closed.load()); // Closing cannot complete while the old callback is
                            // live.
    hook_release = true;
    publisher.join();
    closer.join();
    assert(r->request_seq == (mode == "callback-admitted-close" ? 1u : 0u));
    assert(!(r->gate & 0x1ffu) && r->inflight == 0);
    assert(tl_resource_open_admission(r) == 0 &&
           (r->gate >> 9) == old_epoch + 1);
    callback(ctx, opaque);
    assert(r->request_seq == (mode == "callback-admitted-close" ? 2u : 1u));
    end(r);
    return 0;
  }
  if (mode == "busy") {
    assert(tl_resource_prepare(false) == TL_ERR_RESOURCE);
    assert(tl_resource_close() == TL_ERR_RESOURCE);
    end(r);
    return 0;
  }
  if (mode == "wrong-thread") {
    tid = 2;
    assert(tl_resource_worker_unlock(r, 0) == TL_ERR_RESOURCE);
    tid = 1;
    assert(tl_resource_finish(0) == TL_ERR_RESOURCE);
    assert(held && tl_resource_close() != 0);
    return 0;
  }
  if (mode == "unlock" || mode == "unsafe-dma" || mode == "missing-terminal") {
    if (mode == "unlock") {
      failure = mode;
      assert(tl_resource_worker_unlock(r, 0) == TL_ERR_HMX);
    }
    if (mode == "unsafe-dma")
      tl_resource_fail(r, TL_ERR_DMA, true);
    int expected = mode == "unlock"       ? TL_ERR_HMX
                   : mode == "unsafe-dma" ? TL_ERR_DMA
                                          : TL_ERR_RESOURCE;
    assert(tl_resource_finish(0) == expected);
    assert(tl_resource_close() != 0);
    assert(held && gives == 0);
    return 0;
  }
  if (mode == "admission-closed") {
    assert(tl_resource_close_admission(r) == 0);
    callback(ctx, opaque);
    assert(r->request_seq == 0);
    assert(tl_resource_open_admission(r) == 0);
    callback(ctx, opaque);
    assert(r->request_seq == 1);
  }
  if (mode == "body-error") {
    tl_resource_fail(r, TL_ERR_GRID);
    end(r, TL_ERR_GRID);
    return 0;
  }
  if (mode == "callback-concurrent") {
    std::vector<std::thread> t;
    for (int j = 0; j < 8; ++j)
      t.emplace_back([] {
        for (int k = 0; k < 10000; ++k)
          callback(ctx, opaque);
      });
    for (auto &x : t)
      x.join();
    assert(tl_resource_request_snapshot(r) == 80000);
  }
  if (mode == "callback-wrong-context") {
    callback(ctx + 1, opaque);
    assert(!tl_resource_request_snapshot(r));
  }
  if (mode == "callback-overflow") {
    r->request_seq = UINT32_MAX;
    callback(ctx, opaque);
    assert(r->request_seq == UINT32_MAX);
    end(r, TL_ERR_RESOURCE);
    return 0;
  }
  if (mode == "epoch-overflow") {
    assert(tl_resource_worker_unlock(r, 0) == 0);
    assert(tl_resource_worker_terminal(r, 0) == 0);
    assert(tl_resource_finish(0) == 0);
    r->gate = 0xfffffe00u;
    assert(tl_resource_prepare(true) == TL_ERR_RESOURCE);
    assert(tl_resource_close() != 0);
    return 0;
  }
  if (mode == "reacquire-fail" || mode == "yield" || mode == "callback-late" ||
      mode == "give") {
    callback(ctx, opaque);
    unsigned snapshot = tl_resource_request_snapshot(r);
    assert(snapshot == 1);
    assert(tl_resource_worker_unlock(r, 0) == 0);
    if (mode == "callback-late")
      callback_on_release = callback_on_acquire = true;
    if (mode == "reacquire-fail")
      failure = "take";
    if (mode == "give")
      failure = "give";
    int rc = tl_resource_owner_yield(r, snapshot);
    if (mode == "give") {
      assert(rc == TL_ERR_VTCM);
      assert(tl_resource_finish(rc) == rc);
      assert(held && tl_resource_close() != 0);
      return 0;
    }
    if (mode == "reacquire-fail") {
      assert(rc == TL_ERR_VTCM && !held);
      assert(tl_resource_worker_terminal(r, 0) == rc);
      assert(tl_resource_finish(rc) == rc);
      failure = "";
      assert(tl_resource_close() == 0);
      return 0;
    }
    assert(rc == 0 && held && r->served_seq == 1);
    if (mode == "callback-late") {
      assert(tl_resource_request_snapshot(r) == 3);
      assert(tl_resource_request_pending(r));
    }
    assert(tl_resource_worker_relock(r, 0) == 0);
  }
  if (mode == "close") {
    assert(tl_resource_worker_unlock(r, 0) == 0);
    assert(tl_resource_worker_terminal(r, 0) == 0);
    assert(tl_resource_finish(0) == 0);
    failure = "close";
    assert(tl_resource_close() != 0 && live);
    return 0;
  }
  end(r);
}
