// Reuse the existing host QuRT persistent pool, not a fresh thread per launch.
#define main worker_regression_main
#include "worker_host.cc"
#undef main
#include <cstring>

enum { TL_ERR_HMX = 2, TL_ERR_DMA = 3, TL_ERR_RESOURCE = 5 };
constexpr int TL_HEXAGON_DMA_OK = 0;
struct MockWorker {
  std::thread::id tid;
  unsigned entered, held, terminal;
};
struct tl_cooperative_run {
  unsigned participants = 0;
  int uses_hmx = 0, first_error = 0, unsafe = 0;
  uint8_t *region = nullptr, *shadow = nullptr;
  unsigned bytes = 0;
  MockWorker worker[QURT_MAX_HTHREAD_LIMIT]{};
  unsigned request_seq = 0, served_seq = 0;
};
static tl_cooperative_run resource;
static std::atomic<uint64_t> ticks{0};
static std::atomic<unsigned> yields{0}, bodies{0}, finished_dma{0};
static std::atomic<unsigned> snapshot_reads{0}, clock_reads{0}, dma_waits{0};
static std::vector<unsigned> selected_requests;
static std::string scenario;
static thread_local bool engine_idle = true;
static bool tl_hexagon_dma_engine_idle() { return engine_idle; }
static uint64_t HAP_perf_get_qtimer_count() {
  ++clock_reads;
  return ticks.load();
}
static tl_cooperative_run *tl_resource_current() { return &resource; }
static int tl_resource_error(tl_cooperative_run *r) {
  return __atomic_load_n(&r->first_error, __ATOMIC_ACQUIRE);
}
static int tl_resource_fail(tl_cooperative_run *r, int error,
                            bool unsafe = false) {
  int expected = 0;
  __atomic_compare_exchange_n(&r->first_error, &expected, error, false,
                              __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE);
  if (unsafe)
    __atomic_store_n(&r->unsafe, 1, __ATOMIC_RELEASE);
  return tl_resource_error(r);
}
static unsigned tl_resource_request_snapshot(tl_cooperative_run *r) {
  ++snapshot_reads;
  return __atomic_load_n(&r->request_seq, __ATOMIC_ACQUIRE);
}
static bool tl_resource_request_pending(tl_cooperative_run *r) {
  // Publish at deterministic boundaries of the actual header's two reads.
  // This is a fake callback publication, not SDK callback scheduling evidence.
  if (scenario == "between-snapshots" && snapshot_reads == 1)
    __atomic_add_fetch(&r->request_seq, 1u, __ATOMIC_RELEASE);
  bool pending = tl_resource_request_snapshot(r) != r->served_seq;
  if (scenario == "after-negative" && snapshot_reads == 2) {
    assert(!pending);
    __atomic_add_fetch(&r->request_seq, 1u, __ATOMIC_RELEASE);
  }
  return pending;
}
static int tl_resource_group_admit(tl_cooperative_run *r, unsigned nw,
                                   bool hmx) {
  assert(!r->participants);
  r->participants = nw;
  r->uses_hmx = hmx;
  return 0;
}
static int tl_resource_worker_relock(tl_cooperative_run *r, unsigned wid) {
  auto &w = r->worker[wid];
  assert(w.entered && w.tid == std::this_thread::get_id() && !w.held);
  if (scenario == "relock-error" && yields && wid == 1) {
    w.held = 2;
    return tl_resource_fail(r, TL_ERR_HMX, true);
  }
  w.held = r->uses_hmx;
  return 0;
}
static int tl_resource_worker_enter(tl_cooperative_run *r, unsigned wid) {
  auto &w = r->worker[wid];
  assert(!w.entered);
  w.entered = 1;
  w.tid = std::this_thread::get_id();
  return tl_resource_error(r) ? tl_resource_error(r)
                              : tl_resource_worker_relock(r, wid);
}
static int tl_resource_worker_unlock(tl_cooperative_run *r, unsigned wid) {
  auto &w = r->worker[wid];
  assert(w.entered && w.tid == std::this_thread::get_id());
  if (scenario == "unlock-error" && wid == 1 && w.held) {
    w.held = 2;
    return tl_resource_fail(r, TL_ERR_HMX, true);
  }
  w.held = 0;
  return 0;
}
static int tl_resource_worker_terminal(tl_cooperative_run *r, unsigned wid) {
  auto &w = r->worker[wid];
  assert(w.entered && w.tid == std::this_thread::get_id());
  if (w.held)
    return tl_resource_fail(r, TL_ERR_HMX, true);
  assert(!w.terminal);
  w.terminal = 1;
  return tl_resource_error(r);
}
static std::vector<uint8_t> expected_bytes;
static int tl_resource_owner_yield(tl_cooperative_run *r, unsigned snapshot) {
  assert(worker_id == 0 && !tl_resource_error(r));
  for (unsigned w = 0; w < r->participants; ++w)
    assert(r->worker[w].entered && !r->worker[w].held);
  assert(std::memcmp(r->shadow, expected_bytes.data(), r->bytes) == 0);
  ++yields;
  selected_requests.push_back(snapshot);
  if (scenario == "release-error")
    return tl_resource_fail(r, TL_ERR_RESOURCE);
  std::memset(r->region, 0xa5, r->bytes);
  r->served_seq = snapshot;
  if (scenario == "late-request" && yields == 1)
    __atomic_add_fetch(&r->request_seq, 1u, __ATOMIC_RELEASE);
  return 0;
}

struct FakeDMA {
  unsigned pending = 2, committed = 2;
  FakeDMA() { engine_idle = false; }
  int wait(unsigned) {
    ++dma_waits;
    if (scenario == "dma-wait-error" && worker_id == 1)
      return -4;
    if (scenario == "committed-error" && worker_id == 1)
      tl_resource_fail(&resource, 37);
    pending -= committed;
    committed = 0;
    engine_idle = !pending;
    return 0;
  }
  bool quiescent() const { return !pending && engine_idle; }
  int finish() {
    ++finished_dma;
    if (scenario == "dma-finish-error" && worker_id == 1)
      return -4;
    pending = committed = 0;
    engine_idle = true;
    return 0;
  }
  ~FakeDMA() {
    // Model legacy fallback cleanup: a later reset/idle must not clear unsafe.
    engine_idle = true;
  }
};
#include <tl_templates/hexagon/cooperative.h>

alignas(128) static uint8_t vtcm_bytes[128 * 35 + 37];
alignas(128) static uint8_t saved_bytes[sizeof(vtcm_bytes)];
static void setup() {
  resource = tl_cooperative_run{};
  resource.region = vtcm_bytes;
  resource.shadow = saved_bytes;
  resource.bytes = sizeof(vtcm_bytes);
  std::memset(saved_bytes, 0x5a, sizeof(saved_bytes));
  for (size_t i = 0; i < sizeof(vtcm_bytes); ++i)
    vtcm_bytes[i] = static_cast<uint8_t>(i * 47u + i / 128u);
  expected_bytes.assign(vtcm_bytes, vtcm_bytes + sizeof(vtcm_bytes));
}

int main(int argc, char **argv) {
  assert(argc == 2);
  scenario = argv[1];
  unsigned nw = 4;
  if (scenario.find("roundtrip-") == 0) {
    nw = std::stoi(scenario.substr(10));
    hardware_threads = nw;
    hardware_units = (nw << 8) | 8;
  }
  setup();
  bool elapsed = scenario.find("elapsed-") == 0;
  bool race = scenario == "between-snapshots" || scenario == "after-negative";
  bool need_yield =
      scenario != "no-yield" && !elapsed && scenario != "body-before-grid" &&
      scenario != "body-after-grid" && scenario != "dma-finish-error" &&
      scenario != "early-uncommitted";
  if (need_yield && !race)
    resource.request_seq = 1;
  if (scenario == "elapsed-below")
    ticks = 191999;
  if (scenario == "elapsed-at")
    ticks = 192000;
  if (scenario == "elapsed-long")
    ticks = 19200000; // One second: elapsed time alone must not cause a yield.
  if (scenario == "partial-pool")
    fail_create_after = 1;
  tl_worker_arenas a{vtcm_bytes, 128, nw};
  if (scenario == "arena-count")
    a.count = nw - 1;
  if (scenario == "arena-bounds")
    a.bytes = resource.bytes;
  auto body = [&](int wid, uint8_t *arena) -> int {
    ++bodies;
    if (scenario != "no-arena")
      assert(arena == vtcm_bytes + wid * 128);
    if (scenario == "body-before-grid" && wid == 0) {
      while (__atomic_load_n(&tl_cooperative_active->arrived,
                             __ATOMIC_ACQUIRE) != nw - 1)
        std::this_thread::yield();
      return 37;
    }
    if (scenario == "epoch-overflow")
      tl_cooperative_active->passed[wid] = UINT32_MAX - 1u;
    FakeDMA dma;
    tl_cooperative_dma_scope<FakeDMA> scope(dma, wid);
    if (scenario == "early-uncommitted") {
      dma.committed = 0;
      return 37;
    }
    if (scenario == "dma-uncommitted" && wid == 1)
      dma.committed = 1;
    unsigned iterations = scenario == "no-yield" || elapsed ? 257 : 3;
    for (unsigned i = 0; i < iterations; ++i) {
      if (scenario == "no-yield" && wid == 0 && i % 3 == 0)
        for (unsigned delay = 0; delay < 50; ++delay)
          std::this_thread::yield();
      int rc = tl_cooperative_checkpoint(wid, dma);
      if (rc)
        return rc;
      if (scenario == "body-after-grid" && wid == 0 && i == 1)
        return 37;
      if (elapsed || scenario == "no-yield" ||
          (scenario == "after-negative" && i == 0)) {
        assert(!yields && dma.pending == 2 && dma.committed == 2);
        assert(!dma.quiescent());
      }
      if (scenario == "between-snapshots" && i == 0)
        assert(yields == 1 && resource.served_seq == 0 &&
               __atomic_load_n(&resource.request_seq, __ATOMIC_ACQUIRE) == 1);
      if (yields)
        assert(std::memcmp(vtcm_bytes, expected_bytes.data(), resource.bytes) ==
               0);
    }
    return 0;
  };
  int rc;
  if (scenario == "no-arena") {
    rc = tl_cooperative_parallel_for<false>(nw, nw, [&](int wid) -> int {
      ++bodies;
      return tl_cooperative_checkpoint(wid);
    });
  } else {
    rc = tl_cooperative_parallel_for<true>(nw, nw, a, body);
  }
  if (scenario == "partial-pool" || scenario == "arena-count" ||
      scenario == "arena-bounds") {
    assert(rc && !bodies && !resource.participants && !yields);
  } else if (scenario.find("error") != std::string::npos ||
             scenario.find("body-") == 0 || scenario == "dma-uncommitted" ||
             scenario == "early-uncommitted" || scenario == "epoch-overflow") {
    assert(rc != 0 && tl_resource_error(&resource) == rc);
    if (scenario == "dma-wait-error" || scenario == "dma-uncommitted" ||
        scenario == "dma-finish-error")
      assert(resource.unsafe && !yields);
    if (scenario == "committed-error")
      assert(!yields && !resource.unsafe);
    if (!resource.unsafe)
      for (unsigned w = 0; w < nw; ++w)
        assert(resource.worker[w].terminal && !resource.worker[w].held);
  } else {
    assert(rc == 0 && !resource.unsafe && bodies == nw);
    unsigned expected_yields =
        scenario == "late-request" || scenario == "between-snapshots" ? 2u
        : need_yield                                                  ? 1u
                                                                      : 0u;
    assert(yields == expected_yields);
    if (scenario == "between-snapshots")
      assert((selected_requests == std::vector<unsigned>{0, 1}));
    if (scenario == "after-negative")
      assert((selected_requests == std::vector<unsigned>{1}));
    if (race)
      assert(snapshot_reads == 6 && resource.served_seq == 1);
    if (elapsed || scenario == "no-yield") {
      assert(!dma_waits && snapshot_reads == 514 && !clock_reads);
      assert(finished_dma == nw);
      for (auto value : saved_bytes)
        assert(value == 0x5a);
    }
    for (unsigned w = 0; w < nw; ++w)
      assert(resource.worker[w].terminal && !resource.worker[w].held);
    if (scenario == "repeated-run") {
      std::array<std::thread::id, 8> ids;
      for (unsigned w = 0; w < nw; ++w)
        ids[w] = resource.worker[w].tid;
      setup();
      resource.request_seq = 1;
      assert(tl_cooperative_parallel_for<true>(nw, nw, a, body) == 0);
      for (unsigned w = 0; w < nw; ++w)
        assert(ids[w] == resource.worker[w].tid);
    }
  }
  assert(!tl_cooperative_active);
  finish();
}
