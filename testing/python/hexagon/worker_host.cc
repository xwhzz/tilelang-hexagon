// Actual worker.h with host QuRT primitives; no DSP scheduling is modelled.
#include <array>
#include <atomic>
#include <cassert>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

enum { TL_OK = 0, TL_ERR_VTCM = 1, TL_ERR_GRID = 4, QURT_EOK = 0 };
constexpr unsigned QURT_MAX_HTHREAD_LIMIT = 8;
constexpr unsigned TL_VTCM_HMX_ALIGNMENT = 2048;
struct qurt_sysenv_max_hthreads_t {
  unsigned max_hthreads;
};
static int query_status, hardware_threads = 6, hardware_units = (4 << 8) | 8;
static unsigned hardware_queries, hvx_queries, create_attempts;
static int fail_create_after = -1;
static int qurt_sysenv_get_max_hw_threads(qurt_sysenv_max_hthreads_t *h) {
  ++hardware_queries;
  h->max_hthreads = hardware_threads;
  return query_status;
}
static int qurt_hvx_get_units() {
  ++hvx_queries;
  return hardware_units;
}

struct qurt_sem_t {
  std::mutex mutex;
  std::condition_variable ready;
  unsigned count = 0;
  bool stopping = false;
};
struct qurt_thread_attr_t {
  void *stack = nullptr;
  unsigned bytes = 0;
  int priority = 0;
};
using qurt_thread_t = std::thread *;
static std::vector<qurt_sem_t *> semaphores;
static std::vector<std::thread *> threads;
static thread_local int worker_id;
struct StopMockThread {};
static void qurt_sem_init_val(qurt_sem_t *s, unsigned count) {
  s->count = count;
  semaphores.push_back(s);
}
static void qurt_sem_up(qurt_sem_t *s) {
  std::lock_guard<std::mutex> lock(s->mutex);
  ++s->count;
  s->ready.notify_one();
}
static void qurt_sem_down(qurt_sem_t *s) {
  std::unique_lock<std::mutex> lock(s->mutex);
  s->ready.wait(lock, [s] { return s->count || s->stopping; });
  if (s->stopping)
    throw StopMockThread{};
  --s->count;
}
static qurt_thread_t qurt_thread_get_id() { return nullptr; }
static int qurt_thread_get_priority(qurt_thread_t) { return 240; }
static void qurt_thread_attr_init(qurt_thread_attr_t *a) { *a = {}; }
static void qurt_thread_attr_set_stack_addr(qurt_thread_attr_t *a, void *p) {
  a->stack = p;
}
static void qurt_thread_attr_set_stack_size(qurt_thread_attr_t *a, unsigned n) {
  a->bytes = n;
}
static void qurt_thread_attr_set_priority(qurt_thread_attr_t *a, int p) {
  a->priority = p;
}
static int qurt_thread_create(qurt_thread_t *out, qurt_thread_attr_t *a,
                              void (*entry)(void *), void *arg) {
  ++create_attempts;
  assert(a->stack && a->bytes == 64 * 1024 && a->priority == 240);
  if (fail_create_after >= 0 && (int)threads.size() == fail_create_after)
    return -1;
  *out = new std::thread([=] {
    worker_id = (int)(intptr_t)arg;
    try {
      entry(arg);
    } catch (const StopMockThread &) {
      // Test-only shutdown after the real dispatcher has joined the job.
    }
  });
  threads.push_back(*out);
  return QURT_EOK;
}
static uint8_t *tl_vtcm_arena() { std::abort(); }
static unsigned tl_vtcm_arena_bytes() { std::abort(); }

#include <tl_templates/hexagon/worker.h>

static std::array<uint8_t, QURT_MAX_HTHREAD_LIMIT * 16> arena_storage{};
static tl_worker_arenas arenas(unsigned count = QURT_MAX_HTHREAD_LIMIT) {
  return {arena_storage.data(), 16, count};
}
static std::atomic<unsigned> opened{0}, closed{0}, drained{0}, callbacks{0};
static thread_local bool resource_live;
static thread_local unsigned pending;
static int fail_scope = -1;
struct TrackedScope {
  std::thread::id owner = std::this_thread::get_id();
  int status = worker_id == fail_scope ? 37 : TL_OK;
  TrackedScope() {
    assert(!resource_live);
    resource_live = true;
    ++opened;
  }
  ~TrackedScope() {
    assert(resource_live && owner == std::this_thread::get_id());
    drained += pending;
    pending = 0;
    resource_live = false;
    ++closed;
  }
};
static void check_limit(int expected, bool hvx_queried = true) {
  assert(tl_num_workers() == expected);
  hardware_threads = 1;
  hardware_units = 0;
  query_status = -1;
  assert(tl_num_workers() == expected);
  assert(hardware_queries == 1 && hvx_queries == (hvx_queried ? 1u : 0u));
  assert(create_attempts == 0 && callbacks == 0);
}
static void batch(int requested, int arena_count, int expected_workers) {
  std::array<int, 17> values{};
  std::array<std::atomic<unsigned>, 17> visits{};
  for (auto &visit : visits)
    visit = 0;
  auto body = [&](int i, uint8_t *arena) {
    assert(resource_live);
    assert(arena == arena_storage.data() + (i % expected_workers) * 16);
    ++visits[i];
    values[i] = 3 * i + 7;
    ++callbacks;
    ++pending;
    return TL_OK;
  };
  unsigned before = opened;
  assert(tl_parallel_for<TrackedScope>(17, requested, arenas(arena_count),
                                       body) == TL_OK);
  assert(opened == before + expected_workers && opened == closed);
  for (int i = 0; i < 17; ++i)
    assert(visits[i] == 1 && values[i] == 3 * i + 7);
  assert(drained == callbacks);
}
static void rejected(int extent, int count, int requested = 6) {
  auto body = [](int, uint8_t *) {
    ++callbacks;
    return TL_OK;
  };
  assert((tl_parallel_for<TrackedScope, true>(extent, requested, arenas(count),
                                              body) == TL_ERR_GRID));
  assert(callbacks == 0 && opened == 0 && create_attempts == 0);
  assert(!tl_pool.tried && threads.empty());
}
static void visibility(int participants, int requested) {
  std::array<int, QURT_MAX_HTHREAD_LIMIT> values{};
  auto body = [&](int i, uint8_t *) -> int {
    assert(resource_live);
    for (int generation = 1; generation <= 64; ++generation) {
      values[i] = generation * 10 + i;
      if (int status = tl_grid_sync(i))
        return status;
      for (int j = 0; j < participants; ++j)
        assert(values[j] == generation * 10 + j);
      // Prevent a next-generation write from racing another reader.
      if (int status = tl_grid_sync(i))
        return status;
    }
    ++callbacks;
    return TL_OK;
  };
  assert((tl_parallel_for<TrackedScope, true>(participants, requested, arenas(),
                                              body) == TL_OK));
  assert(callbacks == (unsigned)participants && opened == callbacks &&
         closed == opened);
}
static void finish() {
  assert(opened == closed);
  // Unlike a DSP PD, this process has C++ static objects. Stop and join every
  // mock pool thread before destroying its semaphore storage.
  for (auto s : semaphores) {
    std::lock_guard<std::mutex> lock(s->mutex);
    s->stopping = true;
    s->ready.notify_all();
  }
  for (auto thread : threads) {
    thread->join();
    delete thread;
  }
  std::free(tl_pool.blob);
}
int main(int argc, char **argv) {
  assert(argc == 2);
  std::string name = argv[1];
  if (name == "limit-four") {
    check_limit(4);
  } else if (name == "limit-six") {
    hardware_units = 6 << 8;
    check_limit(6);
  } else if (name == "limit-eight") {
    hardware_threads = 8;
    hardware_units = 8 << 8;
    check_limit(8);
  } else if (name == "limit-hardware") {
    hardware_threads = 2;
    check_limit(2);
  } else if (name == "limit-software") {
    hardware_threads = 99;
    hardware_units = 255 << 8;
    check_limit(8);
  } else if (name == "query-failed" || name == "hardware-zero") {
    query_status = name == "query-failed" ? -1 : 0;
    hardware_threads = name == "query-failed" ? 6 : 0;
    check_limit(1, false);
  } else if (name == "hvx-negative" || name == "hvx-zero" ||
             name == "hvx-64-only") {
    hardware_units = name == "hvx-negative" ? -1 : name == "hvx-zero" ? 0 : 8;
    check_limit(1);
  } else if (name == "fallback-one") {
    hardware_units = 0;
    rejected(2, 6);
    batch(6, 6, 1);
    assert(!create_attempts && !tl_pool.blob && !tl_pool.tried);
    assert(hardware_queries == 1 && hvx_queries == 1);
  } else if (name == "empty") {
    auto body = [](int, uint8_t *) {
      std::abort();
      return TL_OK;
    };
    assert(
        (tl_parallel_for<TrackedScope, true>(0, 6, arenas(0), body) == TL_OK));
    assert(!hardware_queries && !hvx_queries && !opened && !create_attempts);
  } else if (name == "last-int32-program") {
    auto body = [](int32_t i, uint8_t *arena) {
      assert(i == INT32_MAX - 2 && arena == arena_storage.data());
      assert(resource_live);
      ++callbacks;
      return TL_OK;
    };
    tl_parallel_for_job<TrackedScope, decltype(body)> job{
        &body, INT32_MAX, {arena_storage.data(), 0u, 1u}};
    assert(job.run(&job, INT32_MAX - 2, 4) == TL_OK);
    assert(callbacks == 1 && opened == 1 && closed == 1);
    assert(!hardware_queries && !hvx_queries && !create_attempts);
  } else if (name == "batch-four" || name == "batch-request-two" ||
             name == "batch-arena-two") {
    int expected = name == "batch-four" ? 4 : 2;
    batch(name == "batch-request-two" ? 2 : 6,
          name == "batch-arena-two" ? 2 : 6, expected);
    assert(threads.size() == 3 && create_attempts == 3);
    // Reuse the persistent pool; every program is still visited exactly once.
    batch(6, 6, 4);
    assert(create_attempts == 3 && hardware_queries == 1 && hvx_queries == 1);
  } else if (name == "batch-eight" || name == "barrier-eight" ||
             name == "reject-nine") {
    hardware_threads = 8;
    hardware_units = 8 << 8;
    if (name == "reject-nine") {
      rejected(9, 8, 9);
    } else {
      if (name == "batch-eight") {
        batch(9, 8, 8);
        batch(8, 8, 8);
      } else {
        visibility(8, 8);
      }
      assert(threads.size() == 7 && create_attempts == 7);
    }
  } else if (name == "reject-six" || name == "reject-arena") {
    rejected(name == "reject-six" ? 6 : 3, name == "reject-six" ? 6 : 2);
  } else if (name == "extent-four-request-six" || name == "barrier-four" ||
             name == "barrier-six") {
    int n = name == "barrier-six" ? 6 : 4;
    hardware_units = n << 8;
    visibility(n, name == "extent-four-request-six" ? 6 : n);
    assert(threads.size() == (unsigned)n - 1);
  } else if (name == "partial-batch" || name == "partial-core") {
    fail_create_after = 1;
    if (name == "partial-core") {
      auto body = [](int, uint8_t *) {
        ++callbacks;
        return TL_OK;
      };
      assert((tl_parallel_for<TrackedScope, true>(4, 6, arenas(), body) ==
              TL_ERR_GRID));
      assert(callbacks == 0 && opened == 0);
    } else {
      batch(6, 6, 4);
    }
    assert(create_attempts == 2 && threads.size() == 1);
    fail_create_after = -1;
    batch(6, 6, 4);
    assert(create_attempts == 2); // A failed spawn is never retried.
  } else if (name == "scope-failure" || name == "body-failure" ||
             name == "barrier-failure") {
    if (name == "scope-failure")
      fail_scope = 0;
    auto body = [&](int i, uint8_t *) {
      assert(resource_live);
      ++callbacks;
      ++pending;
      if (name == "body-failure")
        return 37 + i;
      if (name == "barrier-failure" && i == 0)
        return 37;
      return tl_grid_sync(i);
    };
    assert((tl_parallel_for<TrackedScope, true>(4, 4, arenas(), body) == 37));
    assert(opened == 4 && closed == 4 && drained == callbacks);
    assert(callbacks == (name == "scope-failure" ? 3u : 4u));
  } else {
    assert(false && "unknown case");
  }
  finish();
}
