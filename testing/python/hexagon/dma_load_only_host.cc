// Exercise the real DMA queue/state machine. Only DSP/QuRT operations are
// mocked; these tests do not claim hardware completion or cache ordering.
#include <cassert>
#include <cstdint>
#include <cstring>
#include <functional>
#include <new>
#include <string>
#include <type_traits>
#include <vector>

using qurt_mem_cache_op_t = int;
using qurt_addr_t = uint32_t;
using qurt_size_t = uint32_t;
constexpr int QURT_MEM_CACHE_FLUSH = 1;
constexpr int QURT_MEM_CACHE_FLUSH_ALL = 2;
constexpr int QURT_MEM_CACHE_INVALIDATE = 3;
constexpr int QURT_MEM_DCACHE = 4;
constexpr int TL_OK = 0;
constexpr int TL_ERR_DMA = -70;

static std::vector<std::string> events;
static uint32_t engine_state;
static int polls, resets, cache_calls, fail_cache_call;
static std::function<void()> poll_hook, cache_hook;

static uint32_t Q6_R_dmpoll() {
  events.push_back("poll");
  ++polls;
  if (poll_hook)
    poll_hook();
  return engine_state;
}
static uint32_t Q6_R_dmpause() {
  events.push_back("reset");
  ++resets;
  return engine_state = 0;
}
static void dma_release(void *) { events.push_back("release"); }
static void dma_sync() { events.push_back("sync"); }
static void Q6_dmstart_A(void *) {
  events.push_back("start");
  engine_state = 1;
}
static void Q6_dmlink_AA(void *, void *) { events.push_back("link"); }
static void Q6_dccleana_A(void *) { events.push_back("clean-line"); }
static int qurt_mem_cache_clean(qurt_addr_t, qurt_size_t,
                                qurt_mem_cache_op_t op, int cache) {
  assert(cache == QURT_MEM_DCACHE);
  events.push_back(op == QURT_MEM_CACHE_INVALIDATE ? "invalidate" : "flush");
  ++cache_calls;
  if (cache_hook)
    cache_hook();
  return fail_cache_call == cache_calls ? 1 : 0;
}

#include <tl_templates/hexagon/dma.h>

using TryPop = int (*)(tl_hexagon_dma_queue *, tl_hexagon_dma_transfer *);
static_assert(std::is_same_v<decltype(&tl_hexagon_dma_queue_try_pop), TryPop>);
static_assert(std::is_same_v<decltype(&tl_hexagon_dma_queue_pop), TryPop>);

// Explicit two-argument function addresses retain the generic API types.
using Generic = tl_hexagon_dma_async_context<1, 1>;
using Submit = int (*)(Generic &, void *, const void *, uint32_t, uint32_t,
                       uint32_t, uint32_t, int);
using Commit = int (*)(Generic &);
using Wait = int (*)(Generic &, uint32_t);
static_assert(
    std::is_same_v<decltype(&tl_hexagon_dma_async_submit<1, 1>), Submit>);
static_assert(
    std::is_same_v<decltype(&tl_hexagon_dma_async_commit<1, 1>), Commit>);
static_assert(std::is_same_v<decltype(&tl_hexagon_dma_async_wait<1, 1>), Wait>);

// Non-PIE host binaries put these objects below 4 GiB, so the actual DMA
// address validation and managed constructor run without a mocked shortcut.
alignas(128) static uint8_t src[512], dst[512];

static void reset_observations() {
  events.clear();
  polls = resets = cache_calls = fail_cache_call = 0;
  poll_hook = cache_hook = {};
}

template <uint32_t D, uint32_t G> struct Managed {
#if LOAD_ONLY
  using Context = tl_hexagon_dma_async_context<D, G, true>;
#else
  using Context = tl_hexagon_dma_async_context<D, G>;
#endif
  alignas(Context) static uint8_t storage[sizeof(Context)];
  Context *p;
  Managed() {
    engine_state = 0;
    std::memset(storage, 0xa5, sizeof(storage));
    p = new (storage) Context;
    assert(p->status == TL_HEXAGON_DMA_OK);
    assert(p->queue.capacity == D && p->queue.descriptors == p->descriptors &&
           p->queue.transfers == p->transfers);
    // Mark live padding after object construction; no indeterminate-byte
    // assumption.
    for (auto &descriptor : p->descriptors)
      std::memset(descriptor.padding, 0xa5, sizeof(descriptor.padding));
    reset_observations();
  }
  ~Managed() { p->~Context(); }
};
template <uint32_t D, uint32_t G>
alignas(typename Managed<D, G>::Context)
    uint8_t Managed<D, G>::storage[sizeof(typename Managed<D, G>::Context)];

template <typename Context>
static void submit(Context &c, uint32_t bytes = 64) {
  assert(c.submit(dst, src, bytes, bytes, bytes, 1,
                  TL_HEXAGON_DMA_DDR_TO_VTCM_UNWRITTEN) == TL_HEXAGON_DMA_OK);
}

template <typename Context> static void finish(Context &c) {
  for (uint32_t n = 0; n < c.queue.count; ++n)
    tl_hexagon_dma_descriptor_mark_done(
        &c.descriptors[tl_hexagon_dma_queue_index(&c.queue, n)]);
  engine_state = 0;
}

static void assert_cleared(const tl_hexagon_dma_descriptor &d,
                           const tl_hexagon_dma_transfer &t) {
  for (auto word : d.word)
    assert(word == 0);
  for (auto byte : d.padding)
    assert(byte == 0xa5);
  assert(t.dst == nullptr && t.src == nullptr && t.dst_stride == 0 &&
         t.src_stride == 0 && t.row_bytes == 0 && t.rows == 0 &&
         t.direction == TL_HEXAGON_DMA_NO_CACHE);
}

static void busy_then_done() {
  Managed<1, 1> m;
  auto &c = *m.p;
  submit(c);
  assert(c.commit() == TL_HEXAGON_DMA_OK);
  reset_observations();
  poll_hook = [&] {
    // The first try observes busy via two polls; the next first poll makes
    // completion visible. Reading done before that poll cannot pass.
    if (polls == 3)
      finish(c);
  };
  assert(c.wait(0) == TL_HEXAGON_DMA_OK);
  assert(polls == 3 &&
         events == std::vector<std::string>({"poll", "poll", "poll"}));
  assert(c.retired == 1 && c.queue.count == 0);
  poll_hook = {};
}

static void poll_before_done() {
  Managed<1, 1> m;
  auto &c = *m.p;
  submit(c);
  assert(c.commit() == TL_HEXAGON_DMA_OK);
  reset_observations();
  poll_hook = [&] {
    tl_hexagon_dma_descriptor_mark_done(&c.descriptors[0]);
    engine_state = TL_HEXAGON_DMA_STATUS_ERROR;
  };
  // Once this descriptor is done, the original path does not inspect the
  // engine-status bits. A first done read before the poll takes the error path.
  assert(c.wait(0) == TL_HEXAGON_DMA_OK);
  assert(polls == 1 && c.retired == 1 && c.queue.count == 0);
  poll_hook = {};
}

static void store_cache_error() {
  Managed<1, 1> m;
  auto &c = *m.p;
  assert(c.submit(dst, src, 64, 64, 32, 2, TL_HEXAGON_DMA_VTCM_TO_DDR) ==
         TL_HEXAGON_DMA_OK);
  assert(c.commit() == TL_HEXAGON_DMA_OK);
  finish(c);
  auto descriptor = c.descriptors[0];
  auto transfer = c.transfers[0];
  reset_observations();
  fail_cache_call = 2;
  cache_hook = [&] {
    assert(std::memcmp(&descriptor, &c.descriptors[0], sizeof(descriptor)) ==
           0);
    assert(std::memcmp(&transfer, &c.transfers[0], sizeof(transfer)) == 0);
    assert(c.queue.head == 0 && c.queue.count == 1 && c.retired == 0 &&
           c.group_count == 1 && c.queue.tail == &c.descriptors[0]);
  };
  assert(c.wait(0) == TL_HEXAGON_DMA_ERR_CACHE);
  assert(events ==
         std::vector<std::string>({"poll", "invalidate", "invalidate"}));
  cache_hook();
  reset_observations();
  assert(c.wait(0) == TL_HEXAGON_DMA_OK);
  assert(events ==
         std::vector<std::string>({"poll", "invalidate", "invalidate"}));
  assert_cleared(c.descriptors[0], c.transfers[0]);
}

static void error_drain() {
  {
    Managed<1, 1> m;
    auto &c = *m.p;
    submit(c);
    assert(c.commit() == TL_HEXAGON_DMA_OK);
    reset_observations();
    engine_state = TL_HEXAGON_DMA_STATUS_ERROR;
    assert(c.wait(0) == TL_HEXAGON_DMA_ERR_ENGINE);
    assert(c.queue.count == 1 && c.retired == 0 && c.group_count == 1);
    assert(events == std::vector<std::string>({"poll", "poll"}));
  }
  assert(resets == 1 &&
         events == std::vector<std::string>(
                       {"poll", "poll", "poll", "poll", "reset"}));
}

static void cached_load() {
  Managed<2, 2> m;
  auto &c = *m.p;
  assert(c.submit(dst, src, 256, 256, 256, 1, TL_HEXAGON_DMA_DDR_TO_VTCM) == 0);
  assert(events == std::vector<std::string>({"clean-line", "clean-line",
                                             "clean-line", "clean-line", "sync",
                                             "poll", "release", "start"}));
  assert(c.commit() == 0);
  finish(c);
  reset_observations();
  assert(c.wait(0) == 0);
  assert(events == std::vector<std::string>({"poll"}));
  assert(c.queue.count == 0 && c.group_count == 0 && c.submitted == 1 &&
         c.retired == 1);
}

static void reject_direction(bool pending) {
  Managed<2, 2> m;
  auto &c = *m.p;
  if (pending) {
    submit(c);
    assert(c.commit() == 0);
  }
  reset_observations();
  std::vector<uint8_t> before(sizeof(c));
  std::memcpy(before.data(), &c, sizeof(c));
  for (int direction : {-1, 1, 2, 4, 99}) {
    volatile int dynamic_direction = direction;
    assert(c.submit(dst, src, 256, 256, 256, 1, dynamic_direction) ==
           TL_HEXAGON_DMA_ERR_ARGUMENT);
    assert(events.empty());
    assert(std::memcmp(before.data(), &c, sizeof(c)) == 0);
  }
  if (pending) {
    finish(c);
    assert(c.wait(0) == 0);
  }
}

static void early_success() {
  {
    Managed<2, 2> m;
    auto &c = *m.p;
    submit(c);
    assert(c.commit() == 0);
    finish(c);
    reset_observations();
  }
  assert(events == std::vector<std::string>({"poll"}));
  assert(resets == 0);
}

static void mixed_generic() {
  Managed<2, 2> m;
  auto &c = *m.p;
  assert(c.submit(dst, src, 64, 64, 64, 1, TL_HEXAGON_DMA_DDR_TO_VTCM) == 0);
  assert(c.commit() == 0);
  assert(c.submit(dst, src, 64, 64, 64, 1, TL_HEXAGON_DMA_VTCM_TO_DDR) == 0);
  assert(c.commit() == 0);
  finish(c);
  reset_observations();
  assert(c.wait(1) == 0);
  assert(events == std::vector<std::string>({"poll"}));
  assert(c.queue.count == 1 && c.retired == 1 && c.group_count == 1);
  reset_observations();
  assert(c.wait(0) == 0);
  assert(events == std::vector<std::string>({"poll", "invalidate"}));
}

int main(int argc, char **argv) {
  assert(argc == 2);
  std::string name = argv[1];
  if (name == "cached-load")
    cached_load();
  else if (name == "reject-direction")
    reject_direction(false);
  else if (name == "reject-pending")
    reject_direction(true);
  else if (name == "early-success")
    early_success();
  else if (name == "mixed-generic")
    mixed_generic();
  else if (name == "busy-then-done")
    busy_then_done();
  else if (name == "poll-before-done")
    poll_before_done();
  else if (name == "store-cache-error")
    store_cache_error();
  else if (name == "error-drain")
    error_drain();
  else
    assert(false);
}
