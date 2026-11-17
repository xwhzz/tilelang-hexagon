// The test fixture supplies existing host SDK/QuRT providers; all four runtime
// headers below are the production implementations, with only pause replaced.
#include "resource_collective_providers.h"
#include <tl_templates/hexagon/worker.h>
static constexpr int TL_HEXAGON_DMA_OK = 0;
static bool tl_hexagon_dma_engine_idle() { return true; }
#include <tl_templates/hexagon/cooperative.h>

int main(int argc, char **argv) {
  assert(argc == 2);
  std::string mode = argv[1];
  granted =
      8193; // Exercise the final scalar tail, as well as every HVX stripe.
  assert(tl_vtcm_open(1) == 0);
  for (unsigned call = 0; call < (mode == "repeated" ? 4u : 1u); ++call) {
    assert(tl_resource_prepare(true) == 0);
    auto *run = tl_resource_current();
    for (unsigned i = 0; i < run->bytes; ++i)
      run->region[i] = uint8_t(i * 17u + call);
    std::vector<uint8_t> expected(run->region, run->region + run->bytes);
    callback(ctx, opaque);
    if (mode == "release-error")
      failure = "give";
    int rc = tl_cooperative_parallel_for<true>(4, 4, [&](int wid) -> int {
      if (mode == "body-error" && wid == 0)
        return 37;
      int status = tl_cooperative_checkpoint(wid);
      if (status)
        return status;
      assert(std::memcmp(run->region, expected.data(), run->bytes) == 0);
      return 0;
    });
    if (mode == "release-error") {
      assert(rc == TL_ERR_VTCM && run->unsafe && held);
      assert(tl_resource_finish(rc) == rc);
      assert(tl_resource_close() != 0);
    } else {
      assert(rc == (mode == "body-error" ? 37 : TL_OK));
      for (unsigned w = 0; w < 4; ++w)
        assert(run->worker[w].terminal && !run->worker[w].held);
      assert(tl_resource_finish(rc) == rc && !held);
    }
  }
  if (mode != "release-error") {
    assert(tl_resource_close() == TL_OK);
    assert(opens == closes && takes == gives && locks == unlocks);
  }
  // Test-only pool shutdown; production keeps it for the PD lifetime.
  for (auto *sem : semaphores) {
    std::lock_guard<std::mutex> lock(sem->mutex);
    sem->stopping = true;
    sem->ready.notify_all();
  }
  for (auto *thread : threads) {
    thread->join();
    delete thread;
  }
  std::free(tl_pool.blob);
}
