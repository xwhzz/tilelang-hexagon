// Exercise the generated FastRPC policy and error paths, without an SDK or DSP.
#include <cassert>
#include <string>
#include <type_traits>
#include <vector>

using AEEResult = int;
enum { AEE_SUCCESS = 0, TL_OK = 0, TL_ERR_VTCM = 1, TL_ERR_RESOURCE = 5 };
#define TL_HEXAGON_USES_VTCM 1
#define TL_HEXAGON_USES_COOPERATIVE 1
static std::vector<std::string> calls;
static bool prepared, cooperative, held;
static int prepare_error, take_error, body_error, finish_error, close_error,
    give_error;
static int tl_resource_state;
static int tl_resource_fail(int *first_error, int status) {
  if (!*first_error)
    *first_error = status;
  return *first_error;
}

static int tl_resource_prepare(bool mode) {
  assert(!prepared && !held);
  calls.push_back(mode ? "prepare-cooperative" : "prepare-ordinary");
  if (prepare_error)
    return prepare_error;
  prepared = true;
  cooperative = mode;
  held = mode;
  tl_resource_state = 0;
  return TL_OK;
}
static int tl_resource_finish(int status) {
  assert(prepared);
  calls.push_back("finish");
  prepared = held = false;
  if (status)
    tl_resource_fail(&tl_resource_state, status);
  if (finish_error)
    tl_resource_fail(&tl_resource_state, finish_error);
  return tl_resource_state;
}
static int tl_resource_close() {
  assert(!prepared && !held);
  calls.push_back("close-resource");
  return close_error;
}
static int tl_vtcm_open(int) { return 0; }
static int open_error;
static int tl_resource_release_callback(unsigned, void *) { return 0; }
static int tl_vtcm_open_mode(int, int (*callback)(unsigned, void *),
                             void *opaque) {
  assert(callback == tl_resource_release_callback &&
         opaque == &tl_resource_state);
  calls.push_back("open-callback");
  return open_error;
}
static int tl_vtcm_take() {
  assert(prepared && !cooperative && !held);
  calls.push_back("take");
  if (take_error)
    return take_error;
  held = true;
  return 0;
}
static int tl_vtcm_give() {
  assert(prepared && !cooperative && held);
  calls.push_back("give");
  held = false;
  if (give_error)
    tl_resource_fail(&tl_resource_state, TL_ERR_VTCM);
  return give_error;
}
static int tl_vtcm_close() {
  calls.push_back("close-vtcm");
  return 0;
}
static void tl_power_vote() {}
static void tl_power_unvote() { calls.push_back("unvote"); }

// GENERATED_RUN_KERNEL

static int body() {
  assert(prepared);
  calls.push_back("body");
  return body_error;
}

int main(int argc, char **argv) {
  assert(argc == 2);
  const std::string mode = argv[1];
  bool coop = mode.find("cooperative") != std::string::npos;
  if (mode.find("prepare-error") != std::string::npos)
    prepare_error = TL_ERR_RESOURCE;
  if (mode.find("take-error") != std::string::npos)
    take_error = -1;
  if (mode.find("body-error") != std::string::npos)
    body_error = 37;
  if (mode.find("finish-error") != std::string::npos)
    finish_error = TL_ERR_RESOURCE;
  if (mode.find("give-error") != std::string::npos)
    give_error = -1;
  bool free = mode.find("resource-free") == 0;
  auto invoke = [&]() {
    assert(held == !free);
    return body();
  };
  int result = free   ? tl_hexagon_run_kernel<false, false>(invoke)
               : coop ? tl_hexagon_run_kernel<true>(invoke)
                      : tl_hexagon_run_kernel(invoke);
  int expected = prepare_error ? prepare_error
                 : take_error  ? TL_ERR_VTCM
                 : body_error  ? body_error
                 : give_error  ? TL_ERR_VTCM
                               : finish_error;
  assert(result == (expected ? 0x1000 + expected : AEE_SUCCESS));
  std::vector<std::string> order{coop ? "prepare-cooperative"
                                      : "prepare-ordinary"};
  if (!prepare_error) {
    if (!coop && !free)
      order.push_back("take");
    if (!take_error) {
      order.push_back("body");
      if (!coop && !free)
        order.push_back("give");
    }
    order.push_back("finish");
  }
  assert(calls == order);
  assert(!prepared && !held);

  calls.clear();
  close_error = TL_ERR_RESOURCE;
  assert(tl_hexagon_close() == TL_ERR_RESOURCE);
  assert(calls == std::vector<std::string>{"close-resource"});
  calls.clear();
  close_error = 0;
  assert(tl_hexagon_close() == TL_OK);
  assert((calls ==
          std::vector<std::string>{"close-resource", "close-vtcm", "unvote"}));

  // A Session with a cooperative kernel opens its context with the release
  // callback; when that fails, open gives everything back.
  calls.clear();
  open_error = -1;
  assert(tl_hexagon_open() == -1);
  assert((calls == std::vector<std::string>{"open-callback", "close-resource",
                                            "close-vtcm", "unvote"}));
}
