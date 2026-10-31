// Exercise the actual arena header with a grant-tracking HAP implementation.
// Cached take/give must release each grant, including failed arena binding.
#include <cassert>
#include <cstdint>
#include <cstring>
#include <string>

struct compute_res_attr_t {
  bool cached;
};
using compute_res_vtcm_page_t = unsigned;
static std::string failure;
static bool context_live, grant_live, context_cached;
static unsigned contexts, releases, takes, gives;
alignas(2048) static unsigned char arena[8192];
static int HAP_compute_res_query_VTCM(int, unsigned *total, unsigned *,
                                      unsigned *, unsigned *) {
  *total = sizeof(arena);
  return failure == "query" ? -1 : 0;
}
static void HAP_compute_res_attr_init(compute_res_attr_t *req) {
  req->cached = false;
}
static void HAP_compute_res_attr_set_vtcm_param(compute_res_attr_t *, unsigned,
                                                int) {}
static void *HAP_compute_res_attr_get_vtcm_ptr(compute_res_attr_t *) {
  return failure == "bind" ? nullptr : arena;
}
static int HAP_compute_res_acquire(compute_res_attr_t *req, unsigned) {
  assert(!context_live && !grant_live);
  if (failure == "acquire")
    return 0;
  context_live = true;
  context_cached = req->cached;
  grant_live = !context_cached;
  ++contexts;
  if (grant_live)
    ++takes;
  return 17;
}
static void HAP_compute_res_release(int ctx) {
  assert(ctx == 17 && context_live);
  assert(grant_live == !context_cached);
  if (grant_live)
    ++gives;
  context_live = grant_live = false;
  ++releases;
}
static void HAP_compute_res_attr_set_hmx_param(compute_res_attr_t *, int) {}
static void HAP_compute_res_attr_set_cache_mode(compute_res_attr_t *req,
                                                int cached) {
  req->cached = cached;
}
static int HAP_compute_res_acquire_cached(int ctx, unsigned) {
  assert(ctx == 17 && context_live && context_cached && !grant_live);
  if (failure == "take")
    return -1;
  grant_live = true;
  ++takes;
  return 0;
}
static void HAP_compute_res_release_cached(int ctx) {
  assert(ctx == 17 && grant_live);
  grant_live = false;
  ++gives;
}
#include <tl_templates/hexagon/vtcm.h>

int main() {
  for (const char *mode : {"none", "query", "acquire", "bind", "take"}) {
    failure = mode;
    int opened = tl_vtcm_open(0);
    if (opened == 0) {
      int got = tl_vtcm_take();
      bool expected = failure == "none";
      assert((got == 0) == expected);
      if (got == 0) {
        assert(grant_live && tl_vtcm_arena() == arena);
        assert(tl_vtcm_arena_bytes() == sizeof(arena));
        tl_vtcm_give();
        assert(!grant_live && !tl_vtcm_arena() && !tl_vtcm_arena_bytes());
        assert(tl_vtcm_take() == 0);
        tl_vtcm_give();
      }
    } else {
      assert(failure == "query" || failure == "acquire");
    }
    tl_vtcm_close();
    assert(!context_live && !grant_live && !tl_vtcm_arena());
    assert(contexts == releases && takes == gives);
  }
}
