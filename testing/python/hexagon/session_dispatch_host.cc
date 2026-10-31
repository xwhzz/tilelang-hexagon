// Host lifecycle tests for generated dispatch code. QuRT's primitives are
// modelled by real host threads/semaphores; device tests verify stack capacity.
#include <assert.h>
#include <condition_variable>
#include <cstdlib>
#include <mutex>
#include <string>
#include <thread>
#include <type_traits>

using AEEResult = int;
enum { AEE_SUCCESS, AEE_EFAILED, AEE_ENOMEMORY, AEE_EBADPARM, QURT_EOK = 0 };
using qurt_thread_t = std::thread *;
struct qurt_thread_attr_t {
  unsigned stack_size;
  void *stack_addr;
  unsigned priority;
};
struct qurt_sem_t {
  std::mutex mutex;
  std::condition_variable ready;
  unsigned count = 0;
};
static std::string failure;
static unsigned caller_stack, allocations, frees, created, joined, semaphores;
static unsigned opened, closed;
static unsigned votes, unvotes;
static std::thread::id owner, power_owner;
static bool fail_join, resources_live;

static void qurt_sem_init_val(qurt_sem_t *s, unsigned count) {
  s->count = count;
  ++semaphores;
}
static void qurt_sem_destroy(qurt_sem_t *) { --semaphores; }
static void qurt_sem_up(qurt_sem_t *s) {
  std::lock_guard<std::mutex> lock(s->mutex);
  ++s->count;
  s->ready.notify_one();
}
static void qurt_sem_down(qurt_sem_t *s) {
  std::unique_lock<std::mutex> lock(s->mutex);
  s->ready.wait(lock, [s] { return s->count; });
  --s->count;
}
static qurt_thread_t qurt_thread_get_id() { return nullptr; }
static int qurt_thread_attr_get(qurt_thread_t, qurt_thread_attr_t *attr) {
  *attr = {caller_stack, nullptr, 240};
  return failure == "query" ? -1 : QURT_EOK;
}
static void qurt_thread_attr_init(qurt_thread_attr_t *attr) { *attr = {}; }
static void qurt_thread_attr_set_stack_addr(qurt_thread_attr_t *attr, void *p) {
  attr->stack_addr = p;
}
static void qurt_thread_attr_set_stack_size(qurt_thread_attr_t *attr,
                                            unsigned bytes) {
  attr->stack_size = bytes;
}
static void qurt_thread_attr_set_priority(qurt_thread_attr_t *attr,
                                          unsigned priority) {
  attr->priority = priority;
}
static int qurt_thread_create(qurt_thread_t *thread, qurt_thread_attr_t *attr,
                              void (*entry)(void *), void *arg) {
  assert(attr->stack_addr && attr->stack_size == 256 * 1024);
  assert(attr->priority == 240);
  if (failure == "create")
    return -1;
  *thread = new std::thread(entry, arg);
  ++created;
  return QURT_EOK;
}
static int qurt_thread_join(qurt_thread_t thread, int *status) {
  if (fail_join) {
    fail_join = false;
    return -1;
  }
  thread->join();
  delete thread;
  ++joined;
  *status = 0;
  return QURT_EOK;
}
static void *test_malloc(size_t bytes) {
  if (failure == "allocate")
    return nullptr;
  ++allocations;
  return std::malloc(bytes);
}
static void test_free(void *p) {
  assert(p);
  ++frees;
  std::free(p);
}
#define TL_HEXAGON_USES_VTCM 1
#define TL_OK 0
#define TL_ERR_VTCM 1
static bool context_live;
static int tl_vtcm_open(int) {
  assert(votes == 1 && unvotes == 0 && !context_live);
  context_live = true;
  return 0;
}
static int tl_vtcm_take() {
  assert(context_live && !resources_live);
  assert(power_owner == std::this_thread::get_id());
  if (failure == "resources" || failure == "take")
    return -1;
  owner = std::this_thread::get_id();
  resources_live = true;
  ++opened;
  return 0;
}
static void tl_vtcm_give() {
  assert(resources_live && owner == std::this_thread::get_id());
  resources_live = false;
  ++closed;
}
static void tl_vtcm_close() {
  assert(context_live && !resources_live);
  context_live = false;
}
static void tl_power_vote() {
  assert(votes == 0);
  power_owner = std::this_thread::get_id();
  ++votes;
}
static void tl_power_unvote() {
  assert(!resources_live && power_owner == std::this_thread::get_id());
  ++unvotes;
}
static int test_kernel(void **args) {
  assert(resources_live && owner == std::this_thread::get_id());
  assert(votes == 1 && unvotes == 0);
  int *value = static_cast<int *>(args[0]);
  ++*value;
  return *value == 20 ? 37 : AEE_SUCCESS;
}

#define malloc test_malloc
#define free test_free
// GENERATED_DISPATCH
#undef malloc
#undef free

int main(int argc, char **argv) {
  assert(argc == 3);
  caller_stack = std::string(argv[1]) == "direct" ? 512 * 1024 : 16 * 1024;
  failure = argv[2];
  std::string requested = failure;
  if (failure == "take")
    failure = "none";
  int status = tl_session_dispatch_open();
  failure = requested;
  bool startup_failure =
      failure != "none" && failure != "join" && failure != "take";
  if (startup_failure) {
    assert(status == (failure == "allocate" ? AEE_ENOMEMORY : AEE_EFAILED));
  } else {
    assert(status == AEE_SUCCESS);
    int value = 0;
    void *args[] = {&value};
    assert(!resources_live && context_live);
    assert(votes == 1 && unvotes == 0);
    if (failure == "take") {
      failure = "resources";
      assert(tl_session_dispatch_run(0, args) == 0x1001);
      assert(value == 0 && opened == closed && !resources_live);
      failure = "none";
    }
    for (int i = 1; i <= 20; ++i) {
      assert(tl_session_dispatch_run(0, args) ==
             (i == 20 ? 0x1000 + 37 : AEE_SUCCESS));
      assert(value == i);
      assert(!resources_live);
      assert(opened == closed);
      assert(votes == 1 && unvotes == 0);
    }
    assert(tl_session_dispatch_run(1, args) == AEE_EBADPARM);
    if (failure == "join") {
      fail_join = true;
      assert(tl_session_dispatch_close() == AEE_EFAILED);
      assert(allocations == 1 && frees == 0);
    }
    assert(tl_session_dispatch_close() == AEE_SUCCESS);
  }
  assert(allocations == frees && created == joined);
  assert(semaphores == 0 && opened == closed && votes == unvotes);
  assert(!resources_live);
  if (caller_stack >= 256 * 1024)
    assert(allocations == 0 && created == 0);
}
