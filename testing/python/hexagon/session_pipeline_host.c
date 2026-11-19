// Exercise the generated ARM pipeline handler, with real pthread launch/join.
// FastRPC and wire I/O are injected; this does not model DSP turn visibility.
#include <assert.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef int AEEResult;
typedef uint64_t remote_handle64;
#define AEE_SUCCESS 0
#define MAX_PROCESSES 4
#define MAX_BUFFERS 4
static int g_processes;
static remote_handle64 g_handles[MAX_PROCESSES];
static uint32_t turn;
static uint8_t *g_turn = (uint8_t *)&turn;
static int g_turn_fd = 99;
static int64_t g_last_run_ns;
static struct {
  void *ptr;
  int fd;
  unsigned procs;
} g_buffers[MAX_BUFFERS];
static int fail_sync, sync_calls, replies, reply_status, allocations, frees;
static int injected[MAX_PROCESSES];
static atomic_int launched;
static pthread_t caller;
static int32_t wire[64];
static size_t wire_size, wire_read;

static int read_full(int fd, void *out, size_t size) {
  assert(fd == 42 && wire_read + size <= wire_size);
  memcpy(out, (uint8_t *)wire + wire_read, size);
  wire_read += size;
  return 0;
}
static int reply(int fd, unsigned char status, const void *data, size_t size) {
  assert(fd == 42 && data == NULL && size == 0);
  ++replies;
  reply_status = status;
  return 0;
}
static void *counted_malloc(size_t size) {
  void *p = malloc(size);
  assert(p);
  ++allocations;
  return p;
}
static void counted_free(void *p) {
  assert(p);
  ++frees;
  free(p);
}
static int tl_session_x_sync(remote_handle64 h, int fd, uint32_t offset,
                             uint32_t size, int writeback) {
  assert(fd == g_turn_fd && offset == 0 && size == 128 && writeback == 0);
  assert(h == (remote_handle64)sync_calls && atomic_load(&launched) == 0);
  assert(turn == 0);
  return sync_calls++ == fail_sync ? 123 : AEE_SUCCESS;
}
static AEEResult tl_session_x_pipeline(remote_handle64 h, int kernel,
                                       const int32_t *args, int args_len,
                                       int steps, int stage, int stages,
                                       int turn_fd) {
  assert(sync_calls == g_processes && fail_sync < 0);
  assert(h == (remote_handle64)stage && kernel == 7 && stages == g_processes);
  assert(args_len == 6 && args[0] == TEST_RESIDENT && args[1] == 100 + stage &&
         args[2] == 128 && args[3] == TEST_STEP && args[4] == 0 &&
         args[5] == 0);
  assert(steps == 4 && turn_fd == g_turn_fd && turn == 0);
  assert((stages == 1) == !!pthread_equal(pthread_self(), caller));
  atomic_fetch_add(&launched, 1);
  return injected[stage];
}

#define malloc counted_malloc
#define free counted_free
// GENERATED_PIPELINE_LAUNCH
static int serve_pipeline(int cli, unsigned char op) {
  // GENERATED_PIPELINE_HANDLER
  return 1;
}
#undef malloc
#undef free

static void run_case(int processes, int sync_failure, int a, int b, int c,
                     int expected) {
  g_processes = processes;
  fail_sync = sync_failure;
  sync_calls = replies = allocations = frees = 0;
  atomic_store(&launched, 0);
  injected[0] = a;
  injected[1] = b;
  injected[2] = c;
  turn = UINT32_MAX; // Every attempt must initialize the turn anew.
  caller = pthread_self();
  wire[0] = processes;
  wire[1] = 4;
  size_t n = 2;
  for (int p = 0; p < processes; ++p) {
    g_handles[p] = p;
    g_buffers[p].ptr = &turn;
    g_buffers[p].fd = 100 + p;
    g_buffers[p].procs = 1u << p;
    int32_t stage[] = {7, p, 2, TEST_RESIDENT, p, 128, TEST_STEP, 0, 0};
    memcpy(wire + n, stage, sizeof(stage));
    n += sizeof(stage) / sizeof(stage[0]);
  }
  wire_size = n * sizeof(wire[0]);
  wire_read = 0;
  assert(serve_pipeline(42, TEST_PIPELINE_OP) == 0);
  assert(wire_read == wire_size && replies == 1 && reply_status == expected);
  assert(allocations == processes && frees == allocations);
  assert(atomic_load(&launched) == (sync_failure < 0 ? processes : 0));
  assert(sync_calls == (sync_failure < 0 ? processes : sync_failure + 1));
}

int main(int argc, char **argv) {
  // Isolate either fault when checking that an older handler fails this test.
  if (argc == 2) {
    assert(strcmp(argv[1], "sync") == 0 || strcmp(argv[1], "giveup") == 0);
    if (strcmp(argv[1], "sync") == 0)
      run_case(1, 0, 0, 0, 0, 255);
    else
      run_case(1, -1, TEST_TURN_GAVE_UP, 0, 0, 255);
    return 0;
  }
  assert(argc == 1);
  run_case(1, -1, 0, 0, 0, 0);
  run_case(3, -1, 0, 0, 0, 0);
  run_case(1, 0, 0, 0, 0, 255);
  run_case(3, 1, 0, 0, 0, 255);
  run_case(3, 2, 0, 0, 0, 255);
  run_case(1, -1, TEST_TURN_GAVE_UP, 0, 0, 255);
  run_case(3, -1, 0, TEST_TURN_GAVE_UP, 0, 255);
  run_case(3, -1, TEST_TURN_GAVE_UP, TEST_TURN_GAVE_UP, TEST_TURN_GAVE_UP, 255);
  run_case(3, -1, TEST_TURN_GAVE_UP, TEST_KERNEL_STATUS_BASE + 7,
           TEST_KERNEL_STATUS_BASE + 8, 7);
  run_case(3, -1, TEST_KERNEL_STATUS_BASE + 8, TEST_TURN_GAVE_UP,
           TEST_KERNEL_STATUS_BASE + 7, 8);
  run_case(3, -1, TEST_TURN_GAVE_UP, 123, 0, 255);
  // Failures must not retain request allocations or poison the next request.
  run_case(1, -1, 0, 0, 0, 0);
  run_case(3, -1, 0, 0, 0, 0);
  puts("13 generated pipeline cases passed");
  return 0;
}
