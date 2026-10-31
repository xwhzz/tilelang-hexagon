// Execute the generated ARM agent natively with a cache-aware fake DSP.
// Each allocation has distinct CPU and DSP bytes. Omitting either side of
// Copy's cache protocol corrupts the resident recurrence tested by Python.
#include <pthread.h>
static int test_pthread_create(pthread_t *, const pthread_attr_t *,
                               void *(*)(void *), void *);
static int test_pthread_join(pthread_t, void **);
#define pthread_create test_pthread_create
#define pthread_join test_pthread_join
#include "agent.c"
#undef pthread_create
#undef pthread_join
#include <assert.h>

struct allocation {
  uint8_t *host, *dsp[MAX_PROCESSES], *valid[MAX_PROCESSES],
      *dirty[MAX_PROCESSES];
  size_t size, lines;
  int fd;
};
static struct allocation memory[1024];
static int next_fd = 100, runs, acquires, releases, lease, sync_failure;
static int pipeline_calls, thread_attempts, threads_created, threads_joined;
static int test_pthread_create(pthread_t *thread, const pthread_attr_t *attr,
                               void *(*entry)(void *), void *arg) {
  const char *fail_after = getenv("TL_TEST_FAIL_CREATE_AFTER");
  if (fail_after && thread_attempts++ == atoi(fail_after))
    return 11;
  int status = pthread_create(thread, attr, entry, arg);
  if (status == 0)
    ++threads_created;
  return status;
}
static int test_pthread_join(pthread_t thread, void **result) {
  int status = pthread_join(thread, result);
  if (status == 0)
    ++threads_joined;
  return status;
}

static struct allocation *by_fd(int fd) {
  for (int i = 0; i < 1024; ++i)
    if (memory[i].host && memory[i].fd == fd)
      return &memory[i];
  return 0;
}
void *rpcmem_alloc(int heap, int flags, int size) {
  (void)heap;
  (void)flags;
  for (int i = 0; i < 1024; ++i)
    if (!memory[i].host) {
      struct allocation *m = &memory[i];
      m->size = (size_t)size;
      m->lines = (m->size + 127) / 128;
      m->host = calloc(1, m->size);
      for (int p = 0; p < MAX_PROCESSES; ++p) {
        m->dsp[p] = calloc(1, m->size);
        m->valid[p] = calloc(m->lines, 1);
        m->dirty[p] = calloc(m->lines, 1);
        assert(m->dsp[p] && m->valid[p] && m->dirty[p]);
      }
      m->fd = next_fd++;
      assert(m->host);
      return m->host;
    }
  return 0;
}
int rpcmem_to_fd(void *p) {
  for (int i = 0; i < 1024; ++i)
    if (memory[i].host == p)
      return memory[i].fd;
  return -1;
}
void rpcmem_free(void *p) {
  for (int i = 0; i < 1024; ++i)
    if (memory[i].host == p) {
      free(memory[i].host);
      for (int p = 0; p < MAX_PROCESSES; ++p) {
        free(memory[i].dsp[p]);
        free(memory[i].valid[p]);
        free(memory[i].dirty[p]);
      }
      memset(&memory[i], 0, sizeof(memory[i]));
      return;
    }
  assert(!p);
}
int fastrpc_mmap(int d, int fd, void *p, int offset, uint64_t n, int flags) {
  (void)d;
  (void)fd;
  (void)p;
  (void)offset;
  (void)n;
  (void)flags;
  return 0;
}
int fastrpc_munmap(int d, int fd, void *p, uint64_t n) {
  (void)d;
  (void)fd;
  (void)p;
  (void)n;
  return 0;
}
int seq_test_open(const char *uri, remote_handle64 *h) {
  (void)uri;
  static int opened;
  *h = (remote_handle64)opened++;
  return 0;
}
int seq_test_close(remote_handle64 h) {
  (void)h;
  assert(!lease && acquires == releases);
  assert(threads_created == threads_joined);
  return 0;
}
int seq_test_release(remote_handle64 h, int32_t fd) {
  (void)h;
  (void)fd;
  return 0;
}

static void load_line(struct allocation *m, int p, size_t line) {
  if (m->valid[p][line])
    return;
  size_t start = line * 128, n = m->size - start;
  if (n > 128)
    n = 128;
  memcpy(m->dsp[p] + start, m->host + start, n);
  m->valid[p][line] = 1;
}
int seq_test_sync(remote_handle64 h, int32_t fd, uint32_t off, uint32_t size,
                  int32_t writeback) {
  int p = (int)h;
  if (sync_failure && --sync_failure == 0)
    return (int)0x80000073u;
  struct allocation *m = by_fd(fd);
  if (!m || off > m->size || size > m->size - off)
    return AEE_EBADPARM;
  if (!size)
    return 0;
  for (size_t line = off / 128; line <= ((size_t)off + size - 1) / 128;
       ++line) {
    size_t start = line * 128, n = m->size - start;
    if (n > 128)
      n = 128;
    if (m->valid[p][line] && m->dirty[p][line])
      memcpy(m->host + start, m->dsp[p] + start, n);
    m->dirty[p][line] = 0;
    if (!writeback)
      m->valid[p][line] = 0;
  }
  return 0;
}
int seq_test_run(remote_handle64 h, int32_t kernel, const int32_t *args, int n,
                 const uint8_t *in, int ni, uint8_t *out, int no) {
  int p = (int)h;
  (void)in;
  (void)ni;
  (void)out;
  (void)no;
  assert(n == (kernel == 3 ? 3 : 6));
  assert(!lease);
  lease = 1;
  ++acquires;
  struct allocation *m = by_fd(args[1]);
  assert(m && args[0] == 0);
  size_t offset = (uint32_t)args[2];
  load_line(m, p, offset / 128);
  int32_t *values = (int32_t *)(m->dsp[p] + offset);
  int status = 0, scalar = kernel == 3 ? 0 : args[4];
  if (kernel == 0) {
    values[1] = values[0] + scalar;
    values[31] +=
        1; // Dirty adjacent bytes must survive a four-byte feedback copy.
  } else if (kernel == 1) {
    values[2] += 1;
    status = scalar;
  } else if (kernel == 2) {
    sync_failure = scalar;
  } else {
    values[0] = runs;
    values[1] = acquires - 1;
    values[2] = releases;
    values[3] = pipeline_calls;
    values[4] = threads_created;
    values[5] = threads_joined;
  }
  m->dirty[p][offset / 128] = 1;
  ++runs;
  ++releases;
  lease = 0;
  return status;
}

int remote_session_control(int op, void *arg, unsigned int size) {
  (void)size;
  if (op == FASTRPC_RESERVE_NEW_SESSION) {
    remote_rpc_reserve_new_session_t *r = arg;
    r->effective_domain_id = 11;
    r->session_id = 1;
  } else if (op == FASTRPC_GET_URI) {
    remote_rpc_get_uri_t *u = arg;
    u->uri[0] = 0;
  }
  return AEE_SUCCESS;
}
int seq_test_pipeline(remote_handle64 h, int32_t kernel, const int32_t *args,
                      int n, int32_t steps, int32_t stage, int32_t stages,
                      int32_t turn) {
  (void)h;
  (void)kernel;
  (void)args;
  (void)n;
  (void)stage;
  (void)stages;
  (void)turn;
  __atomic_add_fetch(&pipeline_calls, 1, __ATOMIC_SEQ_CST);
  return steps == 0 ? AEE_SUCCESS : AEE_EBADPARM;
}
