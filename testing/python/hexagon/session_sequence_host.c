// Execute the generated ARM agent natively with a cache-aware fake DSP.
// Each allocation has distinct CPU and DSP bytes. Omitting either side of
// Copy's cache protocol corrupts the resident recurrence tested by Python.
#include "agent.c"
#include <assert.h>

struct allocation {
  uint8_t *host, *dsp, *valid, *dirty;
  size_t size, lines;
  int fd;
};
static struct allocation memory[1024];
static int next_fd = 100, runs, acquires, releases, lease, sync_failure;

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
      m->dsp = calloc(1, m->size);
      m->valid = calloc(m->lines, 1);
      m->dirty = calloc(m->lines, 1);
      m->fd = next_fd++;
      assert(m->host && m->dsp && m->valid && m->dirty);
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
      free(memory[i].dsp);
      free(memory[i].valid);
      free(memory[i].dirty);
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
  *h = 1;
  return 0;
}
int seq_test_close(remote_handle64 h) {
  (void)h;
  assert(!lease && acquires == releases);
  return 0;
}
int seq_test_release(remote_handle64 h, int32_t fd) {
  (void)h;
  (void)fd;
  return 0;
}

static void load_line(struct allocation *m, size_t line) {
  if (m->valid[line])
    return;
  size_t start = line * 128, n = m->size - start;
  if (n > 128)
    n = 128;
  memcpy(m->dsp + start, m->host + start, n);
  m->valid[line] = 1;
}
int seq_test_sync(remote_handle64 h, int32_t fd, uint32_t off, uint32_t size,
                  int32_t writeback) {
  (void)h;
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
    if (m->valid[line] && m->dirty[line])
      memcpy(m->host + start, m->dsp + start, n);
    m->dirty[line] = 0;
    if (!writeback)
      m->valid[line] = 0;
  }
  return 0;
}
int seq_test_run(remote_handle64 h, int32_t kernel, const int32_t *args, int n,
                 const uint8_t *in, int ni, uint8_t *out, int no) {
  (void)h;
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
  load_line(m, offset / 128);
  int32_t *values = (int32_t *)(m->dsp + offset);
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
  }
  m->dirty[offset / 128] = 1;
  ++runs;
  ++releases;
  lease = 0;
  return status;
}
