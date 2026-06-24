// Auto-generated persistent FastRPC agent for a tilelang Hexagon kernel.
#include <AEEStdErr.h>
#include <remote.h>
#include "rpcmem.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include "dsp_capabilities_utils.h"
#include "tl_matmul_kernel.h"

static remote_handle64 g_handle = -1;

static int read_full(int fd, void* buf, size_t n) {
  size_t got = 0;
  while (got < n) { ssize_t r = read(fd, (char*)buf + got, n - got); if (r <= 0) return -1; got += (size_t)r; }
  return 0;
}
static int write_full(int fd, const void* buf, size_t n) {
  size_t put = 0;
  while (put < n) { ssize_t w = write(fd, (const char*)buf + put, n - put); if (w <= 0) return -1; put += (size_t)w; }
  return 0;
}

static int open_session(int domain_id) {
  domain* d = get_domain(domain_id);
  if (!d) { fprintf(stderr, "get_domain failed\n"); return -1; }
  if (&remote_session_control) {
    struct remote_rpc_control_unsigned_module c; c.domain = domain_id; c.enable = 1;
    int e = remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &c, sizeof(c));
    if (e != AEE_SUCCESS) { fprintf(stderr, "unsigned-PD failed: 0x%x\n", e); return -1; }
  }
  int len = strlen(tl_matmul_kernel_URI) + MAX_DOMAIN_URI_SIZE;
  char* uri = (char*)malloc(len);
  snprintf(uri, len, "%s%s", tl_matmul_kernel_URI, d->uri);
  int e = tl_matmul_kernel_open(uri, &g_handle); free(uri);
  if (e != AEE_SUCCESS) { fprintf(stderr, "open failed: 0x%x\n", e); return -1; }
  return 0;
}

int main(int argc, char** argv) {
  int port = (argc > 1) ? atoi(argv[1]) : 9777;
  if (open_session(CDSP_DOMAIN_ID) != 0) return 1;
  unsigned char* rbuf_p0 = (unsigned char*)rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, 131072);
  if (!rbuf_p0) { fprintf(stderr, "rpcmem_alloc failed\n"); return 3; }
  memset(rbuf_p0, 0, 131072);  // avoid garbage if a buffer is masked-out before first send
  unsigned char* rbuf_p1 = (unsigned char*)rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, 131072);
  if (!rbuf_p1) { fprintf(stderr, "rpcmem_alloc failed\n"); return 3; }
  memset(rbuf_p1, 0, 131072);  // avoid garbage if a buffer is masked-out before first send
  unsigned char* rbuf_p2 = (unsigned char*)rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, 131072);
  if (!rbuf_p2) { fprintf(stderr, "rpcmem_alloc failed\n"); return 3; }
  memset(rbuf_p2, 0, 131072);  // avoid garbage if a buffer is masked-out before first send
  int srv = socket(AF_INET, SOCK_STREAM, 0);
  int opt = 1; setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
  struct sockaddr_in addr; memset(&addr, 0, sizeof(addr));
  addr.sin_family = AF_INET; addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK); addr.sin_port = htons((unsigned short)port);
  if (bind(srv, (struct sockaddr*)&addr, sizeof(addr)) < 0) { perror("bind"); return 2; }
  listen(srv, 1);
  printf("AGENT_READY port=%d\n", port); fflush(stdout);
  int running = 1;
  while (running) {
    int cli = accept(srv, 0, 0);
    if (cli < 0) continue;
    for (;;) {
      unsigned char op, mask;
      if (read_full(cli, &op, 1)) break;
      if (op == 0) break;
      if (op == 2) { running = 0; break; }
      if (read_full(cli, &mask, 1)) break;
      if ((mask >> 0) & 1) { if (read_full(cli, rbuf_p0, 131072)) break; }
      if ((mask >> 1) & 1) { if (read_full(cli, rbuf_p1, 131072)) break; }
      int e = tl_matmul_kernel_run(g_handle, rbuf_p0, 131072, rbuf_p1, 131072, rbuf_p2, 131072);
      unsigned char status = (e == AEE_SUCCESS) ? 0 : 1;  // status byte so the client never hangs on a failed run
      if (write_full(cli, &status, 1)) break;
      if (e == AEE_SUCCESS) {
        write_full(cli, rbuf_p2, 131072);
      }
    }
    close(cli);
  }
  rpcmem_free(rbuf_p0);
  rpcmem_free(rbuf_p1);
  rpcmem_free(rbuf_p2);
  tl_matmul_kernel_close(g_handle);
  return 0;
}
