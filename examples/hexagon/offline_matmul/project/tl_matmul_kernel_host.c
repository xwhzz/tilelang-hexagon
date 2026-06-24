// Auto-generated HLOS host driver for a tilelang Hexagon kernel.
#include <AEEStdErr.h>
#include <remote.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "dsp_capabilities_utils.h"
#include "tl_matmul_kernel.h"

static remote_handle64 g_handle = -1;

static unsigned char* readfile(const char* path, long* len) {
  FILE* f = fopen(path, "rb"); if (!f) { fprintf(stderr, "open %s failed\n", path); exit(2); }
  fseek(f, 0, SEEK_END); *len = ftell(f); fseek(f, 0, SEEK_SET);
  unsigned char* b = (unsigned char*)malloc(*len);
  if (fread(b, 1, *len, f) != (size_t)*len) { fprintf(stderr, "read failed\n"); exit(2); }
  fclose(f); return b;
}

static void writefile(const char* path, const unsigned char* b, int len) {
  FILE* f = fopen(path, "wb"); if (!f) { fprintf(stderr, "write %s failed\n", path); exit(2); }
  fwrite(b, 1, len, f); fclose(f);
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
  (void)argc;
  if (open_session(CDSP_DOMAIN_ID) != 0) return 1;
  long p0_len = 0;
  unsigned char* p0 = readfile(argv[1], &p0_len);
  long p1_len = 0;
  unsigned char* p1 = readfile(argv[2], &p1_len);
  unsigned char* p2 = (unsigned char*)malloc(131072);
  const char* p2_path = argv[3];
  int e = tl_matmul_kernel_run(g_handle, p0, (int)p0_len, p1, (int)p1_len, p2, 131072);
  if (e != AEE_SUCCESS) { fprintf(stderr, "run failed: 0x%x\n", e); }
  else {
  writefile(p2_path, p2, 131072);
  }
  free(p0);
  free(p1);
  free(p2);
  tl_matmul_kernel_close(g_handle);
  return e ? 1 : 0;
}
