"""Device sessions: several kernels and their resident buffers in one FastRPC session.

A :class:`Session` holds one agent process on the HLOS and one FastRPC
session on the cDSP (one PD, one VTCM grant, one worker pool), with every
kernel it was given built into one skel. Its :class:`DeviceBuffer` objects live
in the session: rpcmem the agent maps into the DSP once (``fastrpc_mmap`` with
``FASTRPC_MAP_FD``), which kernels then reach by file descriptor, without the
cache maintenance FastRPC does for the buffers of every call. Weights, KV
caches and recurrent state stay on the device across calls and kernels.

A kernel parameter takes, per call, a :class:`DeviceBuffer` (resident: no
copy), a tensor (copied in, as with a standalone kernel), or for an int32
scalar a Python int. Its ``out_idx`` parameters come back as tensors unless a
:class:`DeviceBuffer` is passed for them too.

The host writes a buffer's bytes with :meth:`DeviceBuffer.write`, or has the
agent read them from a byte range of a file on the device
(:meth:`DeviceBuffer.load`); the DSP first writes back and drops its copies of
the lines written, so neither a stale copy nor a dirty line a kernel left
survives the write.
:meth:`DeviceBuffer.read` writes the DSP's lines back first. The buffers are
uncached on the CPU, so the host needs no cache maintenance of its own.

    session = Session([(prefill, [3]), (decode, [4])])
    weights = session.buffer(nbytes)
    weights.load("/data/local/tmp/model.bin")
    logits = session.kernels[0](tokens, weights, state)
"""

from __future__ import annotations

import atexit
import contextlib
import functools
import hashlib
import json
import operator
import os
import re
import select
import shutil
import socket
import struct
import tempfile
import time
import weakref
import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

from . import config, fastrpc
from .device import DEVICE_LOCK
from . import sequence
from .sequence import Copy, Run, SequenceError, SequenceResult, SequenceTiming  # noqa: F401

# Seconds one call to the agent may take, its transfers included.
CALL_TIMEOUT = 600.0

# Parameter kinds of a run, with their two values.
_RESIDENT = 0  # (fd, byte offset)
_INPUT = 1  # (byte offset into the inputs, 0)
_OUTPUT = 2  # (byte offset into the outputs, 0)
_SCALAR = 3  # (value, 0)

# Agent operations.
_OP_CLOSE, _OP_RUN, _OP_SHUTDOWN, _OP_ALLOC, _OP_FREE, _OP_WRITE, _OP_READ, _OP_LOAD, _OP_TIMING = range(9)

# Inline arguments start on register boundaries, as kernels assume.
_ALIGN = 128
_MAX_BUFFERS = 256

_KERNEL_RE = re.compile(r"^\s*(?:void|int|int32_t)\s+(\w+)\s*\(", re.MULTILINE)


def _align(n: int) -> int:
    return (n + _ALIGN - 1) // _ALIGN * _ALIGN


# ---------------------------------------------------------------------------
# The project: IDL, skel and agent of a set of kernels.
# ---------------------------------------------------------------------------


def gen_idl(iface: str) -> str:
    return (
        '#include "AEEStdDef.idl"\n'
        '#include "remote.idl"\n\n'
        f"interface {iface} : remote_handle64 {{\n"
        "   AEEResult run(in int32 kernel, in sequence<int32> args, in sequence<uint8> inputs, rout sequence<uint8> outputs);\n"
        "   AEEResult sync(in int32 fd, in uint32 offset, in uint32 size, in int32 writeback);\n"
        "   AEEResult release(in int32 fd);\n"
        "};\n"
    )


def _dispatch_source(calls: str) -> str:
    """Keep large kernel frames off a FastRPC thread whose requested stack
    was not applied. The Session protocol serializes calls; semaphore handoffs
    retain arguments until completion and publish the kernel's status."""
    return f"""
static constexpr bool tl_session_resources_per_call = {str(config.resource_lifetime() == "call").lower()};

// Prevent kernel frames from being inlined into the FastRPC dispatcher.
static __attribute__((noinline)) AEEResult tl_session_invoke(int kernel, void** a) {{
  switch (kernel) {{
{calls}  default: return AEE_EBADPARM;
  }}
}}

static AEEResult tl_session_resources_open() {{
{fastrpc.SKEL_RESOURCES_OPEN}  return AEE_SUCCESS;
}}
static void tl_session_resources_close() {{
{fastrpc.SKEL_RESOURCES_CLOSE}}}

static AEEResult tl_session_execution_open() {{
  tl_power_vote();
  return tl_session_resources_per_call ? AEE_SUCCESS : tl_session_resources_open();
}}
static void tl_session_execution_close() {{
  if (!tl_session_resources_per_call) tl_session_resources_close();
  tl_power_unvote();
}}

static AEEResult tl_session_execute(int kernel, void** args) {{
  if (!tl_session_resources_per_call) return tl_session_invoke(kernel, args);
  AEEResult status = tl_session_resources_open();
  if (status == AEE_SUCCESS) status = tl_session_invoke(kernel, args);
  // invoke has joined its workers and drained their DMA queues before return.
  // Release even after an acquisition or kernel error, before reporting done.
  tl_session_resources_close();
  return status;
}}

// A session executes either on its FastRPC caller or one persistent thread.
// HMX enable/disable must run on the same execution thread as the kernels.
static struct {{
  qurt_thread_t thread;
  qurt_sem_t go, done;
  void* stack;
  int started, stopping, kernel;
  void** args;
  AEEResult status;
}} tl_session_dispatch;

static void tl_session_worker(void*) {{
  tl_session_dispatch.status = tl_session_execution_open();
  qurt_sem_up(&tl_session_dispatch.done);
  if (tl_session_dispatch.status != AEE_SUCCESS) {{
    tl_session_execution_close();
    return;
  }}
  for (;;) {{
    qurt_sem_down(&tl_session_dispatch.go);
    if (tl_session_dispatch.stopping) break;
    tl_session_dispatch.status = tl_session_execute(tl_session_dispatch.kernel, tl_session_dispatch.args);
    qurt_sem_up(&tl_session_dispatch.done);
  }}
  tl_session_execution_close();
}}

static AEEResult tl_session_join_worker() {{
  int status;
  // Never free a stack whose thread has not been joined successfully.
  if (qurt_thread_join(tl_session_dispatch.thread, &status) != QURT_EOK) return AEE_EFAILED;
  tl_session_dispatch.started = 0;
  qurt_sem_destroy(&tl_session_dispatch.go);
  qurt_sem_destroy(&tl_session_dispatch.done);
  free(tl_session_dispatch.stack);
  tl_session_dispatch.stack = 0;
  return AEE_SUCCESS;
}}

static AEEResult tl_session_dispatch_open() {{
  qurt_thread_attr_t caller;
  if (qurt_thread_attr_get(qurt_thread_get_id(), &caller) != QURT_EOK) return AEE_EFAILED;
  if (caller.stack_size >= {fastrpc.PD_THREAD_STACK}) {{
    AEEResult status = tl_session_execution_open();
    if (status != AEE_SUCCESS) tl_session_execution_close();
    return status;
  }}
  tl_session_dispatch.stack = malloc({fastrpc.PD_THREAD_STACK});
  if (!tl_session_dispatch.stack) return AEE_ENOMEMORY;
  tl_session_dispatch.stopping = 0;
  qurt_sem_init_val(&tl_session_dispatch.go, 0);
  qurt_sem_init_val(&tl_session_dispatch.done, 0);
  qurt_thread_attr_t worker;
  qurt_thread_attr_init(&worker);
  qurt_thread_attr_set_stack_addr(&worker, tl_session_dispatch.stack);
  qurt_thread_attr_set_stack_size(&worker, {fastrpc.PD_THREAD_STACK});
  qurt_thread_attr_set_priority(&worker, caller.priority);
  if (qurt_thread_create(&tl_session_dispatch.thread, &worker, tl_session_worker, 0) != QURT_EOK) {{
    qurt_sem_destroy(&tl_session_dispatch.go);
    qurt_sem_destroy(&tl_session_dispatch.done);
    free(tl_session_dispatch.stack);
    tl_session_dispatch.stack = 0;
    return AEE_EFAILED;
  }}
  tl_session_dispatch.started = 1;
  qurt_sem_down(&tl_session_dispatch.done);
  AEEResult status = tl_session_dispatch.status;
  if (status != AEE_SUCCESS) {{
    AEEResult joined = tl_session_join_worker();
    if (joined != AEE_SUCCESS) return joined;
  }}
  return status;
}}

static AEEResult tl_session_dispatch_run(int kernel, void** args) {{
  if (!tl_session_dispatch.started) return tl_session_execute(kernel, args);
  tl_session_dispatch.kernel = kernel;
  tl_session_dispatch.args = args;
  qurt_sem_up(&tl_session_dispatch.go);
  qurt_sem_down(&tl_session_dispatch.done);
  return tl_session_dispatch.status;
}}

static AEEResult tl_session_dispatch_close() {{
  if (!tl_session_dispatch.started) {{
    tl_session_execution_close();
    return AEE_SUCCESS;
  }}
  tl_session_dispatch.stopping = 1;
  qurt_sem_up(&tl_session_dispatch.go);
  return tl_session_join_worker();
}}
"""


def gen_dsp(iface: str, kernels: Sequence[tuple[str, str, list]]) -> str:
    """The skel of `kernels`, (name, source, plans) each: one translation unit,
    every kernel's entry renamed to its index."""
    parts = [
        "// Auto-generated FastRPC skel of a tilelang Hexagon session.\n",
        "#include <AEEStdErr.h>\n#include <HAP_mem.h>\n#include <qurt.h>\n#include <stdint.h>\n#include <stdlib.h>\n#include <type_traits>\n",
        f'#include "{iface}.h"\n\n',
        fastrpc.SKEL_INCLUDES,
    ]
    for i, (name, source, _) in enumerate(kernels):
        parts.append(f"\n// ---- kernel {i}: {name} ----\n#define {name} tl_session_kernel_{i}\n{source}\n#undef {name}\n")
    parts.append("\n" + fastrpc.FORWARD_KERNEL_STATUS)
    for i, (_, _, plans) in enumerate(kernels):
        args = ", ".join(f"({pl.ctype})(intptr_t)a[{j}]" if pl.is_scalar else f"({pl.ctype}*)a[{j}]" for j, pl in enumerate(plans))
        parts.append(
            f"static AEEResult tl_session_call_{i}(void** a) {{\n"
            f"  (void)a;\n  return tl_hexagon_forward_kernel_status([&]() {{ return tl_session_kernel_{i}({args}); }});\n}}\n"
        )
    calls = "".join(f"  case {i}: return tl_session_call_{i}(a);\n" for i in range(len(kernels)))
    max_params = max([len(plans) for _, _, plans in kernels] + [1])
    parts.append(_dispatch_source(calls))
    parts.append(f"""
// The DSP address of each resident buffer, by its fd.
#define TL_SESSION_MAPS 256
static struct {{ int fd; uint8_t* base; }} tl_session_maps[TL_SESSION_MAPS];
static int tl_session_nmaps;

static uint8_t* tl_session_address(int fd) {{
  for (int i = 0; i < tl_session_nmaps; ++i)
    if (tl_session_maps[i].fd == fd) return tl_session_maps[i].base;
  void* vaddr = 0;
  uint64 paddr = 0;
  if (tl_session_nmaps == TL_SESSION_MAPS || HAP_mmap_get(fd, &vaddr, &paddr) != 0) return 0;
  tl_session_maps[tl_session_nmaps].fd = fd;
  tl_session_maps[tl_session_nmaps].base = (uint8_t*)vaddr;
  ++tl_session_nmaps;
  return (uint8_t*)vaddr;
}}

#ifdef __cplusplus
extern "C" {{
#endif

AEEResult {iface}_open(const char* uri, remote_handle64* h) {{
  (void)uri; *h = 0;
  return tl_session_dispatch_open();
}}

AEEResult {iface}_release(remote_handle64 h, int32 fd) {{
  (void)h;
  for (int i = 0; i < tl_session_nmaps; ++i) {{
    if (tl_session_maps[i].fd != fd) continue;
    HAP_mmap_put(fd);
    tl_session_maps[i] = tl_session_maps[--tl_session_nmaps];
    return AEE_SUCCESS;
  }}
  return AEE_SUCCESS;
}}

AEEResult {iface}_close(remote_handle64 h) {{
  (void)h;
  while (tl_session_nmaps > 0) {{ HAP_mmap_put(tl_session_maps[tl_session_nmaps - 1].fd); --tl_session_nmaps; }}
  return tl_session_dispatch_close();
}}

AEEResult {iface}_sync(remote_handle64 h, int32 fd, uint32 offset, uint32 size, int32 writeback) {{
  (void)h;
  uint8_t* base = tl_session_address(fd);
  if (!base) return AEE_EBADPARM;
  if (size == 0) return AEE_SUCCESS;
  // Before the host reads the bytes, the DSP's dirty lines of them go to DDR;
  // before it writes them, they go there too and the DSP's copies go. A dirty
  // line left behind would overwrite the host's bytes when evicted, and an
  // invalidate on its own writes dirty lines back.
  int rc = qurt_mem_cache_clean((qurt_addr_t)(base + offset), size, writeback ? QURT_MEM_CACHE_FLUSH : QURT_MEM_CACHE_FLUSH_INVALIDATE, QURT_MEM_DCACHE);
  return rc == QURT_EOK ? AEE_SUCCESS : AEE_EFAILED;
}}

AEEResult {iface}_run(remote_handle64 h, int32 kernel, const int32* args, int argsLen, const uint8* inputs, int inputsLen, uint8* outputs, int outputsLen) {{
  (void)h; (void)inputsLen; (void)outputsLen;
  void* a[{max_params}];
  int n = argsLen / 3;
  if (n > {max_params}) return AEE_EBADPARM;
  for (int i = 0; i < n; ++i) {{
    int kind = args[3 * i], v = args[3 * i + 1], o = args[3 * i + 2];
    if (kind == {_RESIDENT}) {{
      uint8_t* base = tl_session_address(v);
      if (!base) return AEE_EBADPARM;
      a[i] = base + (unsigned int)o;
    }} else if (kind == {_INPUT}) {{
      a[i] = (void*)(inputs + v);
    }} else if (kind == {_OUTPUT}) {{
      a[i] = (void*)(outputs + v);
    }} else {{
      a[i] = (void*)(intptr_t)v;
    }}
  }}
  return tl_session_dispatch_run(kernel, a);
}}

#ifdef __cplusplus
}}
#endif
""")
    return "".join(parts)


def gen_agent(iface: str, kernels=()) -> str:
    """The agent: one FastRPC session, its resident buffers, and a TCP socket
    (reached through the transport's loopback tunnel) for host operations."""
    return f"""// Auto-generated agent of a tilelang Hexagon session.
#include <AEEStdErr.h>
#include <remote.h>
#include "rpcmem.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/uio.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include "dsp_capabilities_utils.h"
#include "{iface}.h"

static remote_handle64 g_handle = -1;
static int g_domain = CDSP_DOMAIN_ID;

static int read_full(int fd, void* buf, size_t n) {{
  size_t got = 0;
  while (got < n) {{ ssize_t r = read(fd, (char*)buf + got, n - got); if (r <= 0) return -1; got += (size_t)r; }}
  return 0;
}}
static int write_full(int fd, const void* buf, size_t n) {{
  size_t put = 0;
  while (put < n) {{ ssize_t w = write(fd, (const char*)buf + put, n - put); if (w <= 0) return -1; put += (size_t)w; }}
  return 0;
}}

// A status byte and its payload in one write: adb relays each write as a
// packet, and a small packet followed by another waits out a delayed ACK.
static int reply(int fd, unsigned char status, const void* data, size_t n) {{
  struct iovec io[2] = {{{{&status, 1}}, {{(void*)data, n}}}};
  ssize_t w = writev(fd, io, n ? 2 : 1);
  if (w <= 0) return -1;
  // The rest of the payload, if writev stopped short.
  size_t sent = (size_t)w - 1;
  return sent < n ? write_full(fd, (const char*)data + sent, n - sent) : 0;
}}

// Resident buffers: rpcmem, uncached on the CPU, mapped into the DSP once.
#define MAX_BUFFERS {_MAX_BUFFERS}
static struct {{ uint8_t* ptr; int fd; uint64_t size, generation; }} g_buffers[MAX_BUFFERS];

// The duration of the last run's FastRPC call.
static int64_t g_last_run_ns;

// The inline arguments of a run, grown as calls need.
static uint8_t* g_inputs; static uint32_t g_inputs_size;
static uint8_t* g_outputs; static uint32_t g_outputs_size;

static uint8_t* grow(uint8_t* buf, uint32_t* size, uint32_t need) {{
  if (need <= *size) return buf;
  if (buf) rpcmem_free(buf);
  uint32_t n = need < 4096 ? 4096 : need;
  buf = (uint8_t*)rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, n);
  *size = buf ? n : 0;
  return buf;
}}

static int open_session(int domain_id) {{
  domain* d = get_domain(domain_id);
  if (!d) {{ fprintf(stderr, "get_domain failed\\n"); return -1; }}
{fastrpc.PD_SETUP}  int len = strlen({iface}_URI) + MAX_DOMAIN_URI_SIZE;
  char* uri = (char*)malloc(len);
  snprintf(uri, len, "%s%s", {iface}_URI, d->uri);
  int e = {iface}_open(uri, &g_handle); free(uri);
  if (e != AEE_SUCCESS) {{ fprintf(stderr, "open failed: 0x%x\\n", e); return -1; }}
  return 0;
}}

static void free_buffer(int id) {{
  if (id < 0 || id >= MAX_BUFFERS || !g_buffers[id].ptr) return;
  {iface}_release(g_handle, g_buffers[id].fd);
  fastrpc_munmap(g_domain, g_buffers[id].fd, g_buffers[id].ptr, g_buffers[id].size);
  rpcmem_free(g_buffers[id].ptr);
  g_buffers[id].ptr = 0;
}}

{sequence.agent_source(iface, kernels)}

// One operation of the host; nonzero ends the connection.
static int serve(int cli, unsigned char op, int* running) {{
  if (op == {_OP_CLOSE}) return 1;
  if (op == {_OP_SHUTDOWN}) {{ *running = 0; return 1; }}
  if (op == {sequence.OP_SEQUENCE}) return tl_serve_sequence(cli);
  if (op == {_OP_RUN}) {{
    int32_t head[3];  // kernel, number of args, inline output bytes
    if (read_full(cli, head, sizeof(head))) return 1;
    int32_t n = head[1];
    uint32_t out_bytes = (uint32_t)head[2];
    int32_t* args = (int32_t*)malloc(sizeof(int32_t) * 3 * (n > 0 ? n : 1));
    uint32_t in_bytes = 0;
    if (read_full(cli, args, sizeof(int32_t) * 3 * n) || read_full(cli, &in_bytes, 4)) {{ free(args); return 1; }}
    g_inputs = grow(g_inputs, &g_inputs_size, in_bytes ? in_bytes : 1);
    g_outputs = grow(g_outputs, &g_outputs_size, out_bytes ? out_bytes : 1);
    if (!g_inputs || !g_outputs) {{ free(args); return 1; }}
    if (read_full(cli, g_inputs, in_bytes)) {{ free(args); return 1; }}
    // Resident arguments name buffers; the DSP takes their fds.
    for (int i = 0; i < n; ++i) {{
      if (args[3 * i] != {_RESIDENT}) continue;
      int id = args[3 * i + 1];
      args[3 * i + 1] = (id >= 0 && id < MAX_BUFFERS && g_buffers[id].ptr) ? g_buffers[id].fd : -1;
    }}
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    int e = {iface}_run(g_handle, head[0], args, 3 * n, g_inputs, (int)in_bytes, g_outputs, (int)out_bytes);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    g_last_run_ns = (int64_t)(t1.tv_sec - t0.tv_sec) * 1000000000 + (t1.tv_nsec - t0.tv_nsec);
    free(args);
    // 0, a kernel's TL_ERR_* status, or 255 for another error.
    unsigned char status = e == AEE_SUCCESS ? 0 : (e > {fastrpc.KERNEL_STATUS_BASE:#x} && e < {fastrpc.KERNEL_STATUS_BASE + 255:#x}) ? (unsigned char)(e - {fastrpc.KERNEL_STATUS_BASE:#x}) : 255;
    return reply(cli, status, g_outputs, status == 0 ? out_bytes : 0);
  }}
  if (op == {_OP_TIMING}) return write_full(cli, &g_last_run_ns, 8);
  if (op == {_OP_ALLOC}) {{
    uint64_t size = 0;
    if (read_full(cli, &size, 8)) return 1;
    int32_t id = -1;
    // rpcmem_alloc takes int; validate before narrowing or zeroing its result.
    if (size == 0 || size >= (UINT64_C(1) << 31)) return write_full(cli, &id, 4);
    for (int i = 0; i < MAX_BUFFERS; ++i) if (!g_buffers[i].ptr) {{ id = i; break; }}
    if (id >= 0) {{
      uint8_t* p = (uint8_t*)rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_FLAG_UNCACHED, (int)size);
      int fd = p ? rpcmem_to_fd(p) : -1;
      if (!p || fd < 0 || fastrpc_mmap(g_domain, fd, p, 0, size, FASTRPC_MAP_FD) != 0) {{
        if (p) rpcmem_free(p);
        id = -1;
      }} else {{
        memset(p, 0, size);
        g_buffers[id].ptr = p; g_buffers[id].fd = fd; g_buffers[id].size = size;
        ++g_buffers[id].generation;
      }}
    }}
    return write_full(cli, &id, 4);
  }}
  if (op == {_OP_FREE}) {{
    int32_t id;
    if (read_full(cli, &id, 4)) return 1;
    free_buffer(id);
    return 0;
  }}
  if (op == {_OP_WRITE} || op == {_OP_READ} || op == {_OP_LOAD}) {{
    int32_t id; uint64_t offset, size;
    if (read_full(cli, &id, 4) || read_full(cli, &offset, 8) || read_full(cli, &size, 8)) return 1;
    int ok = id >= 0 && id < MAX_BUFFERS && g_buffers[id].ptr && offset <= g_buffers[id].size && size <= g_buffers[id].size - offset;
    if (op == {_OP_LOAD}) {{
      // The file's bytes from `start` on into the `size` bytes at `offset`,
      // as many as it has: the start, the path's length and the path follow.
      uint64_t start; uint32_t n;
      char path[1024];
      if (read_full(cli, &start, 8) || read_full(cli, &n, 4) || n >= sizeof(path) || read_full(cli, path, n)) return 1;
      path[n] = 0;
      int64_t got = -1;
      FILE* f = ok ? fopen(path, "rb") : 0;
      if (f) {{
        if (fseeko(f, (off_t)start, SEEK_SET) == 0) {{
          {iface}_sync(g_handle, g_buffers[id].fd, (uint32_t)offset, (uint32_t)size, 0);
          got = (int64_t)fread(g_buffers[id].ptr + offset, 1, size, f);
        }}
        fclose(f);
      }}
      return write_full(cli, &got, 8);
    }}
    if (op == {_OP_WRITE}) {{
      // The bytes come either way, so the wire stays in step.
      uint8_t* dst = ok ? g_buffers[id].ptr + offset : 0;
      if (ok) {iface}_sync(g_handle, g_buffers[id].fd, (uint32_t)offset, (uint32_t)size, 0);
      uint8_t chunk[65536];
      for (uint64_t done = 0; done < size;) {{
        uint64_t n = size - done < sizeof(chunk) ? size - done : sizeof(chunk);
        if (read_full(cli, dst ? dst + done : chunk, n)) return 1;
        done += n;
      }}
      unsigned char status = ok ? 0 : 1;
      return write_full(cli, &status, 1);
    }}
    if (ok) {iface}_sync(g_handle, g_buffers[id].fd, (uint32_t)offset, (uint32_t)size, 1);
    return reply(cli, ok ? 0 : 1, ok ? g_buffers[id].ptr + offset : 0, ok ? size : 0);
  }}
  return 1;
}}

int main(int argc, char** argv) {{
  // The port to listen on, 0 for one the kernel picks, which it reports.
  int port = (argc > 1) ? atoi(argv[1]) : 0;
  if (argc > 2) g_domain = atoi(argv[2]);
  if (open_session(g_domain) != 0) return 1;
  int srv = socket(AF_INET, SOCK_STREAM, 0);
  int opt = 1; setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
  struct sockaddr_in addr; memset(&addr, 0, sizeof(addr));
  addr.sin_family = AF_INET; addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK); addr.sin_port = htons((unsigned short)port);
  if (bind(srv, (struct sockaddr*)&addr, sizeof(addr)) < 0) {{ perror("bind"); return 2; }}
  socklen_t len = sizeof(addr);
  getsockname(srv, (struct sockaddr*)&addr, &len);
  listen(srv, 1);
  printf("AGENT_READY port=%d\\n", ntohs(addr.sin_port)); fflush(stdout);
  int running = 1;
  while (running) {{
    int cli = accept(srv, 0, 0);
    if (cli < 0) continue;
    setsockopt(cli, IPPROTO_TCP, TCP_NODELAY, &opt, sizeof(opt));
    for (;;) {{
      unsigned char op;
      if (read_full(cli, &op, 1) || serve(cli, op, &running)) break;
    }}
    close(cli);
  }}
  for (int i = 0; i < MAX_BUFFERS; ++i) free_buffer(i);
  if (g_inputs) rpcmem_free(g_inputs);
  if (g_outputs) rpcmem_free(g_outputs);
  {iface}_close(g_handle);
  return 0;
}}
"""


def write_project(workdir: str, kernels: Sequence[tuple[str, str, list]]) -> tuple[str, str]:
    """Write the session's FastRPC project; returns (project_dir, iface)."""
    from tilelang.env import TILELANG_TEMPLATE_PATH

    digest = hashlib.sha256("\0".join(source for _, source, _ in kernels).encode()).hexdigest()[:10]
    iface = f"tl_session_{digest}"
    os.makedirs(workdir, exist_ok=True)
    files = {
        f"{iface}.idl": gen_idl(iface),
        f"{iface}_dsp.cc": gen_dsp(iface, kernels),
        f"{iface}_agent.c": gen_agent(iface, kernels),
        "CMakeLists.txt": fastrpc.gen_cmake(iface, TILELANG_TEMPLATE_PATH, agent=True),
    }
    for fn, content in files.items():
        with open(os.path.join(workdir, fn), "w") as f:
            f.write(content)
    return workdir, iface


# ---------------------------------------------------------------------------
# The host side.
# ---------------------------------------------------------------------------


@dataclass
class _BufferAllocation:
    id: int
    nbytes: int
    generation: int
    alive: bool = True


@dataclass
class DeviceBuffer:
    """``nbytes`` of resident memory of a :class:`Session`, from ``offset``."""

    session: Session
    id: int
    nbytes: int
    offset: int = 0
    _allocation: _BufferAllocation | None = None

    def _live_allocation(self) -> _BufferAllocation:
        allocation = self._allocation
        if (
            allocation is None
            or not allocation.alive
            or allocation.id != self.id
            or getattr(self.session, "_closed", True)
            or getattr(self.session, "_allocations", {}).get(self.id) is not allocation
        ):
            raise ValueError("the resident buffer allocation is no longer live")
        if self.offset < 0 or self.nbytes < 0 or self.offset > allocation.nbytes or self.nbytes > allocation.nbytes - self.offset:
            raise ValueError("the resident buffer view is outside its allocation")
        return allocation

    def at(self, offset: int, nbytes: int | None = None) -> DeviceBuffer:
        """The bytes from ``offset`` on, as an argument: a register boundary."""
        allocation = self._live_allocation()
        if offset % _ALIGN:
            raise ValueError(f"a resident argument starts on a {_ALIGN}-byte boundary, not at {offset}")
        n = self.nbytes - offset if nbytes is None else nbytes
        if offset < 0 or n < 0 or offset + n > self.nbytes:
            raise ValueError(f"bytes [{offset}, {offset + n}) are not in a buffer of {self.nbytes}")
        return DeviceBuffer(self.session, self.id, n, self.offset + offset, allocation)

    def write(self, data, offset: int = 0) -> None:
        """Write a tensor's (or a byte string's) bytes at ``offset``."""
        self._live_allocation()
        raw = _bytes(data)
        if offset < 0 or offset + len(raw) > self.nbytes:
            raise ValueError(f"{len(raw)} bytes at {offset} overrun a buffer of {self.nbytes}")
        self.session._write(self.id, self.offset + offset, raw)

    def read(self, nbytes: int | None = None, offset: int = 0) -> bytes:
        self._live_allocation()
        n = self.nbytes - offset if nbytes is None else nbytes
        if offset < 0 or n < 0 or offset > self.nbytes or n > self.nbytes - offset:
            raise ValueError("read range is outside its resident buffer view")
        return self.session._read(self.id, self.offset + offset, n)

    def load(self, device_path: str, start: int = 0) -> int:
        """Read the file at ``device_path`` on the device, from byte ``start``
        on, into the buffer: as many bytes as the buffer holds, or as the file
        has past ``start``. Returns how many it read."""
        self._live_allocation()
        return self.session._load(self.id, self.offset, self.nbytes, device_path, start)

    def free(self) -> None:
        self.session._free(self.id, self._live_allocation())


def _bytes(data) -> bytes:
    if isinstance(data, (bytes, bytearray, memoryview)):
        return bytes(data)
    if isinstance(data, torch.Tensor):
        data = data.detach().to("cpu").contiguous()
        if data.dtype == torch.bfloat16:
            data = data.view(torch.int16)
        return data.numpy().tobytes()
    return np.ascontiguousarray(data).tobytes()


class SessionKernel:
    """A kernel of a :class:`Session`, called like a compiled kernel."""

    def __init__(self, session: Session, index: int, name: str, params, out_idx):
        self.session = session
        self.index = index
        self.name = name
        self.params = list(params)
        n = len(self.params)
        self.out_idx = [i if i >= 0 else n + i for i in out_idx]
        self.plans = fastrpc._plan(self.params, [])

    def __call__(self, *args):
        """``args`` fill the parameters outside ``out_idx`` in order, and the
        ``out_idx`` ones come back as tensors, in ``out_idx`` order (one
        output alone, not in a list). Passing every parameter instead
        decides each one: an ``out_idx`` parameter given a DeviceBuffer is
        written there, one given None comes back."""
        if len(args) == len(self.params):
            given = list(args)
        elif len(args) == len(self.params) - len(self.out_idx):
            it = iter(args)
            given = [None if i in self.out_idx else next(it) for i in range(len(self.params))]
        else:
            raise TypeError(
                f"{self.name} takes {len(self.params) - len(self.out_idx)} arguments (or all {len(self.params)}), got {len(args)}"
            )
        return self.session._run(self, given)


@functools.cache
def _compiler_stamp() -> str:
    """What a kernel's lowering depends on besides the kernel: tilelang's
    version and native libraries (as the kernel cache stamps them), and its
    Python sources, where operators such as the GEMMs lower."""
    import tilelang
    from tilelang.cache.kernel_cache import KernelCache

    digest = hashlib.sha256(f"{tilelang.__version__}|{KernelCache._get_tilelang_lib_stamp()}".encode())
    root = os.path.dirname(os.path.abspath(tilelang.__file__))
    for path, dirs, files in os.walk(root):
        dirs.sort()
        for name in sorted(files):
            if name.endswith(".py"):
                digest.update(os.path.relpath(os.path.join(path, name), root).encode() + b"\0")
                with open(os.path.join(path, name), "rb") as f:
                    digest.update(f.read())
    return digest.hexdigest()


def _lower(func, target, configs: dict) -> tuple[str, str]:
    """The name and the source of a kernel lowered under `configs`, from the
    cache when it holds them: a megakernel lowers in about a minute, its key
    takes a fraction of a second (its script, the target, the configs and
    the compiler)."""
    import tilelang
    from tilelang import tvm
    from tilelang.env import env

    path = None
    if env.is_cache_enabled():
        key = hashlib.sha256()
        for part in (func.script(show_meta=True), str(target), json.dumps(configs, sort_keys=True, default=str), _compiler_stamp()):
            key.update(part.encode() + b"\0")
        path = os.path.join(env.TILELANG_CACHE_DIR, "hexagon_lowered", key.hexdigest() + ".json")
        with contextlib.suppress(OSError, ValueError, KeyError):
            with open(path) as f:
                entry = json.load(f)
            return entry["name"], entry["source"]
    with tvm.target.Target(target), tvm.transform.PassContext(opt_level=3, config=configs):
        art = tilelang.lower(func, target=target, enable_host_codegen=False, enable_device_compile=False)
    source = str(art.kernel_source)
    match = _KERNEL_RE.search(source)
    if not match:
        raise RuntimeError("could not find the kernel function in the generated source")
    if path is not None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "w") as f:
            json.dump({"name": match.group(1), "source": source}, f)
        os.replace(tmp, path)
    return match.group(1), source


def lower_kernels(kernels, target="hexagon") -> list[tuple[str, str, list, list]]:
    """The (name, source, params, out_idx) of each (prim_func, out_idx) pair,
    each lowered under its own pass configs (``T.annotate_pass_configs``), as
    ``tilelang.compile`` lowers it. Lowered sources are cached on disk
    (``$TILELANG_CACHE_DIR/hexagon_lowered``), as built skels are."""
    from tilelang.backend.target import determine_target
    from tilelang.engine.lower import extrac_params

    target = determine_target(target, return_object=True)
    lowered = []
    for func, out_idx in kernels:
        configs = dict(func.attrs["tilelang_pass_configs"]) if func.attrs and "tilelang_pass_configs" in func.attrs else {}
        name, source = _lower(func, target, configs)
        out = list(out_idx) if isinstance(out_idx, (list, tuple)) else ([] if out_idx is None else [out_idx])
        lowered.append((name, source, extrac_params(func), out))
    return lowered


class Session:
    """Kernels sharing one FastRPC session and its resident buffers.

    TILELANG_HEXAGON_RESOURCE_LIFETIME=call limits VTCM/HMX ownership to each
    kernel invocation. The default, session, retains them until close. DDR
    buffers, execution threads and DSP clock votes live for the whole session.
    Call-scoped resources do not provide preemption inside a long kernel.
    Calls on one Session must be serialized; its socket protocol is not
    thread-safe, including buffer operations and run_sequence.
    """

    _port_seq = 0

    def __init__(self, kernels, *, target="hexagon", serial: str | None = None, verbose: bool = False, ready_timeout: float = 30.0):
        """``kernels``: (prim_func, out_idx) pairs, compiled into one skel for
        the DSP the target names."""
        config.resource_lifetime()  # Reject invalid profiles before lowering.
        from tilelang.backend.target import determine_target
        from tilelang.hexagon.target import hexagon_arch

        target = determine_target(target, return_object=True)
        self._open(lower_kernels(kernels, target), None, hexagon_arch(target), serial, verbose, ready_timeout)

    @classmethod
    def of_lowered(
        cls,
        lowered,
        *,
        sdk=None,
        dsp_arch: str | None = None,
        serial: str | None = None,
        verbose: bool = False,
        ready_timeout: float = 30.0,
    ) -> Session:
        """A session of kernels lowered already, (name, source, params,
        out_idx) each as :func:`lower_kernels` gives them: a compiled kernel
        (tilelang.compile) is a session of one."""
        session = cls.__new__(cls)
        session._open(lowered, sdk, dsp_arch, serial, verbose, ready_timeout)
        return session

    def _open(self, lowered, sdk, dsp_arch: str | None, serial: str | None, verbose: bool, ready_timeout: float) -> None:
        config.resource_lifetime()  # Also cover of_lowered, before build/deploy.
        from .build import build_cmake_fastrpc
        from .config import device_serial, domain_id, hlos, remote_dir
        from .device import AdbConnection
        from .env import HexagonSDK

        self._closed = True
        self._allocations = {}
        self._buffer_generations = {}
        self.verbose = verbose
        # The session's FastRPC project and build tree, removed with it.
        self._workdir = tempfile.mkdtemp(prefix="tl_hexagon_session_")
        weakref.finalize(self, shutil.rmtree, self._workdir, True)
        project, iface = write_project(self._workdir, [(name, source, fastrpc._plan(params, [])) for name, source, params, _ in lowered])
        sdk = (sdk or (HexagonSDK(dsp_arch=dsp_arch) if dsp_arch else HexagonSDK())).validate()
        self._artifacts = build_cmake_fastrpc(project, iface, sdk=sdk, dsp_arch=dsp_arch, verbose=verbose)
        self.kernels = [SessionKernel(self, i, name, params, out) for i, (name, _, params, out) in enumerate(lowered)]
        self._domain = domain_id()
        if hlos() == "linux":
            from .linux_runtime import LinuxConnection

            self._conn = LinuxConnection.from_environment(serial)
        else:
            Session._port_seq += 1
            self._conn = AdbConnection(
                serial=serial or device_serial(),
                workdir=f"{remote_dir('android')}/session_{os.getpid()}_{Session._port_seq}",
            )
        self._device_dir = self._conn.workdir
        self._holds_device = DEVICE_LOCK.acquire()
        self._closed = False
        try:
            self._start(ready_timeout)
        except BaseException:
            self.close()
            raise
        atexit.register(self.close)

    # ---- lifecycle --------------------------------------------------------
    def _start(self, ready_timeout: float) -> None:
        for local in (self._artifacts.dsp_skel, self._artifacts.agent_exe):
            self._conn.push(local)
        exe = os.path.join(self._device_dir, os.path.basename(self._artifacts.agent_exe))
        self._proc = self._conn.start_host_binary(exe, ["0", str(self._domain)])
        # select() keeps a silent agent from blocking past the deadline, and
        # the reads take what the pipe holds, unbuffered: a buffered readline
        # could leave AGENT_READY where select() does not see it. What the
        # agent prints before it is ready is its diagnosis, should it not get
        # there.
        deadline = time.time() + ready_timeout
        printed = b""
        while (ready_line := re.search(rb"AGENT_READY port=(\d+)\n", printed)) is None:
            if self._proc.poll() is not None:
                printed += self._proc.stdout.read()
                raise RuntimeError(
                    f"Hexagon session agent exited ({self._proc.returncode}) before ready:\n{printed.decode(errors='replace')}"
                )
            remaining = deadline - time.time()
            if remaining <= 0:
                raise RuntimeError(f"Hexagon session agent did not report ready in time:\n{printed.decode(errors='replace')}")
            ready, _, _ = select.select([self._proc.stdout], [], [], min(0.2, remaining))
            if ready:
                printed += os.read(self._proc.stdout.fileno(), 4096)
        self._host_port = self._conn.forward_port(int(ready_line.group(1)))
        for _ in range(100):
            try:
                self._sock = socket.create_connection(("127.0.0.1", self._host_port), timeout=2.0)
                self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                # A call may move hundreds of MB or run long: not the connect's timeout.
                self._sock.settimeout(CALL_TIMEOUT)
                return
            except OSError:
                time.sleep(0.05)
        raise RuntimeError("could not connect to the Hexagon session agent")

    def close(self) -> None:
        if getattr(self, "_closed", True):
            return
        self._closed = True
        for allocation in getattr(self, "_allocations", {}).values():
            allocation.alive = False
        self._allocations = {}
        with contextlib.suppress(Exception):
            atexit.unregister(self.close)
        try:
            shutdown_sent = False
            if getattr(self, "_sock", None) is not None:
                with contextlib.suppress(Exception):
                    self._sock.sendall(bytes([_OP_SHUTDOWN]))
                    shutdown_sent = True
                with contextlib.suppress(Exception):
                    self._sock.close()
            if getattr(self, "_proc", None) is not None:
                # EOF also wakes the remote watchdog if no socket was opened.
                if not shutdown_sent and self._proc.stdin is not None:
                    with contextlib.suppress(Exception):
                        self._proc.stdin.close()
                try:
                    self._proc.wait(timeout=10)
                except Exception:
                    if self._proc.stdin is not None:
                        with contextlib.suppress(Exception):
                            self._proc.stdin.close()
                    with contextlib.suppress(Exception):
                        self._proc.terminate()
                        self._proc.wait(timeout=5)
                    if self._proc.poll() is None:
                        with contextlib.suppress(Exception):
                            self._proc.kill()
                            self._proc.wait()
                if self._proc.stdin is not None:
                    with contextlib.suppress(Exception):
                        self._proc.stdin.close()
                with contextlib.suppress(Exception):
                    self._proc.stdout.close()
            if getattr(self, "_host_port", None):
                with contextlib.suppress(Exception):
                    self._conn.remove_forward(self._host_port)
            try:
                self._conn.cleanup()
            except Exception as error:
                # A lost network can prevent immediate remote removal, but
                # must never keep the host device lock held or hide startup's error.
                warnings.warn(f"Hexagon session cleanup failed: {error}", RuntimeWarning, stacklevel=2)
        finally:
            if self._holds_device:
                self._holds_device = False
                DEVICE_LOCK.release()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        with contextlib.suppress(Exception):
            self.close()

    # ---- wire -------------------------------------------------------------
    def _recv(self, n: int) -> bytes:
        chunks, got = [], 0
        while got < n:
            c = self._sock.recv(min(n - got, 1 << 22))
            if not c:
                raise RuntimeError("the Hexagon session agent closed the connection")
            chunks.append(c)
            got += len(c)
        return b"".join(chunks)

    def _status(self, what: str) -> None:
        status = self._recv(1)[0]
        if status != 0:
            reason = fastrpc.KERNEL_STATUS.get(status, f"status {status}") if status != 255 else "the FastRPC call failed"
            raise RuntimeError(f"Hexagon session: {what} failed: {reason}")

    # ---- buffers ----------------------------------------------------------
    def buffer(self, nbytes: int) -> DeviceBuffer:
        """``nbytes`` of zeroed resident memory, under 2 GiB (rpcmem_alloc
        takes an int). Large buffers fragment the DSP's 4 GB of addresses: on
        a v79, seven of 512 MiB map and an eighth does not, while 3.5 GB of
        model weights map as a buffer per tensor, the largest first."""
        try:
            nbytes = operator.index(nbytes)
        except TypeError:
            raise ValueError("a resident buffer size must be an integer") from None
        if not 0 < nbytes < 1 << 31:
            raise ValueError(f"a resident buffer holds 1 byte to 2 GiB - 1, not {nbytes}")
        if self._closed:
            raise ValueError("the Hexagon session is closed")
        request = bytes([_OP_ALLOC]) + struct.pack("<Q", nbytes)
        try:
            self._sock.sendall(request)
            (bid,) = struct.unpack("<i", self._recv(4))
        except BaseException:
            # An allocation reply lost in transit leaves generation counters
            # unsynchronized. Do not retry on this session.
            self.close()
            raise
        if bid < 0:
            raise RuntimeError(f"Hexagon session: could not allocate and map {nbytes} bytes")
        if bid >= _MAX_BUFFERS or bid in self._allocations:
            self.close()
            raise RuntimeError("invalid Hexagon session allocation reply")
        # Freed by DeviceBuffer.free, or with the session.
        generation = self._buffer_generations.get(bid, 0) + 1
        self._buffer_generations[bid] = generation
        allocation = _BufferAllocation(bid, nbytes, generation)
        self._allocations[bid] = allocation
        return DeviceBuffer(self, bid, nbytes, _allocation=allocation)

    def _free(self, bid: int, allocation: _BufferAllocation) -> None:
        try:
            self._sock.sendall(bytes([_OP_FREE]) + struct.pack("<i", bid))
        except BaseException:
            self.close()
            raise
        allocation.alive = False
        del self._allocations[bid]

    def _write(self, bid: int, offset: int, raw: bytes) -> None:
        self._sock.sendall(bytes([_OP_WRITE]) + struct.pack("<iQQ", bid, offset, len(raw)) + raw)
        self._status("a buffer write")

    def _read(self, bid: int, offset: int, n: int) -> bytes:
        self._sock.sendall(bytes([_OP_READ]) + struct.pack("<iQQ", bid, offset, n))
        self._status("a buffer read")
        return self._recv(n)

    def _load(self, bid: int, offset: int, nbytes: int, path: str, start: int) -> int:
        raw = path.encode()
        self._sock.sendall(bytes([_OP_LOAD]) + struct.pack("<iQQQI", bid, offset, nbytes, start, len(raw)) + raw)
        (got,) = struct.unpack("<q", self._recv(8))
        if got < 0:
            raise RuntimeError(f"Hexagon session: could not read {path} on the device")
        return got

    def last_run_seconds(self) -> float:
        """How long the last kernel run's FastRPC call took, timed by the agent
        on the device: the run without the host's round trip."""
        self._sock.sendall(bytes([_OP_TIMING]))
        (ns,) = struct.unpack("<q", self._recv(8))
        return ns * 1e-9

    # ---- runs -------------------------------------------------------------
    def run_sequence(self, operations) -> SequenceResult:
        """Execute a bounded, explicit Run/Copy list with one agent round trip.

        All descriptors are validated before execution. A failed operation
        stops the sequence and raises SequenceError with partial timings;
        even the failed operation may have partial side effects. Effects
        remain and are never replayed automatically.
        Each Run retains the normal per-call DSP resource scope. Copies use
        byte offsets, while Run buffer arguments remain 128-byte aligned.
        """
        if self._closed:
            raise ValueError("the Hexagon session is closed")
        operations, wire = sequence.encode(self, operations)
        try:
            self._sock.sendall(wire)
            return sequence.receive(self, operations)
        except SequenceError:
            raise
        except BaseException:
            # Do not reuse a socket after a partial send or truncated reply.
            self.close()
            raise

    def _run(self, kernel: SessionKernel, given: list):
        args, inputs, outputs = [], bytearray(), []
        out_bytes = 0
        for i, plan in enumerate(kernel.plans):
            value = given[i]
            if plan.is_scalar:
                args += [_SCALAR, int(value), 0]
            elif isinstance(value, DeviceBuffer):
                if value.session is not self:
                    raise ValueError(f"argument {i} of {kernel.name} is a buffer of another session")
                value._live_allocation()
                if value.nbytes < plan.nbytes:
                    raise ValueError(f"argument {i} of {kernel.name} takes {plan.nbytes} bytes, its buffer has {value.nbytes}")
                args += [_RESIDENT, value.id, value.offset]
            elif value is None:
                args += [_OUTPUT, out_bytes, 0]
                outputs.append((i, out_bytes))
                out_bytes = _align(out_bytes + plan.nbytes)
            else:
                raw = _bytes(value)
                if len(raw) != plan.nbytes:
                    raise ValueError(f"argument {i} of {kernel.name} has {len(raw)} bytes, the kernel takes {plan.nbytes}")
                args += [_INPUT, len(inputs), 0]
                inputs += raw + bytes(_align(len(raw)) - len(raw))
        # The outputs come back in out_idx order, as a compiled kernel's do;
        # any other parameter given None after them.
        rank = {i: k for k, i in enumerate(kernel.out_idx)}
        outputs.sort(key=lambda output: rank.get(output[0], len(rank) + output[0]))
        head = struct.pack("<iii", kernel.index, len(args) // 3, out_bytes)
        # One write: adb relays each as a packet (see the agent's reply()).
        self._sock.sendall(b"".join([bytes([_OP_RUN]), head, struct.pack(f"<{len(args)}i", *args), struct.pack("<I", len(inputs)), inputs]))
        self._status(f"kernel {kernel.name}")
        raw = self._recv(out_bytes)
        results = []
        for i, offset in outputs:
            param = kernel.params[i]
            plan = kernel.plans[i]
            chunk = raw[offset : offset + plan.nbytes]
            dtype = str(param.dtype)
            if dtype in ("int4", "uint4"):
                results.append(torch.frombuffer(bytearray(chunk), dtype=torch.uint8))
                continue
            results.append(torch.frombuffer(bytearray(chunk), dtype=getattr(torch, dtype)).reshape([int(s) for s in param.shape]))
        if not results:
            return None
        return results[0] if len(results) == 1 else results
