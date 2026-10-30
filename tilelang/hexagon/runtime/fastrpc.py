"""The FastRPC pieces of a kernel's project: what its skel adds around the
kernels' sources (``SKEL_OPEN``, ``SKEL_CLOSE``), the PD's setup, the kernel
status, and the CMakeLists. A session's project (:mod:`.session`) builds its
skel and agent with them.

On a Linux BSP, where there is no agent, a kernel's own project
(:func:`write_project`) adds an IDL with the kernel's buffers and a one-shot
host driver that runs one call (:class:`~.linux_runtime.OneShotKernel`).
Buffers cross the FastRPC boundary as ``sequence<uint8>`` (``in`` for inputs,
``rout`` for outputs), scalars as ``in int32``; the driver's byte sizes come
from the static parameter shapes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from tvm import DataType

# tilelang/numpy dtype -> the C scalar type the kernel pointer is cast to.
_CTYPE = {
    "float32": "float",
    "float16": "__fp16",
    "float64": "double",
    "int32": "int32_t",
    "int64": "int64_t",
    "int16": "int16_t",
    "int8": "int8_t",
    "uint8": "uint8_t",
    # 4-bit integers: two to a byte, the first in its low nibble.
    "int4": "int8_t",
    "uint4": "uint8_t",
    "bool": "uint8_t",
}


def _ctype(dtype) -> str:
    name = str(dtype)
    if name not in _CTYPE:
        raise NotImplementedError(f"Hexagon adapter: unsupported dtype {name}")
    return _CTYPE[name]


def _nbytes(param) -> int:
    """Static byte size of a buffer param (raises on dynamic shapes for now).

    Elements narrower than a byte are packed: a 4-bit param of n elements
    takes n / 2 bytes, passed as any tensor of those bytes."""
    n = 1
    for s in param.shape:
        if not isinstance(s, int):
            raise NotImplementedError("Hexagon adapter: dynamic shapes not supported yet")
        n *= int(s)
    bits = DataType(str(param.dtype)).bits
    if n * bits % 8:
        raise NotImplementedError(f"Hexagon adapter: {param.dtype} param of {n} elements is not whole bytes")
    return n * bits // 8


@dataclass
class BufferPlan:
    """Marshaling plan for one kernel parameter."""

    index: int  # position in the kernel signature
    name: str  # p0, p1, ... or s0, s1, ...
    is_scalar: bool
    is_output: bool
    ctype: str
    nbytes: int  # for buffers


def _plan(params, result_idx) -> list[BufferPlan]:
    out = set(result_idx)
    plans = []
    for i, p in enumerate(params):
        if p.is_scalar():
            dt = str(p.dtype)
            if dt not in ("int32", "uint32"):
                raise NotImplementedError(
                    f"Hexagon adapter: scalar param dtype '{dt}' is not supported yet "
                    "(only int32/uint32). float/int64 scalars would be silently truncated "
                    "across the FastRPC int32 boundary."
                )
            plans.append(BufferPlan(i, f"s{i}", True, False, _ctype(p.dtype), 0))
        else:
            plans.append(BufferPlan(i, f"p{i}", False, i in out, _ctype(p.dtype), _nbytes(p)))
    return plans


def gen_idl(iface: str, plans: list[BufferPlan]) -> str:
    args = []
    for pl in plans:
        if pl.is_scalar:
            args.append(f"      in int32 {pl.name}")
        elif pl.is_output:
            args.append(f"      rout sequence<uint8> {pl.name}")
        else:
            args.append(f"      in sequence<uint8> {pl.name}")
    body = ",\n".join(args)
    return (
        '#include "AEEStdDef.idl"\n'
        '#include "remote.idl"\n\n'
        f"interface {iface} : remote_handle64 {{\n"
        f"   AEEResult run(\n{body}\n   );\n"
        "};\n"
    )


# What a skel adds around its kernels' sources, which include the template
# headers they call: its _open takes the DSP process's resources and its _close
# gives them up (tl_hexagon_open and tl_hexagon_close, RUN_KERNEL).
SKEL_INCLUDES = "#include <tl_templates/hexagon/power.h>\n"
SKEL_OPEN = "  if (tl_hexagon_open() != 0) return AEE_EFAILED;\n"
SKEL_CLOSE = "  tl_hexagon_close();\n"


# A kernel's status as the FastRPC result: a void kernel always succeeds.
# Bytes of stack of the FastRPC thread, which runs a kernel's first program:
# its default, 16 KB, overflows at a large kernel's frames.
PD_THREAD_STACK = 256 * 1024

# Before the host opens the session's handle: an unsigned PD, and the stack of
# its FastRPC thread. In open_session(domain_id) of the host programs.
PD_SETUP = f"""  if (&remote_session_control) {{
    struct remote_rpc_control_unsigned_module c; c.domain = domain_id; c.enable = 1;
    int e = remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &c, sizeof(c));
    if (e != AEE_SUCCESS) {{ fprintf(stderr, "unsigned-PD failed: 0x%x\\n", e); return -1; }}
    struct remote_rpc_thread_params t; t.domain = domain_id; t.prio = -1; t.stack_size = {PD_THREAD_STACK};
    e = remote_session_control(FASTRPC_THREAD_PARAMS, &t, sizeof(t));
    if (e != AEE_SUCCESS) {{ fprintf(stderr, "thread params failed: 0x%x\\n", e); return -1; }}
  }}
"""

# A kernel's nonzero status (TL_ERR_*, common.h) comes back from run() as
# KERNEL_STATUS_BASE + status, an error FastRPC passes through.
KERNEL_STATUS_BASE = 0x1000
# The TL_ERR_* statuses, by code.
KERNEL_STATUS = {1: "TL_ERR_VTCM (no VTCM region fits)", 2: "TL_ERR_HMX", 3: "TL_ERR_DMA", 4: "TL_ERR_GRID (too few workers for the grid)"}

RUN_KERNEL = f"""// The DSP process's resources: its power votes (power.h) and, for kernels
// whose sources say they use HMX or VTCM (the codegen defines
// TL_HEXAGON_USES_HMX and TL_HEXAGON_USES_VTCM), its compute-resource context
// (vtcm.h). A run takes the context when it starts and gives it back when it
// ends, so that another DSP process of the session can run in between.
// tl_hexagon_take returns 0 or a TL_ERR_* status.
static inline int tl_hexagon_take(void) {{
#if defined(TL_HEXAGON_USES_HMX)
  return tl_hmx_take();
#elif defined(TL_HEXAGON_USES_VTCM)
  return tl_vtcm_take() == 0 ? TL_OK : TL_ERR_VTCM;
#else
  return TL_OK;
#endif
}}
static inline void tl_hexagon_give(void) {{
#if defined(TL_HEXAGON_USES_HMX)
  tl_hmx_give();
#elif defined(TL_HEXAGON_USES_VTCM)
  tl_vtcm_give();
#endif
}}

// _close gives the resources up, or they leak for the agent's life.
static inline void tl_hexagon_close(void) {{
#if defined(TL_HEXAGON_USES_HMX)
  tl_hmx_session_deinit();
#elif defined(TL_HEXAGON_USES_VTCM)
  tl_vtcm_close();
#endif
  tl_power_unvote();
}}

// _open takes the context once, so that it fails loudly when another client
// holds the resources (e.g. a leaked agent), rather than a run returning silent
// zeros; a failed open gives up what it took. 0, or nonzero.
static inline int tl_hexagon_open(void) {{
  tl_power_vote();
#if defined(TL_HEXAGON_USES_HMX)
  tl_hmx_session_init();
  bool ok = tl_hmx_session_ok();
#elif defined(TL_HEXAGON_USES_VTCM)
  bool ok = tl_vtcm_open(0) == 0;
#else
  bool ok = true;
#endif
  if (ok && tl_hexagon_take() == TL_OK) {{
    tl_hexagon_give();
    return 0;
  }}
  tl_hexagon_close();
  return -1;
}}

// A kernel's run, between take and give; a nonzero status comes back from
// run() as KERNEL_STATUS_BASE + status, an error FastRPC passes through.
template <typename Fn>
static AEEResult tl_hexagon_run_kernel(Fn&& fn) {{
  int tl_rc = tl_hexagon_take();
  if (tl_rc == TL_OK) {{
    if constexpr (std::is_void_v<decltype(fn())>)
      fn();
    else
      tl_rc = (int)fn();
    tl_hexagon_give();
  }}
  return tl_rc == TL_OK ? AEE_SUCCESS : (AEEResult)({KERNEL_STATUS_BASE:#x} + tl_rc);
}}
"""


def gen_dsp(iface: str, kernel_name: str, kernel_source: str, plans: list[BufferPlan]) -> str:
    # qaic skel signature for run(), in the types of its header: in seq ->
    # (const uint8*, int); rout seq -> (uint8*, int); in int32 -> (int32), which
    # is not an int on the DSP.
    skel_args = ["remote_handle64 _h"]
    call_args = []
    for pl in plans:
        if pl.is_scalar:
            skel_args.append(f"int32 {pl.name}")
            call_args.append(f"({pl.ctype}){pl.name}")
        elif pl.is_output:
            # QAIC maps `rout sequence<uint8>` to (ptr, len); the length is fixed
            # by the caller and the buffer is copied back, so no written-length.
            skel_args.append(f"uint8* {pl.name}, int {pl.name}Len")
            call_args.append(f"({pl.ctype}*){pl.name}")
        else:
            skel_args.append(f"const uint8* {pl.name}, int {pl.name}Len")
            call_args.append(f"({pl.ctype}*){pl.name}")
    return (
        "// Auto-generated FastRPC skel impl for a tilelang Hexagon kernel.\n"
        "#include <AEEStdErr.h>\n"
        "#include <type_traits>\n"
        f'#include "{iface}.h"   // QAIC-generated handler prototypes\n\n'
        f"{SKEL_INCLUDES}"
        "// ---- the tilelang-generated device kernel ----\n"
        f"{kernel_source}\n\n"
        f"{RUN_KERNEL}\n"
        '#ifdef __cplusplus\nextern "C" {\n#endif\n\n'
        f"AEEResult {iface}_open(const char* uri, remote_handle64* h) {{\n  (void)uri; *h = 0;\n{SKEL_OPEN}  return AEE_SUCCESS;\n}}\n\n"
        f"AEEResult {iface}_close(remote_handle64 h) {{\n  (void)h;\n{SKEL_CLOSE}  return AEE_SUCCESS;\n}}\n\n"
        f"AEEResult {iface}_run({', '.join(skel_args)}) {{\n"
        f"  return tl_hexagon_run_kernel([&]() {{ return {kernel_name}({', '.join(call_args)}); }});\n"
        "}\n\n"
        "#ifdef __cplusplus\n}\n#endif\n"
    )


def gen_host(iface: str, plans: list[BufferPlan]) -> str:
    # argv order == kernel param order: buffers -> file path, scalars -> int value.
    reads, calls, writes, frees = [], ["g_handle"], [], []
    argv = 1
    for pl in plans:
        if pl.is_scalar:
            calls.append(f"atoi(argv[{argv}])")
            argv += 1
        elif pl.is_output:
            calls.append(f"{pl.name}")
            calls.append(f"{pl.nbytes}")
            reads.append(f"  unsigned char* {pl.name} = (unsigned char*)malloc({pl.nbytes});\n  const char* {pl.name}_path = argv[{argv}];")
            writes.append(f"  writefile({pl.name}_path, {pl.name}, {pl.nbytes});")
            frees.append(f"  free({pl.name});")
            argv += 1
        else:
            reads.append(f"  long {pl.name}_len = 0;\n  unsigned char* {pl.name} = readfile(argv[{argv}], &{pl.name}_len);")
            calls.append(f"{pl.name}")
            calls.append(f"(int){pl.name}_len")
            frees.append(f"  free({pl.name});")
            argv += 1
    return (
        "// Auto-generated HLOS host driver for a tilelang Hexagon kernel.\n"
        "#include <AEEStdErr.h>\n#include <remote.h>\n"
        "#include <stdio.h>\n#include <stdlib.h>\n#include <string.h>\n"
        '#include "dsp_capabilities_utils.h"\n'
        f'#include "{iface}.h"\n\n'
        "static remote_handle64 g_handle = -1;\n\n"
        "static unsigned char* readfile(const char* path, long* len) {\n"
        '  FILE* f = fopen(path, "rb"); if (!f) { fprintf(stderr, "open %s failed\\n", path); exit(2); }\n'
        "  fseek(f, 0, SEEK_END); *len = ftell(f); fseek(f, 0, SEEK_SET);\n"
        "  unsigned char* b = (unsigned char*)malloc(*len);\n"
        '  if (fread(b, 1, *len, f) != (size_t)*len) { fprintf(stderr, "read failed\\n"); exit(2); }\n'
        "  fclose(f); return b;\n}\n\n"
        "static void writefile(const char* path, const unsigned char* b, int len) {\n"
        '  FILE* f = fopen(path, "wb"); if (!f) { fprintf(stderr, "write %s failed\\n", path); exit(2); }\n'
        "  fwrite(b, 1, len, f); fclose(f);\n}\n\n"
        "static int open_session(int domain_id) {\n"
        "  domain* d = get_domain(domain_id);\n"
        '  if (!d) { fprintf(stderr, "get_domain failed\\n"); return -1; }\n'
        f"{PD_SETUP}"
        "  int len = strlen(" + iface + "_URI) + MAX_DOMAIN_URI_SIZE;\n"
        "  char* uri = (char*)malloc(len);\n"
        f'  snprintf(uri, len, "%s%s", {iface}_URI, d->uri);\n'
        f"  int e = {iface}_open(uri, &g_handle); free(uri);\n"
        '  if (e != AEE_SUCCESS) { fprintf(stderr, "open failed: 0x%x\\n", e); return -1; }\n'
        "  return 0;\n}\n\n"
        "int main(int argc, char** argv) {\n"
        "  (void)argc;\n"
        "  if (open_session(CDSP_DOMAIN_ID) != 0) return 1;\n" + "\n".join(reads) + "\n" + f"  int e = {iface}_run({', '.join(calls)});\n"
        '  if (e != AEE_SUCCESS) { fprintf(stderr, "run failed: 0x%x\\n", e); }\n'
        "  else {\n" + "\n".join(writes) + "\n  }\n" + "\n".join(frees) + "\n"
        f"  {iface}_close(g_handle);\n"
        "  return e ? 1 : 0;\n}\n"
    )


def gen_cmake(iface: str, template_dir: str, agent: bool) -> str:
    # The DSP skel (skel + impl, compiled with HVX/HMX enabled) and, for a
    # session, its Android agent (stub + agent). ``template_dir`` is the parent
    # of ``tl_templates`` so the kernel's <tl_templates/hexagon/...> include
    # resolves (as nvcc/hipcc get -I TILELANG_TEMPLATE_PATH). A Linux BSP builds
    # its host driver itself (linux_build.py).
    # Kernels clean up by scope (DMA queues, worker HMX scopes) and never throw;
    # -fno-exceptions keeps those destructors from importing the C++ unwinder,
    # which the DSP does not have.
    agent_block = (
        """
    add_executable(IFACE_agent
        ${CMAKE_CURRENT_BINARY_DIR}/IFACE_stub.c
        ${HEXAGON_SDK_ROOT}/utils/examples/dsp_capabilities_utils.c
        ${CMAKE_CURRENT_SOURCE_DIR}/IFACE_agent.c
    )
    build_idl(IFACE.idl IFACE_agent)
    set_common_compile_and_link_options(IFACE_agent)
    if(${CMAKE_SYSTEM_NAME} MATCHES "Android")
        target_link_options(IFACE_agent PUBLIC -llog -ldl)
    endif()
    choose_dsprpc(${DSP_TYPE} dsprpc)
    link_custom_library(IFACE_agent ${dsprpc})
    copy_binaries(IFACE_agent)""".replace("IFACE", iface)
        if agent
        else ""
    )
    return f"""cmake_minimum_required(VERSION 3.14.3)
project({iface} C CXX ASM)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
option(TL_HEXAGON_LEGACY_COMPUTE_RES "Use the legacy compute-resource API for a BSP without v2" OFF)

if(HEXAGON_SDK_ROOT)
    include(${{HEXAGON_SDK_ROOT}}/build/cmake/hexagon_fun.cmake)
else()
    include(${{HEXAGON_CMAKE_ROOT}}/hexagon_fun.cmake)
endif()

include_directories(
    ${{CMAKE_CURRENT_BINARY_DIR}}
    ${{HEXAGON_SDK_ROOT}}/incs/
    ${{HEXAGON_SDK_ROOT}}/incs/stddef/
    ${{HEXAGON_SDK_ROOT}}/rtos/qurt/
    ${{HEXAGON_SDK_ROOT}}/utils/examples/
    {template_dir}
)

if(${{OS_TYPE}} MATCHES "HLOS"){agent_block}
else()
    add_library({iface}_skel SHARED
        ${{CMAKE_CURRENT_BINARY_DIR}}/{iface}_skel.c
        ${{CMAKE_CURRENT_SOURCE_DIR}}/{iface}_dsp.cc
    )
    build_idl({iface}.idl {iface}_skel)
    if(TL_HEXAGON_LEGACY_COMPUTE_RES)
        target_compile_definitions({iface}_skel PRIVATE TL_HEXAGON_LEGACY_COMPUTE_RES=1)
    endif()
    # No -mhvx-ieee-fp: the SM8750 (v79) cDSP returns zeros for the IEEE .sf/.hf
    # HVX arithmetic it selects, although hexagon-sim computes it. Vector float
    # ops stay a qfloat op plus a conversion.
    # --disable-handle-qfp: hexagon-clang 19's rewrite of qfloat spills and
    # refills after register allocation corrupts spilled qf32 registers (an
    # HVX GEMM's accumulators turned NaN); the registers spill as they are.
    set(CMAKE_C_FLAGS "${{CMAKE_C_FLAGS}} -mhmx -mhvx -Wno-error -mllvm --disable-handle-qfp")
    set(CMAKE_CXX_FLAGS "${{CMAKE_CXX_FLAGS}} -mhmx -mhvx -Wno-error -fno-exceptions -mllvm --disable-handle-qfp")
    copy_binaries({iface}_skel)
endif()
"""


def write_project(workdir: str, kernel_name: str, kernel_source: str, params, result_idx) -> tuple[str, str]:
    """Write a kernel's own FastRPC project, its one-shot host driver's, under
    *workdir*. Returns (project_dir, iface)."""
    from tilelang.env import TILELANG_TEMPLATE_PATH

    iface = "tl_" + kernel_name
    plans = _plan(params, result_idx)
    os.makedirs(workdir, exist_ok=True)
    # The kernel #includes <tl_templates/hexagon/common.h>; resolve via tilelang's
    # canonical template root (works in both source tree and installed wheel).
    if not TILELANG_TEMPLATE_PATH or not os.path.exists(os.path.join(TILELANG_TEMPLATE_PATH, "tl_templates", "hexagon", "common.h")):
        raise RuntimeError(
            "tilelang Hexagon templates not found under TILELANG_TEMPLATE_PATH="
            f"{TILELANG_TEMPLATE_PATH!r}; set TL_TEMPLATE_PATH to the directory containing "
            "tl_templates/hexagon/common.h"
        )
    files = {
        f"{iface}.idl": gen_idl(iface, plans),
        f"{iface}_dsp.cc": gen_dsp(iface, kernel_name, kernel_source, plans),
        f"{iface}_host.c": gen_host(iface, plans),
        "CMakeLists.txt": gen_cmake(iface, TILELANG_TEMPLATE_PATH, agent=False),
    }
    for fn, content in files.items():
        with open(os.path.join(workdir, fn), "w") as f:
            f.write(content)
    return workdir, iface
