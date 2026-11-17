"""Actual resource ownership code with simulated SDK failures and callbacks."""

from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.fixture(scope="module")
def resource_host(tmp_path_factory):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("a host C++ compiler is required")
    directory = tmp_path_factory.mktemp("resource-host")
    headers = directory / "tl_templates" / "hexagon"
    headers.mkdir(parents=True)
    for stub in ["HAP_compute_res.h", "HAP_perf.h", "qurt.h", "tl_templates/hexagon/common.h"]:
        (directory / stub).write_text("")
    root = Path(__file__).resolve().parents[3]
    resource = (root / "src/tl_templates/hexagon/resource.h").read_text()
    # Pause the actual callback at two publication boundaries. The host-only
    # insertions add no state changes; production has no test hook/API.
    for anchor, phase in [("  bool admitted = false;", 1), ("  if (admitted) {", 2)]:
        assert resource.count(anchor) == 1
        resource = resource.replace(anchor, anchor + f"\n  resource_test_hook({phase});")
    (headers / "resource.h").write_text(resource)
    hmx = (root / "src/tl_templates/hexagon/hmx.h").read_text()
    # Compile the exact resource-facing HMX functions. The matrix instruction
    # helpers are not part of this host ownership test.
    enable = hmx[hmx.index("TL_HMX_INLINE int tl_hmx_enable") : hmx.index("TL_HMX_INLINE void tl_hmx_unit_acquire")]
    lifecycle = hmx[hmx.index("static int tl_hmx_inited;") :]
    (headers / "hmx_lifecycle.h").write_text("#define TL_HMX_INLINE static inline\n" + enable + lifecycle)
    binary = directory / "resource-host"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-pthread",
            "-fsanitize=undefined",
            "-fno-sanitize-recover=all",
            "-I",
            str(directory),
            "-I",
            str(root / "src"),
            str(Path(__file__).with_name("resource_host.cc")),
            "-o",
            str(binary),
        ],
        check=True,
        timeout=30,
    )
    return binary


@pytest.mark.parametrize(
    "case",
    [
        "normal",
        "dynamic-size",
        "no-hmx",
        "parallel-workers",
        "ordinary-switch",
        "query",
        "attribute",
        "open",
        "take",
        "bind",
        "callback-attribute",
        "lock",
        "unlock",
        "busy",
        "wrong-thread",
        "unsafe-dma",
        "missing-terminal",
        "body-error",
        "callback-concurrent",
        "callback-wrong-context",
        "callback-overflow",
        "epoch-overflow",
        "yield",
        "callback-late",
        "reacquire-fail",
        "give",
        "close",
        "four-runs",
        "admission-closed",
        "ordinary-hmx",
        "ordinary-hmx-unlock-fail",
        "ordinary-hmx-body-pool-fail",
        "ordinary-hmx-body-give-fail",
        "callback-before-cas-close",
        "callback-admitted-close",
    ],
)
def test_resource_ownership(resource_host, case):
    subprocess.run([str(resource_host), case], check=True, timeout=10)


@pytest.fixture(scope="module")
def resource_collective_host(tmp_path_factory):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("a host C++ compiler is required")
    out = tmp_path_factory.mktemp("resource-collective-host")
    headers = out / "tl_templates" / "hexagon"
    headers.mkdir(parents=True)
    root = Path(__file__).resolve().parents[3]
    here = Path(__file__).parent
    # Reuse the existing persistent-pool SDK providers. No resource ownership,
    # barrier, snapshot, or worker-dispatch implementation is mocked here.
    worker = (here / "worker_host.cc").read_text().split("static uint8_t *tl_vtcm_arena()")[0]
    worker = worker.replace("constexpr unsigned TL_VTCM_HMX_ALIGNMENT = 2048;", "")
    worker = worker.replace(
        "static qurt_thread_t qurt_thread_get_id() { return nullptr; }",
        "static qurt_thread_t qurt_thread_get_id() { return reinterpret_cast<qurt_thread_t>(uintptr_t(worker_id + 1)); }",
    )
    sdk = (here / "resource_host.cc").read_text()
    sdk = sdk[sdk.index("struct compute_res_attr_t {") : sdk.index("#include <tl_templates/hexagon/resource.h>")]
    assert sdk.count("  held = true;") == 1
    sdk = sdk.replace("  held = true;", "  if (takes) std::memset(arena, 0xa5, sizeof(arena));\n  held = true;")
    providers = worker + "\n#include <cstring>\nenum { TL_ERR_HMX=2, TL_ERR_DMA=3, TL_ERR_RESOURCE=5, HAP_COMPUTE_RES_HMX_SHARED=1 };\n"
    providers += "static uint64_t HAP_perf_get_qtimer_count() { return 100; }\n" + sdk
    providers += "\n#include <tl_templates/hexagon/resource.h>\n"
    (out / "resource_collective_providers.h").write_text(providers)
    for name in ("worker.h", "cooperative.h"):
        source = (root / "src/tl_templates/hexagon" / name).read_text()
        (headers / name).write_text(source.replace('asm volatile("pause(#8)");', "std::this_thread::yield();"))
    for stub in [
        "HAP_compute_res.h",
        "HAP_perf.h",
        "qurt.h",
        "hexagon_types.h",
        "tl_templates/hexagon/common.h",
        "tl_templates/hexagon/dma.h",
    ]:
        (out / stub).write_text("")
    binary = out / "resource-collective-host"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-pthread",
            "-fsanitize=undefined",
            "-fno-sanitize-recover=all",
            "-I",
            str(out),
            "-I",
            str(root / "src"),
            str(here / "resource_collective_host.cc"),
            "-o",
            str(binary),
        ],
        check=True,
        timeout=30,
    )
    return binary


@pytest.mark.parametrize("case", ["repeated", "body-error", "release-error"])
def test_resource_collective_integration(resource_collective_host, case):
    subprocess.run([str(resource_collective_host), case], check=True, timeout=10)
