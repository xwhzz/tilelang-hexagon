"""Runtime worker limits and pool lifecycle, using the actual worker header."""

from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.fixture(scope="module")
def worker_host(tmp_path_factory):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("a host C++ compiler is required")
    directory = tmp_path_factory.mktemp("worker-host")
    headers = directory / "tl_templates" / "hexagon"
    headers.mkdir(parents=True)
    root = Path(__file__).resolve().parents[3]
    worker = (root / "src/tl_templates/hexagon/worker.h").read_text()
    # Execute the actual limit, dispatcher, jobs and barrier. Only the target
    # pause instruction is replaced; QuRT/resource interfaces are host mocks.
    pause = 'asm volatile("pause(#8)");'
    assert pause in worker
    (headers / "worker.h").write_text(worker.replace(pause, "std::this_thread::yield();"))
    for stub in (directory / "qurt.h", headers / "common.h", headers / "vtcm.h"):
        stub.write_text("")
    binary = directory / "worker-host"
    source = Path(__file__).with_name("worker_host.cc")
    subprocess.run(
        [compiler, "-std=c++17", "-O2", "-pthread", "-I", str(directory), str(source), "-o", str(binary)],
        check=True,
        timeout=30,
    )
    return binary


@pytest.mark.parametrize(
    "case",
    [
        "limit-four",
        "limit-six",
        "limit-eight",
        "limit-hardware",
        "limit-software",
        "query-failed",
        "hardware-zero",
        "hvx-negative",
        "hvx-zero",
        "hvx-64-only",
        "fallback-one",
        "empty",
        "last-int32-program",
        "batch-four",
        "batch-request-two",
        "batch-arena-two",
        "batch-eight",
        "reject-six",
        "reject-arena",
        "reject-nine",
        "extent-four-request-six",
        "partial-batch",
        "partial-core",
        "scope-failure",
        "body-failure",
        "barrier-four",
        "barrier-six",
        "barrier-eight",
        "barrier-failure",
    ],
)
def test_worker_runtime_on_host(worker_host, case):
    # Every case has a fresh process, including the real function-local cache.
    subprocess.run([str(worker_host), case], check=True, timeout=10)
