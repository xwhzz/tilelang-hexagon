"""Actual cooperative/worker headers; SDK, DMA and resource API are host fakes.

The resource API itself is covered independently by test_tilelang_hexagon_resource_host.
These tests model barriers, admission, byte preservation and checked error
propagation. They neither access a device nor prove SDK scheduling/latency.
"""

from pathlib import Path
import shutil
import subprocess
import pytest


@pytest.fixture(scope="module")
def cooperative_host(tmp_path_factory):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler required")
    out = tmp_path_factory.mktemp("cooperative-host")
    headers = out / "tl_templates/hexagon"
    headers.mkdir(parents=True)
    root = Path(__file__).resolve().parents[3]
    for name in ("worker.h", "cooperative.h"):
        source = (root / "src/tl_templates/hexagon" / name).read_text()
        (headers / name).write_text(source.replace('asm volatile("pause(#8)");', "std::this_thread::yield();"))
    for name in ("common.h", "vtcm.h", "resource.h", "dma.h"):
        (headers / name).write_text("")
    (out / "qurt.h").write_text("")
    (out / "hexagon_types.h").write_text("")
    binary = out / "cooperative-host"
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
            str(Path(__file__).with_name("cooperative_host.cc")),
            "-o",
            str(binary),
        ],
        check=True,
        timeout=45,
    )
    return binary


@pytest.mark.parametrize(
    "case",
    [
        "roundtrip-1",
        "roundtrip-2",
        "roundtrip-3",
        "roundtrip-4",
        "roundtrip-6",
        "roundtrip-8",
        "no-yield",
        "no-arena",
        "elapsed-below",
        "elapsed-at",
        "elapsed-long",
        "between-snapshots",
        "after-negative",
        "late-request",
        "repeated-run",
        "partial-pool",
        "arena-count",
        "arena-bounds",
        "body-before-grid",
        "body-after-grid",
        "dma-wait-error",
        "dma-uncommitted",
        "dma-finish-error",
        "early-uncommitted",
        "release-error",
        "relock-error",
        "unlock-error",
        "committed-error",
        "epoch-overflow",
    ],
)
def test_actual_cooperative_header(cooperative_host, case):
    subprocess.run([str(cooperative_host), case], check=True, timeout=15)
