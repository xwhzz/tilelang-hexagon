"""Grouped lane broadcasts must fit the destination, including partial vectors."""

import os
from pathlib import Path
import shutil
import subprocess

import pytest

from test_tilelang_hexagon_convert_sim import _tool

ROOT = Path(__file__).resolve().parents[3]
SOURCE = Path(__file__).with_name("hvx_broadcast_bounds.cc")


@pytest.fixture(scope="module")
def broadcast_host(tmp_path_factory):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("a host C++ compiler is required")
    directory = tmp_path_factory.mktemp("broadcast-host")
    common = directory / "tl_templates/hexagon/common.h"
    common.parent.mkdir(parents=True)
    common.write_text("")
    (directory / "hexagon_types.h").write_text("")
    binary = directory / "broadcast-host"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O0",
            "-fsanitize=address",
            "-fno-omit-frame-pointer",
            "-DTL_LANES_HOST_TEST",
            "-I",
            str(directory),
            "-I",
            str(ROOT / "src"),
            str(SOURCE),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        timeout=60,
    )
    return binary


@pytest.mark.parametrize("group", [1, 2, 4])
def test_grouped_broadcast_stays_within_output_on_host(broadcast_host, group):
    # Actual header, without an optimizer removing its invalid local memcpy.
    subprocess.run([str(broadcast_host), str(group)], check=True, capture_output=True, timeout=30)


@pytest.mark.skipif(os.environ.get("TL_HEXAGON_SIMULATOR_TEST") != "1", reason="requires the Hexagon SDK simulator")
@pytest.mark.parametrize("arch", ["v73", "v79"])
def test_grouped_broadcast_on_simulator(arch, tmp_path):
    binary = tmp_path / "broadcast"
    subprocess.run(
        [
            _tool("hexagon-clang++"),
            f"-m{arch}",
            "-mhvx",
            "-mhvx-length=128B",
            "-O2",
            "-std=c++17",
            "-I",
            str(ROOT / "src"),
            str(SOURCE),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        timeout=180,
    )
    for group in [1, 2, 4]:
        subprocess.run(
            [_tool("hexagon-sim"), f"-m{arch}", "--simulated_returnval", str(binary), "--", str(group)],
            cwd=tmp_path,
            check=True,
            capture_output=True,
            timeout=180,
        )
