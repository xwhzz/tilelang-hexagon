"""HVX math accuracy on real v73/v79 instruction simulation, without a device.

Opt in with TL_HEXAGON_SIMULATOR_TEST=1 and HEXAGON_TOOLS_ROOT set to the SDK
toolchain. Its host shared libraries must be on LD_LIBRARY_PATH. No TileLang
native library or mocked intrinsic results are used.
"""

import os
from pathlib import Path
import shutil
import struct
import subprocess

import numpy as np
import pytest


pytestmark = pytest.mark.skipif(
    os.environ.get("TL_HEXAGON_SIMULATOR_TEST") != "1",
    reason="set TL_HEXAGON_SIMULATOR_TEST=1 to run the Hexagon SDK simulator",
)


def _tool(name):
    root = os.environ.get("HEXAGON_TOOLS_ROOT")
    path = Path(root) / "Tools/bin" / name if root else Path(shutil.which(name) or name)
    if not path.is_file():
        pytest.fail(f"{name} is required for the requested simulator test: {path}")
    return str(path)


def _inputs():
    # The device regression's logspace, plus every adjacent float32 around
    # one on both sides. Cancellation near one amplifies a lost input ulp.
    device = np.logspace(-6, 6, 4096, dtype=np.float64).astype(np.float32)
    near_one = np.arange(0x3F800000 - 32768, 0x3F800000 + 32769, dtype=np.uint32).view(np.float32)
    # Accurate log/log2 must preserve classification after the final scale:
    # signed zero -> -inf, +inf -> +inf, negative inputs and NaNs -> NaN.
    # Include quiet/signalling NaNs of either sign, without requiring payloads.
    specials = np.array(
        [0, 0x80000000, 0xBF800000, 0x7F800000, 0xFF800000, 0x7FC00000, 0xFFC00000, 0x7F800001, 0xFF800001],
        dtype=np.uint32,
    ).view(np.float32)
    log = np.concatenate([device, near_one, specials])
    np.random.default_rng(73).shuffle(log)
    log = np.pad(log, (0, (-len(log)) % 32), constant_values=1).astype("<f4")

    # All finite half bit patterns within the fast functions' supported
    # device-test domains, including signed zero and subnormal inputs.
    # Append the original 4096-point grids, then mix signs and magnitudes
    # between lanes to expose even/odd widening and narrowing mistakes.
    all_half = np.arange(65536, dtype=np.uint16).view(np.float16)
    halves = []
    for lo, hi in [(-9, 10), (-13, 15), (-8, 8)]:
        selected = all_half[np.isfinite(all_half) & (all_half >= lo) & (all_half <= hi)]
        device = np.linspace(lo, hi, 4096, dtype=np.float64).astype(np.float16)
        values = np.concatenate([selected, device])
        np.random.default_rng(79).shuffle(values)
        halves.append(np.pad(values, (0, (-len(values)) % 64)).astype("<f2"))
    return log, halves


def _check_error(name, actual, expected, kind, bound, zero_bound):
    # These are the acceptance rules in test_tilelang_hexagon_math_device:
    # relative error away from zero, absolute error near zero, and absolute
    # error throughout for the bounded fp16 sigmoid.
    assert np.isfinite(actual).all(), f"{name}: non-finite output in the supported domain"
    absolute = np.abs(actual.astype(np.float64) - expected)
    if kind == "abs":
        error = absolute.max()
    else:
        near_zero = np.abs(expected) < 1e-3
        error = (absolute[~near_zero] / np.abs(expected[~near_zero])).max()
        if near_zero.any():
            assert absolute[near_zero].max() <= zero_bound, f"{name}: absolute error near zero"
    print(f"{name}: max {kind} error {error:.9g}, limit {bound:.9g}")
    assert error <= bound, f"{name}: {error} exceeds {bound}"


@pytest.mark.parametrize("arch", ["v73", "v79"])
def test_hvx_math_accuracy_on_simulator(arch, tmp_path):
    root = Path(__file__).resolve().parents[3]
    executable = tmp_path / f"math-{arch}"
    compilation = subprocess.run(
        [
            _tool("hexagon-clang++"),
            f"-m{arch}",
            "-mhvx",
            "-mhvx-length=128B",
            "-O2",
            "-std=c++17",
            "-I",
            str(root / "src"),
            str(Path(__file__).with_name("hvx_math_sim.cc")),
            "-o",
            str(executable),
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert compilation.returncode == 0, compilation.stdout + compilation.stderr
    log, halves = _inputs()
    counts = [log.size, *(h.size for h in halves)]
    input_path, output_path = tmp_path / "input.bin", tmp_path / "output.bin"
    input_path.write_bytes(struct.pack("<4I", *counts) + log.tobytes() + b"".join(h.tobytes() for h in halves))
    simulation = subprocess.run(
        [_tool("hexagon-sim"), f"-m{arch}", "--simulated_returnval", str(executable), "--", str(input_path), str(output_path)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert simulation.returncode == 0, simulation.stdout + simulation.stderr
    assert "HVX_MATH_SIM_OK" in simulation.stdout
    data = output_path.read_bytes()
    assert struct.unpack_from("<6I", data) == (0x4856584D, int(arch[1:]), *counts)
    offset = 24
    results = np.frombuffer(data, dtype="<f4", count=4 * log.size, offset=offset).reshape(-1, 4, 32)
    offset += results.nbytes
    results = results.transpose(1, 0, 2).reshape(4, -1)
    for name, actual, reference, fast in zip(
        ["log", "log2", "log_fast", "log2_fast"], results, [np.log, np.log2, np.log, np.log2], [False, False, True, True]
    ):
        supported = np.isfinite(log) & (log > 0)
        _check_error(
            f"{arch} {name}",
            actual[supported],
            reference(log[supported].astype(np.float64)),
            "rel",
            3e-5 if fast else 1e-6,
            1e-4 if fast else 1e-5,
        )
        if not fast:
            with np.errstate(divide="ignore", invalid="ignore"):
                expected = reference(log[~supported].astype(np.float64))
            special = actual[~supported]
            np.testing.assert_array_equal(np.isnan(special), np.isnan(expected))
            np.testing.assert_array_equal(np.isinf(special), np.isinf(expected))
            np.testing.assert_array_equal(np.signbit(special[np.isinf(expected)]), np.signbit(expected[np.isinf(expected)]))

    for name, values, reference, kind, bound in zip(
        ["exp_fast_hf", "exp2_fast_hf", "sigmoid_fast_hf"],
        halves,
        [np.exp, np.exp2, lambda x: 1 / (1 + np.exp(-x))],
        ["rel", "rel", "abs"],
        [8e-4, 8e-4, 1.2e-3],
    ):
        actual = np.frombuffer(data, dtype="<f2", count=values.size, offset=offset)
        offset += actual.nbytes
        _check_error(f"{arch} {name}", actual, reference(values.astype(np.float64)), kind, bound, 1e-4)
    assert offset == len(data), "unexpected fixture output layout"
