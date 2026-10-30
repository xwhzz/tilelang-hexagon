"""IEEE half conversions on actual v73/v79 Hexagon instruction simulation.

Opt in with TL_HEXAGON_SIMULATOR_TEST=1 and HEXAGON_TOOLS_ROOT set to the SDK
toolchain. The simulator's host shared libraries must be on LD_LIBRARY_PATH.
This does not use Android, a board, FastRPC, or mocked intrinsic results.
"""

import math
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


def _float_inputs():
    # Every positive adjacent finite-half pair has a midpoint exactly
    # representable in float32. Its immediate float32 neighbours distinguish
    # tie-to-even from truncation and either directed rounding mode.
    halves = np.arange(0x7C00, dtype=np.uint16).view(np.float16).astype(np.float32)
    midpoint = ((halves[:-1].astype(np.float64) + halves[1:]) / 2).astype(np.float32)
    midpoint = np.append(midpoint, np.float32(65520))  # finite -> infinity tie
    around = (
        np.stack(
            [np.nextafter(midpoint, np.float32(-np.inf)), midpoint, np.nextafter(midpoint, np.float32(np.inf))],
            axis=1,
        )
        .ravel()
        .view(np.uint32)
    )
    boundary = np.concatenate([around, around | np.uint32(0x80000000)])
    special = np.array(
        [
            0,
            0x80000000,
            1,
            0x80000001,
            0x007FFFFF,
            0x00800000,
            0x33000000,
            0x33800000,
            0x387FC000,
            0x38800000,
            0x477FE000,
            0x477FF000,
            0x7F7FFFFF,
            0x7F800000,
            0xFF800000,
            0x7F800001,
            0x7FC00000,
            0x7FFFFFFF,
            0xFF800001,
            0xFFC00000,
            0xFFFFFFFF,
        ],
        dtype=np.uint32,
    )
    random = np.random.default_rng(0x8650).integers(0, 1 << 32, 1 << 18, dtype=np.uint32)
    bits = np.concatenate([boundary, special, random])
    # Fixed permutation mixes magnitudes, signs, specials, and midpoint cases
    # across even/odd lanes instead of validating broadcasts only.
    np.random.default_rng(73).shuffle(bits)
    return np.pad(bits, (0, (-len(bits)) % 64)).astype("<u4")


@pytest.mark.parametrize("arch", ["v73", "v79"])
def test_ieee_conversion_and_math_on_simulator(arch, tmp_path):
    root = Path(__file__).resolve().parents[3]
    fixture = Path(__file__).with_name("hvx_convert_sim.cc")
    executable = tmp_path / f"convert-{arch}"
    compile_result = subprocess.run(
        [
            _tool("hexagon-clang++"),
            f"-m{arch}",
            "-mhvx",
            "-mhvx-length=128B",
            "-O2",
            "-std=c++17",
            "-I",
            str(root / "src"),
            str(fixture),
            "-o",
            str(executable),
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert compile_result.returncode == 0, compile_result.stdout + compile_result.stderr
    inputs = _float_inputs()
    input_path, output_path = tmp_path / "input.bin", tmp_path / "output.bin"
    input_path.write_bytes(struct.pack("<I", inputs.size) + inputs.tobytes())
    simulation = subprocess.run(
        [_tool("hexagon-sim"), f"-m{arch}", "--simulated_returnval", str(executable), "--", str(input_path), str(output_path)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert simulation.returncode == 0, simulation.stdout + simulation.stderr
    assert "HVX_CONVERT_SIM_OK" in simulation.stdout
    data = output_path.read_bytes()
    assert struct.unpack_from("<III", data) == (0x48565843, int(arch[1:]), inputs.size)
    offset = 12
    copies = 2 if arch == "v79" else 1
    all_half = np.arange(65536, dtype=np.uint16)
    half_rows = np.frombuffer(data, dtype=np.uint8, count=1024 * 384 * copies, offset=offset)
    half_rows = half_rows.reshape(1024, copies, 384)
    offset += half_rows.size
    wide = half_rows[:, 0, :256].copy().view("<u4").ravel()
    back = half_rows[:, 0, 256:].copy().view("<u2").ravel()
    expected_wide = all_half.view(np.float16).astype(np.float32).view(np.uint32)
    half_nan = ((all_half & 0x7C00) == 0x7C00) & ((all_half & 0x03FF) != 0)
    np.testing.assert_array_equal(wide[~half_nan], expected_wide[~half_nan])
    expected_back = all_half.copy()
    if arch == "v79":
        # Native v79 vcvt turns an exact -0.0f into +0 half. Preserve this
        # pre-existing instruction behavior; v73 promises IEEE signed zero.
        expected_back[0x8000] = 0
    np.testing.assert_array_equal(back[~half_nan], expected_back[~half_nan])
    assert np.isnan(wide[half_nan].view(np.float32)).all()
    assert np.isnan(back[half_nan].view(np.float16)).all()
    if arch == "v79":
        np.testing.assert_array_equal(half_rows[:, 0], half_rows[:, 1])

    count = inputs.size * copies
    narrowed = np.frombuffer(data, dtype="<u2", count=count, offset=offset).reshape(-1, copies, 64)
    offset += count * 2
    actual = narrowed[:, 0].ravel()
    floats = inputs.view("<f4")
    nan = np.isnan(floats)
    with np.errstate(over="ignore", invalid="ignore"):
        expected = floats.astype(np.float16).view(np.uint16)
    if arch == "v79":
        expected[inputs == 0x80000000] = 0
    np.testing.assert_array_equal(actual[~nan], expected[~nan])
    assert np.isnan(actual[nan].view(np.float16)).all()
    assert ((actual[nan] & 0x0200) != 0).all(), "narrowed NaNs must be quiet"
    if arch == "v79":
        np.testing.assert_array_equal(narrowed[:, 0], narrowed[:, 1])

    math_result = np.frombuffer(data, dtype="<f4", count=8 * 32, offset=offset).reshape(8, 32)
    offset += math_result.nbytes
    x = np.arange(1, 33, dtype=np.float64) / 8
    reference = np.array([x * 0.5, 1 / np.sqrt(x), np.exp2(x), np.exp(x), np.log2(x), np.log(x), np.tanh(x), [math.erf(t) for t in x]])
    np.testing.assert_allclose(math_result, reference, rtol=4e-4, atol=2e-5)
    reductions = np.frombuffer(data, dtype="<f4", count=6, offset=offset)
    row = (np.arange(320) % 32 + 1) / 8
    reference_reduce = np.array([row[:n].sum() for n in (63, 64, 65, 255, 256, 257)])
    np.testing.assert_allclose(reductions, reference_reduce, rtol=2e-5, atol=2e-5)
    assert offset + reductions.nbytes == len(data), "unexpected fixture output layout"
