"""Generated half-source reductions on the Hexagon instruction simulator."""

import os
from pathlib import Path
import struct
import subprocess

import numpy as np
import pytest

from tilelang.hexagon import language as T
from test_tilelang_hexagon_convert_sim import _tool
from test_tilelang_hexagon_reduce import _source


SIMULATOR = pytest.mark.skipif(
    os.environ.get("TL_HEXAGON_SIMULATOR_TEST") != "1",
    reason="set TL_HEXAGON_SIMULATOR_TEST=1 to run the Hexagon SDK simulator",
)


def _make_accumulating_extrema(op, rows):
    @T.prim_func
    def kernel(A: T.Tensor((rows, 80), "float16"), Init: T.Tensor((76,), "float32"), B: T.Tensor((76,), "float32")):
        with T.Kernel(1):
            src = T.alloc_shared((rows, 80), "float16")
            dst = T.alloc_shared((76,), "float32")
            T.copy(A, src)
            T.copy(Init, dst)
            getattr(T, f"reduce_{op}")(src[:, 3:73], dst[2:72], dim=0, clear=False)
            T.copy(dst, B)

    return kernel


def _inputs():
    # Every ordered pair appears with each fp32 initial value. Neighbouring
    # lanes differ, including across the vector block and its scalar tail.
    halves = np.array(
        [0, 0x8000, 1, 0x8001, 0x03FF, 0x83FF, 0x0400, 0x8400, 0x3C00, 0xBC00, 0x7C00, 0xFC00, 0x7C01, 0x7E01, 0xFC01, 0xFE01],
        dtype=np.uint16,
    )
    initial = np.array(
        [
            0,
            0x80000000,
            1,
            0x80000001,
            0x337FFFFF,
            0x33800001,
            0xB37FFFFF,
            0xB3800001,
            0x3F800001,
            0xBF800001,
            0x3F7FFFFF,
            0xBF7FFFFF,
            0x477FE001,
            0xC77FE001,
            0x7F800000,
            0xFF800000,
            0x7F800001,
            0x7FC00001,
            0xFF800001,
            0xFFC00001,
        ],
        dtype=np.uint32,
    )
    count = initial.size * halves.size**2
    indices = np.arange(count).reshape(-1, 64)
    # Tail lanes include both zero orders and all-NaN pairs as well.
    tail = (indices[:, :1] + [0, 17, 64, 136, 204, 255]) % count
    indices = np.concatenate([indices, tail], axis=1)
    first = halves[(indices // halves.size) % halves.size]
    second = halves[indices % halves.size]
    init = initial[indices // halves.size**2]
    return first, second, init


@SIMULATOR
@pytest.mark.parametrize("arch", ["v73", "v75", "v79"])
def test_accumulating_half_extrema_preserve_float_initial_values(arch, tmp_path):
    root = Path(__file__).resolve().parents[3]
    cases = [(op, rows, True) for op in ("max", "min") for rows in (1, 2, 48)]
    cases += [(op, 2, False) for op in ("max", "min")]
    sources, entries = [], []
    for op, rows, vectorized in cases:
        source = _source(
            _make_accumulating_extrema(op, rows),
            {"kind": "hexagon", "mcpu": f"hexagon{arch}"},
            pass_configs={"tirx.disable_vectorize": not vectorized},
        )
        assert "dma" not in source and "sync_grid" not in source
        name = f"reduce_{op}_{rows}_{int(vectorized)}"
        assert "kernel_kernel(" in source
        sources.append(source.replace("#include <tl_templates/hexagon/vtcm.h>", "").replace("kernel_kernel(", f"{name}("))
        entries.append(f"{{{rows}, {'true' if op == 'max' else 'false'}, {'true' if vectorized else 'false'}, {name}}}")
    sources.append("static const ReductionCase reduction_cases[] = {" + ",\n".join(entries) + "};\n")
    (tmp_path / "generated_reductions.inc").write_text("\n".join(sources))
    executable = tmp_path / "reduce-accumulate"
    compiled = subprocess.run(
        [
            _tool("hexagon-clang++"),
            f"-m{arch}",
            "-mhvx",
            "-mhvx-length=128B",
            "-O2",
            "-std=c++17",
            "-mllvm",
            "--disable-handle-qfp",
            "-mllvm",
            "-hexagon-insert=false",
            "-I",
            str(root / "src"),
            "-I",
            str(tmp_path),
            str(Path(__file__).with_name("hvx_reduce_accumulate_sim.cc")),
            "-o",
            str(executable),
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr

    first, second, initial_bits = _inputs()
    batches = first.shape[0]
    inputs, initials = [], []
    data = bytearray(struct.pack("<I", batches))
    for op, rows, _ in cases:
        # Opposing infinities outside the source region catch origin errors.
        values = np.full((batches, rows, 80), 0x7C00 if op == "max" else 0xFC00, dtype=np.uint16)
        values[:, :, 3:73] = np.where(np.arange(rows)[None, :, None] % 2, second[:, None], first[:, None])
        initial = np.broadcast_to(np.arange(76, dtype=np.float32) + 1024, (batches, 76)).copy()
        initial[:, 2:72] = initial_bits.view(np.float32)
        for value, init in zip(values, initial):
            data.extend(value.tobytes())
            data.extend(init.tobytes())
        inputs.append(values.view(np.float16))
        initials.append(initial)
    input_path, output_path = tmp_path / "input.bin", tmp_path / "output.bin"
    input_path.write_bytes(data)
    simulated = subprocess.run(
        [_tool("hexagon-sim"), f"-m{arch}", "--simulated_returnval", str(executable), "--", str(input_path), str(output_path)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert simulated.returncode == 0, simulated.stdout + simulated.stderr
    results = np.frombuffer(output_path.read_bytes(), dtype="<f4").reshape(len(cases), batches, 2, 76)
    for case, (op, rows, vectorized) in enumerate(cases):
        actual, ordered = results[case, :, 0], results[case, :, 1]
        nonnan = ~np.isnan(ordered)
        np.testing.assert_array_equal(
            actual[nonnan].view(np.uint32),
            ordered[nonnan].view(np.uint32),
            err_msg=f"{op}, rows={rows}, vectorized={vectorized}",
        )
        np.testing.assert_array_equal(np.isnan(actual), np.isnan(ordered))
        # An independent mathematical oracle also checks all non-NaN lanes.
        values = inputs[case][:, :, 3:73].astype(np.float32)
        init = initials[case][:, 2:72]
        finite = ~np.isnan(values).any(axis=1) & ~np.isnan(init)
        combine = np.maximum if op == "max" else np.minimum
        expected = combine(init, combine.reduce(values, axis=1))
        np.testing.assert_array_equal(actual[:, 2:72][finite], expected[finite])
