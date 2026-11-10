"""Parallel maps into regions of physically laid-out buffers."""

import os
from pathlib import Path
import re
import subprocess

import numpy as np
import pytest

import tilelang
from tilelang import tvm
from tilelang.hexagon import language as T
from test_tilelang_hexagon_convert_sim import _tool
from test_tilelang_hexagon_crouton import DEVICE, KINDS

SCALAR_TILE_STORE = re.compile(r"(?:\(\(half\*\)tiles\)|\btiles)\[[^\]]*\] =")


def _map_region(kind, region="aligned", dtype="float16", guarded=False):
    transposed = kind.endswith("_t")
    row, rows, col, cols = {
        "aligned": (4, 4, 0, 96),
        "columns": (4, 4, 32, 64),
        "dynamic": (4, 4, 0, 96),
        "cross_tile": (30, 4, 0, 96),
        "tail": (3, 5, 5, 87),
    }[region]
    if transposed:
        row, rows, col, cols = col, cols, row, rows
    shape = (96, 64) if transposed else (64, 96)
    repeats = 3 if region == "dynamic" else 1

    @T.prim_func
    def kernel(X: T.Tensor(shape, "float16"), Y: T.Tensor(shape, "float16")):
        with T.Kernel(1):
            tiles = T.alloc_shared(shape, "float16")
            T.annotate_layout({tiles: KINDS[kind](tiles)})
            T.copy(X, tiles)
            for batch in T.serial(repeats):
                for i, j in T.Parallel(rows, cols):
                    r = row + i + (batch * 4 if not transposed else 0)
                    c = col + j + (batch * 4 if transposed else 0)
                    if not guarded or (i + j) % 3 != 0:
                        tiles[r, c] = T.cast(T.cast(tiles[r, c], dtype) * T.cast(2, dtype) + T.cast(1, dtype), "float16")
            T.copy(tiles, Y)

    return kernel


def _source(func, arch="v73"):
    target = tvm.target.Target({"kind": "hexagon", "mcpu": f"hexagon{arch}"})
    with target, tvm.transform.PassContext(opt_level=3):
        return str(tilelang.lower(func, target=target, enable_host_codegen=False, enable_device_compile=False).kernel_source)


@pytest.mark.parametrize("kind", ["activation", "activation_t"])
@pytest.mark.parametrize("region", ["aligned", "columns", "dynamic"])
@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_aligned_region_maps_use_vector_stores(kind, region, dtype):
    source = _source(_map_region(kind, region, dtype))
    assert re.search(r"\*\(half(?:32|64)\*\)", source)
    assert not SCALAR_TILE_STORE.search(source)


@pytest.mark.parametrize("kind", ["weight", "weight_t"])
def test_weight_regions_keep_the_logical_fallback(kind):
    # These strided physical coordinates have holes. They are not a dense
    # bijection of the region, even though the logical stores are injective.
    source = _source(_map_region(kind))
    assert SCALAR_TILE_STORE.search(source)


def _norm_region(kind, rows, workers=4):
    """The model's worker/row-block/slot offsets and FP32 scale-to-FP16 map."""
    r, c = 128, 96
    rw = r // workers
    shape = (c, r) if kind.endswith("_t") else (r, c)

    @T.prim_func
    def kernel(
        X: T.Tensor((r, c), "float32"),
        U: T.Tensor((r,), "float32"),
        V: T.Tensor((c,), "float32"),
        Y: T.Tensor(shape, "float16"),
    ):
        with T.Kernel(workers, num_workers=workers) as w:
            tiles = T.alloc_global(shape, "float16", scope="global.vtcm")
            xr = T.alloc_shared((2, rows, c), "float32")
            ss = T.alloc_fragment((rows,), "float32")
            nw = T.alloc_shared((c,), "float32")
            T.annotate_layout({tiles: KINDS[kind](tiles)})
            T.copy(V, nw)
            for rb in T.serial(rw // rows):
                start = w * rw + rb * rows
                T.copy(X[start : start + rows, :], xr[rb % 2, :, :])
                T.copy(U[start : start + rows], ss)
                for i, k in T.Parallel(rows, c):
                    value = T.cast(xr[rb % 2, i, k] * ss[i] * nw[k], "float16")
                    if kind.endswith("_t"):
                        tiles[k, start + i] = value
                    else:
                        tiles[start + i, k] = value
            if kind.endswith("_t"):
                T.copy(tiles[:, w * rw : (w + 1) * rw], Y[:, w * rw : (w + 1) * rw])
            else:
                T.copy(tiles[w * rw : (w + 1) * rw, :], Y[w * rw : (w + 1) * rw, :])

    return kernel


@pytest.mark.parametrize("kind", ["activation", "activation_t"])
@pytest.mark.parametrize("rows", [2, 4])
@pytest.mark.parametrize("arch", ["v73", "v79"])
def test_model_scale_regions_use_vector_stores(kind, rows, arch):
    source = _source(_norm_region(kind, rows), arch)
    assert re.search(r"\*\(half(?:32|64)\*\)", source), source
    assert not SCALAR_TILE_STORE.search(source), source
    assert "tl_hvx_load_interleave<2," in source, source


def _norm_inputs():
    x = ((np.arange(128 * 96) % 251 - 125) / 4).astype(np.float32).reshape(128, 96)
    u = np.ldexp(np.ones(128, dtype=np.float32), np.arange(128) % 3 - 1)
    v = np.ldexp(np.ones(96, dtype=np.float32), np.arange(96) % 3 - 1)
    expected = ((x * u[:, None]) * v[None, :]).astype(np.float16)
    return (x, u, v), expected


@DEVICE
@pytest.mark.parametrize("kind", ["activation", "activation_t"])
@pytest.mark.parametrize("rows", [2, 4])
def test_model_scale_regions_preserve_values_and_guards(kind, rows):
    from tilelang.hexagon.runtime.session import Session

    inputs, expected = _norm_inputs()
    if kind.endswith("_t"):
        expected = expected.T.copy()
    guard = bytes([0xA5]) * 128
    before = [value.tobytes() + guard for value in inputs]
    with Session([(_norm_region(kind, rows), [])], target="hexagon") as session:
        resident = [session.buffer(len(raw)) for raw in before]
        for buffer, raw in zip(resident, before):
            buffer.write(raw)
        output = session.buffer(expected.nbytes + len(guard))
        output.write(bytes([0xA5]) * (expected.nbytes + len(guard)))
        session.kernels[0](*resident, output)
        assert output.read() == expected.tobytes() + guard
        for buffer, raw in zip(resident, before):
            assert buffer.read() == raw


def _input_and_expected(kind, region, guarded):
    shape = (96, 64) if kind.endswith("_t") else (64, 96)
    # Exact binary fractions distinguish every row/column without depending
    # on a particular target's arithmetic rounding or NaN policy.
    x = ((np.arange(np.prod(shape)) % 251 - 125) / 4).astype(np.float16).reshape(shape)
    expected = x.copy()
    row, rows, col, cols = {
        "aligned": (4, 4, 0, 96),
        "columns": (4, 4, 32, 64),
        "dynamic": (4, 4, 0, 96),
        "cross_tile": (30, 4, 0, 96),
        "tail": (3, 5, 5, 87),
    }[region]
    transposed = kind.endswith("_t")
    if transposed:
        row, rows, col, cols = col, cols, row, rows
    for batch in range(3 if region == "dynamic" else 1):
        r = row + (0 if transposed else batch * 4)
        c = col + (batch * 4 if transposed else 0)
        part = expected[r : r + rows, c : c + cols]
        active = (np.indices(part.shape).sum(axis=0) % 3 != 0) if guarded else np.ones(part.shape, dtype=bool)
        part[active] = (part[active].astype(np.float32) * 2 + 1).astype(np.float16)
    return x, expected


@DEVICE
@pytest.mark.parametrize("kind", sorted(KINDS))
@pytest.mark.parametrize("region,guarded", [("aligned", False), ("dynamic", False), ("cross_tile", False), ("tail", True)])
def test_region_maps_preserve_unwritten_elements(kind, region, guarded):
    import torch

    x, expected = _input_and_expected(kind, region, guarded)
    kernel = tilelang.compile(_map_region(kind, region, "float32", guarded), out_idx=[1], target="hexagon")
    try:
        actual = kernel(torch.from_numpy(x)).numpy()
        np.testing.assert_array_equal(actual.view(np.uint16), expected.view(np.uint16))
    finally:
        kernel.adapter.close()


@pytest.mark.skipif(os.environ.get("TL_HEXAGON_SIMULATOR_TEST") != "1", reason="requires the Hexagon SDK simulator")
@pytest.mark.parametrize("arch", ["v73", "v75", "v79"])
def test_region_maps_on_simulator(arch, tmp_path):
    cases = [
        (kind, region, guarded)
        for kind in sorted(KINDS)
        for region, guarded in [("aligned", False), ("columns", False), ("dynamic", False), ("cross_tile", False), ("tail", True)]
    ]
    sources = []
    for index, (kind, region, guarded) in enumerate(cases):
        source = _source(_map_region(kind, region, "float32", guarded), arch)
        sources.append(source.replace("#include <tl_templates/hexagon/vtcm.h>", "").replace("kernel_kernel(", f"region_{index}("))
    calls = "\n".join(
        f"if (region_{i}(input, reinterpret_cast<half *>(output))) return 3;\nif (fwrite(output, sizeof(output), 1, out) != 1) return 4;"
        for i in range(len(cases))
    )
    code = (
        """
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <tl_templates/hexagon/common.h>
alignas(128) static uint8_t arena[65536];
static uint8_t *tl_vtcm_arena() { return arena; }
static unsigned tl_vtcm_arena_bytes() { return sizeof(arena); }
static unsigned tl_vtcm_shared_high_water;
"""
        + "\n".join(sources)
        + """
alignas(128) static half input[64 * 96];
alignas(128) static uint8_t output[64 * 96 * 2 + 128];
int main(int argc, char **argv) {
  if (argc != 3) return 1;
  FILE *in = fopen(argv[1], "rb"), *out = fopen(argv[2], "wb");
  if (!in || !out || fread(input, sizeof(input), 1, in) != 1) return 2;
  memset(output, 0xa5, sizeof(output));
"""
        + calls
        + "\nfclose(in); fclose(out); return 0; }\n"
    )
    cpp, executable = tmp_path / "regions.cc", tmp_path / "regions"
    cpp.write_text(code)
    root = Path(__file__).resolve().parents[3]
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
            str(cpp),
            "-o",
            str(executable),
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    inputs, outputs = tmp_path / "input.bin", tmp_path / "output.bin"
    inputs.write_bytes(_input_and_expected(*cases[0])[0].tobytes())
    simulated = subprocess.run(
        [_tool("hexagon-sim"), f"-m{arch}", "--simulated_returnval", str(executable), "--", str(inputs), str(outputs)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert simulated.returncode == 0, simulated.stdout + simulated.stderr
    raw = np.frombuffer(outputs.read_bytes(), dtype=np.uint8).reshape(len(cases), 64 * 96 * 2 + 128)
    for result, case in zip(raw, cases):
        expected = _input_and_expected(*case)[1]
        np.testing.assert_array_equal(result[:-128].view(np.uint16).reshape(expected.shape), expected.view(np.uint16), err_msg=str(case))
        np.testing.assert_array_equal(result[-128:], np.full(128, 0xA5, dtype=np.uint8))
