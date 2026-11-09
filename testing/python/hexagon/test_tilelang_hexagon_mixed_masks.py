"""Mixed-width logical predicates select lanes without changing their bits."""

import itertools
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

import tilelang
from tilelang import tvm
from tilelang.backend.target import determine_target
from tilelang.hexagon import language as T

DEVICE = pytest.mark.skipif(os.environ.get("TL_HEXAGON_DEVICE_TEST") != "1", reason="requires an authorized Hexagon device and SDK")
SIMULATOR = pytest.mark.skipif(
    os.environ.get("TL_HEXAGON_SIMULATOR_TEST") != "1",
    reason="set TL_HEXAGON_SIMULATOR_TEST=1 to run the Hexagon SDK simulator",
)
N, GUARD = 259, 32
CASES = list(itertools.product(("float16", "float32"), ("and", "or"), ("float16", "float32")))


def _kernel(left_dtype, operation, output_dtype, matching=False):
    right_dtype = left_dtype if matching else "float32" if left_dtype == "float16" else "float16"
    combine = T.And if operation == "and" else T.Or

    @T.prim_func
    def kernel(
        A: T.Tensor((N,), left_dtype),
        B: T.Tensor((N,), right_dtype),
        Yes: T.Tensor((N,), output_dtype),
        No: T.Tensor((N,), output_dtype),
        Out: T.Tensor((N + GUARD,), output_dtype),
    ):
        with T.Kernel(1):
            for i in T.Parallel(256):
                Out[i] = T.Select(combine(A[i] > T.cast(0, left_dtype), B[i] < T.cast(1, right_dtype)), Yes[i], No[i])
            for i in T.serial(256, N):
                Out[i] = T.Select(combine(A[i] > T.cast(0, left_dtype), B[i] < T.cast(1, right_dtype)), Yes[i], No[i])

    return kernel


def _source(func, arch="v79"):
    target = determine_target({"kind": "hexagon", "mcpu": f"hexagon{arch}"}, return_object=True)
    with tvm.target.Target(target), tvm.transform.PassContext(opt_level=3):
        return str(tilelang.lower(func, target=target, enable_host_codegen=False, enable_device_compile=False).kernel_source)


def _inputs(left_dtype, operation, output_dtype, matching=False):
    right_dtype = left_dtype if matching else "float32" if left_dtype == "float16" else "float16"
    index = np.arange(N)

    def predicates(dtype, offset):
        # Threshold neighbours, signed zeros, subnormals and infinities. This
        # test does not assign a new contract to floating-point NaN comparisons.
        if dtype == "float16":
            bits = np.array([0, 0x8000, 1, 0x8001, 0x3C00, 0xBC00, 0x3BFF, 0x3C01, 0x7C00, 0xFC00], np.uint16)
        else:
            bits = np.array(
                [0, 0x80000000, 1, 0x80000001, 0x3F800000, 0xBF800000, 0x3F7FFFFF, 0x3F800001, 0x7F800000, 0xFF800000], np.uint32
            )
        return bits[(index * (3 if offset else 1) + offset) % len(bits)].view(dtype).copy()

    a, b = predicates(left_dtype, 0), predicates(right_dtype, 3)
    values = np.array([0.0, -0.0, 1.0001, -1.0001, 2.0**-20, -(2.0**-20), np.inf, -np.inf], dtype=output_dtype)
    yes, no = values[index % len(values)].copy(), values[(index + 3) % len(values)].copy()
    left, right = a > 0, b < 1
    condition = left & right if operation == "and" else left | right
    # Select bytes from the inputs, preserving signed zero and subnormal bits.
    word = np.uint16 if output_dtype == "float16" else np.uint32
    expected = np.where(condition, yes.view(word), no.view(word)).astype(word).tobytes()
    return (a, b, yes, no), expected + b"\xa5" * (GUARD * np.dtype(output_dtype).itemsize)


@pytest.mark.parametrize("left_dtype,operation,output_dtype", CASES)
def test_mixed_predicates_lower_to_vector_logical_operations(left_dtype, operation, output_dtype):
    source = _source(_kernel(left_dtype, operation, output_dtype))
    assert "tl_vec_select(" in source and (" && " if operation == "and" else " || ") in source
    assert "*(half64*)(" in source and "*(float64*)(" in source
    assert "tl_parallel_for" not in source and "tl_hexagon_dma" not in source


@DEVICE
@pytest.mark.parametrize("left_dtype,operation,output_dtype", CASES)
def test_mixed_predicates_preserve_selected_bits_and_guards(left_dtype, operation, output_dtype):
    from tilelang.hexagon.runtime.session import Session

    arrays, expected = _inputs(left_dtype, operation, output_dtype)
    with Session([(_kernel(left_dtype, operation, output_dtype), [])], target="hexagon") as session:
        inputs = [session.buffer(a.nbytes) for a in arrays]
        for buffer, array in zip(inputs, arrays):
            buffer.write(array.tobytes())
        output = session.buffer(len(expected))
        output.write(b"\xa5" * len(expected))
        session.kernels[0](*inputs, output)
        assert output.read() == expected


def _tool(name):
    root = os.environ.get("HEXAGON_TOOLS_ROOT")
    path = Path(root) / "Tools/bin" / name if root else Path(shutil.which(name) or name)
    if not path.is_file():
        pytest.fail(f"{name} is required for the requested simulator test: {path}")
    return str(path)


@SIMULATOR
@pytest.mark.parametrize("arch", ["v73", "v75", "v79"])
def test_generated_mixed_predicates_on_simulator(arch, tmp_path):
    root = Path(__file__).resolve().parents[3]
    source = "#include <hexagon_standalone.h>\n#include <stdio.h>\n#include <cstring>\n"
    calls, inputs, expected = [], bytearray(), bytearray()
    # A nonzero wide operand must not become false by narrowing it before
    # converting to a predicate (65536 and -65536 have zero low 16 bits).
    narrow = np.resize(np.array([0, 1, -1, 7], np.int16), 64)
    wide = np.resize(np.array([65536, -65536, 0, 1, -1], np.int32), 64)
    inputs.extend(narrow.tobytes() + wide.tobytes())
    calls += [
        "if (fread(a, 1, 128, in) != 128 || fread(b, 1, 256, in) != 256) return 7;",
        "if (check_raw_masks((int16_t*)a, (int32_t*)b)) return 8;",
    ]
    # Matching-width controls use the same normal lowering and input rules.
    for index, (matching, case) in enumerate(itertools.product((False, True), CASES)):
        left_dtype, operation, output_dtype = case
        right_dtype = left_dtype if matching else "float32" if left_dtype == "float16" else "float16"
        body = _source(_kernel(*case, matching=matching), arch)
        source += body.replace("kernel_kernel", f"case_{index}")
        arrays, reference = _inputs(*case, matching=matching)
        expected.extend(reference)
        for name, array in zip(("a", "b", "yes", "no"), arrays):
            inputs.extend(array.tobytes())
            calls.append(f"if (fread({name}, 1, {array.nbytes}, in) != {array.nbytes}) return 4;")
        types = ["half" if dtype == "float16" else "float" for dtype in (left_dtype, right_dtype, output_dtype, output_dtype, output_dtype)]
        args = ", ".join(f"({typ}*){name}" for typ, name in zip(types, ("a", "b", "yes", "no", "out")))
        calls += [
            "memset(out, 0xa5, sizeof(out));",
            f"if (case_{index}({args})) return 5;",
            f"if (fwrite(out, 1, {len(reference)}, output) != {len(reference)}) return 6;",
        ]
    source += (
        """
__attribute__((noinline)) int check_raw_masks(const int16_t* a, const int32_t* b) {
  int16_t64 narrow;
  int32_t64 wide;
  memcpy(&narrow, a, sizeof(narrow));
  memcpy(&wide, b, sizeof(wide));
  auto narrow_and = narrow && wide;
  auto wide_and = wide && narrow;
  auto narrow_or = narrow || wide;
  auto wide_or = wide || narrow;
  for (int i = 0; i < 64; ++i) {
    int expected_and = (a[i] != 0 && b[i] != 0) ? -1 : 0;
    int expected_or = (a[i] != 0 || b[i] != 0) ? -1 : 0;
    if (narrow_and.v[i] != expected_and || wide_and.v[i] != expected_and ||
        narrow_or.v[i] != expected_or || wide_or.v[i] != expected_or) return 1;
  }
  return 0;
}
alignas(128) static unsigned char a[2048], b[2048], yes[2048], no[2048], out[2048];
int main(int argc, char** argv) {
  if (argc != 3) return 1;
  SIM_ACQUIRE_HVX;
  SIM_SET_HVX_DOUBLE_MODE;
  FILE* in = fopen(argv[1], "rb"), *output = fopen(argv[2], "wb");
  if (!in || !output) return 2;
"""
        + "\n".join(calls)
        + "\nfclose(in); return fclose(output);\n}\n"
    )
    cpp, executable = tmp_path / "mixed.cc", tmp_path / "mixed"
    input_path, output_path = tmp_path / "input.bin", tmp_path / "output.bin"
    cpp.write_text(source)
    input_path.write_bytes(inputs)
    command = [
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
    ]
    compiled = subprocess.run(command, capture_output=True, text=True, timeout=180)
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    simulated = subprocess.run(
        [_tool("hexagon-sim"), f"-m{arch}", "--simulated_returnval", str(executable), "--", str(input_path), str(output_path)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert simulated.returncode == 0, simulated.stdout + simulated.stderr
    assert output_path.read_bytes() == bytes(expected)
