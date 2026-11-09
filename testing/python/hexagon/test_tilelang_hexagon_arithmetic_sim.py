"""Generated arithmetic kernels on the simulator and an opt-in v73 device.

Only v73 enables the device-qualified local IEEE feature. The other targets'
compilation checks ensure the policy cannot leak into their generated code.
"""

import importlib.util
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess

import numpy as np
import pytest

import tilelang
from tilelang import tvm
from tilelang.hexagon import language as T

SIMULATOR = pytest.mark.skipif(
    os.environ.get("TL_HEXAGON_SIMULATOR_TEST") != "1",
    reason="set TL_HEXAGON_SIMULATOR_TEST=1 to run the Hexagon SDK simulator",
)
DEVICE = pytest.mark.skipif(
    os.environ.get("TL_HEXAGON_DEVICE_TEST") != "1" or os.environ.get("HEXAGON_DSP_ARCH", "v79") != "v73",
    reason="requires an authorized v73 device and SDK",
)


def _tool(name):
    root = os.environ.get("HEXAGON_TOOLS_ROOT")
    path = Path(root) / "Tools/bin" / name if root else Path(shutil.which(name) or name)
    if not path.is_file():
        pytest.fail(f"{name} is required: {path}")
    return str(path)


def _value(kind, a, b, dtype):
    if kind == "add":
        return a + b
    if kind == "mul":
        return a * b
    if kind == "mul_add":
        return a * b + a
    if kind == "affine":
        return a * T.cast(2, dtype) + T.cast(1, dtype)
    if kind == "leaky":
        return T.Select(a > T.cast(0, dtype), a, a * T.cast(0.125, dtype))
    if kind == "cast":
        other = "float32" if dtype == "float16" else "float16"
        return T.cast(T.cast(a, other), dtype)
    if kind == "mixed":
        # A 32-bit comparison mask selects half lanes as well as float lanes;
        # the two comparison operands share a mask type before conjunction.
        mask = T.And(T.cast(a, "float32") > T.float32(0), T.cast(b, "float32") < T.float32(1))
        return T.Select(mask, T.cast(T.cast(a, "int16"), dtype) + b, b * T.cast(0.125, dtype))
    if kind == "exp2":
        return T.exp2(a)
    if kind == "log":
        return T.log(a)
    raise ValueError(kind)


def _kernel(kind, dtype, width=1031, workers=None):
    blocks = 4 if workers else 1
    vector_width = width // 64 * 64

    @T.prim_func
    def kernel(A: T.Tensor((blocks, width), dtype), B: T.Tensor((blocks, width), dtype), C: T.Tensor((blocks, width), dtype)):
        with T.Kernel(blocks, num_workers=workers) as bx:
            for i in T.Parallel(vector_width):
                C[bx, i] = _value(kind, A[bx, i], B[bx, i], dtype)
            for i in T.serial(vector_width, width):
                C[bx, i] = _value(kind, A[bx, i], B[bx, i], dtype)

    return kernel


def _source(func, arch="v73", scalar=False):
    target = tvm.target.Target({"kind": "hexagon", "mcpu": f"hexagon{arch}"})
    with target, tvm.transform.PassContext(opt_level=3, config={"tirx.disable_vectorize": scalar}):
        return str(tilelang.lower(func, target=target, enable_host_codegen=False, enable_device_compile=False).kernel_source)


def _integer_cast_kernel(dtype, lanes, workers=None):
    blocks = 4 if workers else 1

    @T.prim_func
    def kernel(A: T.Tensor((blocks, 256), "float16"), C: T.Tensor((blocks, 256), dtype)):
        with T.Kernel(blocks, num_workers=workers) as bx:
            for tile in T.serial(256 // lanes):
                for i in T.vectorized(lanes):
                    C[bx, tile * lanes + i] = T.cast(A[bx, tile * lanes + i], dtype)

    return kernel


def _integer_inputs(dtype, n):
    halves = np.arange(65536, dtype=np.uint16).view(np.float16)
    limits = np.iinfo(dtype)
    with np.errstate(invalid="ignore"):
        truncated = np.trunc(halves.astype(np.float64))
    values = halves[np.isfinite(halves) & (truncated >= limits.min) & (truncated <= limits.max)]
    if n is None:
        return np.pad(values, (0, (-values.size) % 128))
    # Sampling includes both ends of the representable domain, signed zeros,
    # subnormals, integers, and fractions on either side of half-integers.
    selected = values[np.linspace(0, values.size - 1, n, dtype=np.int64)]
    boundaries = np.array(
        [-1.75, -1.5, -1.25, -0.75, -0.5, -0.25, -0.0, 0.0, 0.25, 0.5, 0.75, 1.25, 1.5, 1.75, 126.5, 127.5, 255.5], np.float16
    )
    valid = boundaries[(np.trunc(boundaries.astype(np.float64)) >= limits.min) & (np.trunc(boundaries.astype(np.float64)) <= limits.max)]
    selected[: valid.size] = valid
    return selected


def test_arithmetic_feature_is_on_kernel_and_worker_bodies():
    serial = _source(_kernel("affine", "float32"))
    worker = _source(_kernel("leaky", "float16", 64, workers=2))
    assert "TL_HEXAGON_IEEE_ARITHMETIC\nint32_t kernel_kernel" in serial
    assert ") TL_HEXAGON_IEEE_ARITHMETIC -> int32_t {" in worker
    assert "float32" in serial and "half64" in worker
    # The dispatcher does not evaluate the worker's floating operations.
    assert worker.count("TL_HEXAGON_IEEE_ARITHMETIC") == 1


def _copy_or_fill_kernel(fill, workers=None, dtype="float32"):
    @T.prim_func
    def kernel(A: T.Tensor((4, 64), dtype), C: T.Tensor((4, 64), dtype)):
        with T.Kernel(4, num_workers=workers) as bx:
            for i in T.Parallel(64):
                C[bx, i] = T.cast(0.125, dtype) if fill else A[bx, i]

    return kernel


@pytest.mark.parametrize("fill", [False, True])
@pytest.mark.parametrize("workers", [None, 2])
@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_float_data_movement_does_not_change_instruction_policy(fill, workers, dtype):
    source = _source(_copy_or_fill_kernel(fill, workers, dtype))
    assert ("half64" if dtype == "float16" else "float32") in source
    assert "TL_HEXAGON_IEEE_ARITHMETIC" not in source
    assert ("tl_parallel_for" in source) == (workers is not None)


@pytest.mark.parametrize("unroll", [False, True])
def test_scalar_float_operations_keep_feature_before_clang_vectorization(unroll):
    @T.prim_func
    def kernel(A: T.Tensor((64,), "float32"), C: T.Tensor((64,), "float32")):
        with T.Kernel(1):
            for i in T.serial(64):
                if unroll:
                    for j in T.unroll(2):
                        C[i] = A[i] * T.float32(0.125) + T.cast(j, "float32")
                else:
                    C[i] = A[i] * T.float32(0.125)

    source = _source(kernel, scalar=True)
    assert "TL_HEXAGON_IEEE_ARITHMETIC\nint32_t kernel_kernel" in source


def test_panel_worker_keeps_resource_helpers_in_default_scope():
    path = Path(__file__).with_name("test_tilelang_hexagon_gemm_int.py")
    spec = importlib.util.spec_from_file_location("arithmetic_panel_test", path)
    panel = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(panel)
    source = _source(panel._blockscaled_hmx(96, 64, 512, 64, "int4", "float32", False, 2, False))
    assert "tl_hexagon_hmx_gemm_blockscaled" in source
    assert "tl_hexagon_dma_async_submit" in source
    assert "tl_parallel_for<tl_hmx_worker_scope>" in source
    assert "TL_HEXAGON_IEEE_ARITHMETIC" not in source


def test_arithmetic_feature_does_not_leak_between_module_functions():
    from tilelang.engine.lower import _prepare_device_codegen_mod

    target = tvm.target.Target({"kind": "hexagon", "mcpu": "hexagonv73"})
    funcs = {}
    with target, tvm.transform.PassContext(opt_level=3):
        for name, func in (("a_arithmetic", _kernel("affine", "float32")), ("z_copy", _copy_or_fill_kernel(False))):
            artifact = tilelang.lower(func, target=target, enable_host_codegen=False, enable_device_compile=False)
            (lowered,) = _prepare_device_codegen_mod(artifact.device_mod).functions.values()
            funcs[name] = lowered.with_attr("global_symbol", name)
        module = tvm.get_global_func("target.build.tilelang_hexagon")(tvm.IRModule(funcs), target)
    source = module.inspect_source()
    assert "TL_HEXAGON_IEEE_ARITHMETIC\nint32_t a_arithmetic" in source
    assert "int32_t z_copy" in source
    assert source.count("TL_HEXAGON_IEEE_ARITHMETIC") == 1


def _float_ramp_source(arch):
    # Ramp's multiply/add is introduced by the C++ helper, not explicit TIR
    # Add/Mul nodes. Exercise the codegen boundary directly before LLVM.
    tir = tvm.tirx
    a = tir.decl_buffer((32,), "float32", name="A")
    c = tir.decl_buffer((32,), "float32", name="C")
    value = tir.Ramp(tir.BufferLoad(a, [0]), tir.BufferLoad(a, [1]), 32)
    body = tir.BufferStore(c, value, [tir.Ramp(0, 1, 32)])
    func = tir.PrimFunc([a.data, c.data], body, buffer_map={a.data: a, c.data: c}).with_attr("global_symbol", "generated_ramp")
    target = tvm.target.Target({"kind": "hexagon", "mcpu": f"hexagon{arch}"})
    module = tvm.get_global_func("target.build.tilelang_hexagon")(tvm.IRModule({"generated_ramp": func}), target)
    return module.inspect_source()


def test_float_ramp_marks_implicit_native_multiply_add():
    source = _float_ramp_source("v73")
    assert "TL_HEXAGON_IEEE_ARITHMETIC\nint32_t generated_ramp" in source
    assert "tl_vec_ramp<float, 32>" in source


def _direct_scope_case(kind, dtype="float32", worker=False):
    tir = tvm.tirx
    name = f"direct_{kind}_{dtype}" + ("_worker" if worker else "")
    a = tir.decl_buffer((2,), dtype, name="A")
    buffers = [a]
    index = tir.Var("bx", "int32") if worker else tir.IntImm("int32", 0)
    output_dtype = "bool" if kind in ("compare", "isnan") else "int32" if kind == "cast" else dtype
    c = tir.decl_buffer((2,), output_dtype, name="C")
    buffers.append(c)
    x = tir.BufferLoad(a, [index])
    if kind == "compare":
        expr = tir.GT(x, tir.BufferLoad(a, [1 - index]))
        body = tir.BufferStore(c, expr, [index])
        markers = (">",)
    elif kind == "isnan":
        body = tir.BufferStore(c, tir.isnan(x), [index])
        markers = ("isnan",)
    elif kind == "cast":
        body = tir.BufferStore(c, tir.Cast("int32", x), [index])
        markers = ("int32_t",)
    elif kind == "entry_then_copy_worker":
        assert not worker and dtype == "float32"
        d = tir.decl_buffer((1,), dtype, name="Entry")
        buffers.append(d)
        entry = tir.BufferStore(d, tir.Add(x, tir.FloatImm(dtype, 1)), [0])
        bx = tir.Var("bx", "int32")
        copy = tir.BufferStore(c, tir.BufferLoad(a, [bx]), [bx])
        loop = tir.For(bx, 0, 2, tir.ForKind.PARALLEL, copy, annotations={"hexagon.num_workers": 2})
        body = tir.SeqStmt([entry, loop])
        markers = ("Entry[0]", "tl_parallel_for", "C_w[bx] = A_w[bx]")
    elif kind == "opaque_void":
        # Pointer-only void call: the return dtype and argument types reveal
        # nothing about floating work in an unknown implementation.
        body = tir.Evaluate(tir.call_extern("void", "opaque_float_effect", a.data, c.data))
        markers = ("opaque_float_effect(A, C)",)
    elif kind == "void_reduce":
        # This inline numerical intrinsic also has a void result. No outer
        # Add/Mul/Cast exists to accidentally make the test pass.
        body = tir.Evaluate(
            tir.call_intrin(
                "void",
                tvm.ir.Op.get("tl.hexagon_reduce_rows"),
                tir.StringImm("TL_REDUCE_SUM"),
                tir.type_annotation("float32"),
                tir.StringImm("false"),
                tir.address_of(tir.BufferLoad(c, [0])),
                tir.address_of(tir.BufferLoad(a, [0])),
                1,
                2,
                2,
                1,
                tir.IntImm("bool", 1),
            )
        )
        markers = ("tl_hvx_reduce_rows<TL_REDUCE_SUM, float, false>",)
    else:
        raise ValueError(kind)
    if worker:
        body = tir.For(index, 0, 2, tir.ForKind.PARALLEL, body, annotations={"hexagon.num_workers": 2})
    func = tir.PrimFunc([b.data for b in buffers], body, buffer_map={b.data: b for b in buffers}).with_attr("global_symbol", name)
    return name, func, not worker, worker, markers


@pytest.mark.parametrize(
    "kind,dtype,worker",
    [(kind, dtype, False) for kind in ("compare", "cast") for dtype in ("float16", "float32")]
    + [
        ("compare", "float32", True),
        ("isnan", "float32", False),
        ("entry_then_copy_worker", "float32", False),
        ("opaque_void", "float32", False),
        ("void_reduce", "float32", False),
    ],
)
def test_direct_float_operation_scopes(kind, dtype, worker):
    # Direct final IR prevents simplification from turning a compare-only or
    # void-call test into unrelated arithmetic that would also enable IEEE.
    name, func, entry_expected, worker_expected, markers = _direct_scope_case(kind, dtype, worker)
    target = tvm.target.Target({"kind": "hexagon", "mcpu": "hexagonv73"})
    module = tvm.get_global_func("target.build.tilelang_hexagon")(tvm.IRModule({name: func}), target)
    source = module.inspect_source()
    entry = f"TL_HEXAGON_IEEE_ARITHMETIC\nint32_t {name}" in source
    worker = bool(re.search(r"\[&\]\([^\n]*\) TL_HEXAGON_IEEE_ARITHMETIC -> int32_t", source))
    assert (entry, worker) == (entry_expected, worker_expected)
    assert source.count("TL_HEXAGON_IEEE_ARITHMETIC") == int(entry) + int(worker)
    for marker in markers:
        assert marker in source


def _cases():
    for dtype in ("float16", "float32"):
        for kind in ("add", "mul", "mul_add", "affine", "leaky", "cast", "mixed", "exp2", "log"):
            yield kind, dtype, 1031, None, False
        # The scalarized form and a worker body are distinct feature boundaries.
        yield "affine", dtype, 19, None, True
        yield "leaky", dtype, 64, 2, False


def _inputs(dtype, n, kind=None):
    dt = np.dtype(dtype)
    rng = np.random.default_rng(230)
    a = rng.standard_normal(n).astype(dt)
    b = rng.standard_normal(n).astype(dt)
    if kind == "mixed":
        return a, b
    if kind == "exp2":
        return np.linspace(-8, 8, n).astype(dt), b
    if kind == "log":
        return np.geomspace(1e-4, 1e4, n).astype(dt), b
    patterns = (
        [0, 0x8000, 1, 0x8001, 0x03FF, 0x0400, 0x3C00, 0x3C01, 0x7BFF, 0x7C00, 0xFC00, 0x7C01, 0x7E35]
        if dt == np.float16
        else [
            0,
            0x80000000,
            1,
            0x80000001,
            0x007FFFFF,
            0x00800000,
            0x3F800000,
            0x3F800001,
            0x7F7FFFFF,
            0x7F800000,
            0xFF800000,
            0x7F800001,
            0x7FC12345,
        ]
    )
    special = np.array(patterns, np.uint16 if dt == np.float16 else np.uint32).view(dt)
    # Include all pairings and repeat special values in the final scalar tail.
    count = min(n, len(special) ** 2)
    a[:count] = np.repeat(special, len(special))[:count]
    b[:count] = np.tile(special, len(special))[:count]
    if n < len(special) ** 2:
        # The small explicitly scalarized kernel still sees every category.
        a[: len(special)] = special
        b[: len(special)] = np.roll(special, 5)
    a[-7:] = special[[0, 1, 2, 4, 8, 11, 12]]
    return a, b


def _expected(kind, a, b):
    with np.errstate(all="ignore"):
        if kind == "add":
            return (a.astype(np.float64) + b.astype(np.float64)).astype(a.dtype)
        if kind == "mul":
            return (a.astype(np.float64) * b.astype(np.float64)).astype(a.dtype)
        if kind == "mul_add":
            product = (a.astype(np.float64) * b.astype(np.float64)).astype(a.dtype)
            return (product.astype(np.float64) + a.astype(np.float64)).astype(a.dtype)
        if kind == "affine":
            doubled = (a.astype(np.float64) * 2).astype(a.dtype)
            return (doubled.astype(np.float64) + 1).astype(a.dtype)
        if kind == "leaky":
            scaled = (a.astype(np.float64) * 0.125).astype(a.dtype)
            return np.where(a > 0, a, scaled)
        if kind == "mixed":
            mask = (a.astype(np.float32) > 0) & (b.astype(np.float32) < 1)
            summed = (a.astype(np.int16).astype(np.float64) + b.astype(np.float64)).astype(a.dtype)
            return np.where(mask, summed, (b.astype(np.float64) * 0.125).astype(a.dtype))
        if kind in ("exp2", "log"):
            return getattr(np, kind)(a.astype(np.float64)).astype(a.dtype)
        other = np.float32 if a.dtype == np.float16 else np.float16
        return a.astype(other).astype(a.dtype)


def _check_result(kind, dtype, actual, reference):
    bits = np.uint16 if dtype == "float16" else np.uint32
    not_nan = ~np.isnan(reference)
    if kind in ("exp2", "log"):
        # Keep the existing math-function accuracy contract inside the new
        # feature scope, including explicit Qfloat polynomial helpers.
        a, _ = _inputs(dtype, actual.size, kind)
        exact = getattr(np, kind)(a.astype(np.float64))
        absolute = np.abs(actual.astype(np.float64) - exact)
        near_zero = np.abs(exact) < 1e-3
        bound = (1e-5 if kind == "exp2" else 1e-6) + (2**-11 if dtype == "float16" else 0)
        assert np.all(absolute[~near_zero] / np.abs(exact[~near_zero]) <= bound), (kind, dtype)
        assert np.all(absolute[near_zero] <= 1e-5), (kind, dtype)
    else:
        np.testing.assert_array_equal(actual[not_nan].view(bits), reference[not_nan].view(bits), err_msg=f"{kind}/{dtype}")
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(reference), err_msg=f"{kind}/{dtype}")


def _program(tmp_path, arch, scalar=False):
    sources, calls, data, expected = [], [], bytearray(), []
    for index, (kind, dtype, width, workers, scalar_case) in enumerate(_cases()):
        n = width * (4 if workers else 1)
        source = _source(_kernel(kind, dtype, width, workers), arch, scalar=scalar or scalar_case)
        source = source.replace("kernel_kernel(", f"generated_{index}(")
        # Execute generated worker bodies serially in the standalone simulator;
        # their lambda feature attribute and arithmetic body remain unchanged.
        source = source.replace("#include <tl_templates/hexagon/worker.h>", "")
        sources.append(source)
        a, b = _inputs(dtype, n, kind)
        data += a.tobytes() + b.tobytes()
        expected.append((kind, dtype, _expected(kind, a, b)))
        ctype = "half" if dtype == "float16" else "float"
        size = n * a.dtype.itemsize
        calls.append(
            f"if (fread(a, {size}, 1, in) != 1 || fread(b, {size}, 1, in) != 1) return 3;\n"
            f"memset(c, 0xA5, sizeof(c));\n"
            f"if (generated_{index}(({ctype}*)a, ({ctype}*)b, ({ctype}*)c)) return 4;\n"
            f"if (fwrite(c, {size + 128}, 1, out) != 1) return 5;\n"
        )
    # Compile and execute the implicit floating operations inside tl_vec_ramp
    # too. A power-of-two stride makes the reference independent of contraction.
    sources.append(_float_ramp_source(arch))
    ramp_input = np.zeros(32, np.float32)
    ramp_input[:2] = [0.1234567, 0.03125]
    data += ramp_input.tobytes()
    ramp_reference = (ramp_input[0].astype(np.float64) + np.arange(32) * float(ramp_input[1])).astype(np.float32)
    expected.append(("ramp", "float32", ramp_reference))
    calls.append(
        "if (fread(a, 128, 1, in) != 1) return 3;\n"
        "memset(c, 0xA5, sizeof(c));\n"
        "if (generated_ramp((float*)a, (float*)c)) return 4;\n"
        "if (fwrite(c, 256, 1, out) != 1) return 5;\n"
    )
    prelude = """
#include <hexagon_standalone.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
template<class Fn> static int tl_parallel_for(int n, int workers, Fn fn) {
  (void)workers;
  for (int i = 0; i < n; ++i) { int rc = fn(i); if (rc) return rc; }
  return 0;
}
"""
    main = """
alignas(128) static uint8_t a[8192], b[8192], c[8192];
int main(int argc, char** argv) {
  if (argc != 3) return 1;
  SIM_ACQUIRE_HVX;
  SIM_SET_HVX_DOUBLE_MODE;
  FILE* in = fopen(argv[1], "rb"); FILE* out = fopen(argv[2], "wb");
  if (!in || !out) return 2;
"""
    path = tmp_path / "generated.cc"
    path.write_text(prelude + "\n".join(sources) + main + "\n".join(calls) + "fclose(in); return fclose(out);\n}\n")
    inputs = tmp_path / "input.bin"
    inputs.write_bytes(data)
    return path, inputs, expected


@SIMULATOR
@pytest.mark.parametrize("arch", ["v73", "v75", "v79"])
def test_generated_arithmetic_feature_compilation(arch, tmp_path):
    root = Path(__file__).resolve().parents[3]
    source, _, _ = _program(tmp_path, arch)
    ir = tmp_path / "generated.ll"
    command = [
        _tool("hexagon-clang++"),
        f"-m{arch}",
        "-mhvx",
        "-mhvx-length=128B",
        "-O2",
        "-std=c++17",
        "-mllvm",
        "--disable-handle-qfp",
        "-I",
        str(root / "src"),
        str(source),
        "-S",
        "-emit-llvm",
        "-o",
        str(ir),
    ]
    compiled = subprocess.run(command, capture_output=True, text=True, timeout=180)
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    text = ir.read_text()
    assert ("+hvx-ieee-fp" in text) == (arch == "v73")
    # Wrappers must not introduce one out-of-line C++ call per vector operation.
    assert not re.search(r"call[^\n]*@[^\n]*(?:vec_type|plERKS|mlERKS)", text)


@SIMULATOR
@pytest.mark.parametrize("columns", [256, 512])
@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_reduction_keeps_one_qfloat_boundary(columns, dtype, tmp_path):
    @T.prim_func
    def reduce(A: T.Tensor((64, columns), dtype), C: T.Tensor((64,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((64, columns), dtype)
            c = T.alloc_shared((64,), "float32")
            T.copy(A, a)
            T.reduce_sum(a, c, dim=1)
            T.copy(c, C)

    # Declaration-only VTCM entry points let the normal generated arithmetic
    # compile without a QURT runtime. No VTCM operation is simulated here.
    generated = _source(reduce).replace("#include <tl_templates/hexagon/vtcm.h>", "")
    source, ir = tmp_path / "reduction.cc", tmp_path / "reduction.ll"
    source.write_text(
        "#include <stdint.h>\nuint8_t *tl_vtcm_arena();\n"
        "unsigned tl_vtcm_arena_bytes();\nstatic unsigned tl_vtcm_shared_high_water;\n" + generated
    )
    root = Path(__file__).resolve().parents[3]
    compiled = subprocess.run(
        [
            _tool("hexagon-clang++"),
            "-mv73",
            "-mhvx",
            "-mhvx-length=128B",
            "-O2",
            "-std=c++17",
            "-mllvm",
            "--disable-handle-qfp",
            "-I",
            str(root / "src"),
            str(source),
            "-S",
            "-emit-llvm",
            "-o",
            str(ir),
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    # Each half-source row crosses the feature boundary once. Its widening and
    # accumulation stay together; FP32-source rows retain their inline path.
    # A feature mismatch used to outline the first/next-step lambdas instead.
    text = ir.read_text()
    assert not re.search(r"call[^\n]*@[^\n]*tl_reduce_qf32_sum", text)
    assert bool(re.search(r"call[^\n]*@[^\n]*tl_reduce_qf32_half_sum", text)) == (dtype == "float16")


def _strided_half_sum_kernel():
    @T.prim_func
    def kernel(A: T.Tensor((5, 266), "float16"), Init: T.Tensor((5, 3), "float32"), B: T.Tensor((5, 3), "float32")):
        with T.Kernel(1):
            src = T.alloc_shared((5, 266), "float16")
            dst = T.alloc_shared((5, 3), "float32")
            T.copy(A, src)
            T.copy(Init, dst)
            T.reduce_sum(src[:, 3:262], dst[:, 2:3], dim=1)
            T.copy(dst, B)

    return kernel


def _strided_half_sum_inputs():
    # Reproduce a cancellation-heavy sample from the seeded tail/stride sweep.
    # The 256-element prefix and three-element tail can round differently at
    # an out-of-line boundary; the contract is the original FP64 sum tolerance.
    rng = np.random.default_rng(7304)
    rng.standard_normal(17027)
    values = (rng.standard_normal(1397) * 4).astype(np.float16)
    initial = (rng.standard_normal(128) * 3).astype(np.float32)[:15].reshape(5, 3)
    source = values[: 5 * 266].reshape(5, 266)
    source[:, 3:11] = np.array([0, 0x8000, 1, 0x8001, 0x03FF, 0x83FF, 0x7BFF, 0xFBFF], np.uint16).view(np.float16)
    reference = source[:, 3:262].astype(np.float64).sum(axis=1)
    return source, initial, reference


def _check_strided_half_sum(raw, initial, reference):
    assert len(raw) == initial.nbytes + 128
    assert raw[initial.nbytes :] == b"\xa5" * 128
    actual = np.frombuffer(raw[: initial.nbytes], np.float32).reshape(initial.shape)
    np.testing.assert_array_equal(actual[:, :2].view(np.uint32), initial[:, :2].view(np.uint32))
    np.testing.assert_allclose(actual[:, 2].astype(np.float64), reference, rtol=1e-5, atol=259 * 1e-5)


@SIMULATOR
def test_generated_half_sum_tail_and_strides(tmp_path):
    source = _source(_strided_half_sum_kernel())
    assert "5, 259, 266, 3, (bool)1" in source
    assert "tl_hvx_reduce_rows<TL_REDUCE_SUM, float, false>" in source
    # Keep the normal generated body; only provide a standalone arena in
    # place of the QURT VTCM allocator. Device coverage uses the real Session.
    source = source.replace("#include <tl_templates/hexagon/vtcm.h>", "")
    prelude = """
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <hexagon_standalone.h>
alignas(128) static uint8_t arena[8192];
static uint8_t *tl_vtcm_arena() { return arena; }
static unsigned tl_vtcm_arena_bytes() { return sizeof(arena); }
static unsigned tl_vtcm_shared_high_water;
"""
    main = """
alignas(128) static half input[5 * 266];
alignas(128) static float initial[15];
alignas(128) static uint8_t output[15 * 4 + 128];
int main(int argc, char **argv) {
  if (argc != 3) return 1;
  SIM_ACQUIRE_HVX; SIM_SET_HVX_DOUBLE_MODE;
  FILE *in = fopen(argv[1], "rb"), *out = fopen(argv[2], "wb");
  if (!in || !out) return 2;
  if (fread(input, sizeof(input), 1, in) != 1 || fread(initial, sizeof(initial), 1, in) != 1) return 3;
  memset(output, 0xA5, sizeof(output));
  if (kernel_kernel(input, initial, reinterpret_cast<float *>(output))) return 4;
  if (fwrite(output, sizeof(output), 1, out) != 1) return 5;
  fclose(in); return fclose(out);
}
"""
    program, executable, inputs, output = (tmp_path / name for name in ("tail.cc", "tail", "input.bin", "output.bin"))
    program.write_text(prelude + source + main)
    values, initial, reference = _strided_half_sum_inputs()
    inputs.write_bytes(values.tobytes() + initial.tobytes())
    root = Path(__file__).resolve().parents[3]
    compiled = subprocess.run(
        [
            _tool("hexagon-clang++"),
            "-mv73",
            "-mhvx",
            "-mhvx-length=128B",
            "-O2",
            "-std=c++17",
            "-mllvm",
            "--disable-handle-qfp",
            "-I",
            str(root / "src"),
            str(program),
            "-o",
            str(executable),
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    simulated = subprocess.run(
        [_tool("hexagon-sim"), "-mv73", "--simulated_returnval", str(executable), "--", str(inputs), str(output)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert simulated.returncode == 0, simulated.stdout + simulated.stderr
    _check_strided_half_sum(output.read_bytes(), initial, reference)


@DEVICE
def test_generated_half_sum_tail_and_strides_on_device(tmp_path):
    from tilelang.hexagon.runtime.session import Session

    source, initial, reference = _strided_half_sum_inputs()
    with Session([(_strided_half_sum_kernel(), [2])]) as session:
        buffers = [session.buffer(size + 128) for size in (source.nbytes, initial.nbytes, initial.nbytes)]
        try:
            for buffer, data in zip(buffers, (source.tobytes(), initial.tobytes(), b"\xa5" * initial.nbytes)):
                buffer.write(data + b"\xa5" * 128)
            session.kernels[0](*buffers)
            raw = buffers[2].read()
            (tmp_path / "half-sum-tail-strides.bin").write_bytes(raw)
            _check_strided_half_sum(raw, initial, reference)
        finally:
            for buffer in reversed(buffers):
                buffer.free()


@SIMULATOR
@pytest.mark.parametrize("example", ["flash_attention", "ffn"])
def test_hmx_weight_interface_stays_inline_in_ieee_kernel(example, tmp_path):
    root = Path(__file__).resolve().parents[3]
    spec = importlib.util.spec_from_file_location(example, root / f"examples/hexagon/example_{example}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    func = module.make_flash(64, 256, 64, 64) if example == "flash_attention" else module.make_ffn(64, 128, 256, 128)
    source, ir = tmp_path / "hmx.cc", tmp_path / "hmx.ll"
    source.write_text(_source(func))
    sdk = Path(os.environ["HEXAGON_SDK_ROOT"])
    includes = [root / "src", sdk / "incs", sdk / "incs/stddef", sdk / "rtos/qurt"]
    includes.extend(sdk / "rtos/qurt/computev73/include" / subdir for subdir in ("", "qurt", "posix"))
    command = [
        _tool("hexagon-clang++"),
        "-mv73",
        "-mhmx",
        "-mhvx",
        "-mhvx-length=128B",
        "-O2",
        "-std=c++17",
        "-mllvm",
        "--disable-handle-qfp",
        *[argument for directory in includes for argument in ("-I", str(directory))],
        str(source),
        "-S",
        "-emit-llvm",
        "-o",
        str(ir),
    ]
    compiled = subprocess.run(command, capture_output=True, text=True, timeout=180)
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    # Empty weight hooks, column lookup and tile lambdas must not become calls
    # inside the GEMM loops solely because the enclosing kernel enables IEEE.
    assert not re.search(r"call[^\n]*@[^\n]*(?:tl_hmx_weight_tiles|tl_hmx_gemm)", ir.read_text())


@SIMULATOR
def test_generated_v73_arithmetic_matches_ieee(tmp_path):
    root = Path(__file__).resolve().parents[3]
    source, inputs, expected = _program(tmp_path, "v73")
    executable, output = tmp_path / "generated", tmp_path / "output.bin"
    compiled = subprocess.run(
        [
            _tool("hexagon-clang++"),
            "-mv73",
            "-mhvx",
            "-mhvx-length=128B",
            "-O2",
            "-std=c++17",
            "-mllvm",
            "--disable-handle-qfp",
            "-I",
            str(root / "src"),
            str(source),
            "-o",
            str(executable),
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    simulated = subprocess.run(
        [_tool("hexagon-sim"), "-mv73", "--simulated_returnval", str(executable), "--", str(inputs), str(output)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert simulated.returncode == 0, simulated.stdout + simulated.stderr
    raw, offset = output.read_bytes(), 0
    for kind, dtype, reference in expected:
        size = reference.nbytes
        actual = np.frombuffer(raw[offset : offset + size], reference.dtype)
        assert raw[offset + size : offset + size + 128] == b"\xa5" * 128, (kind, dtype)
        _check_result(kind, dtype, actual, reference)
        offset += size + 128
    assert offset == len(raw)


@SIMULATOR
def test_half_to_integer_truncates_across_vector_widths(tmp_path):
    root = Path(__file__).resolve().parents[3]
    source = Path(__file__).with_name("hvx_arithmetic_cast_sim.cc")
    executable, inputs, output = (tmp_path / name for name in ("casts", "input.bin", "output.bin"))
    compiled = subprocess.run(
        [
            _tool("hexagon-clang++"),
            "-mv73",
            "-mhvx",
            "-mhvx-length=128B",
            "-O2",
            "-std=c++17",
            "-mllvm",
            "--disable-handle-qfp",
            "-I",
            str(root / "src"),
            str(source),
            "-o",
            str(executable),
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    expected, data = [], bytearray()
    for bits in (8, 16, 32, 64):
        for sign in ("int", "uint"):
            dtype = np.dtype(f"{sign}{bits}")
            # Conversion is defined only when the truncated finite value fits.
            values = _integer_inputs(dtype, None)
            data += struct.pack("<I", values.size) + values.tobytes()
            expected.append(values.astype(dtype))
    inputs.write_bytes(data)
    simulated = subprocess.run(
        [_tool("hexagon-sim"), "-mv73", "--simulated_returnval", str(executable), "--", str(inputs), str(output)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert simulated.returncode == 0, simulated.stdout + simulated.stderr
    raw, offset = output.read_bytes(), 0
    for reference in expected:
        for width in (2, 4, 8, 16, 32, 64, 128):
            size = reference.nbytes
            actual = np.frombuffer(raw[offset : offset + size], reference.dtype)
            np.testing.assert_array_equal(actual, reference, err_msg=f"{reference.dtype}/{width}")
            assert raw[offset + size : offset + size + 128] == b"\xa5" * 128
            offset += size + 128
    # bool is an integral C++ type, but its conversion means nonzero rather
    # than truncation. Both signed zeros are false; fractions and NaNs true.
    boolean = ((np.arange(65536, dtype=np.uint16) & 0x7FFF) != 0).astype(np.uint8)
    for width in (2, 4, 8, 16, 32, 64, 128):
        actual = np.frombuffer(raw[offset : offset + 65536], np.uint8)
        np.testing.assert_array_equal(actual, boolean, err_msg=f"bool/{width}")
        offset += 65536
    assert offset == len(raw)


@DEVICE
def test_generated_v73_arithmetic_on_device(tmp_path):
    from tilelang.hexagon.runtime.session import Session

    cases = list(_cases())
    kernels = []
    for kind, dtype, width, workers, scalar in cases:
        func = _kernel(kind, dtype, width, workers)
        if scalar:
            func = func.with_attr("tilelang_pass_configs", {"tirx.disable_vectorize": True})
        kernels.append((func, [2]))
    # One session keeps every normally lowered serial/worker variant in the
    # same generated project. Inputs and outputs have 128-byte guard regions.
    with Session(kernels) as session:
        for index, (kind, dtype, width, workers, _) in enumerate(cases):
            n = width * (4 if workers else 1)
            a, b = _inputs(dtype, n, kind)
            size = a.nbytes
            buffers = [session.buffer(size + 128) for _ in range(3)]
            try:
                for buffer, data in zip(buffers, (a.tobytes(), b.tobytes(), b"\xa5" * size)):
                    buffer.write(data + b"\xa5" * 128)
                session.kernels[index](*buffers)
                raw = buffers[2].read()
                (tmp_path / f"case-{index}-{kind}-{dtype}.bin").write_bytes(raw)
                assert raw[size:] == b"\xa5" * 128
                actual = np.frombuffer(raw[:size], np.dtype(dtype))
                _check_result(kind, dtype, actual, _expected(kind, a, b))
            finally:
                for buffer in reversed(buffers):
                    buffer.free()


@DEVICE
@pytest.mark.parametrize("dtype", [f"{sign}{bits}" for bits in (8, 16, 32, 64) for sign in ("int", "uint")])
def test_generated_v73_integer_casts_on_device(dtype, tmp_path):
    from tilelang.engine.param import KernelParam
    from tilelang.hexagon.runtime.session import Session, lower_kernels

    cases = [(lanes, None) for lanes in (1, 2, 4, 8, 16, 32, 64, 128)] + [(64, 2)]
    kernels = [(_integer_cast_kernel(dtype, lanes, workers), [1]) for lanes, workers in cases]
    lowered = lower_kernels(kernels)
    if dtype.startswith("uint") and dtype != "uint8":
        # Unsigned 16/32/64-bit storage is not an adapter dtype. Keep the
        # normally generated arithmetic body intact and expose its bytes
        # through an equivalent signed pointer in a thin C entry wrapper.
        wrapped = []
        storage = dtype.replace("uint", "int")
        for index, (name, source, params, out_idx) in enumerate(lowered):
            implementation = f"{name}_impl_{dtype}_{index}"
            source = source.replace(f"{name}(", f"{implementation}(")
            source += (
                f'extern "C" int32_t {name}(half* A, {storage}_t* C) {{ return {implementation}(A, reinterpret_cast<{dtype}_t*>(C)); }}\n'
            )
            params[1] = KernelParam(storage, params[1].shape)
            wrapped.append((name, source, params, out_idx))
        lowered = wrapped
    with Session.of_lowered(lowered, dsp_arch="v73") as session:
        for index, (lanes, workers) in enumerate(cases):
            n = 256 * (4 if workers else 1)
            values = _integer_inputs(np.dtype(dtype), n)
            reference = values.astype(dtype)
            src, dst = session.buffer(values.nbytes + 128), session.buffer(reference.nbytes + 128)
            try:
                src.write(values.tobytes() + b"\xa5" * 128)
                dst.write(b"\xa5" * (reference.nbytes + 128))
                session.kernels[index](src, dst)
                raw = dst.read()
                (tmp_path / f"{dtype}-{lanes}-{workers}.bin").write_bytes(raw)
                assert raw[reference.nbytes :] == b"\xa5" * 128
                np.testing.assert_array_equal(np.frombuffer(raw[: reference.nbytes], np.dtype(dtype)), reference)
            finally:
                dst.free()
                src.free()
