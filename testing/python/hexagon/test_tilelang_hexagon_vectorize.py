"""Hexagon maps vectorize through the standard lowering onto whole HVX registers."""

import pytest

import tilelang
from tilelang import tvm
from tilelang.backend.target import determine_target
from tilelang.hexagon import language as T


def _source(func):
    target = determine_target("hexagon", return_object=True)
    with tvm.target.Target(target), tvm.transform.PassContext(opt_level=3):
        return str(tilelang.lower(func, target=target, enable_host_codegen=False, enable_device_compile=False).kernel_source)


@pytest.mark.parametrize("dtype,vector", [("float16", "half64"), ("float32", "float32"), ("int8", "int8_t128")])
def test_maps_use_whole_hvx_registers(dtype, vector):
    @T.prim_func
    def add(A: T.Tensor((4, 512), dtype), B: T.Tensor((4, 512), dtype), C: T.Tensor((4, 512), dtype)):
        with T.Kernel(4) as bx:
            for j in T.Parallel(512):
                C[bx, j] = A[bx, j] + B[bx, j]

    source = _source(add)
    # One 1024-bit register per vector, as clang ext_vector arithmetic.
    assert f"*({vector}*)(C + " in source
    assert f"(*({vector}*)(A + " in source


def test_a_narrow_input_fills_whole_registers():
    @T.prim_func
    def widen(A: T.Tensor((4, 512), "uint8"), B: T.Tensor((4, 512), "float16")):
        with T.Kernel(4) as bx:
            for j in T.Parallel(512):
                B[bx, j] = T.Cast("float16", A[bx, j])

    source = _source(widen)
    # 128 uint8 lanes fill one register; the fp16 result takes two.
    assert "*(uint8_t128*)(A + " in source
    assert "((half128)*(uint8_t128*)" in source
    assert "uint8_t64" not in source


def test_a_loop_too_short_for_a_register_is_scalar():
    @T.prim_func
    def widen(A: T.Tensor((64,), "uint8"), B: T.Tensor((64,), "float16")):
        with T.Kernel(1):
            for j in T.Parallel(64):
                B[j] = T.Cast("float16", A[j])

    source = _source(widen)
    # 64 uint8 lanes are half a register, which HVX cannot operate on.
    assert "uint8_t64" not in source
    assert "half64" not in source


@pytest.mark.parametrize("dtype,vector", [("float32", "float32"), ("float16", "half64")])
def test_vector_division_uses_the_hvx_reciprocal(dtype, vector):
    @T.prim_func
    def divide(A: T.Tensor((4, 256), dtype), B: T.Tensor((4, 256), dtype), C: T.Tensor((4, 256), dtype)):
        with T.Kernel(4) as bx:
            for j in T.Parallel(256):
                C[bx, j] = A[bx, j] / B[bx, j]
            # A scalar division stays a C division.
            C[bx, 0] = A[bx, 0] / B[bx, 1]

    source = _source(divide)
    assert f"tl_hvx_div(*({vector}*)(A + " in source
    assert "(A[(bx * 256)] / B[((bx * 256) + 1)])" in source


@pytest.mark.parametrize("dtype,vector", [("float32", "float32"), ("float16", "half64"), ("int8", "int8_t128")])
def test_vector_min_and_max_are_hvx_instructions(dtype, vector):
    @T.prim_func
    def clamp(A: T.Tensor((4, 256), dtype), B: T.Tensor((4, 256), dtype)):
        with T.Kernel(4) as bx:
            for j in T.Parallel(256):
                B[bx, j] = T.min(T.max(A[bx, j], T.cast(0, dtype)), T.cast(1, dtype))
            # A scalar min stays a C conditional.
            B[bx, 0] = T.min(A[bx, 1], A[bx, 2])

    source = _source(clamp)
    assert f"tl_vec_min(tl_vec_max(*({vector}*)(A + " in source
    assert ") < (" in source


# DSL function: the hvx_math.h function it prints as.
MATH = {fn: f"tl_hvx_{fn}" for fn in ["exp", "exp2", "log", "log2", "sqrt", "rsqrt", "tanh", "sigmoid", "erf"]}
MATH["abs"] = "tl_hvx_fabs"


@pytest.mark.parametrize("fn", sorted(MATH))
@pytest.mark.parametrize("dtype,vector", [("float32", "float32"), ("float16", "half64")])
def test_math_vectorizes_to_hvx_functions(fn, dtype, vector):
    op = getattr(T, fn)

    @T.prim_func
    def math(A: T.Tensor((4, 256), dtype), B: T.Tensor((4, 256), dtype)):
        with T.Kernel(4) as bx:
            for j in T.Parallel(256):
                B[bx, j] = op(A[bx, j])
            # A scalar runs the same HVX kernel.
            B[bx, 0] = op(A[bx, 1])

    source = _source(math)
    assert f"{MATH[fn]}(*({vector}*)(A + " in source
    assert f"{MATH[fn]}(A[" in source
    # No libm call, scalar or vector.
    for libm in ("expf", "exp2f", "logf", "log2f", "sqrtf", "tanhf", "erff", "fabsf"):
        assert f"{libm}(" not in source
