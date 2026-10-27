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
