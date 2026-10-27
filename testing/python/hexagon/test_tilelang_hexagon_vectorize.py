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


# ---------------------------------------------------------------------------
# Per-row scalars broadcast from registers (BroadcastFromRegisters).
# ---------------------------------------------------------------------------


def _row_map(rows, cols, row_dtype):
    """S[i, j] = exp(S[i, j] - r[i]) in fp32 over an fp16 tile, as in softmax."""

    @T.prim_func
    def kernel(X: T.Tensor((rows, cols), "float16"), R: T.Tensor((rows,), row_dtype), Y: T.Tensor((rows, cols), "float16")):
        with T.Kernel(1):
            s = T.alloc_shared((rows, cols), "float16")
            r = T.alloc_shared((rows,), row_dtype)
            T.copy(X, s)
            T.copy(R, r)
            for i, j in T.Parallel(rows, cols):
                s[i, j] = T.cast(T.exp(T.cast(s[i, j], "float32") - T.cast(r[i], "float32")), "float16")
            T.copy(s, Y)

    return kernel


def test_row_scalars_come_from_one_register_per_strip():
    source = _source(_row_map(64, 64, "float32"))
    # Strips of 32 rows: one register of fp32 scalars each, broadcast per row.
    assert "for (int32_t i_strip = 0; i_strip < 2; ++i_strip) {" in source
    assert "float32 r_lanes = tl_hvx_load_lanes<32>((&(((float*)r)[(i_strip * 32)])), 32);" in source
    assert "for (int32_t i_lane = 0; i_lane < 32; ++i_lane) {" in source
    assert "tl_hvx_broadcast_lane<32, 1>(r_lanes, i_lane)" in source
    # The row loop reads no scalar of r.
    assert "((float*)r)[" not in source.split("i_strip")[1].split("tl_hvx_load_lanes")[0]


def test_fp16_row_scalars_fill_a_strip_of_64_rows():
    source = _source(_row_map(128, 64, "float16"))
    assert "half64 r_lanes = tl_hvx_load_lanes<64>((&(((half*)r)[(i_strip * 64)])), 64);" in source


def test_the_last_strip_loads_only_its_rows():
    source = _source(_row_map(70, 64, "float32"))
    assert "for (int32_t i_strip = 0; i_strip < 3; ++i_strip) {" in source
    assert "int32_t v_ = 70 - (i_strip * 32);" in source
    assert "tl_hvx_load_lanes<32>((&(((float*)r)[(i_strip * 32)])), ((32) < (v_) ? (32) : (v_)));" in source


def test_rows_of_several_registers_split_into_rows_and_columns():
    # 256 fp16 columns are 4 registers per row: T.Parallel fuses them into one
    # loop over all 256 vectors, which splits back into rows of 4.
    source = _source(_row_map(64, 256, "float32"))
    assert "for (int32_t i_strip = 0; i_strip < 2; ++i_strip) {" in source
    assert "for (int32_t i_lane = 0; i_lane < 32; ++i_lane) {" in source
    assert "for (int32_t i_col = 0; i_col < 4; ++i_col) {" in source
    assert "tl_hvx_broadcast_lane<32, 1>(r_lanes, i_lane)" in source


def test_scalars_that_do_not_step_with_the_row_stay_loads():
    @T.prim_func
    def kernel(X: T.Tensor((64, 64), "float32"), R: T.Tensor((128,), "float32"), Y: T.Tensor((64, 64), "float32")):
        with T.Kernel(1):
            s = T.alloc_shared((64, 64), "float32")
            r = T.alloc_shared((128,), "float32")
            T.copy(X, s)
            T.copy(R, r)
            for i, j in T.Parallel(64, 64):
                s[i, j] = s[i, j] * r[i * 2]
            T.copy(s, Y)

    source = _source(kernel)
    assert "tl_hvx_broadcast_lane" not in source


@pytest.mark.skipif(__import__("os").environ.get("TL_HEXAGON_DEVICE_TEST") != "1", reason="requires an authorized Hexagon device and SDK")
@pytest.mark.parametrize(
    "rows,cols,row_dtype", [(64, 64, "float32"), (70, 64, "float32"), (128, 64, "float16"), (100, 64, "float16"), (48, 256, "float32")]
)
def test_row_broadcast_maps_match_torch(rows, cols, row_dtype):
    import torch

    kernel = tilelang.compile(_row_map(rows, cols, row_dtype), out_idx=[2], target="hexagon")
    try:
        gen = torch.Generator().manual_seed(3)
        x = (torch.randn(rows, cols, generator=gen) * 0.5).half()
        r = torch.randn(rows, generator=gen).to(getattr(torch, row_dtype))
        y = kernel(x, r)
        expected = torch.exp(x.float() - r.float()[:, None]).half()
        torch.testing.assert_close(y, expected, rtol=2e-3, atol=2e-3)
    finally:
        kernel.adapter.close()
