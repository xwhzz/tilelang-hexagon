"""T.reduce_* on Hexagon: in place on shared buffers and fragments, any axis."""

import functools
import os

import pytest

import tilelang
from tilelang import tvm
from tilelang.backend.target import determine_target
from tilelang.hexagon import language as T

DEVICE = pytest.mark.skipif(os.environ.get("TL_HEXAGON_DEVICE_TEST") != "1", reason="requires an authorized Hexagon device and SDK")

_FLOAT_OPS = ("sum", "max", "min", "abssum", "absmax")
_BIT_OPS = ("bitand", "bitor", "bitxor")


def _emit(op, src, dst, dim, clear):
    if op == "abssum":
        T.reduce_abssum(src, dst, dim=dim)
    else:
        getattr(T, f"reduce_{op}")(src, dst, dim=dim, clear=clear)


def _make(op, shape, dim, src_dtype, dst_dtype, clear=True, scope="shared"):
    """A[shape] -> B, reducing `dim` of a shared or fragment copy of A into a
    copy of Init (so clear=False accumulates onto Init)."""
    dst_shape = tuple(e for d, e in enumerate(shape) if d != dim)
    alloc = T.alloc_shared if scope == "shared" else T.alloc_fragment

    @T.prim_func
    def kernel(A: T.Tensor(shape, src_dtype), Init: T.Tensor(dst_shape, dst_dtype), B: T.Tensor(dst_shape, dst_dtype)):
        with T.Kernel(1):
            src = alloc(shape, src_dtype)
            dst = alloc(dst_shape, dst_dtype)
            T.copy(A, src)
            T.copy(Init, dst)
            _emit(op, src, dst, dim, clear)
            T.copy(dst, B)

    return kernel


def _source(func):
    target = determine_target("hexagon", return_object=True)
    with tvm.target.Target(target), tvm.transform.PassContext(opt_level=3):
        return str(tilelang.lower(func, target=target, enable_host_codegen=False, enable_device_compile=False).kernel_source)


@pytest.mark.parametrize("scope", ["shared", "fragment"])
@pytest.mark.parametrize("op", _FLOAT_OPS)
def test_innermost_axis_is_one_call_over_the_rows(op, scope):
    source = _source(_make(op, (64, 256), 1, "float32", "float32", scope=scope))
    kind = f"TL_REDUCE_{op.upper()}"
    assert f"tl_hvx_reduce_rows<{kind}, float, true>((&(((float*)dst)[0])), (&(((float*)src)[0])), 64, 256, 256, 1, (bool)1);" in source
    # The source stays in the arena: no fragment copy of it.
    assert source.count("buf_dyn_shmem +") == 2


def test_rows_of_any_length_use_the_same_call():
    # Rows shorter than a register, and rows that do not start on a register
    # boundary (200 floats), take the call's tail and unaligned paths.
    assert (
        "tl_hvx_reduce_rows<TL_REDUCE_SUM, float, false>((&(((float*)dst)[0])), (&(((float*)src)[0])), 64, 20, 20, 1, (bool)1);"
        in _source(_make("sum", (64, 20), 1, "float32", "float32"))
    )
    assert (
        "tl_hvx_reduce_rows<TL_REDUCE_MAX, float, false>((&(((float*)dst)[0])), (&(((float*)src)[0])), 64, 200, 200, 1, (bool)1);"
        in _source(_make("max", (64, 200), 1, "float32", "float32"))
    )


def test_outer_axes_loop_around_the_call():
    source = _source(_make("sum", (4, 24, 96), 2, "float32", "float32"))
    assert "for (int32_t i0 = 0; i0 < 4; ++i0) {" in source
    assert (
        "tl_hvx_reduce_rows<TL_REDUCE_SUM, float, true>((&(((float*)dst)[(i0 * 24)])), (&(((float*)src)[(i0 * 2304)])), 24, 96, 96, 1, (bool)1);"
        in source
    )


def _make_regions(op, clear=True):
    """Reduces columns [8, 200) of a 64x256 tile into column 1 of a 64x4 one."""

    @T.prim_func
    def kernel(A: T.Tensor((64, 256), "float32"), Init: T.Tensor((64, 4), "float32"), B: T.Tensor((64, 4), "float32")):
        with T.Kernel(1):
            src = T.alloc_shared((64, 256), "float32")
            dst = T.alloc_shared((64, 4), "float32")
            T.copy(A, src)
            T.copy(Init, dst)
            getattr(T, f"reduce_{op}")(src[:, 8:200], dst[:, 1:2], dim=1, clear=clear)
            T.copy(dst, B)

    return kernel


def test_regions_pass_their_origin_and_strides():
    source = _source(_make_regions("max"))
    assert (
        "tl_hvx_reduce_rows<TL_REDUCE_MAX, float, false>((&(((float*)dst)[1])), (&(((float*)src)[8])), 64, 192, 256, 4, (bool)1);" in source
    )


def test_crouton_tiles_reduce_register_by_register():
    from tilelang.hexagon.intrinsics.crouton import make_hmx_output_layout

    @T.prim_func
    def kernel(A: T.Tensor((64, 64), "float16"), B: T.Tensor((64,), "float32")):
        with T.Kernel(1):
            src = T.alloc_shared((64, 64), "float16")
            dst = T.alloc_shared((64,), "float32")
            T.annotate_layout({src: make_hmx_output_layout(src)})
            T.copy(A, src)
            T.reduce_max(src, dst, dim=1)
            T.copy(dst, B)

    source = _source(kernel)
    assert "tl_crouton_reduce<TL_REDUCE_MAX, half, true, false>(" in source
    assert "tl_hvx_reduce_rows" not in source


def test_other_layouts_reduce_through_them():
    # A layout without a lowering of its own reduces one element at a time.
    @T.prim_func
    def kernel(A: T.Tensor((64, 64), "float16"), B: T.Tensor((64,), "float32")):
        with T.Kernel(1):
            src = T.alloc_shared((64, 64), "float16")
            dst = T.alloc_shared((64,), "float32")
            T.annotate_layout({src: T.Layout((64, 64), lambda i, j: [j, i])})
            T.copy(A, src)
            T.reduce_max(src, dst, dim=1)
            T.copy(dst, B)

    source = _source(kernel)
    assert "tl_hvx_reduce_rows" not in source
    assert "for (int32_t k = 0; k < 64; ++k) {" in source


def test_outer_axis_accumulates_whole_registers():
    source = _source(_make("max", (64, 256), 0, "float32", "float32"))
    assert "*(float32*)(acc + 0) = tl_vec_max(*(float32*)(acc + 0), *(float32*)(((float*)src) + ((k * 256) + (j0 * 32))));" in source
    assert "tl_hvx_lane" not in source


def test_fp16_max_stays_in_fp16():
    # Max is exact, so fp16 lanes need no widening; a sum accumulates in fp32.
    assert "tl_hvx_reduce_rows<TL_REDUCE_MAX, half, true>(" in _source(_make("max", (64, 256), 1, "float16", "float32"))
    assert "tl_hvx_reduce_rows<TL_REDUCE_SUM, float, true>(" in _source(_make("sum", (64, 256), 1, "float16", "float32"))
    # The fp32 values already in the destination join an accumulating max.
    assert "tl_hvx_reduce_rows<TL_REDUCE_MAX, float, true>(" in _source(_make("max", (64, 256), 1, "float16", "float32", clear=False))


def test_accumulating_fp16_max_into_fp32_widens():
    # The fp32 values already in the destination join the reduction.
    source = _source(_make("max", (64, 256), 0, "float16", "float32", clear=False))
    assert "*(float64*)(acc + 0) = tl_vec_max(*(float64*)(acc + 0), ((float64)" in source
    assert "half acc" not in source


def test_reduce_rejects_global_buffers():
    with pytest.raises(ValueError, match="Invalid buffer scopes: global"):

        @T.prim_func
        def kernel(A: T.Tensor((64, 64), "float32"), B: T.Tensor((64,), "float32")):
            with T.Kernel(1):
                T.reduce_sum(A, B, dim=1)


def test_reduce_rejects_nan_propagation():
    @T.prim_func
    def kernel(A: T.Tensor((64, 64), "float16"), B: T.Tensor((64,), "float16")):
        with T.Kernel(1):
            a = T.alloc_shared((64, 64), "float16")
            b = T.alloc_shared((64,), "float16")
            T.copy(A, a)
            T.reduce_max(a, b, dim=1, annotations={"nan_propagate": True})
            T.copy(b, B)

    with pytest.raises(Exception, match="does not propagate NaN"):
        _source(kernel)


# ---------------------------------------------------------------------------
# Device numerics against torch.
# ---------------------------------------------------------------------------


def _reference(torch, A, op, dim):
    if op == "sum":
        return A.sum(dim=dim)
    if op == "max":
        return A.amax(dim=dim)
    if op == "min":
        return A.amin(dim=dim)
    if op == "abssum":
        return A.abs().sum(dim=dim)
    if op == "absmax":
        return A.abs().amax(dim=dim)
    binop = {"bitand": torch.bitwise_and, "bitor": torch.bitwise_or, "bitxor": torch.bitwise_xor}[op]
    return functools.reduce(lambda acc, i: binop(acc, A.select(dim, i)), range(1, A.shape[dim]), A.select(dim, 0).clone())


def _combine(torch, op, init, value):
    if op in ("sum", "abssum"):
        return init + value
    if op in ("max", "absmax"):
        return torch.maximum(init, value)
    if op == "min":
        return torch.minimum(init, value)
    return {"bitand": torch.bitwise_and, "bitor": torch.bitwise_or, "bitxor": torch.bitwise_xor}[op](init, value)


def _check(op, shape, dim, src_dtype, dst_dtype, clear=True, scope="shared"):
    import torch

    kernel = tilelang.compile(_make(op, shape, dim, src_dtype, dst_dtype, clear, scope), out_idx=[2], target="hexagon")
    try:
        gen = torch.Generator().manual_seed(7)
        dst_shape = tuple(e for d, e in enumerate(shape) if d != dim)
        if src_dtype.startswith("int"):
            A = torch.randint(-50, 50, shape, generator=gen).to(getattr(torch, src_dtype))
            init = torch.randint(-50, 50, dst_shape, generator=gen).to(getattr(torch, dst_dtype))
        else:
            A = torch.randn(shape, generator=gen).to(getattr(torch, src_dtype))
            init = torch.randn(dst_shape, generator=gen).to(getattr(torch, dst_dtype))
        out = kernel(A, init)
        wide = torch.float64 if dst_dtype.startswith("float") else torch.int64
        expected = _reference(torch, A.to(wide), op, dim)
        if not clear:
            expected = _combine(torch, op, init.to(wide), expected)
        if dst_dtype.startswith("int"):
            torch.testing.assert_close(out, expected.to(out.dtype), rtol=0, atol=0)
        elif op in ("sum", "abssum"):
            # fp32 accumulation, then one rounding to the destination.
            tol = 1e-3 if dst_dtype == "float16" else 1e-5
            torch.testing.assert_close(out.double(), expected.to(out.dtype).double(), rtol=tol, atol=tol * shape[dim])
        else:
            torch.testing.assert_close(out, expected.to(out.dtype), rtol=0, atol=0)
    finally:
        kernel.adapter.close()


@DEVICE
@pytest.mark.parametrize("dim", [0, 1])
@pytest.mark.parametrize("op", _FLOAT_OPS)
def test_float_reductions_match_torch(op, dim):
    # 200 = 3 * 64 + 8: whole registers and a tail on either axis.
    _check(op, (64, 200), dim, "float16", "float32")


@DEVICE
@pytest.mark.parametrize("op", ["sum", "max"])
@pytest.mark.parametrize("src_dtype,dst_dtype", [("float32", "float32"), ("float16", "float16"), ("float32", "float16")])
def test_float_types_match_torch(op, src_dtype, dst_dtype):
    _check(op, (32, 256), 1, src_dtype, dst_dtype, scope="fragment")


@DEVICE
@pytest.mark.parametrize("dim", [0, 1])
@pytest.mark.parametrize("op", ["sum", "max", "min", "absmax"])
def test_accumulating_reductions_match_torch(op, dim):
    _check(op, (48, 160), dim, "float16", "float32", clear=False, scope="fragment")


@DEVICE
@pytest.mark.parametrize("dim", [0, 1])
@pytest.mark.parametrize("op", _BIT_OPS + ("sum", "max"))
def test_integer_reductions_match_torch(op, dim):
    _check(op, (16, 288), dim, "int32", "int32")


@DEVICE
def test_int8_sums_into_int32():
    _check("sum", (16, 256), 1, "int8", "int32")


@DEVICE
@pytest.mark.parametrize("op", ["sum", "bitxor"])
def test_int32_reduces_into_int8(op):
    # Four source registers fill one int8 accumulator register.
    _check(op, (16, 300), 1, "int32", "int8")


def _make_unaligned_output(op):
    """Reduces 64 rows into elements [3, 67) of a 70-element buffer."""

    @T.prim_func
    def kernel(A: T.Tensor((64, 96), "float32"), Init: T.Tensor((70,), "float32"), B: T.Tensor((70,), "float32")):
        with T.Kernel(1):
            src = T.alloc_shared((64, 96), "float32")
            dst = T.alloc_shared((70,), "float32")
            T.copy(A, src)
            T.copy(Init, dst)
            getattr(T, f"reduce_{op}")(src, dst[3:67], dim=1, clear=False)
            T.copy(dst, B)

    return kernel


@DEVICE
@pytest.mark.parametrize("op", ["sum", "max"])
def test_unaligned_outputs_keep_their_neighbours(op):
    import torch

    kernel = tilelang.compile(_make_unaligned_output(op), out_idx=[2], target="hexagon")
    try:
        gen = torch.Generator().manual_seed(13)
        A = torch.randn(64, 96, generator=gen)
        init = torch.randn(70, generator=gen)
        out = kernel(A, init)
        value = A.double().sum(dim=1) if op == "sum" else A.double().amax(dim=1)
        expected = init.double().clone()
        expected[3:67] = expected[3:67] + value if op == "sum" else torch.maximum(expected[3:67], value)
        torch.testing.assert_close(out.double(), expected, rtol=1e-5, atol=1e-4)
    finally:
        kernel.adapter.close()


@DEVICE
@pytest.mark.parametrize("op", ["sum", "absmax"])
def test_short_rows_match_torch(op):
    _check(op, (8, 20), 1, "float32", "float32")


@DEVICE
@pytest.mark.parametrize("op", ["sum", "max", "min", "absmax"])
@pytest.mark.parametrize("rows", [1, 7, 70])
def test_partial_row_blocks_match_torch(op, rows):
    # A block is 32 fp32 rows; 70 = 2 blocks + 6 rows, and a 1-row block folds
    # within one register.
    _check(op, (rows, 96), 1, "float32", "float32")
    _check(op, (rows, 96), 1, "float16", "float16", scope="fragment")


@DEVICE
@pytest.mark.parametrize("op", ["sum", "max"])
@pytest.mark.parametrize("clear", [True, False])
def test_regions_match_torch(op, clear):
    import torch

    kernel = tilelang.compile(_make_regions(op, clear), out_idx=[2], target="hexagon")
    try:
        gen = torch.Generator().manual_seed(11)
        A = torch.randn(64, 256, generator=gen)
        init = torch.randn(64, 4, generator=gen)
        out = kernel(A, init)
        part = A[:, 8:200].double()
        value = part.sum(dim=1) if op == "sum" else part.amax(dim=1)
        expected = init.double().clone()
        expected[:, 1] = value if clear else (expected[:, 1] + value if op == "sum" else torch.maximum(expected[:, 1], value))
        torch.testing.assert_close(out.double(), expected, rtol=1e-5, atol=1e-4)
    finally:
        kernel.adapter.close()


def _make_vector(op, n, src_dtype, dst_dtype):
    """Reduces a vector of n elements into a one-element buffer."""

    @T.prim_func
    def kernel(A: T.Tensor((n,), src_dtype), B: T.Tensor((1,), dst_dtype)):
        with T.Kernel(1):
            src = T.alloc_shared((n,), src_dtype)
            dst = T.alloc_shared((1,), dst_dtype)
            T.copy(A, src)
            getattr(T, f"reduce_{op}")(src, dst, dim=0)
            T.copy(dst, B)

    return kernel


def test_a_vector_is_one_row():
    source = _source(_make_vector("sum", 300, "float32", "float32"))
    assert (
        "tl_hvx_reduce_rows<TL_REDUCE_SUM, float, false>((&(((float*)dst)[0])), (&(((float*)src)[0])), 1, 300, 300, 1, (bool)1);" in source
    )


@DEVICE
@pytest.mark.parametrize("op,src_dtype", [("sum", "float32"), ("max", "float16")])
def test_vector_reductions_match_torch(op, src_dtype):
    import torch

    kernel = tilelang.compile(_make_vector(op, 300, src_dtype, "float32"), out_idx=[1], target="hexagon")
    try:
        A = torch.randn(300, generator=torch.Generator().manual_seed(5)).to(getattr(torch, src_dtype))
        out = kernel(A)
        expected = A.double().sum() if op == "sum" else A.double().max()
        torch.testing.assert_close(out.double(), expected.reshape(1), rtol=1e-5, atol=1e-4)
    finally:
        kernel.adapter.close()


@DEVICE
@pytest.mark.parametrize("dim", [0, 1, 2])
def test_three_dimensional_reductions_match_torch(dim):
    _check("sum", (4, 24, 96), dim, "float32", "float32")
    _check("max", (4, 24, 96), dim, "float32", "float32", scope="fragment")
