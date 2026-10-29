"""Crouton layouts and T.copy between Crouton and row-major buffers on Hexagon."""

import os
import re

import pytest

import tilelang
from tilelang import tvm
from tilelang.backend.target import determine_target
from tilelang.hexagon import language as T
from tilelang.hexagon.intrinsics.crouton import (
    CroutonLayout,
    make_hmx_activation_layout,
    make_hmx_output_layout,
    make_hmx_weight_layout,
)

DEVICE = pytest.mark.skipif(os.environ.get("TL_HEXAGON_DEVICE_TEST") != "1", reason="requires an authorized Hexagon device and SDK")

KINDS = {
    "activation": lambda b: make_hmx_activation_layout(b),
    "weight": lambda b: make_hmx_weight_layout(b),
    "activation_t": lambda b: make_hmx_activation_layout(b, transposed=True),
    "weight_t": lambda b: make_hmx_weight_layout(b, transposed=True),
}
R, C = 64, 96
REGIONS = {"full": (0, R, 0, C), "partial": (3, 61, 5, 91)}


def _source(func):
    target = determine_target("hexagon", return_object=True)
    with tvm.target.Target(target), tvm.transform.PassContext(opt_level=3):
        return str(tilelang.lower(func, target=target, enable_host_codegen=False, enable_device_compile=False).kernel_source)


def _round_trip(kind, dtype, endpoint, region):
    """X[region] -> zeroed Crouton tiles -> Y, the whole tile matrix."""
    r0, r1, c0, c1 = REGIONS[region]

    @T.prim_func
    def kernel(X: T.Tensor((R, C), dtype), Y: T.Tensor((R, C), dtype)):
        with T.Kernel(1):
            tiles = T.alloc_shared((R, C), "float16")
            T.annotate_layout({tiles: KINDS[kind](tiles)})
            T.fill(tiles, 0)
            if endpoint == "vtcm":
                x = T.alloc_shared((R, C), dtype)
                y = T.alloc_shared((R, C), dtype)
                T.copy(X, x)
                T.copy(x[r0:r1, c0:c1], tiles[r0:r1, c0:c1])
                T.copy(tiles, y)
                T.copy(y, Y)
            else:
                T.copy(X[r0:r1, c0:c1], tiles[r0:r1, c0:c1])
                T.copy(tiles, Y)

    return kernel


def test_hmx_layouts_are_crouton_layout_objects():
    activation = make_hmx_activation_layout((64, 128))
    weight = make_hmx_weight_layout((128, 96), transposed=True)
    assert isinstance(activation, CroutonLayout)
    assert (bool(activation.column_tiles), bool(activation.transposed)) == (False, False)
    assert (bool(weight.column_tiles), bool(weight.transposed)) == (True, True)
    assert isinstance(make_hmx_output_layout((2, 32, 64)), CroutonLayout)


def test_a_region_copy_passes_its_origin_and_strides():
    source = _source(_round_trip("activation", "float16", "ddr", "partial"))
    # Rows [3, 61) and columns [5, 91) of a 2 x 3 tile matrix, from X[3, 5].
    assert "tl_crouton_copy<true, false>((&(tiles[0])), 2, 3, 3, 5, 58, 86, (&(X[293])), 96, 1);" in source
    assert "tl_crouton_copy<false, false>((&(tiles[0])), 2, 3, 0, 0, 64, 96, (&(Y[0])), 96, 1);" in source


def test_a_transposed_layout_swaps_the_matrix_axes():
    source = _source(_round_trip("weight_t", "float32", "ddr", "partial"))
    # Storage [64, 96] holds the matrix [96, 64]: its rows run down the
    # storage columns.
    assert "tl_crouton_copy<true, true>((&(tiles[0])), 3, 2, 5, 3, 86, 58, (&(X[293])), 1, 96);" in source


def test_batched_crouton_buffers_copy_one_matrix():
    @T.prim_func
    def kernel(X: T.Tensor((64, 64), "float32"), Y: T.Tensor((64, 64), "float32")):
        with T.Kernel(1):
            tiles = T.alloc_shared((2, 64, 64), "float16")
            T.annotate_layout({tiles: make_hmx_activation_layout(tiles)})
            T.copy(X, tiles[1, :, :])
            T.copy(tiles[1, :, :], Y)

    source = _source(kernel)
    assert "tl_crouton_copy<true, false>((&(tiles[4096])), 2, 2, 0, 0, 64, 64, (&(X[0])), 64, 1);" in source
    assert "tl_crouton_copy<false, false>((&(tiles[4096])), 2, 2, 0, 0, 64, 64, (&(Y[0])), 64, 1);" in source


def _stack_round_trip(kind, dtype):
    """X[1:3, 3:61, 5:91] -> a stack of three zeroed Crouton matrices -> Y."""

    @T.prim_func
    def kernel(X: T.Tensor((4, R, C), dtype), Y: T.Tensor((3, R, C), dtype)):
        with T.Kernel(1):
            tiles = T.alloc_shared((3, R, C), "float16")
            T.annotate_layout({tiles: KINDS[kind](tiles)})
            T.fill(tiles, 0)
            T.copy(X[1:3, 3:61, 5:91], tiles[0:2, 3:61, 5:91])
            T.copy(tiles, Y)

    return kernel


def test_stacked_crouton_copies_take_one_call_per_matrix():
    source = _source(_stack_round_trip("activation", "float16"))
    lines = [line.strip() for line in source.splitlines()]
    # Each matrix of the region packs with one call, its stack index a loop:
    # tile matrices are R x C = 6144 elements apart, X's too, and X[1, 3, 5]
    # is 6437 elements in.
    (pack,) = [i for i, line in enumerate(lines) if "tl_crouton_copy<true, false>(" in line]
    assert lines[pack - 1] == "for (int32_t m0 = 0; m0 < 2; ++m0) {"
    assert lines[pack] == "tl_crouton_copy<true, false>((&(tiles[(m0 * 6144)])), 2, 3, 3, 5, 58, 86, (&(X[((m0 * 6144) + 6437)])), 96, 1);"
    (unpack,) = [i for i, line in enumerate(lines) if "tl_crouton_copy<false, false>(" in line]
    assert lines[unpack - 1] == "for (int32_t m0_1 = 0; m0_1 < 3; ++m0_1) {"
    assert "for (int32_t i" not in "\n".join(lines[pack:])


def test_crouton_to_crouton_copies_go_element_by_element():
    @T.prim_func
    def kernel(X: T.Tensor((64, 64), "float16"), Y: T.Tensor((64, 64), "float16")):
        with T.Kernel(1):
            a = T.alloc_shared((64, 64), "float16")
            w = T.alloc_shared((64, 64), "float16")
            T.annotate_layout({a: make_hmx_activation_layout(a), w: make_hmx_weight_layout(w)})
            T.copy(X, a)
            T.copy(a, w)
            T.copy(w, Y)

    source = _source(kernel)
    assert source.count("tl_crouton_copy<") == 2


@DEVICE
@pytest.mark.parametrize("dtype", ["float16", "float32"])
@pytest.mark.parametrize("kind", sorted(KINDS))
def test_stacked_round_trips_match_torch(kind, dtype):
    import torch

    kernel = tilelang.compile(_stack_round_trip(kind, dtype), out_idx=[1], target="hexagon")
    try:
        x = torch.randn(4, R, C, generator=torch.Generator().manual_seed(23)).to(getattr(torch, dtype))
        expected = torch.zeros(3, R, C, dtype=x.dtype)
        expected[0:2, 3:61, 5:91] = x[1:3, 3:61, 5:91].half().to(x.dtype)
        torch.testing.assert_close(kernel(x), expected, rtol=0, atol=0)
    finally:
        kernel.adapter.close()


@DEVICE
@pytest.mark.parametrize("region", sorted(REGIONS))
@pytest.mark.parametrize("endpoint", ["ddr", "vtcm"])
@pytest.mark.parametrize("dtype", ["float16", "float32"])
@pytest.mark.parametrize("kind", sorted(KINDS))
def test_region_round_trips_match_torch(kind, dtype, endpoint, region):
    import torch

    kernel = tilelang.compile(_round_trip(kind, dtype, endpoint, region), out_idx=[1], target="hexagon")
    try:
        x = torch.randn(R, C, generator=torch.Generator().manual_seed(17)).to(getattr(torch, dtype))
        y = kernel(x)
        r0, r1, c0, c1 = REGIONS[region]
        expected = torch.zeros(R, C, dtype=x.dtype)
        expected[r0:r1, c0:c1] = x[r0:r1, c0:c1].half().to(x.dtype)
        torch.testing.assert_close(y, expected, rtol=0, atol=0)
    finally:
        kernel.adapter.close()


def _gemm(transpose_a, transpose_b, a_dtype):
    """C = A @ B through Crouton copies and HMX, A from fp16 or fp32."""
    a_shape = (128, 64) if transpose_a else (64, 128)
    b_shape = (96, 128) if transpose_b else (128, 96)

    @T.prim_func
    def kernel(A: T.Tensor(a_shape, a_dtype), B: T.Tensor(b_shape, "float16"), C: T.Tensor((64, 96), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared(a_shape, "float16")
            b = T.alloc_shared(b_shape, "float16")
            c = T.alloc_shared((64, 96), "float16")
            T.copy(A, a)
            T.copy(B, b)
            T.gemm(a, b, c, transpose_A=transpose_a, transpose_B=transpose_b, clear_accum=True)
            T.copy(c, C)

    return kernel


@DEVICE
@pytest.mark.parametrize("a_dtype", ["float16", "float32"])
@pytest.mark.parametrize("transpose_b", [False, True])
@pytest.mark.parametrize("transpose_a", [False, True])
def test_hmx_gemm_operands_match_torch(transpose_a, transpose_b, a_dtype):
    import torch

    kernel = tilelang.compile(_gemm(transpose_a, transpose_b, a_dtype), out_idx=[2], target="hexagon")
    try:
        gen = torch.Generator().manual_seed(23)
        a = (torch.randn(64, 128, generator=gen) * 0.5).half()
        b = (torch.randn(128, 96, generator=gen) * 0.5).half()
        a_in = (a.T.contiguous() if transpose_a else a).to(getattr(torch, a_dtype))
        b_in = b.T.contiguous() if transpose_b else b
        c = kernel(a_in, b_in)
        torch.testing.assert_close(c, (a.float() @ b.float()).half().float(), rtol=2e-3, atol=2e-3)
    finally:
        kernel.adapter.close()


# Maps over Crouton buffers run in the tiles' physical order, a register (a
# row pair of a tile) at a time. Row operands then read [r0, r1, r0, r1, ...]
# and column operands [c0, c0, c1, c1, ...].


def _map(kind, dtype):
    """Y = X * u[i] + v[j] on Crouton tiles of layout `kind`, computed in `dtype`."""

    @T.prim_func
    def kernel(
        X: T.Tensor((R, C), "float16"),
        U: T.Tensor((R,), dtype),
        V: T.Tensor((C,), dtype),
        Y: T.Tensor((R, C), "float16"),
    ):
        with T.Kernel(1):
            tiles = T.alloc_shared((R, C), "float16")
            u = T.alloc_shared((R,), dtype)
            v = T.alloc_shared((C,), dtype)
            T.annotate_layout({tiles: KINDS[kind](tiles)})
            T.copy(X, tiles)
            T.copy(U, u)
            T.copy(V, v)
            for i, j in T.Parallel(R, C):
                tiles[i, j] = T.cast(T.cast(tiles[i, j], dtype) * u[i] + v[j], "float16")
            T.copy(tiles, Y)

    return kernel


# A scalar store to the tiles, which a vectorized map has none of.
SCALAR_TILE_STORE = re.compile(r"\(\(half\*\)tiles\)\[[^\]]*\] =")


def test_maps_over_crouton_tiles_use_whole_registers():
    source = _source(_map("activation", "float16"))
    # One register per row pair, in loops over tiles and their row pairs.
    assert re.search(r"\*\(half64\*\)\(\(\(half\*\)tiles\) \+ \(\(i\w* \* 1024\) \+ \(j\w* \* 64\)\)\) = ", source)
    # The rows' values: a tile's 32 rows in one register, two of them per row
    # pair. The columns' values: 32 elements, each repeated twice.
    assert "half64 u_lanes = tl_hvx_load_lanes<64>((&(((half*)u)[" in source
    assert "tl_hvx_broadcast_lane<64, 2>(u_lanes, j_lane)" in source
    assert "tl_hvx_load_repeat<2, 64, true>((&(((half*)v)[" in source
    assert not SCALAR_TILE_STORE.search(source)


def test_fp32_maps_over_crouton_tiles_widen_the_windows():
    source = _source(_map("weight", "float32"))
    assert "float32 u_lanes = tl_hvx_load_lanes<32>((&(((float*)u)[" in source
    assert "tl_hvx_broadcast_lane<32, 2>(u_lanes, j_lane)" in source
    assert "tl_hvx_load_repeat<2, 32, true>((&(((float*)v)[" in source
    assert not SCALAR_TILE_STORE.search(source)


@DEVICE
@pytest.mark.parametrize("dtype", ["float16", "float32"])
@pytest.mark.parametrize("kind", sorted(KINDS))
def test_maps_over_crouton_tiles_match_torch(kind, dtype):
    import torch

    kernel = tilelang.compile(_map(kind, dtype), out_idx=[3], target="hexagon")
    try:
        gen = torch.Generator().manual_seed(29)
        x = torch.randn(R, C, generator=gen).half()
        u = torch.randn(R, generator=gen).to(getattr(torch, dtype))
        v = torch.randn(C, generator=gen).to(getattr(torch, dtype))
        y = kernel(x, u, v)
        expected = (x.to(u.dtype) * u[:, None] + v[None, :]).half()
        torch.testing.assert_close(y, expected, rtol=1e-2, atol=1e-2)
    finally:
        kernel.adapter.close()


def _masked(kind):
    """Y = X where j <= i + d, else 0, through Crouton tiles of layout `kind`:
    a map that computes with its indices, which a register's lanes take from
    their position in it (a row pair's lanes alternate between its rows)."""

    @T.prim_func
    def kernel(X: T.Tensor((R, C), "float16"), Y: T.Tensor((R, C), "float16"), d: T.int32):
        with T.Kernel(1):
            tiles = T.alloc_shared((R, C), "float16")
            T.annotate_layout({tiles: KINDS[kind](tiles)})
            T.copy(X, tiles)
            for i, j in T.Parallel(R, C):
                tiles[i, j] = T.if_then_else(j <= i + d, tiles[i, j], T.float16(0))
            T.copy(tiles, Y)

    return kernel


def test_maps_over_crouton_tiles_compute_with_their_indices_in_registers():
    source = _source(_masked("activation"))
    # A lane's row and column: its index's parity and half.
    assert "tl_vec_ramp<int32_t, 64>(0, 1) % " in source
    assert not SCALAR_TILE_STORE.search(source)


@DEVICE
@pytest.mark.parametrize("kind", sorted(KINDS))
def test_maps_over_crouton_tiles_compute_with_their_indices(kind):
    import torch

    kernel = tilelang.compile(_masked(kind), out_idx=[1], target="hexagon")
    try:
        x = torch.randn(R, C, generator=torch.Generator().manual_seed(37)).half()
        i, j = torch.meshgrid(torch.arange(R), torch.arange(C), indexing="ij")
        for d in (0, 5, -7):
            torch.testing.assert_close(kernel(x, d), torch.where(j <= i + d, x, torch.zeros_like(x)), rtol=0, atol=0)
    finally:
        kernel.adapter.close()


def _reduce(kind, op, dim, dtype, shape=(R, C)):
    """Y = op of X along `dim`, from Crouton tiles of layout `kind`."""
    rows, cols = shape
    n = rows if dim == 1 else cols

    @T.prim_func
    def kernel(X: T.Tensor((rows, cols), "float16"), Y: T.Tensor((n,), dtype)):
        with T.Kernel(1):
            tiles = T.alloc_shared((rows, cols), "float16")
            out = T.alloc_shared((n,), dtype)
            T.annotate_layout({tiles: KINDS[kind](tiles)})
            T.copy(X, tiles)
            getattr(T, "reduce_" + op)(tiles, out, dim=dim)
            T.copy(out, Y)

    return kernel


def test_reductions_of_crouton_tiles_are_one_call():
    source = _source(_reduce("activation", "max", 1, "float16"))
    assert "tl_crouton_reduce<TL_REDUCE_MAX, half, true, false>((&(((half*)out)[0])), (&(((half*)tiles)[0])), 2, 3, 1, (bool)1);" in source
    # Columns of a weight layout, in fp32.
    source = _source(_reduce("weight", "sum", 0, "float32"))
    assert "tl_crouton_reduce<TL_REDUCE_SUM, float, false, true>(" in source
    # The rows of a transposed layout's buffer are the matrix's columns.
    source = _source(_reduce("activation_t", "max", 1, "float16"))
    assert "tl_crouton_reduce<TL_REDUCE_MAX, half, false, false>(" in source


@DEVICE
@pytest.mark.parametrize("shape", [(R, C), (64, 64)])
@pytest.mark.parametrize("dim", [0, 1])
@pytest.mark.parametrize("op,dtype", [("max", "float16"), ("sum", "float32"), ("absmax", "float16"), ("min", "float32")])
@pytest.mark.parametrize("kind", sorted(KINDS))
def test_reductions_of_crouton_tiles_match_torch(kind, op, dtype, dim, shape):
    import torch

    kernel = tilelang.compile(_reduce(kind, op, dim, dtype, shape), out_idx=[1], target="hexagon")
    try:
        x = torch.randn(*shape, generator=torch.Generator().manual_seed(31)).half()
        y = kernel(x)
        xf = x.float()
        expected = {
            "max": lambda: xf.amax(dim),
            "min": lambda: xf.amin(dim),
            "sum": lambda: xf.sum(dim),
            "absmax": lambda: xf.abs().amax(dim),
        }[op]().to(getattr(torch, dtype))
        torch.testing.assert_close(y, expected, rtol=1e-3, atol=1e-2)
    finally:
        kernel.adapter.close()
