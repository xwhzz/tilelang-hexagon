"""Integer T.gemm on HVX: vrmpy over a B in the panel layout, exact in int32."""

import os

import pytest
import torch

import tilelang
from tilelang import tvm
from tilelang.backend.target import determine_target
from tilelang.hexagon import language as T
from tilelang.hexagon.intrinsics.panel import make_hvx_panel_layout, pack_panel, unpack_panel

DEVICE = pytest.mark.skipif(os.environ.get("TL_HEXAGON_DEVICE_TEST") != "1", reason="requires an authorized Hexagon device and SDK")

BITS = {"int8": 8, "uint8": 8, "int16": 16, "int4": 4, "uint4": 4}


def _lower(func):
    target = determine_target("hexagon", return_object=True)
    with tvm.target.Target(target), tvm.transform.PassContext(opt_level=3):
        return tilelang.lower(func, target=target, enable_host_codegen=False, enable_device_compile=False)


def _gemm(M, N, K, a_dtype, b_dtype, clear):
    """C = C0 + A B^T (or A B^T with clear), B packed in the panel layout."""

    @T.prim_func
    def gemm(
        A: T.Tensor((M, K), a_dtype),
        B: T.Tensor((N, K), b_dtype),
        C0: T.Tensor((M, N), "int32"),
        C: T.Tensor((M, N), "int32"),
    ):
        with T.Kernel(1):
            b = T.alloc_shared((N, K), b_dtype)
            c = T.alloc_fragment((M, N), "int32")
            T.annotate_layout({B: make_hvx_panel_layout(B), b: make_hvx_panel_layout(b)})
            T.copy(B, b)
            T.copy(C0, c)
            T.gemm(A, b, c, transpose_B=True, clear_accum=clear)
            T.copy(c, C)

    return gemm


def _random(shape, dtype):
    """Random integers of `dtype`, with its extremes among them."""
    bits = BITS[dtype]
    lo, hi = (-(1 << (bits - 1)), (1 << (bits - 1)) - 1) if dtype.startswith("int") else (0, (1 << bits) - 1)
    v = torch.randint(lo, hi + 1, shape, dtype=torch.int64)
    v.view(-1)[0], v.view(-1)[-1] = lo, hi
    return v


def test_panel_layout_places_each_element():
    rows, cols, lane = 64, 24, 4
    layout = make_hvx_panel_layout((rows, cols), "int8")
    for r, c in ((0, 0), (1, 0), (0, 1), (31, 3), (32, 4), (37, 23), (63, 23)):
        expected = [r // 32, c // lane, r % 32, c % lane]
        assert [int(x) for x in layout.map_forward_index([r, c])] == expected


@pytest.mark.parametrize("bits,signed", [(8, True), (8, False), (4, True), (4, False)])
def test_pack_panel_round_trips(bits, signed):
    dtype = ("int" if signed else "uint") + str(bits)
    values = _random((2, 64, 32), dtype)
    packed = pack_panel(values, bits)
    assert packed.dtype == torch.uint8 and packed.numel() == values.numel() * bits // 8
    torch.testing.assert_close(unpack_panel(packed, values.shape, bits, signed), values)


def test_pack_panel_puts_a_lane_in_a_word():
    # Row 1, columns 0..7 of a 4-bit panel: one 32-bit lane, the first column
    # in the low nibble of its first byte.
    values = torch.zeros(32, 8, dtype=torch.int64)
    values[1] = torch.arange(1, 9)
    packed = pack_panel(values, 4)
    assert packed[4:8].tolist() == [0x21, 0x43, 0x65, 0x87]


def test_panel_copy_moves_whole_panels_by_dma():
    source = _lower(_gemm(1, 128, 256, "int16", "uint4", True)).kernel_source
    # 128 x 256 4-bit elements, contiguous on both sides: one 16 KB transfer.
    assert source.count("tl_hexagon_dma_async_submit(") == 1
    assert "(uint32_t)16384, (uint32_t)16384, (uint32_t)16384, (uint32_t)1" in source


def test_integer_gemm_is_one_vrmpy_call():
    source = _lower(_gemm(2, 96, 136, "int8", "int4", False)).kernel_source
    # 4-bit signed B, A rows 136 apart, C rows 96 apart; panels of 32 rows
    # by 136 4-bit columns are 2176 bytes apart.
    assert "tl_hvx_gemm_int<2, 96, 136, 4, true, 136, 96, true>(" in source
    assert ", 2176, (bool)0);" in source


def test_integer_b_needs_whole_panels():
    with pytest.raises(Exception, match="multiples of 32"):
        _lower(_gemm(1, 48, 64, "int16", "uint4", True))


def test_integer_b_region_starts_on_a_panel():
    @T.prim_func
    def offset(A: T.Tensor((1, 64), "int16"), B: T.Tensor((64, 64), "uint4"), C: T.Tensor((1, 32), "int32")):
        with T.Kernel(1):
            b = T.alloc_shared((64, 64), "uint4")
            c = T.alloc_fragment((1, 32), "int32")
            T.annotate_layout({B: make_hvx_panel_layout(B)})
            T.copy(B, b)
            # Rows [16, 48) straddle two panels; HVX does not run it.
            T.gemm(A, b[16:48, :], c, transpose_B=True, clear_accum=True)
            T.copy(c, C)

    with pytest.raises(Exception, match="start and end on multiples of 32"):
        _lower(offset)


CASES = [
    ("int16", "uint4"),
    ("int16", "int4"),
    ("int8", "int4"),
    ("uint8", "uint4"),
    ("int16", "int8"),
    ("int8", "int8"),
    ("uint8", "uint8"),
]


@DEVICE
@pytest.mark.parametrize("a_dtype,b_dtype", CASES)
@pytest.mark.parametrize("M,N,K", [(1, 160, 1040), (3, 96, 136)])
def test_integer_gemms_match_torch(a_dtype, b_dtype, M, N, K):
    torch.manual_seed(M * N + K)
    kernel = tilelang.compile(_gemm(M, N, K, a_dtype, b_dtype, True), out_idx=[3], target="hexagon")
    try:
        a, b = _random((M, K), a_dtype), _random((N, K), b_dtype)
        c = kernel(
            a.to(torch.int16 if a_dtype == "int16" else torch.int8 if a_dtype == "int8" else torch.uint8),
            pack_panel(b, BITS[b_dtype]),
            torch.zeros(M, N, dtype=torch.int32),
        )
        assert torch.equal(c.to(torch.int64), a @ b.t())
    finally:
        kernel.adapter.close()


@DEVICE
def test_integer_gemm_accumulates_into_c():
    M, N, K = 2, 64, 512
    kernel = tilelang.compile(_gemm(M, N, K, "int16", "uint4", False), out_idx=[3], target="hexagon")
    try:
        a, b = _random((M, K), "int16"), _random((N, K), "uint4")
        c0 = torch.randint(-(1 << 20), 1 << 20, (M, N), dtype=torch.int32)
        c = kernel(a.to(torch.int16), pack_panel(b, 4), c0)
        assert torch.equal(c.to(torch.int64), c0.to(torch.int64) + a @ b.t())
    finally:
        kernel.adapter.close()
