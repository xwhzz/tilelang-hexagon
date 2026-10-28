"""Integer T.gemm on HVX: vrmpy over a B in the panel layout, exact in int32;
and T.gemm_blockscaled of 8- and 4-bit B: on HVX by integer A, on HMX by fp16
A."""

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


def _blockscaled(M, N, K, G, a_dtype, b_dtype, clear, GA=None, split=1):
    """C = C0 + (A * SA) @ (B * SB)^T: an fp32 factor per GA columns of each row
    of A (all K by default), an fp16 one per G columns of each row of B
    (group-major), in `split` calls over K."""
    GA = GA or K
    KS = K // split

    @T.prim_func
    def gemm(
        A: T.Tensor((M, K), a_dtype),
        SA: T.Tensor((M, K // GA), "float32"),
        B: T.Tensor((N, K), b_dtype),
        SB: T.Tensor((K // G, N), "float16"),
        C0: T.Tensor((M, N), "float32"),
        C: T.Tensor((M, N), "float32"),
    ):
        with T.Kernel(1):
            b = T.alloc_shared((N, K), b_dtype)
            sb = T.alloc_shared((K // G, N), "float16")
            c = T.alloc_fragment((M, N), "float32")
            T.annotate_layout({B: make_hvx_panel_layout(B)})
            T.copy(B, b)
            T.copy(SB, sb)
            T.copy(C0, c)
            for s in list(range(split)):
                cols = slice(s * KS, (s + 1) * KS)
                T.gemm_blockscaled(
                    A[:, cols],
                    b[:, cols],
                    c,
                    SA,
                    sb,
                    transpose_B=True,
                    clear_accum=clear and s == 0,
                    k_start=s * KS,
                    sf_a_granularity_k=GA,
                    sf_b_granularity_k=G,
                )
            T.copy(c, C)

    return gemm


def test_blockscaled_gemm_is_one_fused_call():
    source = _lower(_blockscaled(1, 128, 2048, 64, "int16", "int4", True)).kernel_source
    # Groups of 64 for B, one factor per row of A; SFB rows of 128 factors.
    assert "tl_hvx_gemm_int_scaled<1, 128, 2048, 4, true, 2048, 128, 2048, 64, true>(" in source
    assert ", 32768, (&(SA[0])), 1, (&(((half*)sb)[0])), 128, (bool)1);" in source


def test_blockscaled_gemm_needs_fp32_and_fp16_factors():
    @T.prim_func
    def wrong(
        A: T.Tensor((1, 64), "int16"),
        SA: T.Tensor((1, 1), "float16"),
        B: T.Tensor((32, 64), "int4"),
        SB: T.Tensor((1, 32), "float16"),
        C: T.Tensor((1, 32), "float32"),
    ):
        with T.Kernel(1):
            b = T.alloc_shared((32, 64), "int4")
            T.annotate_layout({B: make_hvx_panel_layout(B)})
            T.copy(B, b)
            T.gemm_blockscaled(A, b, C, SA, SB, transpose_B=True, k_start=0, sf_a_granularity_k=64, sf_b_granularity_k=64)

    with pytest.raises(Exception, match="scales A by fp32 and B by fp16 factors"):
        _lower(wrong)


@pytest.mark.parametrize(
    "args,error",
    [
        ((1, 64, 512, 64, "int8", "int4", True), "groups of a signed B with the sums of an int16 A"),
        ((1, 64, 512, 32, "int16", "int4", True), "the smaller one of 64"),
        ((1, 64, 768, 512, "int16", "uint4", True), "nor split into whole ones"),
        ((1, 64, 1024, 64, "int16", "int4", True, 768, 2), "nor split into whole ones"),
    ],
)
def test_blockscaled_gemm_rejects_what_hvx_cannot_scale(args, error):
    with pytest.raises(Exception, match=error):
        _lower(_blockscaled(*args))


def test_blockscaled_gemm_splits_k_by_its_start():
    source = _lower(_blockscaled(1, 64, 1024, 64, "int16", "int4", True, 1024, 2)).kernel_source
    # The second half's B factors start at group 8; A's one factor weighs both.
    assert source.count("tl_hvx_gemm_int_scaled<1, 64, 512, 4, true, 1024, 64, 512, 64, true>(") == 2
    assert "(&(SA[0])), 1, (&(((half*)sb)[512])), 64, (bool)0);" in source


@DEVICE
@pytest.mark.parametrize("b_dtype", ["int4", "uint4", "int8"])
@pytest.mark.parametrize(
    "M,N,K,G,GA,split",
    [
        (1, 128, 2048, 64, None, 1),
        (2, 96, 512, 128, None, 1),
        (1, 64, 8192, 64, None, 1),
        # A's factors per group, finer than B's and coarser, over two calls.
        (2, 64, 1024, 256, 64, 1),
        (1, 64, 1024, 64, 256, 2),
        # A's one factor per row across both calls; one panel of B.
        (2, 32, 1024, 128, None, 2),
    ],
)
def test_blockscaled_gemms_match_torch(b_dtype, M, N, K, G, GA, split):
    torch.manual_seed(N + K + G)
    GA = GA or K
    kernel = tilelang.compile(_blockscaled(M, N, K, G, "int16", b_dtype, False, GA, split), out_idx=[5], target="hexagon")
    try:
        a, b = _random((M, K), "int16"), _random((N, K), b_dtype)
        sa = torch.rand(M, K // GA) * 1e-3
        sb = (torch.rand(K // G, N) * 1e-2).half()
        c0 = torch.randn(M, N)
        c = kernel(a.to(torch.int16), sa, pack_panel(b, BITS[b_dtype]), sb, c0)
        g = min(G, GA)
        dots = (a.double().view(M, 1, K // g, g) * b.double().view(1, N, K // g, g)).sum(-1)  # [M, N, groups]
        fa = sa.double().repeat_interleave(GA // g, dim=1)[:, None]  # [M, 1, groups]
        fb = sb.double().t().repeat_interleave(G // g, dim=1)[None]  # [1, N, groups]
        ref = c0.double() + (dots * fa * fb).sum(-1)
        torch.testing.assert_close(c.double(), ref, rtol=1e-5, atol=1e-5 * ref.abs().max().item())
    finally:
        kernel.adapter.close()


def _blockscaled_hmx(M, N, K, G, b_dtype, c_dtype, clear, workers=1):
    """C = C0 + SA * (A @ (B * SB)^T) on HMX, for an fp16 A, a factor per row
    of A and an fp16 one per G columns of each row of B (group-major).
    `workers` programs each multiply their own A, B and C."""

    @T.prim_func
    def gemm(
        A: T.Tensor((workers, M, K), "float16"),
        SA: T.Tensor((workers, M, 1), "float32"),
        B: T.Tensor((workers, N, K), b_dtype),
        SB: T.Tensor((workers, K // G, N), "float16"),
        C0: T.Tensor((workers, M, N), c_dtype),
        C: T.Tensor((workers, M, N), c_dtype),
    ):
        with T.Kernel(workers, num_workers=workers) as p:
            a = T.alloc_shared((M, K), "float16")
            b = T.alloc_shared((N, K), b_dtype)
            sb = T.alloc_shared((K // G, N), "float16")
            c = T.alloc_shared((M, N), c_dtype)
            T.annotate_layout({B: make_hvx_panel_layout(B)})
            T.copy(A[p, :, :], a)
            T.copy(B[p, :, :], b)
            T.copy(SB[p, :, :], sb)
            T.copy(C0[p, :, :], c)
            T.gemm_blockscaled(
                a, b, c, SA[p, :, :], sb, transpose_B=True, clear_accum=clear, k_start=0, sf_a_granularity_k=K, sf_b_granularity_k=G
            )
            T.copy(c, C[p, :, :])

    return gemm


def test_blockscaled_gemm_of_fp16_a_runs_on_hmx():
    source = _lower(_blockscaled_hmx(64, 64, 256, 64, "int4", "float32", True)).kernel_source
    # One call of the out-of-line HMX GEMM: A's row tiles 8 tiles of K apart,
    # B's panels 4096 bytes apart, groups of 64, a factor per row of A.
    assert (
        "tl_hexagon_hmx_gemm_blockscaled<true, true, true, true, false>(64, 64, 256, (&(((float*)c)[0])), 64, "
        "(&(((half*)a)[0])), 8192, ((int8_t*)b + ((0) >> 1)), 4096, (&(((half*)sb)[0])), 64, 64, (&(SA[0])), 1, (bool)1, "
    ) in source
    assert "tl_hvx_gemm_int_scaled" not in source


def test_blockscaled_hmx_gemm_of_a_worker_pool_takes_turns_at_hmx():
    source = _lower(_blockscaled_hmx(64, 64, 256, 64, "uint4", "float32", True, workers=2)).kernel_source
    body = source[source.index("tl_parallel_for<tl_hmx_worker_scope") :]
    # The workers share HMX: a tile each at a time.
    assert "tl_hexagon_hmx_gemm_blockscaled<false, true, true, true, true>(" in body


@pytest.mark.parametrize(
    "args,error",
    [
        ((64, 64, 256, 64, "int8", "float32", True), "an fp16 A by a 4-bit B"),
        ((64, 64, 256, 4, "int4", "float32", True), "groups of 8 columns of K"),
        ((48, 64, 256, 64, "int4", "float32", True), "multiples of 32"),
    ],
)
def test_blockscaled_hmx_gemm_rejects_what_it_cannot_run(args, error):
    with pytest.raises(Exception, match=error):
        _lower(_blockscaled_hmx(*args))


@DEVICE
@pytest.mark.parametrize(
    "M,N,K,G,b_dtype,c_dtype,clear,workers",
    [
        (64, 64, 256, 64, "int4", "float32", True, 1),
        (256, 128, 2048, 64, "int4", "float32", False, 1),
        (32, 96, 512, 128, "uint4", "float32", True, 1),
        (64, 64, 1024, 64, "int4", "float16", False, 1),
        (96, 64, 512, 64, "int4", "float32", False, 2),
    ],
)
def test_blockscaled_hmx_gemms_match_torch(M, N, K, G, b_dtype, c_dtype, clear, workers):
    torch.manual_seed(M + N + K)
    kernel = tilelang.compile(_blockscaled_hmx(M, N, K, G, b_dtype, c_dtype, clear, workers), out_idx=[5], target="hexagon")
    try:
        a = (torch.randn(workers, M, K) * 0.5).half()
        sa = torch.rand(workers, M, 1) + 0.5
        b = torch.stack([_random((N, K), b_dtype) for _ in range(workers)])
        sb = (torch.rand(workers, K // G, N) * 2e-2 + 1e-3).half()
        c0 = torch.randn(workers, M, N).to(getattr(torch, c_dtype))
        c = kernel(a, sa, torch.stack([pack_panel(b[i], 4) for i in range(workers)]), sb, c0)
        # B dequantizes to fp16 exactly as torch rounds the products.
        w = (b.half() * sb.repeat_interleave(G, dim=1).transpose(1, 2)).double()  # [workers, N, K]
        ref = sa.double() * (a.double() @ w.transpose(1, 2)) + (0 if clear else c0.double())
        torch.testing.assert_close(c.double(), ref, rtol=4e-3, atol=4e-3 * ref.abs().max().item())
    finally:
        kernel.adapter.close()
