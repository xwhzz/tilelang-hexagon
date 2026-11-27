"""Integer T.gemm on HVX: vrmpy over a B in the panel layout, exact in int32;
and T.gemm_blockscaled of 8- and 4-bit B: on HVX by integer A, on HMX by fp16
A."""

import json
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


@pytest.mark.parametrize("representation", ["native", "json", "legacy_json"])
def test_panel_layout_places_each_element(representation):
    rows, cols, lane = 64, 24, 4
    layout = make_hvx_panel_layout((rows, cols), "int8")
    if representation != "native":
        graph = json.loads(tvm.ir.save_json(layout))
        if representation == "legacy_json":
            del graph["nodes"][graph["root_index"]]["data"]["placeholder_vars"]
        layout = tvm.ir.load_json(json.dumps(graph))
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


def _sfb_layout_gemm(hmx, storage):
    M, N, K = (32 if hmx else 1), 32, 128
    ad, bd = ("float16", "uint4") if hmx else ("int16", "uint8")
    shape = (2, 5, 64)
    raw = storage.startswith(("strided", "pitched"))
    strides = (320, 1, 5) if storage.startswith("strided") else (400, 80, 1)
    sb_type = T.StridedTensor(shape, strides, "float16") if raw else T.Tensor(shape, "float16")

    @T.prim_func
    def gemm(
        A: T.Tensor((M, K), ad), B: T.Tensor((N, K), bd), SA: T.Tensor((M, 1), "float32"), SB: sb_type, C: T.Tensor((M, N), "float32")
    ):
        with T.Kernel(1):
            a = T.alloc_shared((M, K), ad)
            b = T.alloc_shared((N, K), bd)
            c = T.alloc_shared((M, N), "float32")
            sb = T.alloc_shared(shape, "float16")
            T.annotate_layout({B: make_hvx_panel_layout(B)})
            if storage in ("strided_identity", "pitched_identity"):
                T.annotate_layout({SB: T.Layout(shape, lambda i, g, n: [i, g, n])})
            elif storage == "strided_conflict":
                T.annotate_layout({SB: T.Layout(shape, lambda i, g, n: [i, n, g])})
            elif storage == "transpose":
                T.annotate_layout({sb: T.Layout(shape, lambda i, g, n: [i, n, g])})
            elif storage == "batch_reverse":
                T.annotate_layout({sb: T.Layout(shape, lambda i, g, n: [1 - i, g, n])})
            elif storage == "identity":
                T.annotate_layout({sb: T.Layout(shape, lambda i, g, n: [i, g, n])})
            T.copy(A, a)
            T.copy(B, b)
            if not raw:
                T.copy(SB, sb)
            scales = SB if raw else sb
            T.gemm_blockscaled(
                a,
                b,
                c,
                SA,
                scales[1, 1:5, 16:48],
                transpose_B=True,
                clear_accum=True,
                k_start=64,
                sf_a_granularity_k=256,
                sf_b_granularity_k=64,
            )
            T.copy(c, C)

    return gemm


@pytest.mark.parametrize("hmx", [False, True])
def test_sfb_layout_prepares_only_non_group_major_scales(hmx):
    direct = _lower(_sfb_layout_gemm(hmx, "dense")).kernel_source
    identity = _lower(_sfb_layout_gemm(hmx, "identity")).kernel_source
    packed = _lower(_sfb_layout_gemm(hmx, "transpose")).kernel_source
    assert direct == identity
    assert "packed_sfb" not in direct
    batch = _lower(_sfb_layout_gemm(hmx, "batch_reverse")).kernel_source
    assert "packed_sfb" not in batch
    assert "[144])), 64," in batch
    assert "packed_sfb" in packed
    strided = _lower(_sfb_layout_gemm(hmx, "strided")).kernel_source
    assert "packed_sfb" in strided
    pitched = _lower(_sfb_layout_gemm(hmx, "pitched")).kernel_source
    assert "packed_sfb" not in pitched
    assert "[576])), 80," in pitched
    assert _lower(_sfb_layout_gemm(hmx, "pitched_identity")).kernel_source == pitched
    assert _lower(_sfb_layout_gemm(hmx, "strided_identity")).kernel_source == strided
    with pytest.raises(Exception, match="combines noncompact global strides with a non-row-major layout"):
        _lower(_sfb_layout_gemm(hmx, "strided_conflict"))


@DEVICE
@pytest.mark.parametrize("hmx", [False, True])
@pytest.mark.parametrize("storage", ["transpose", "strided", "batch_reverse", "strided_identity", "pitched_identity"])
def test_sfb_layout_reads_offset_and_first_group(hmx, storage):
    m = 32 if hmx else 1
    kernel = tilelang.compile(_sfb_layout_gemm(hmx, storage), out_idx=[4], target="hexagon")
    try:
        a = torch.ones((m, 128), dtype=torch.float16 if hmx else torch.int16)
        b = pack_panel(torch.ones((32, 128), dtype=torch.int64), 4 if hmx else 8)
        sa = torch.ones((m, 1))
        sb = torch.arange(2 * 5 * 64).reshape(2, 5, 64).remainder(29).half()
        sb_arg = sb
        if storage == "pitched_identity":
            backing = torch.full((784,), -1, dtype=torch.float16)
            backing.as_strided(sb.shape, (400, 80, 1)).copy_(sb)
            sb_arg = backing.numpy().tobytes()
        result = kernel(a, b, sa, sb_arg)
        expected = 64 * sb[1, 2:4, 16:48].float().sum(0)
        torch.testing.assert_close(result, expected.expand(m, 32), rtol=0, atol=0)
    finally:
        kernel.adapter.close()


def _strided_blockscaled(sa, sc, signed, clear, transpose_sfb=False):
    M, N, K = 3, 32, 128
    ad, bd = ("int16", "int4") if signed else ("uint8", "uint8")
    x_shape, y_shape = (M + 2, K + 8, sa), (M + 2, N + 4, sc)

    @T.prim_func
    def gemm(
        X: T.Tensor(x_shape, ad),
        W: T.Tensor((N, K), bd),
        SA: T.Tensor((M + 2, 4), "float32"),
        SB: T.Tensor((4, N), "float16"),
        Y0: T.Tensor(y_shape, "float32"),
        Y: T.Tensor(y_shape, "float32"),
    ):
        with T.Kernel(1):
            a = T.StridedTensor(x_shape[:2], (x_shape[1] * sa, sa), ad, data=X.data)
            c = T.StridedTensor(y_shape[:2], (y_shape[1] * sc, sc), "float32", data=Y.data)
            b = T.alloc_shared((N, K), bd)
            sb = T.alloc_shared((4, N), "float16")
            T.annotate_layout({W: make_hvx_panel_layout(W)})
            if transpose_sfb:
                T.annotate_layout({sb: T.Layout((4, N), lambda g, n: [n, g])})
            T.copy(W, b)
            T.copy(SB, sb)
            T.copy(Y0, Y)
            T.gemm_blockscaled(
                a[1 : M + 1, 4 : K + 4],
                b,
                c[1 : M + 1, 1 : N + 1],
                SA[1 : M + 1, 1:4],
                sb,
                transpose_B=True,
                clear_accum=clear,
                k_start=64,
                sf_a_granularity_k=64,
                sf_b_granularity_k=64,
            )

    return gemm


@pytest.mark.parametrize("sa,sc,clear", [(1, 1, True), (2, 1, True), (1, 2, True), (2, 2, False)])
def test_strided_blockscaled_stages_only_noncontiguous_rows(sa, sc, clear):
    source = _lower(_strided_blockscaled(sa, sc, False, clear)).kernel_source
    assert ("a_row" in source) == (sa != 1)
    assert ("c_row" in source) == (sc != 1)
    rows = 3 if sa == sc == 1 else 1
    assert f"tl_hvx_gemm_int_scaled<{rows}, 32, 128," in source
    before_call = source.split("tl_hvx_gemm_int_scaled<", 1)[0]
    assert (" = c[" in before_call) == (sc != 1 and not clear)


def test_blockscaled_strided_global_identity_keeps_physical_channels():
    @T.prim_func
    def gemm(
        X: T.StridedTensor((3, 132), (264, 2), "uint8"),
        W: T.Tensor((32, 128), "uint8"),
        SA: T.Tensor((2, 1), "float32"),
        SB: T.Tensor((1, 32), "float16"),
        Y: T.StridedTensor((3, 34), (68, 2), "float32"),
    ):
        with T.Kernel(1):
            b = T.alloc_shared((32, 128), "uint8")
            T.annotate_layout(
                {W: make_hvx_panel_layout(W), X: T.Layout((3, 132), lambda i, k: [i, k]), Y: T.Layout((3, 34), lambda i, j: [i, j])}
            )
            T.copy(W, b)
            T.gemm_blockscaled(
                X[1:3, 2:130],
                b,
                Y[1:3, 1:33],
                SA,
                SB,
                transpose_B=True,
                clear_accum=True,
                k_start=0,
                sf_a_granularity_k=128,
                sf_b_granularity_k=128,
            )

    source = _lower(gemm).kernel_source
    assert "a_row" in source and "c_row" in source
    assert "tl_hvx_gemm_int_scaled<1, 32, 128," in source


@DEVICE
@pytest.mark.parametrize("sa,sc,signed,clear", [(2, 1, False, True), (1, 2, False, True), (1, 2, True, False), (2, 2, True, False)])
def test_strided_blockscaled_respects_channels_and_accumulation(sa, sc, signed, clear):
    M, N, K = 3, 32, 128
    kernel = tilelang.compile(_strided_blockscaled(sa, sc, signed, clear, transpose_sfb=signed), out_idx=[5], target="hexagon")
    try:
        x = (torch.arange((M + 2) * (K + 8) * sa).reshape(M + 2, K + 8, sa) % 11).to(torch.int16 if signed else torch.uint8)
        w = (torch.arange(N * K).reshape(N, K) % 8) - (4 if signed else 0)
        sfa = (1 + torch.arange((M + 2) * 4).reshape(M + 2, 4) % 5).float()
        sfb = (1 + torch.arange(4 * N).reshape(4, N) % 7).half()
        initial = torch.arange((M + 2) * (N + 4) * sc).reshape(M + 2, N + 4, sc).float() - 128
        y = kernel(x, pack_panel(w, 4 if signed else 8), sfa, sfb, initial)
        expected = initial.clone()
        target = expected[1 : M + 1, 1 : N + 1, 0]
        if clear:
            target.zero_()
        for g in range(2):
            k = slice(g * 64, (g + 1) * 64)
            sums = x[1 : M + 1, 4 : K + 4, 0][:, k].float() @ w[:, k].float().T
            target += sums * sfa[1 : M + 1, 2 + g, None] * sfb[1 + g].float()
        torch.testing.assert_close(y, expected, rtol=0, atol=0)
    finally:
        kernel.adapter.close()


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
        # Four-panel blocks with one or three panels left in the tail.
        (1, 160, 2048, 64, None, 1),
        (2, 224, 1024, 128, 64, 2),
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


@DEVICE
@pytest.mark.parametrize(
    "dtype,ga,gb,clear,n",
    [
        ("uint8", 512, 512, True, 64),
        ("uint8", 1280, 640, False, 160),
        ("int8", 576, 1152, True, 224),
        ("int8", 512, 512, False, 64),
        ("int4", 8192, 8192, True, 64),
        ("uint4", 4480, 4480, False, 64),
        ("int8", 8256, 8256, True, 64),
    ],
)
def test_blockscaled_wide_groups_preserve_integer_dots(dtype, ga, gb, clear, n):
    m, k, group = 5, max(ga, gb), min(ga, gb)
    a = torch.empty((m, k), dtype=torch.int16)
    a[0], a[1], a[2] = 20000, -32768, 32767
    w = torch.empty((n, k), dtype=torch.int64)
    bits = BITS[dtype]
    signed = dtype.startswith("int")
    low = -(1 << (bits - 1)) if signed else 0
    high = -low - 1 if signed else (1 << bits) - 1
    extreme = low if signed else high
    w[:16] = 220 if dtype == "uint8" else extreme
    w[16:32] = high
    w[32:] = extreme
    # Distinguish successive high/low activation bytes and packed weight nibbles.
    cols = torch.arange(k)
    a[4] = (cols * 811 + cols // 11) % 65536 - 32768
    w[0] = ((cols * 7 + cols // 5) % (1 << bits)) + low
    for start in range(0, k, group):
        mid = start + group // 2
        a[3, start:mid], a[3, mid : start + group] = 32767, -32767
        a[3, mid] += 1
        w[32:, start], w[32:, mid] = 1, 1
    sa = torch.ones(m, k // ga)
    sb = torch.full((k // gb, n), 1 / 256, dtype=torch.float16)
    sb[:, 32:] = 1
    sa[:, 1:] *= 2
    sb[1:] *= 2
    c0 = (torch.arange(m * n) % 7 - 3).reshape(m, n).float()
    expected = torch.zeros_like(c0)
    for start in range(0, k, group):
        cols = slice(start, start + group)
        dots = a[:, cols].long() @ w[:, cols].T
        # The last row's large opposite partials leave exactly one.
        assert (dots[3, 32:] == 1).all()
        expected += dots.float() * sa[:, start // ga, None] * sb[start // gb].float()
    if not clear:
        expected += c0
    kernel = tilelang.compile(_blockscaled(m, n, k, gb, "int16", dtype, clear, ga), out_idx=[5], target="hexagon")
    try:
        torch.testing.assert_close(kernel(a, sa, pack_panel(w, bits), sb, c0), expected, rtol=0, atol=0)
    finally:
        kernel.adapter.close()


@DEVICE
def test_blockscaled_wide_subnormal_group_scale():
    k, n, ga = 1024, 32, 512
    a = torch.full((1, k), 20000, dtype=torch.int16)
    b = torch.full((n, k), 220, dtype=torch.uint8)
    sa = torch.tensor([[1e-35, 0]], dtype=torch.float32)
    sb = torch.full((1, n), 2**-24, dtype=torch.float16)
    c0 = torch.zeros(1, n)
    kernel = tilelang.compile(_blockscaled(1, n, k, k, "int16", "uint8", True, ga), out_idx=[5], target="hexagon")
    try:
        expected = (sa[0, 0] * sb.float()) * float(ga * 20000 * 220)
        torch.testing.assert_close(kernel(a, sa, pack_panel(b, 8), sb, c0), expected, rtol=0, atol=0)
    finally:
        kernel.adapter.close()


@DEVICE
def test_blockscaled_wide_byte_groups():
    k, n = 33152, 32
    a = torch.full((1, k), 255, dtype=torch.uint8)
    b = torch.full((n, k), 255, dtype=torch.uint8)
    b[::2, -1] = 1
    sa = torch.ones(1, 1)
    sb = torch.ones(1, n, dtype=torch.float16)
    c0 = torch.zeros(1, n)
    kernel = tilelang.compile(_blockscaled(1, n, k, k, "uint8", "uint8", True, k), out_idx=[5], target="hexagon")
    try:
        expected = (a.long() @ b.long().T).float()
        torch.testing.assert_close(kernel(a, sa, pack_panel(b, 8), sb, c0), expected, rtol=0, atol=0)
    finally:
        kernel.adapter.close()


def _blockscaled_hmx(M, N, K, G, b_dtype, c_dtype, clear, workers=1, normal_scales=False):
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
                a,
                b,
                c,
                SA[p, :, :],
                sb,
                transpose_B=True,
                clear_accum=clear,
                k_start=0,
                sf_a_granularity_k=K,
                sf_b_granularity_k=G,
                annotations={"hexagon.normal_scales": normal_scales},
            )
            T.copy(c, C[p, :, :])

    return gemm


def test_blockscaled_gemm_of_fp16_a_runs_on_hmx():
    source = _lower(_blockscaled_hmx(64, 64, 256, 64, "int4", "float32", True)).kernel_source
    # One call of the out-of-line HMX GEMM: A's row tiles 8 tiles of K apart,
    # B's panels 4096 bytes apart, groups of 64, a factor per row of A.
    assert (
        "tl_hexagon_hmx_gemm_blockscaled<true, true, true, true, false, false>(64, 64, 256, (&(((float*)c)[0])), 64, "
        "(&(((half*)a)[0])), 8192, ((int8_t*)b + ((0) >> 1)), 4096, (&(((half*)sb)[0])), 64, 64, (&(SA[0])), 1, (bool)1, "
    ) in source
    assert "tl_hvx_gemm_int_scaled" not in source


def test_blockscaled_normal_scales_specializes_the_dequantizer():
    source = _lower(_blockscaled_hmx(64, 64, 256, 64, "int4", "float32", True, normal_scales=True)).kernel_source
    assert "tl_hexagon_hmx_gemm_blockscaled<true, true, true, true, false, true>(" in source
    with pytest.raises(Exception, match="compile-time boolean"):
        _lower(_blockscaled_hmx(64, 64, 256, 64, "int4", "float32", True, normal_scales=2))


def test_blockscaled_hmx_gemm_of_a_worker_pool_takes_turns_at_hmx():
    source = _lower(_blockscaled_hmx(64, 64, 256, 64, "uint4", "float32", True, workers=2)).kernel_source
    body = source[source.index("tl_parallel_for<tl_hmx_worker_scope") :]
    # The workers share HMX: a tile each at a time.
    assert "tl_hexagon_hmx_gemm_blockscaled<false, true, true, true, true, false>(" in body


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
@pytest.mark.parametrize("normal_scales", [False, True])
def test_blockscaled_hmx_gemms_match_torch(M, N, K, G, b_dtype, c_dtype, clear, workers, normal_scales):
    torch.manual_seed(M + N + K)
    kernel = tilelang.compile(_blockscaled_hmx(M, N, K, G, b_dtype, c_dtype, clear, workers, normal_scales), out_idx=[5], target="hexagon")
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
