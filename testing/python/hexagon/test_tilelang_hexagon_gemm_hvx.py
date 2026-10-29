"""T.gemm on HVX: the GEMMs HMX does not run, one tl_hvx_gemm call (hvx_gemm.h)."""

import logging
import os

import pytest

import tilelang
from tilelang import tvm
from tilelang.backend.target import determine_target
from tilelang.hexagon import language as T
from tilelang.hexagon.intrinsics.crouton import make_hmx_activation_layout, make_hmx_weight_layout

DEVICE = pytest.mark.skipif(os.environ.get("TL_HEXAGON_DEVICE_TEST") != "1", reason="requires an authorized Hexagon device and SDK")


def _source(func):
    target = determine_target("hexagon", return_object=True)
    with tvm.target.Target(target), tvm.transform.PassContext(opt_level=3):
        return str(tilelang.lower(func, target=target, enable_host_codegen=False, enable_device_compile=False).kernel_source)


def _gemm(M, N, K, BK=None, ab="float16", c_dtype="float32", trans_a=False, trans_b=False, clear="fill"):
    """C = A @ B over K in blocks of BK (all of K by default): T.clear and
    accumulating calls ("fill"), or clear_accum on the first block ("first")."""

    BK = BK or K

    @T.prim_func
    def kernel(
        A: T.Tensor((K, M) if trans_a else (M, K), ab),
        B: T.Tensor((N, K) if trans_b else (K, N), ab),
        C: T.Tensor((M, N), c_dtype),
    ):
        with T.Kernel(1):
            a = T.alloc_shared((BK, M) if trans_a else (M, BK), ab)
            b = T.alloc_shared((N, BK) if trans_b else (BK, N), ab)
            c = T.alloc_fragment((M, N), c_dtype)
            if clear == "fill":
                T.clear(c)
            for k in T.serial(K // BK):
                if trans_a:
                    T.copy(A[k * BK, 0], a)
                else:
                    T.copy(A[0, k * BK], a)
                if trans_b:
                    T.copy(B[0, k * BK], b)
                else:
                    T.copy(B[k * BK, 0], b)
                if clear == "fill":
                    T.gemm(a, b, c, transpose_A=trans_a, transpose_B=trans_b)
                else:
                    T.gemm(a, b, c, transpose_A=trans_a, transpose_B=trans_b, clear_accum=(k == 0))
            T.copy(c, C)

    return kernel


def _sub_matrices():
    """C[2:5, 1:66] = A[1:4, 3:40] @ B[3:40, 5:70]: regions off register
    boundaries, in larger buffers."""

    @T.prim_func
    def kernel(A: T.Tensor((8, 64), "float16"), B: T.Tensor((64, 80), "float16"), C: T.Tensor((8, 80), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((8, 64), "float16")
            b = T.alloc_shared((64, 80), "float16")
            c = T.alloc_fragment((8, 80), "float32")
            T.copy(A, a)
            T.copy(B, b)
            T.clear(c)
            T.gemm(a[1:4, 3:40], b[3:40, 5:70], c[2:5, 1:66])
            T.copy(c, C)

    return kernel


def _weight_gemm(M, N, K, BN, BK, c_dtype="float32", trans_b=False):
    """C = A @ B for B the HMX weight operand in DDR ([K, N], or [N, K] with
    trans_b, in Crouton tiles): blocks [BK, BN] of it into VTCM by DMA, each
    multiplied on HVX into its columns of C, the first block of K clearing
    them."""

    @T.prim_func
    def kernel(A: T.Tensor((M, K), "float16"), B: T.Tensor((N, K) if trans_b else (K, N), "float16"), C: T.Tensor((M, N), c_dtype)):
        with T.Kernel(1):
            a = T.alloc_fragment((M, K), "float16")
            b = T.alloc_shared((BN, BK) if trans_b else (BK, BN), "float16")
            c = T.alloc_fragment((M, N), c_dtype)
            T.annotate_layout({B: make_hmx_weight_layout(B, transposed=trans_b), b: make_hmx_weight_layout(b, transposed=trans_b)})
            T.copy(A, a)
            for n in T.serial(N // BN):
                for k in T.serial(K // BK):
                    if trans_b:
                        T.dma_copy(B[n * BN : (n + 1) * BN, k * BK : (k + 1) * BK], b)
                    else:
                        T.dma_copy(B[k * BK : (k + 1) * BK, n * BN : (n + 1) * BN], b)
                    T.dma_wait()
                    T.gemm(a[:, k * BK : (k + 1) * BK], b, c[:, n * BN : (n + 1) * BN], transpose_B=trans_b, clear_accum=(k == 0))
            T.copy(c, C)

    return kernel


def _weight_tiles(b):
    """The bytes of b [K, N] as the HMX weight operand: Crouton tiles by
    columns ([nt, kt]), element (k, n) of a tile at (k % 32 // 2) * 64 +
    n % 32 * 2 + k % 2."""
    K, N = b.shape
    return b.reshape(K // 32, 16, 2, N // 32, 32).permute(3, 0, 1, 4, 2).reshape(K, N).contiguous()


def _calls(source, name="tl_hvx_gemm"):
    return [line.strip() for line in source.splitlines() if f"{name}<" in line]


def test_gemvs_run_on_hvx():
    source = _source(_gemm(1, 1024, 4096, 256))
    assert "tl_hexagon_hmx_" not in source
    (call,) = _calls(source)
    # M, N, K, trans_A, trans_B, the row strides, then whether A, B and C start
    # on register boundaries.
    assert call.startswith("tl_hvx_gemm<1, 1024, 256, false, false, 256, 1024, 1024, true, true, true>(")
    assert call.endswith("(bool)0);")


def test_transposed_operands_pick_the_schedule_in_the_call():
    (call,) = _calls(_source(_gemm(3, 100, 96, trans_b=True)))
    assert call.startswith("tl_hvx_gemm<3, 100, 96, false, true, 96, 96, 100, true, true, true>(")
    (call,) = _calls(_source(_gemm(16, 64, 64, trans_a=True)))
    assert call.startswith("tl_hvx_gemm<16, 64, 64, true, false, 16, 64, 64, true, true, true>(")


def test_fp32_operands_run_on_hvx():
    (call,) = _calls(_source(_gemm(64, 64, 64, ab="float32")))
    assert call.startswith("tl_hvx_gemm<64, 64, 64, false, false, 64, 64, 64,")
    assert "(float*)a" in call


def test_a_dynamic_clear_is_the_last_argument():
    (call,) = _calls(_source(_gemm(1, 256, 512, 128, clear="first")))
    assert call.endswith("(k == 0));")


def test_sub_matrices_start_at_their_first_elements():
    (call,) = _calls(_source(_sub_matrices()))
    # a[1, 3] is 67 fp16 from a's start, b[3, 5] 245 and c[2, 1] 161 fp32:
    # none on a register boundary.
    assert call.startswith("tl_hvx_gemm<3, 65, 37, false, false, 64, 80, 80, false, false, false>(")
    assert "((float*)c)[161]" in call and "((half*)a)[67]" in call and "((half*)b)[245]" in call


def test_gemms_neither_unit_runs_say_why(caplog):
    with caplog.at_level(logging.WARNING, logger="tilelang.hexagon.op.gemm"):
        source = _source(_gemm(16, 64, 64, trans_a=True, trans_b=True))
    assert "tl_hvx_gemm" not in source
    assert (
        "runs as a scalar loop, not on HMX: HMX needs M, N and K multiples of 32, not 16 x 64 x 64; "
        "not on HVX: HVX reads A or B along K, so not both transposed"
    ) in caplog.text


def _laid_out_b(layout):
    @T.prim_func
    def kernel(A: T.Tensor((16, 64), "float16"), B: T.Tensor((64, 64), "float16"), C: T.Tensor((16, 64), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((16, 64), "float16")
            b = T.alloc_shared((64, 64), "float16")
            c = T.alloc_fragment((16, 64), "float32")
            T.annotate_layout({b: layout(b)})
            T.copy(A, a)
            T.copy(B, b)
            T.gemm(a, b, c, clear_accum=True)
            T.copy(c, C)

    return kernel


def test_a_weight_layout_b_runs_on_hvx():
    (call,) = _calls(_source(_laid_out_b(make_hmx_weight_layout)), "tl_hvx_gemm_weight")
    # M, N, K, A's and C's row strides, whether A starts on a register
    # boundary, and the elements between B's panels (its matrix's K tiles).
    assert call.startswith("tl_hvx_gemm_weight<16, 64, 64, 64, 64, true, 2048>(")


def test_operands_in_other_layouts_run_as_scalar_loops(caplog):
    with caplog.at_level(logging.WARNING, logger="tilelang.hexagon.op.gemm"):
        source = _source(_laid_out_b(make_hmx_activation_layout))
    assert "tl_hvx_gemm" not in source
    assert "T.gemm(a, b, c) runs as a scalar loop, not on HVX: b is in a Crouton layout, not the HMX weight operand's" in caplog.text


def test_weight_layout_gemvs_take_regions_of_whole_tiles():
    source = _source(_weight_gemm(1, 256, 7680, 64, 1920))
    (call,) = _calls(source, "tl_hvx_gemm_weight")
    # A block of 60 K tiles, panels of the block 60 tiles apart; A's blocks of
    # 1920 fp16 start on register boundaries.
    assert call.startswith("tl_hvx_gemm_weight<1, 64, 1920, 7680, 256, true, 61440>(")
    assert "(&(((float*)c)[(n * 64)]))" in call and "(&(((half*)a)[(k * 1920)]))" in call
    assert call.endswith("(k == 0));")
    (call,) = _calls(_source(_weight_gemm(2, 64, 128, 64, 64, trans_b=True)), "tl_hvx_gemm_weight")
    assert call.startswith("tl_hvx_gemm_weight<2, 64, 64, 128, 64, true, 2048>(")


def _run(func, *inputs):
    kernel = tilelang.compile(func, out_idx=[len(inputs)], target="hexagon")
    try:
        return kernel(*inputs)
    finally:
        kernel.adapter.close()


def _check(func, M, N, K, ab="float16", c_dtype="float32", trans_a=False, trans_b=False, calls=1):
    import torch

    dtype = getattr(torch, ab)
    a = torch.randn(M, K).to(dtype)
    b = (torch.randn(K, N) * 0.1).to(dtype)
    ref = a.float() @ b.float()
    out = _run(func, a.t().contiguous() if trans_a else a, b.t().contiguous() if trans_b else b).float()
    # qf32 sums; an fp16 C rounds once per call.
    rounding = calls * 2.0**-10 if c_dtype == "float16" else 1e-5
    torch.testing.assert_close(out, ref, atol=rounding * float(ref.abs().max()) + 1e-5, rtol=0)


@DEVICE
@pytest.mark.parametrize("trans_a, trans_b", [(False, False), (True, False), (False, True)])
@pytest.mark.parametrize("c_dtype", ["float32", "float16"])
@pytest.mark.parametrize(
    "M, N, K",
    [
        (1, 1024, 256),  # a GEMV
        (3, 100, 33),  # tails everywhere
        (17, 48, 200),
        (2, 300, 7),
        (5, 1, 64),
        (40, 72, 96),
    ],
)
def test_gemms_match_torch(M, N, K, c_dtype, trans_a, trans_b):
    func = _gemm(M, N, K, c_dtype=c_dtype, trans_a=trans_a, trans_b=trans_b, clear="first")
    _check(func, M, N, K, c_dtype=c_dtype, trans_a=trans_a, trans_b=trans_b)


@DEVICE
@pytest.mark.parametrize("trans_a, trans_b", [(False, False), (True, False), (False, True)])
@pytest.mark.parametrize("M, N, K", [(1, 256, 96), (7, 70, 45), (32, 96, 64)])
def test_fp32_gemms_match_torch(M, N, K, trans_a, trans_b):
    _check(_gemm(M, N, K, ab="float32", trans_a=trans_a, trans_b=trans_b), M, N, K, ab="float32", trans_a=trans_a, trans_b=trans_b)


@DEVICE
@pytest.mark.parametrize("trans_b", [False, True])
@pytest.mark.parametrize("c_dtype", ["float32", "float16"])
def test_k_loops_accumulate_across_calls(c_dtype, trans_b):
    _check(_gemm(2, 160, 512, 128, c_dtype=c_dtype, trans_b=trans_b), 2, 160, 512, c_dtype=c_dtype, trans_b=trans_b, calls=4)


@DEVICE
def test_sub_matrices_match_torch():
    import torch

    a = torch.randn(8, 64).half()
    b = (torch.randn(64, 80) * 0.1).half()
    ref = torch.zeros(8, 80)
    ref[2:5, 1:66] = a[1:4, 3:40].float() @ b[3:40, 5:70].float()
    torch.testing.assert_close(_run(_sub_matrices(), a, b), ref, atol=1e-4, rtol=0)


@DEVICE
@pytest.mark.parametrize("trans_a, trans_b", [(False, False), (True, False), (False, True)])
@pytest.mark.parametrize("c_dtype", ["float32", "float16"])
@pytest.mark.parametrize("clear", ["fill", "first"])
def test_exact_zeros_stay_zero(clear, c_dtype, trans_a, trans_b):
    # qf32 holds zero in many encodings, and adds of two zeros step their
    # exponent up (hvx_gemm.h): rows and columns of zeros, and a block of K
    # whose products are all zero, must still give exact zeros.
    import torch

    M, N, K, BK = 6, 96, 256, 128
    a = torch.randn(M, K).half()
    a[1] = 0
    a[3, :BK] = 0
    b = (torch.randn(K, N) * 0.1).half()
    b[:, 5] = 0
    b[BK:, 40:72] = 0
    a[4, :BK] = 0
    ref = a.float() @ b.float()
    out = _run(
        _gemm(M, N, K, BK, c_dtype=c_dtype, trans_a=trans_a, trans_b=trans_b, clear=clear),
        a.t().contiguous() if trans_a else a,
        b.t().contiguous() if trans_b else b,
    ).float()
    assert torch.all(out[1] == 0) and torch.all(out[:, 5] == 0)
    assert torch.all(out[4, 40:72] == 0)
    rounding = 3 * 2.0**-10 if c_dtype == "float16" else 1e-5
    torch.testing.assert_close(out, ref, atol=rounding * float(ref.abs().max()), rtol=0)


@DEVICE
@pytest.mark.parametrize(
    "M, N, K, BN, BK, c_dtype, trans_b",
    [
        (1, 128, 2048, 32, 2048, "float32", False),  # a decode's GEMV, a panel a block
        (1, 256, 2048, 128, 1024, "float32", False),  # four panels, K in two blocks
        (2, 160, 96, 160, 96, "float32", False),  # five panels; an odd number of K tiles
        (1, 64, 7680, 64, 1920, "float32", False),  # an FFN's w2: K in slices
        (1, 128, 1024, 64, 512, "float16", False),
        (2, 96, 256, 96, 128, "float32", True),
    ],
)
def test_weight_layout_gemms_match_torch(M, N, K, BN, BK, c_dtype, trans_b):
    import torch

    a = torch.randn(M, K).half()
    b = (torch.randn(K, N) * 0.1).half()
    # Columns of zeros, as in LFM3's panels of head gates, come out exact.
    b[:, 3:32] = 0
    ref = a.float() @ b.float()
    tiles = _weight_tiles(b)
    out = _run(_weight_gemm(M, N, K, BN, BK, c_dtype, trans_b), a, tiles.view(N, K) if trans_b else tiles).float()
    assert torch.all(out[:, 3:32] == 0)
    rounding = (K // BK) * 2.0**-10 if c_dtype == "float16" else 1e-5
    torch.testing.assert_close(out, ref, atol=rounding * float(ref.abs().max()) + 1e-5, rtol=0)
