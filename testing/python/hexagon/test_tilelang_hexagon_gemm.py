"""T.gemm on HMX: K loops, C dtypes, scopes and layouts, sub-regions, transposes."""

import logging
import os

import pytest

import tilelang
from tilelang import tvm
from tilelang.backend.target import determine_target
from tilelang.hexagon import language as T
from tilelang.hexagon.intrinsics.crouton import make_hmx_weight_layout

DEVICE = pytest.mark.skipif(os.environ.get("TL_HEXAGON_DEVICE_TEST") != "1", reason="requires an authorized Hexagon device and SDK")


def _source(func):
    target = determine_target("hexagon", return_object=True)
    with tvm.target.Target(target), tvm.transform.PassContext(opt_level=3):
        return str(tilelang.lower(func, target=target, enable_host_codegen=False, enable_device_compile=False).kernel_source)


def _k_loop(M, N, K, BK, c_dtype="float32", scope="fragment", clear="fill", trans_a=False, trans_b=False):
    """C = A @ B over K in blocks of BK: T.clear and accumulating calls
    ("fill"), clear_accum on the first block ("first"), or one call ("one")."""

    @T.prim_func
    def kernel(
        A: T.Tensor((K, M) if trans_a else (M, K), "float16"),
        B: T.Tensor((N, K) if trans_b else (K, N), "float16"),
        C: T.Tensor((M, N), "float32"),
    ):
        with T.Kernel(1):
            a = T.alloc_shared((BK, M) if trans_a else (M, BK), "float16")
            b = T.alloc_shared((N, BK) if trans_b else (BK, N), "float16")
            if scope == "fragment":
                c = T.alloc_fragment((M, N), c_dtype)
            else:
                c = T.alloc_shared((M, N), c_dtype)
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


def _sub_matrices(c_dtype):
    """C[64:128, 96:160] += A[1, 32:96, 64:128] @ B[64:128, 32:96], the rest of
    C as given: the operands are corners of larger buffers, A one of two
    matrices."""

    @T.prim_func
    def kernel(
        A: T.Tensor((2, 96, 128), "float16"),
        B: T.Tensor((128, 160), "float16"),
        C0: T.Tensor((128, 192), c_dtype),
        C: T.Tensor((128, 192), c_dtype),
    ):
        with T.Kernel(1):
            a = T.alloc_shared((2, 96, 128), "float16")
            b = T.alloc_shared((128, 160), "float16")
            c = T.alloc_shared((128, 192), c_dtype)
            T.copy(A, a)
            T.copy(B, b)
            T.copy(C0, c)
            T.gemm(a[1, 32:96, 64:128], b[64:128, 32:96], c[64:128, 96:160])
            T.copy(c, C)

    return kernel


def _c_layout(kind):
    """C = A @ B accumulated twice into a C of another layout: fp16 row-major
    ("rows16"), an fp32 C whose rows do not start on registers ("rows32_odd"),
    fp16 Crouton tiles ordered by columns ("columns16") and a transposed fp32
    C ("transposed32")."""

    dtype = "float16" if kind in ("rows16", "columns16") else "float32"
    cols = 48 if kind == "rows32_odd" else 64

    @T.prim_func
    def kernel(A: T.Tensor((64, 64), "float16"), B: T.Tensor((64, 32), "float16"), C: T.Tensor((64, 32), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((64, 64), "float16")
            b = T.alloc_shared((64, 32), "float16")
            c = T.alloc_shared((64, cols), dtype)
            if kind == "rows16":
                T.annotate_layout({c: T.Layout((64, cols), lambda i, j: [i, j])})
            elif kind == "columns16":
                T.annotate_layout({c: make_hmx_weight_layout(c)})
            elif kind == "transposed32":
                T.annotate_layout({c: T.Layout((64, cols), lambda i, j: [j, i])})
            T.copy(A, a)
            T.copy(B, b)
            T.gemm(a, b, c[:, 0:32], clear_accum=True)
            T.gemm(a, b, c[:, 0:32])
            T.copy(c[:, 0:32], C)

    return kernel


def test_k_loops_accumulate_hmx_tiles_into_an_fp32_fragment():
    source = _source(_k_loop(64, 64, 128, 64))
    assert "tl_hexagon_hmx_mma_atom(" in source
    # Each tile goes through a staging tile into the row-major fp32 C: one
    # store in the tile loop and the last tile's after it.
    assert source.count("tl_crouton_tile_store<true, true, true>(") == 2
    assert "(&(((float*)c)[2080])), 64, (&(((half*)stage)[1024])));" in source


def test_each_tiles_epilogue_overlaps_the_next_tiles_macs():
    lines = [line.strip() for line in _source(_k_loop(64, 64, 128, 64)).splitlines()]

    def first(text, start=0):
        return next(i for i in range(start, len(lines)) if text in lines[i])

    mma = first("tl_hexagon_hmx_mma_atom(")
    store = first("tl_crouton_tile_store<", mma)
    assert lines[store - 1] == "if (0 < t) {"
    convert = first("tl_hexagon_hmx_convert_acc(", mma)
    assert mma < store < convert
    # The last tile's epilogue runs after the accumulator is released.
    release = first("tl_hexagon_hmx_acc_release(", convert)
    assert first("tl_crouton_tile_store<", release) > release


def test_fp16_c_accumulates_in_its_crouton_tiles():
    source = _source(_k_loop(64, 64, 128, 64, "float16"))
    assert "tl_crouton_tile_store<true, false, true>((&(((half*)c)[((t * 1024) - 1024)]))" in source


def test_a_dynamic_clear_writes_the_first_block_and_adds_the_others():
    for dtype, rows in (("float32", "true"), ("float16", "false")):
        source = _source(_k_loop(64, 64, 128, 64, dtype, "shared", "first"))
        assert "if (k == 0) {" in source
        assert f"tl_crouton_tile_store<false, {rows}, true>(" in source
        assert f"tl_crouton_tile_store<true, {rows}, true>(" in source


def test_sub_matrices_start_at_their_tiles():
    source = _source(_sub_matrices("float32"))
    # A's sub-matrix starts at tile 18: matrix 1 of 3 x 4 tiles, row tile 1,
    # column tile 2. B's starts at tile 6: column tile 1 of 4 K tiles, K tile 2.
    assert "a)[(((t >> 1) * 4096) + 18432)]" in source
    assert "b)[(((t & 1) * 4096) + 6144)]" in source
    # C's at row 64, column 96 of its 192 columns.
    assert "c)[(((((t - 1) >> 1) * 6144) + (((t + 1) & 1) * 32)) + 12384)])), 192," in source


def test_c_of_another_layout_is_written_element_by_element():
    source = _source(_c_layout("columns16"))
    assert "tl_hexagon_hmx_mma_atom(" in source
    assert "tl_crouton_tile_store" not in source


def test_rows_off_register_boundaries_are_unaligned_stores():
    assert "tl_crouton_tile_store<true, true, false>(" in _source(_c_layout("rows32_odd"))
    assert "tl_crouton_tile_store<true, true, true>(" in _source(_c_layout("rows16"))


def test_gemms_hmx_cannot_run_go_to_hvx_saying_why(caplog):
    @T.prim_func
    def off_tiles(A: T.Tensor((96, 64), "float16"), B: T.Tensor((64, 64), "float16"), C: T.Tensor((64, 64), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((96, 64), "float16")
            b = T.alloc_shared((64, 64), "float16")
            c = T.alloc_fragment((64, 64), "float32")
            T.copy(A, a)
            T.copy(B, b)
            T.gemm(a[16:80, :], b, c, clear_accum=True)
            T.copy(c, C)

    @T.prim_func
    def fp32_operands(A: T.Tensor((64, 64), "float32"), B: T.Tensor((64, 64), "float32"), C: T.Tensor((64, 64), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((64, 64), "float32")
            b = T.alloc_shared((64, 64), "float32")
            c = T.alloc_fragment((64, 64), "float32")
            T.copy(A, a)
            T.copy(B, b)
            T.gemm(a, b, c, clear_accum=True)
            T.copy(c, C)

    for func, reason in (
        (off_tiles, "A (a)'s region does not start on a 32 x 32 tile"),
        (fp32_operands, "HMX multiplies fp16 operands, not float32 x float32"),
    ):
        caplog.clear()
        with caplog.at_level(logging.DEBUG, logger="tilelang.hexagon.op.gemm"):
            source = _source(func)
        assert "tl_hexagon_hmx_" not in source
        assert "tl_hvx_gemm<" in source
        assert f"runs on HVX, not on HMX: {reason}" in caplog.text


def test_operands_in_other_layouts_are_rejected():
    @T.prim_func
    def kernel(A: T.Tensor((64, 64), "float16"), B: T.Tensor((64, 64), "float16"), C: T.Tensor((64, 64), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((64, 64), "float16")
            b = T.alloc_shared((64, 64), "float16")
            c = T.alloc_fragment((64, 64), "float32")
            T.annotate_layout({a: make_hmx_weight_layout(a)})
            T.copy(A, a)
            T.copy(B, b)
            T.gemm(a, b, c, clear_accum=True)
            T.copy(c, C)

    with pytest.raises(Exception, match=r"reads A \(a\) in the HMX activation layout"):
        _source(kernel)


def _tolerance(ref, calls):
    # Each call rounds its product to fp16 (half an ulp), and an fp16 C each
    # running sum.
    return (calls + 1) * 2.0**-10 * float(ref.abs().max())


def _run(func, *inputs):
    kernel = tilelang.compile(func, out_idx=[len(inputs)], target="hexagon")
    try:
        return kernel(*inputs)
    finally:
        kernel.adapter.close()


@DEVICE
@pytest.mark.parametrize("trans_a, trans_b", [(False, False), (True, False), (False, True), (True, True)])
@pytest.mark.parametrize("clear", ["fill", "first"])
@pytest.mark.parametrize("scope", ["fragment", "shared"])
@pytest.mark.parametrize("c_dtype", ["float32", "float16"])
def test_k_loops_match_torch(c_dtype, scope, clear, trans_a, trans_b):
    import torch

    M, N, K, BK = 64, 96, 128, 64
    a = (torch.randn(M, K) * 0.25).half()
    b = (torch.randn(K, N) * 0.25).half()
    out = _run(
        _k_loop(M, N, K, BK, c_dtype, scope, clear, trans_a, trans_b),
        a.t().contiguous() if trans_a else a,
        b.t().contiguous() if trans_b else b,
    )
    ref = a.float() @ b.float()
    torch.testing.assert_close(out, ref, atol=_tolerance(ref, K // BK), rtol=0)


@DEVICE
@pytest.mark.parametrize(
    "M, N, K, BK",
    [
        (32, 32, 32, 32),  # one tile
        (128, 64, 256, 128),
        (32, 64, 1088, 1088),  # one call of 34 K tiles: two deep packets
        (64, 32, 2048, 1024),  # 32 K tiles per call: one full packet
    ],
)
def test_shapes_match_torch(M, N, K, BK):
    import torch

    a = (torch.randn(M, K) * 0.1).half()
    b = (torch.randn(K, N) * 0.1).half()
    out = _run(_k_loop(M, N, K, BK, clear="first"), a, b)
    ref = a.float() @ b.float()
    torch.testing.assert_close(out, ref, atol=_tolerance(ref, K // BK), rtol=0)


@DEVICE
@pytest.mark.parametrize("c_dtype", ["float32", "float16"])
def test_sub_matrices_match_torch(c_dtype):
    import torch

    a = (torch.randn(2, 96, 128) * 0.25).half()
    b = (torch.randn(128, 160) * 0.25).half()
    c0 = torch.randn(128, 192).to(getattr(torch, c_dtype))
    out = _run(_sub_matrices(c_dtype), a, b, c0).float()
    ref = c0.float().clone()
    ref[64:128, 96:160] += a[1, 32:96, 64:128].float() @ b[64:128, 32:96].float()
    torch.testing.assert_close(out, ref, atol=_tolerance(ref, 1), rtol=0)


@DEVICE
@pytest.mark.parametrize("kind", ["rows16", "rows32_odd", "columns16", "transposed32"])
def test_c_layouts_match_torch(kind):
    import torch

    a = (torch.randn(64, 64) * 0.25).half()
    b = (torch.randn(64, 32) * 0.25).half()
    out = _run(_c_layout(kind), a, b)
    ref = 2 * (a.float() @ b.float())
    torch.testing.assert_close(out, ref, atol=_tolerance(ref, 2), rtol=0)


def _worker_k_loops(NB, M, N, K, BK, num_workers):
    """NB independent K loops of accumulating calls into fp32 C, one per tile
    program, spread over a worker pool."""

    @T.prim_func
    def kernel(A: T.Tensor((NB, M, K), "float16"), B: T.Tensor((NB, K, N), "float16"), C: T.Tensor((NB, M, N), "float32")):
        with T.Kernel(NB, num_workers=num_workers) as bx:
            a = T.alloc_shared((M, K), "float16")
            b = T.alloc_shared((K, N), "float16")
            c = T.alloc_fragment((M, N), "float32")
            T.copy(A[bx, :, :], a)
            T.copy(B[bx, :, :], b)
            for k in T.serial(K // BK):
                T.gemm(a[:, k * BK : (k + 1) * BK], b[k * BK : (k + 1) * BK, :], c, clear_accum=(k == 0))
            T.copy(c, C[bx, :, :])

    return kernel


def _hmx_sections(source):
    """Lines of the HMX acquires and releases, and of the first output-tile loop."""
    lines = [line.strip() for line in source.splitlines()]
    acquire = [i for i, line in enumerate(lines) if "tl_hexagon_hmx_acc_acquire" in line]
    release = [i for i, line in enumerate(lines) if "tl_hexagon_hmx_acc_release" in line]
    tiles = next(i for i, line in enumerate(lines) if line.startswith("for (int32_t t"))
    return acquire, release, tiles


def test_workers_take_hmx_one_output_tile_at_a_time():
    # One tile program holds HMX for its whole call, overlapping each tile's
    # epilogue with the next tile's MACs.
    acquire, release, tiles = _hmx_sections(_source(_worker_k_loops(6, 64, 64, 128, 64, 1)))
    assert acquire[0] < tiles < release[0]
    # A pool's workers share HMX: each takes it for one output tile and runs
    # the tile's epilogue after releasing it.
    source = _source(_worker_k_loops(6, 64, 64, 128, 64, 6))
    acquire, release, tiles = _hmx_sections(source)
    assert tiles < acquire[0] < release[0]
    lines = [line.strip() for line in source.splitlines()]
    assert "tl_hexagon_hmx_load_bias" in lines[acquire[0] + 1]
    assert "tl_crouton_tile_store" in "\n".join(lines[release[0] : release[0] + 3])


@DEVICE
@pytest.mark.parametrize("num_workers", [2, 6])
def test_worker_k_loops_match_torch(num_workers):
    import torch

    NB, M, N, K, BK = 6, 64, 96, 256, 64
    a = (torch.randn(NB, M, K) * 0.1).half()
    b = (torch.randn(NB, K, N) * 0.1).half()
    kernel = tilelang.compile(_worker_k_loops(NB, M, N, K, BK, num_workers), out_idx=[2], target="hexagon")
    try:
        out = kernel(a, b)
    finally:
        kernel.adapter.close()
    ref = torch.bmm(a.float(), b.float())
    torch.testing.assert_close(out, ref, atol=_tolerance(ref, K // BK), rtol=0)
