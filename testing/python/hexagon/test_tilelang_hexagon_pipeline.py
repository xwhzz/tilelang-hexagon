"""T.Pipelined on Hexagon: a pipeline's copies from DDR become DMA transfers."""

import os
import re

import pytest

import tilelang
from tilelang import tvm
from tilelang.backend.target import determine_target
from tilelang.hexagon import language as T

DEVICE = pytest.mark.skipif(os.environ.get("TL_HEXAGON_DEVICE_TEST") != "1", reason="requires an authorized Hexagon device and SDK")


def _source(func):
    target = determine_target("hexagon", return_object=True)
    with tvm.target.Target(target), tvm.transform.PassContext(opt_level=3):
        return str(tilelang.lower(func, target=target, enable_host_codegen=False, enable_device_compile=False).kernel_source)


def _gemm(M, N, K, BK, stages):
    """One BM x BN block of C = A @ B, its K loop pipelined (or serial: 0)."""

    @T.prim_func
    def kernel(A: T.Tensor((M, K), "float16"), B: T.Tensor((K, N), "float16"), C: T.Tensor((M, N), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((M, BK), "float16")
            b = T.alloc_shared((BK, N), "float16")
            acc = T.alloc_fragment((M, N), "float32")
            if stages:
                for k in T.Pipelined(K // BK, num_stages=stages):
                    T.copy(A[0, k * BK], a)
                    T.copy(B[k * BK, 0], b)
                    T.gemm(a, b, acc, clear_accum=(k == 0))
            else:
                for k in T.serial(K // BK):
                    T.copy(A[0, k * BK], a)
                    T.copy(B[k * BK, 0], b)
                    T.gemm(a, b, acc, clear_accum=(k == 0))
            T.copy(acc, C)

    return kernel


def _gemv(K, N, BK):
    @T.prim_func
    def kernel(x: T.Tensor((K,), "float16"), W: T.Tensor((K, N), "float16"), y: T.Tensor((N,), "float32")):
        with T.Kernel(1):
            xs = T.alloc_shared((K,), "float16")
            w = T.alloc_shared((BK, N), "float16")
            acc = T.alloc_fragment((N,), "float32")
            T.copy(x, xs)
            T.clear(acc)
            for k in T.Pipelined(K // BK, num_stages=2):
                T.copy(W[k * BK, 0], w)
                for i in T.serial(BK):
                    for n in T.Parallel(N):
                        acc[n] += T.cast(xs[k * BK + i], "float32") * T.cast(w[i, n], "float32")
            T.copy(acc, y)

    return kernel


def _written_then_streamed(N, BN):
    """Y = 2 X through VTCM, then the sum of Y's blocks streamed back by DMA:
    Y's cache lines are dirty when its DMA starts."""

    @T.prim_func
    def kernel(X: T.Tensor((N,), "float16"), Y: T.Tensor((N,), "float16"), S: T.Tensor((BN,), "float32")):
        with T.Kernel(1):
            x = T.alloc_shared((N,), "float16")
            yb = T.alloc_shared((BN,), "float16")
            acc = T.alloc_fragment((BN,), "float32")
            T.copy(X, x)
            for i in T.Parallel(N):
                x[i] = x[i] * T.float16(2)
            T.copy(x, Y)
            T.clear(acc)
            for k in T.Pipelined(N // BN, num_stages=2):
                T.copy(Y[k * BN], yb)
                for i in T.Parallel(BN):
                    acc[i] += T.cast(yb[i], "float32")
            T.copy(acc, S)

    return kernel


def _loop_lines(source):
    lines = [line.strip() for line in source.splitlines()]
    start = next(i for i, line in enumerate(lines) if line.startswith("for (int32_t k"))
    return lines[start:]


def test_copies_into_row_major_buffers_are_dma_transfers():
    source = _source(_gemv(4096, 1024, 32))
    # A queue for two transfers: one in flight while the other is consumed.
    assert "tl_hexagon_dma_async_context<2, 2> tl_dma_async;" in source
    loop = _loop_lines(source)
    submit = next(i for i, line in enumerate(loop) if "tl_hexagon_dma_async_submit(" in line)
    assert "w)[(((k + 1) & 1) * 32768)]" in loop[submit]
    assert "tl_hexagon_dma_async_commit(" in loop[submit + 1]
    assert "tl_hexagon_dma_async_wait(tl_dma_async, 1)" in loop[submit + 2]


def test_copies_into_hmx_operands_stage_through_dma():
    source = _source(_gemm(128, 128, 2048, 256, 2))
    for operand, pack in (("a", "tl_crouton_copy<true, false>"), ("b", "tl_crouton_copy<true, true>")):
        assert f"void* {operand}_dma = " in source
        assert re.search(rf"tl_hexagon_dma_async_submit\(tl_dma_async, \(&\(\(\(half\*\){operand}_dma\)", source)
        # The operand's Crouton tiles are packed from the staging buffer in VTCM.
        assert re.search(rf"{re.escape(pack)}\(\(&\(\(\(half\*\){operand}\)\[0\]\)\).*\(half\*\){operand}_dma\)", source)


def test_copies_with_short_rows_stay_hvx_copies():
    # A's rows are 64 fp16, 128 bytes: DMA would move them slower than HVX.
    source = _source(_gemm(128, 128, 2048, 64, 2))
    assert "void* a_dma" not in source
    assert "void* b_dma" in source
    assert any("tl_crouton_copy<true, false>(" in line and "(&(A[" in line for line in _loop_lines(source))


def test_dma_from_read_only_arguments_skips_the_writeback():
    # dma.h: 3 is DDR to VTCM from memory the DSP did not write, 0 writes the
    # source's cache lines back first.
    source = _source(_gemv(4096, 1024, 32))
    directions = {line.rsplit(", ", 1)[-1] for line in source.splitlines() if "tl_hexagon_dma_async_submit(" in line}
    assert directions == {"3)) return tl_status;"}
    source = _source(_written_then_streamed(8192, 1024))
    directions = {line.rsplit(", ", 1)[-1] for line in source.splitlines() if "tl_hexagon_dma_async_submit(" in line}
    assert directions == {"0)) return tl_status;"}


def _run(func, *inputs, outputs=1):
    kernel = tilelang.compile(func, out_idx=list(range(len(inputs), len(inputs) + outputs)), target="hexagon")
    try:
        return kernel(*inputs)
    finally:
        kernel.adapter.close()


@DEVICE
@pytest.mark.parametrize("BK, stages", [(64, 2), (256, 2), (256, 3), (512, 2)])
def test_pipelined_gemms_match_torch(BK, stages):
    import torch

    M = N = 128
    K = 2048
    a = (torch.randn(M, K) * 0.1).half()
    b = (torch.randn(K, N) * 0.1).half()
    ref = a.float() @ b.float()
    out = _run(_gemm(M, N, K, BK, stages), a, b)
    torch.testing.assert_close(out, ref, atol=(K // BK + 1) * 2.0**-10 * float(ref.abs().max()), rtol=0)


@DEVICE
def test_pipelined_gemv_matches_torch():
    import torch

    x = torch.randn(4096).half()
    w = (torch.randn(4096, 1024) * 0.05).half()
    torch.testing.assert_close(_run(_gemv(4096, 1024, 32), x, w), x.float() @ w.float(), rtol=1e-4, atol=1e-3)


@DEVICE
def test_dma_after_writes_reads_the_written_bytes():
    import torch

    x = torch.randn(8192).half()
    y, s = _run(_written_then_streamed(8192, 1024), x, outputs=2)
    torch.testing.assert_close(y, x * 2, rtol=0, atol=0)
    torch.testing.assert_close(s, (x * 2).float().reshape(8, 1024).sum(0), rtol=1e-5, atol=1e-3)
