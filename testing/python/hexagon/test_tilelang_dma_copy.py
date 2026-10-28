"""Hexagon async submission, completion, and VTCM allocation contracts."""

import os
import re

import pytest
import tilelang
from tilelang.hexagon import language as T
from tilelang import tvm
from tilelang.backend.target import determine_target
from tilelang.hexagon.intrinsics.hmx import HMXIntrinEmitter


@pytest.mark.skipif(os.environ.get("TL_HEXAGON_DEVICE_TEST") != "1", reason="requires an authorized Hexagon device and SDK")
@pytest.mark.parametrize("size", [32736, 32768, 32769, 524288])
def test_dma_publishes_dirty_ddr_and_preserves_boundary_lines(size):
    """Exercise both sides of the 128 KiB clean threshold on real hardware.

    The DSP dirties the source and destination before DMA. The destination
    begins inside a cache line, so its guards must survive clean/invalidate.
    A DSP readback also checks that stale destination cache lines are removed.
    """
    import torch

    @T.prim_func
    def probe(
        A: T.Tensor((size + 64,), "float32"),
        Scratch: T.Tensor((size + 64,), "float32"),
        Dst: T.Tensor((size + 64,), "float32"),
        Readback: T.Tensor((size + 64,), "float32"),
        Copied: T.Tensor((size,), "float32"),
    ):
        with T.Kernel(1):
            staged = T.alloc_shared((size,), "float32", align=128)
            for j in T.Parallel(size + 64):
                Scratch[j] = A[j] * T.float32(2) + T.float32(1)
                Dst[j] = T.float32(-321.25)
            T.dma_copy(Scratch[1 : size + 1], staged)
            T.dma_wait()
            T.dma_copy(staged, Dst[1 : size + 1])
            T.dma_wait()
            T.copy(staged, Copied)
            T.copy(Dst, Readback)

    kernel = tilelang.compile(probe, out_idx=[1, 2, 3, 4], target="hexagon")
    try:
        for seed in [1, 2, 3]:
            torch.manual_seed(seed)
            src = torch.randn(size + 64)
            scratch, dst, readback, copied = kernel(src)
            expected = src * 2 + 1
            guarded = torch.full_like(src, -321.25)
            guarded[1 : size + 1] = expected[1 : size + 1]
            torch.testing.assert_close(scratch, expected, rtol=0, atol=0)
            torch.testing.assert_close(copied, expected[1 : size + 1], rtol=0, atol=0)
            torch.testing.assert_close(dst, guarded, rtol=0, atol=0)
            torch.testing.assert_close(readback, guarded, rtol=0, atol=0)
    finally:
        kernel.adapter.close()


def lower(func):
    target = determine_target("hexagon", return_object=True)
    with tvm.target.Target(target):
        return str(tilelang.lower(func, target=target, enable_host_codegen=False, enable_device_compile=False).kernel_source)


def make_dma_hmx_matmul(M=256, N=512, K=2048, BM=128, BN=128):
    """An HMX matmul over output blocks of BM x BN, whose A and B rows of the
    next blocks stream in by DMA, two blocks ahead, while HMX multiplies one."""
    assert M % BM == N % BN == K % 32 == BM % 32 == BN % 32 == 0
    E = HMXIntrinEmitter(BM, BN, K)
    count = (M // BM) * (N // BN)

    @T.macro
    def submit(A, B, ar, br, tile):
        mi = tile // (N // BN)
        ni = tile % (N // BN)
        T.dma_copy(A[mi * BM : (mi + 1) * BM, 0:K], ar[tile % 2, 0:BM, 0:K])
        T.dma_copy(B[0:K, ni * BN : (ni + 1) * BN], br[tile % 2, 0:K, 0:BN])

    @T.prim_func
    def dma_hmx_matmul(A: T.Tensor((M, K), "float16"), B: T.Tensor((K, N), "float16"), C: T.Tensor((M, N), "float16")):
        with T.Kernel(1):
            ar = T.alloc_shared((2, BM, K), "float16", align=2048)
            br = T.alloc_shared((2, K, BN), "float16", align=2048)
            an = T.alloc_shared((BM, K), "float16", align=2048)
            bn = T.alloc_shared((K, BN), "float16", align=2048)
            cn = T.alloc_shared((BM, BN), "float16", align=2048)
            config = T.alloc_shared((64,), "uint32", align=256)
            acc = T.hmx.alloc_accumulator()
            cvt = T.hmx.alloc_convert_state()
            bias = T.hmx.alloc_bias_state()
            T.annotate_layout({an: E.activation_layout(an), bn: E.weight_layout(bn), cn: E.output_layout(cn)})
            for i in T.serial(32):
                config[i] = T.uint32(0x3C00)
                config[32 + i] = T.uint32(0)
            # Each output block contributes two queue entries: A and B.
            for tile in T.serial(min(2, count)):
                submit(A, B, ar, br, tile)
            for tile in T.serial(count):
                if tile < count - 1:
                    T.dma_wait(2)
                else:
                    T.dma_wait()
                T.copy(ar[tile % 2, 0, 0], an)
                T.copy(br[tile % 2, 0, 0], bn)
                # Packing has finished reading this slot. Refill it while HMX
                # computes from the separate native Crouton buffers.
                if tile + 2 < count:
                    submit(A, B, ar, br, tile + 2)
                E.acquire(acc)
                E.load_bias(bias, config)
                for mt in T.serial(BM // 32):
                    for nt in T.serial(BN // 32):
                        E.clear(acc)
                        for kt in T.serial(K // 32):
                            E.mma_atom(acc, an, bn, mt, nt, kt)
                        E.convert(cvt, acc, bias, config)
                        E.store(cvt, acc, cn, bias, config, mt, nt)
                E.release(acc)
                T.copy(
                    cn,
                    C[(tile // (N // BN)) * BM : ((tile // (N // BN)) + 1) * BM, (tile % (N // BN)) * BN : ((tile % (N // BN)) + 1) * BN],
                )

    return dma_hmx_matmul


def test_async_manual_ping_pong():
    @T.prim_func
    def probe(A: T.Tensor((768,), "float16")):
        with T.Kernel(1):
            s = T.alloc_shared((2, 256), "float16")
            T.dma_copy(A[0:256], s[0, 0:256])
            T.dma_copy(A[256:512], s[1, 0:256])
            T.dma_wait(1)
            T.dma_copy(A[512:768], s[0, 0:256])
            T.dma_wait()

    source = lower(probe)
    # At most two copies are in flight; each is one descriptor and one group.
    assert "tl_hexagon_dma_async_context<2, 2> tl_dma_async;" in source
    assert source.count("tl_hexagon_dma_async_submit(tl_dma_async, ") == 3
    assert source.count("tl_hexagon_dma_async_commit(tl_dma_async)") == 3
    assert "tl_hexagon_dma_async_wait(tl_dma_async, 1)" in source
    # The program already drains the queue, so no second drain follows.
    assert source.count("tl_hexagon_dma_async_wait(tl_dma_async, 0)") == 1
    assert source.rindex("tl_hexagon_dma_async_wait(tl_dma_async, 0)") > source.rindex("tl_hexagon_dma_async_submit(")
    assert "ptx_" not in source


def test_async_strided_load_store():
    @T.prim_func
    def probe(A: T.Tensor((32, 128), "float16"), B: T.Tensor((32, 128), "float16")):
        with T.Kernel(1):
            s = T.alloc_shared((32, 64), "float16")
            T.dma_copy(A[0:32, 32:96], s)
            T.dma_wait()
            T.dma_copy(s, B[0:32, 32:96])

    source = lower(probe)
    calls = [line for line in source.splitlines() if "tl_hexagon_dma_async_submit(" in line]
    assert len(calls) == 2
    # dst stride, src stride, row bytes, rows, direction: 3 loads from an
    # argument the kernel does not write, with no cache writeback first.
    assert "(uint32_t)128, (uint32_t)256, (uint32_t)128, (uint32_t)32, 3)" in calls[0]
    assert "(uint32_t)256, (uint32_t)128, (uint32_t)128, (uint32_t)32, 1)" in calls[1]
    assert "A[32]" in calls[0] and "B[32]" in calls[1]
    # The store is still in flight at the end, so the program drains it.
    assert source.count("tl_hexagon_dma_async_wait(tl_dma_async, 0)") == 2


def test_async_hmx_dsl_schedule():
    source = lower(make_dma_hmx_matmul())
    # Double buffering: two output blocks of two copies each are in flight.
    assert "tl_hexagon_dma_async_context<4, 4> tl_dma_async;" in source
    assert "tl_hexagon_dma_async_wait(tl_dma_async, 2)" in source
    assert "tl_crouton_copy<true," in source
    assert "tl_hexagon_hmx_mma_atom" in source
    assert "tl_hexagon_dma_copy_hmx_matmul_double_buffer" not in source


def make_queue_probe(workers=None):
    grid = workers or 1

    @T.prim_func
    def probe(A: T.Tensor((grid * 256,), "float16"), B: T.Tensor((grid * 256,), "float16")):
        with T.Kernel(grid, num_workers=workers) as bx:
            s = T.alloc_shared((256,), "float16")
            T.dma_copy(A[bx * 256 : (bx + 1) * 256], s)
            T.dma_wait()
            T.copy(s, B[bx * 256 : (bx + 1) * 256])

    return probe


@pytest.mark.parametrize("pending", [-1, True, 1.5])
def test_reject_invalid_wait_argument(pending):
    with pytest.raises(ValueError, match="nonnegative integer"):
        T.dma_wait(pending)


def test_managed_worker_queue_is_local_and_drained():
    source = lower(make_queue_probe(workers=2))
    entry, program = source.split(" -> int32_t {", 1)
    assert "tl_parallel_for(2, 2, tl_vtcm_worker_arenas(512u), " in entry
    assert "tl_hexagon_dma_async_context" not in entry
    # Each tile program owns its queue and its buffers in the worker's arena.
    assert "tl_hexagon_dma_async_context<1, 1> tl_dma_async;" in program
    assert "half* s = (half*)tl_arena;" in program
    assert program.rindex("tl_hexagon_dma_async_wait(tl_dma_async, 0)") > program.rindex("tl_hexagon_dma_async_submit(")


def test_reject_crouton_async():
    e = HMXIntrinEmitter(32, 32, 32)

    @T.prim_func
    def probe(A: T.Tensor((32, 32), "float16")):
        with T.Kernel(1):
            s = T.alloc_shared((32, 32), "float16")
            T.annotate_layout({s: e.activation_layout(s)})
            T.dma_copy(A, s)
            T.dma_wait()

    with pytest.raises(Exception, match="no Crouton"):
        lower(probe)


def test_reject_out_of_bounds():
    @T.prim_func
    def probe(A: T.Tensor((256,), "float16")):
        with T.Kernel(1):
            s = T.alloc_shared((256,), "float16")
            T.dma_copy(A[128:384], s)
            T.dma_wait()

    with pytest.raises(Exception, match="in-bounds"):
        lower(probe)


@pytest.mark.parametrize("clamped", [True, False])
def test_rows_read_from_a_table_are_in_bounds_once_clamped(clamped):
    # A row the kernel reads from memory has no known value, but clamped to
    # the tensor's rows, it is provably in them.
    @T.prim_func
    def probe(A: T.Tensor((4, 256), "float16"), rows: T.Tensor((2,), "int32"), B: T.Tensor((2, 256), "float16")):
        with T.Kernel(1):
            s = T.alloc_shared((256,), "float16")
            for j in T.serial(2):
                r = T.max(T.min(rows[j], 3), 0) if clamped else rows[j]
                T.dma_copy(A[r, 0:256], s)
                T.dma_wait()
                T.copy(s, B[j, 0:256])

    if clamped:
        assert "tl_hexagon_dma_async_submit" in lower(probe)
    else:
        with pytest.raises(Exception, match="in-bounds"):
            lower(probe)


def test_reject_software_pipeline():
    @T.prim_func
    def probe(A: T.Tensor((4, 256), "float16"), B: T.Tensor((4, 256), "float16")):
        with T.Kernel(1):
            s = T.alloc_shared((256,), "float16")
            for i in T.Pipelined(4, num_stages=2):
                T.dma_copy(A[i, 0:256], s)
                T.dma_wait()
                T.copy(s, B[i, 0:256])

    with pytest.raises(Exception, match="manual schedule"):
        lower(probe)


def test_async_store_source_is_not_recycled_before_wait():
    @T.prim_func
    def probe(A: T.Tensor((256,), "float16"), B: T.Tensor((256,), "float16"), C: T.Tensor((256,), "float16")):
        with T.Kernel(1):
            s = T.alloc_shared((256,), "float16")
            t = T.alloc_shared((256,), "float16")
            T.copy(A, s)
            T.dma_copy(s, B)
            T.fill(t, 2)
            T.copy(t, C)
            T.dma_wait()

    source = lower(probe)
    offsets = re.findall(r"void\* [st] = .*buf_dyn_shmem \+ (\d+)", source)
    assert len(offsets) == 2, source
    assert offsets[0] != offsets[1], source


def test_normal_copy_does_not_create_dma_queue():
    @T.prim_func
    def probe(A: T.Tensor((256,), "float16"), B: T.Tensor((256,), "float16")):
        with T.Kernel(1):
            s = T.alloc_shared((256,), "float16")
            T.copy(A, s)
            T.copy(s, B)

    assert "tl_dma_async" not in lower(probe)


def test_dsp_build_avoids_exception_runtime():
    from tilelang.hexagon.runtime import fastrpc

    # Scope-based cleanup (DMA queues, worker HMX scopes) must not import the
    # C++ unwinder, which the DSP does not have.
    assert "-fno-exceptions" in fastrpc.gen_cmake("test", "/templates")


def test_async_hmx_error_cleanup_is_armed_and_disarmed():
    source = lower(make_dma_hmx_matmul())
    assert "tl_dma_async.release_hmx = tl_hmx_unit_release;" in source
    assert "tl_dma_async.release_hmx = nullptr;" in source


def make_hmx_probe():
    e = HMXIntrinEmitter(32, 32, 64)

    @T.prim_func
    def dma_hmx_workers(
        A: T.Tensor((13 * 32, 64), "float16"),
        B: T.Tensor((64, 32), "float16"),
        Fail: T.Tensor((1,), "int32"),
        C: T.Tensor((13 * 32, 32), "float16"),
    ):
        with T.Kernel(13, num_workers=6) as group:
            ar = T.alloc_shared((32, 64), "float16")
            br = T.alloc_shared((64, 32), "float16")
            an = T.alloc_shared((32, 64), "float16")
            bn = T.alloc_shared((64, 32), "float16")
            cn = T.alloc_shared((32, 32), "float16")
            config = T.alloc_shared((64,), "uint32")
            acc = T.hmx.alloc_accumulator()
            cvt = T.hmx.alloc_convert_state()
            bias = T.hmx.alloc_bias_state()
            T.annotate_layout({an: e.activation_layout(an), bn: e.weight_layout(bn), cn: e.output_layout(cn)})
            for i in T.serial(32):
                config[i] = T.uint32(0x3C00)
                config[i + 32] = T.uint32(0)
            T.dma_copy(A[group * 32 : (group + 1) * 32, :], ar)
            T.dma_copy(B, br)
            T.dma_wait()
            T.copy(ar, an)
            T.copy(br, bn)
            e.acquire(acc)
            if Fail[0] != 0:
                # Deliberate FIFO overflow while holding HMX. The two accepted
                # transfers must finish, and unit/thread leases must be freed.
                T.dma_copy(A[group * 32 : (group + 1) * 32, :], ar)
                T.dma_copy(B, br)
                T.dma_copy(B, br)
            e.load_bias(bias, config)
            e.clear(acc)
            for k in T.serial(2):
                e.mma_atom(acc, an, bn, 0, 0, k)
            e.convert(cvt, acc, bias, config)
            e.store(cvt, acc, cn, bias, config, 0, 0)
            e.release(acc)
            T.copy(cn, C[group * 32 : (group + 1) * 32, :])

    return dma_hmx_workers


def test_worker_hmx_thread_cleanup_survives_dma_errors():
    source = lower(make_hmx_probe())
    # The worker scope holds the thread's HMX enable for all of its programs
    # and drops it on any return; the DMA queue releases the HMX unit lock.
    assert "tl_parallel_for<tl_hmx_worker_scope>(13, 6, " in source
    assert "tl_dma_async.release_hmx = tl_hmx_unit_release;" in source
    assert "tl_hmx_enable" not in source
    assert "tl_hmx_disable" not in source


def test_async_scalar_access_is_not_silently_synchronous():
    @T.prim_func
    def probe(A: T.Tensor((256,), "float16"), B: T.Tensor((256,), "float16")):
        with T.Kernel(1):
            s = T.alloc_shared((256,), "float16")
            T.dma_copy(A[0], s[0])
            T.dma_wait()
            T.copy(s[0], B[0])

    source = lower(probe)
    (call,) = [line for line in source.splitlines() if "tl_hexagon_dma_async_submit(" in line]
    assert "(uint32_t)2, (uint32_t)2, (uint32_t)2, (uint32_t)1, 3)" in call


@pytest.mark.parametrize("shape,dtype", [((2, 8, 8), "float16"), ((256,), "float32")])
def test_reject_unsupported_region_or_dtype(shape, dtype):
    @T.prim_func
    def probe(A: T.Tensor(shape, dtype)):
        with T.Kernel(1):
            s = T.alloc_shared(shape, "float16")
            T.dma_copy(A, s)
            T.dma_wait()

    with pytest.raises(Exception, match="Hexagon T.dma_copy"):
        lower(probe)


def test_reject_mixed_raw_dma_ownership():
    @T.prim_func
    def probe(A: T.Tensor((256,), "float16")):
        with T.Kernel(1):
            s = T.alloc_shared((256,), "float16")
            T.dma_copy(A, s)
            T.call_extern("uint32", "tl_hexagon_dma_poll")
            T.dma_wait()

    with pytest.raises(Exception, match="raw DMA calls"):
        lower(probe)


def test_async_copy_is_separate_from_dma_copy():
    @T.prim_func
    def probe(A: T.Tensor((256,), "float16")):
        with T.Kernel(1):
            s = T.alloc_shared((256,), "float16")
            T.async_copy(A, s)

    assert "T.async_copy(" in probe.script()
    with pytest.raises(Exception, match="requires T.dma_copy instead of T.async_copy"):
        lower(probe)


def test_dma_copy_rejects_cpu_target():
    target = determine_target("llvm", return_object=True)
    with tvm.target.Target(target), pytest.raises(Exception, match="T.dma_copy is supported only on Hexagon"):
        tilelang.lower(make_queue_probe(), target=target, enable_host_codegen=False, enable_device_compile=False)
