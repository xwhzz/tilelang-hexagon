"""The Hexagon launch grid in TIR: a serial loop nest, or one worker-pool loop."""

import os

import pytest

import tilelang
from tilelang import tvm
from tilelang.backend.target import determine_target
from tilelang.hexagon import language as T


def _lower(func):
    target = determine_target("hexagon", return_object=True)
    with tvm.target.Target(target), tvm.transform.PassContext(opt_level=3):
        return tilelang.lower(func, target=target, enable_host_codegen=False, enable_device_compile=False)


def _nodes(stmt, kind):
    found = []
    tvm.tirx.stmt_functor.post_order_visit(stmt, lambda node: found.append(node) if isinstance(node, kind) else None)
    return found


def _device_func(artifact):
    (func,) = artifact.device_mod.functions.values()
    return func


def _parallel_loops(func):
    return [loop for loop in _nodes(func.body, tvm.tirx.For) if loop.kind == tvm.tirx.ForKind.PARALLEL]


def _scale(M, N, num_workers=None):
    @T.prim_func
    def scale(A: T.Tensor((M, N, 64), "float32"), B: T.Tensor((M, N, 64), "float32")):
        with T.Kernel(M, N, num_workers=num_workers) as (bx, by):
            tile = T.alloc_shared((64,), "float32")
            factor = T.alloc_local((1,), "float32")
            factor[0] = T.float32(2)
            T.copy(A[bx, by, :], tile)
            for j in T.Parallel(64):
                B[bx, by, j] = tile[j] * factor[0]

    return scale


def test_serial_grid_is_a_loop_nest():
    artifact = _lower(_scale(4, 8))
    func = _device_func(artifact)
    assert not _parallel_loops(func)
    extents = {loop.loop_var.name: int(loop.extent) for loop in _nodes(func.body, tvm.tirx.For)}
    assert (extents["bx"], extents["by"]) == (4, 8)
    source = artifact.kernel_source
    assert "for (int32_t bx = 0; bx < 4; ++bx) {" in source
    assert "for (int32_t by = 0; by < 8; ++by) {" in source
    assert "tl_parallel_for" not in source


def test_workers_run_the_linearized_grid():
    artifact = _lower(_scale(4, 8, num_workers=6))
    (loop,) = _parallel_loops(_device_func(artifact))
    assert int(loop.extent) == 32
    assert int(loop.annotations["hexagon.num_workers"]) == 6
    assert "tl_parallel_for(32, 6, tl_vtcm_worker_arenas(256u), [&](int32_t bx_by_fused, uint8_t *tl_arena) -> int32_t {" in (
        artifact.kernel_source
    )


def test_programs_own_their_buffers():
    func = _device_func(_lower(_scale(4, 8, num_workers=6)))
    (loop,) = _parallel_loops(func)
    # StorageRewrite hoists allocations out of the grid; each worker must get
    # its own copies, so they are back inside the parallel loop.
    allocations = _nodes(func.body, tvm.tirx.AllocBuffer)
    assert allocations
    assert len(_nodes(loop.body, tvm.tirx.AllocBuffer)) == len(allocations)


def test_a_vtcm_free_worker_takes_no_arena():
    @T.prim_func
    def negate(A: T.Tensor((12, 256), "float32"), B: T.Tensor((12, 256), "float32")):
        with T.Kernel(12, num_workers=6) as bx:
            for j in T.serial(256):
                B[bx, j] = -A[bx, j]

    source = _lower(negate).kernel_source
    assert "tl_parallel_for(12, 6, [&](int32_t bx) -> int32_t {" in source
    assert "tl_vtcm" not in source


@pytest.mark.parametrize("grid,num_workers", [((4, 8), 1), ((1, 1), 4)])
def test_one_worker_or_one_program_stays_serial(grid, num_workers):
    artifact = _lower(_scale(*grid, num_workers=num_workers))
    assert not _parallel_loops(_device_func(artifact))
    assert "tl_parallel_for" not in artifact.kernel_source


def test_grid_axis_below_a_guard_stays_serial_in_the_workers():
    @T.prim_func
    def guarded(A: T.Tensor((4, 8, 64), "float32"), B: T.Tensor((4, 8, 64), "float32"), n: T.int32):
        with T.Kernel(4, 8, num_workers=6) as (bx, by):
            if bx < n:
                for j in T.Parallel(64):
                    B[bx, by, j] = A[bx, by, j] + T.float32(1)

    func = _device_func(_lower(guarded))
    (loop,) = _parallel_loops(func)
    # The guard on bx is hoisted between the grid loops, so only bx is
    # distributed and by stays a serial loop inside each worker.
    assert loop.loop_var.name == "bx"
    assert [inner.loop_var.name for inner in _nodes(loop.body, tvm.tirx.For) if inner.loop_var.name == "by"] == ["by"]


def test_kernel_has_no_thread_index():
    @T.prim_func
    def per_thread(A: T.Tensor((64,), "float32")):
        with T.Kernel(1):
            tx = T.get_thread_binding()
            A[tx] = T.float32(1)

    with pytest.raises(Exception, match="has no SIMT threads"):
        _lower(per_thread)


def test_kernel_has_no_threads_keyword():
    with pytest.raises(TypeError, match="threads"):
        T.Kernel(1, threads=1)


def test_one_kernel_per_prim_func():
    @T.prim_func
    def two_launches(A: T.Tensor((64,), "float32")):
        with T.Kernel(1):
            for j in T.Parallel(64):
                A[j] = T.float32(1)
        with T.Kernel(1):
            for j in T.Parallel(64):
                A[j] = T.float32(2)

    with pytest.raises(Exception, match="one T.Kernel per prim_func"):
        _lower(two_launches)


def _rotate(programs, rounds, num_workers):
    """Each round, every program adds one to its right neighbour's row of the
    previous round: the rows rotate left once per round."""

    @T.prim_func
    def rotate(
        A: T.Tensor((programs, 64), "float32"), rows: T.Tensor((2, programs, 64), "float32"), B: T.Tensor((programs, 64), "float32")
    ):
        with T.Kernel(programs, num_workers=num_workers) as w:
            for j in T.Parallel(64):
                rows[0, w, j] = A[w, j]
            for r in T.serial(rounds):
                T.sync_grid()
                for j in T.Parallel(64):
                    rows[(r + 1) % 2, w, j] = rows[r % 2, (w + 1) % programs, j] + T.float32(1)
            for j in T.Parallel(64):
                B[w, j] = rows[rounds % 2, w, j]

    return rotate


def test_programs_meet_at_sync_grid():
    source = _lower(_rotate(4, 3, 4)).kernel_source
    # The programs run coresident, one per worker, and wait at the barrier
    # with their own program index.
    assert "tl_parallel_for<tl_worker_scope, true>(4, 4, [&](int32_t bx) -> int32_t {" in source
    assert source.count("if (int32_t tl_status = tl_grid_sync(bx)) return tl_status;") == 1


def test_sync_grid_needs_a_worker_per_program():
    with pytest.raises(Exception, match="the grid has 8 programs and num_workers=6"):
        _lower(_rotate(8, 3, 6))


def test_sync_grid_in_one_program_has_nothing_to_wait_for():
    source = _lower(_rotate(1, 3, None)).kernel_source
    assert "tl_grid_sync" not in source
    assert "tl_parallel_for" not in source


@pytest.mark.skipif(os.environ.get("TL_HEXAGON_DEVICE_TEST") != "1", reason="requires an authorized Hexagon device and SDK")
@pytest.mark.parametrize("programs", [2, 6])
def test_programs_exchange_rows_across_sync_grid(programs):
    import torch

    rounds = 61
    kernel = tilelang.compile(_rotate(programs, rounds, programs), out_idx=[2], target="hexagon")
    try:
        a = torch.randn(programs, 64)
        rows = torch.zeros(2, programs, 64)
        expected = torch.roll(a, -rounds, dims=0) + rounds
        torch.testing.assert_close(kernel(a, rows), expected)
    finally:
        kernel.adapter.close()


def _prefetch(programs, n):
    """Each program submits the DMA of its next-phase data before the barrier
    and waits for it after: the transfer overlaps the phase change."""

    @T.prim_func
    def prefetch(A: T.Tensor((programs, n), "float32"), rows: T.Tensor((programs, 64), "float32"), B: T.Tensor((programs, n), "float32")):
        with T.Kernel(programs, num_workers=programs) as w:
            a = T.alloc_shared((n,), "float32")
            T.dma_copy(A[w, :], a)
            for j in T.Parallel(64):
                rows[w, j] = T.Cast("float32", w)
            T.sync_grid()
            T.dma_wait(0)
            for j in T.Parallel(n):
                B[w, j] = a[j] + rows[(w + 1) % programs, j % 64]

    return prefetch


@pytest.mark.skipif(os.environ.get("TL_HEXAGON_DEVICE_TEST") != "1", reason="requires an authorized Hexagon device and SDK")
def test_dma_crosses_sync_grid():
    import torch

    programs, n = 6, 4096
    kernel = tilelang.compile(_prefetch(programs, n), out_idx=[2], target="hexagon")
    try:
        a = torch.randn(programs, n)
        expected = a + ((torch.arange(programs) + 1) % programs).float()[:, None]
        torch.testing.assert_close(kernel(a, torch.zeros(programs, 64)), expected)
    finally:
        kernel.adapter.close()


@pytest.mark.skipif(os.environ.get("TL_HEXAGON_DEVICE_TEST") != "1", reason="requires an authorized Hexagon device and SDK")
def test_hmx_workers_run_on_the_device():
    import torch

    blocks, size = 6, 64

    @T.prim_func
    def batched(
        A: T.Tensor((blocks, size, size), "float16"),
        B: T.Tensor((blocks, size, size), "float16"),
        C: T.Tensor((blocks, size, size), "float16"),
    ):
        # HMX in workers, without DMA: the worker scope's destructor must not
        # pull the C++ unwinder into the skel.
        with T.Kernel(blocks, num_workers=6) as bx:
            A_hmx = T.alloc_shared((size, size), "float16")
            B_hmx = T.alloc_shared((size, size), "float16")
            C_hmx = T.alloc_shared((size, size), "float16")
            T.copy(A[bx, :, :], A_hmx)
            T.copy(B[bx, :, :], B_hmx)
            T.gemm(A_hmx, B_hmx, C_hmx, clear_accum=True)
            T.copy(C_hmx, C[bx, :, :])

    kernel = tilelang.compile(batched, out_idx=[2], target="hexagon")
    try:
        torch.manual_seed(0)
        a = torch.randn(blocks, size, size).half()
        b = torch.randn(blocks, size, size).half()
        torch.testing.assert_close(kernel(a, b).float(), torch.bmm(a.float(), b.float()), rtol=2e-2, atol=2e-2)
    finally:
        kernel.adapter.close()
