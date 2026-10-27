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
