"""The Hexagon launch grid in TIR: a serial loop nest, or one worker-pool loop."""

import os

import pytest

import tilelang
from tilelang import tvm
from tilelang.backend.target import determine_target
from tilelang.hexagon import language as T


def _lower(func, target="hexagon"):
    target = determine_target(target, return_object=True)
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
    assert (
        "tl_parallel_for(32, 6, tl_vtcm_worker_arenas(256u), [&](int32_t bx_by_fused, uint8_t *tl_arena) "
        "TL_HEXAGON_IEEE_ARITHMETIC -> int32_t {"
    ) in artifact.kernel_source


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
    assert "tl_parallel_for(12, 6, [&](int32_t bx) TL_HEXAGON_IEEE_ARITHMETIC -> int32_t {" in source
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


@pytest.mark.parametrize("requested", [0, -1, True, 1.5, 2**31, 2**32, 2**32 + 1])
def test_worker_request_must_fit_positive_int32(requested):
    with pytest.raises(ValueError, match="positive int no greater than 2147483647"):
        T.Kernel(2, num_workers=requested)


@pytest.mark.parametrize("requested", [0, -1, 2**31, 2**32, 2**32 + 1])
def test_native_worker_annotation_rejects_out_of_range_requests(requested):
    @T.prim_func
    def kernel(A: T.Tensor((2,), "int32")):
        with T.Kernel(2) as p:
            T.sblock_attr({"hexagon.num_workers": T.int64(requested)})
            A[p] = p

    with pytest.raises(Exception, match="positive int no greater than 2147483647"):
        _lower(kernel)


def test_largest_worker_request_remains_an_upper_bound():
    source = _lower(_scale(2, 1, num_workers=2**31 - 1)).kernel_source
    assert "tl_parallel_for(2, 2147483647," in source


@pytest.mark.parametrize(
    "grid,message",
    [((65536, 32768), "Hexagon worker grid.*2147483647"), ((2**31, 1), "Literal value 2147483648 exceeds maximum of int32")],
)
@pytest.mark.parametrize("sync", [False, True])
def test_constant_worker_grid_cannot_overflow_the_runtime_extent(grid, message, sync):
    @T.prim_func
    def kernel(A: T.Tensor((1,), "int32")):
        with T.Kernel(*grid, num_workers=2**31 - 1) as (x, y):
            A[0] = T.cast(x + y, "int32")
            if sync:
                T.sync_grid()

    with pytest.raises(Exception, match=message):
        _lower(kernel)


def test_materialize_grid_rejects_a_wide_native_extent():
    from tilelang.hexagon.transform import MaterializeGrid

    tir = tvm.tirx
    loop = tir.For(
        tir.Var("p", "int64"),
        tir.IntImm("int64", 0),
        tir.IntImm("int64", 2**31),
        tir.ForKind.SERIAL,
        tir.Evaluate(0),
        annotations={"hexagon.grid_dim": 0, "hexagon.num_workers": 8},
    )
    module = tvm.IRModule({"main": tir.PrimFunc([], loop)})
    with pytest.raises(Exception, match="Hexagon worker grid extents must be in"):
        MaterializeGrid()(module)


@pytest.mark.parametrize("extent", [0, 2**31 - 1])
def test_worker_grid_accepts_representable_extent_boundaries(extent):
    @T.prim_func
    def kernel(A: T.Tensor((1,), "int32")):
        with T.Kernel(extent, num_workers=8) as p:
            A[0] = p

    source = _lower(kernel).kernel_source
    if extent:
        assert "tl_parallel_for(2147483647, 8," in source
    else:
        assert "tl_parallel_for" not in source


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


def _double(n):
    """B = 2 A, past a buffer and a scalar the kernel never reads."""

    @T.prim_func
    def double(A: T.Tensor((64,), "float32"), unused: T.Tensor((n,), "int8"), count: T.int32, B: T.Tensor((64,), "float32")):
        with T.Kernel(1):
            for j in T.Parallel(64):
                B[j] = A[j] * T.float32(2)

    return double


def test_kernel_takes_every_parameter_in_order():
    # The FastRPC skel calls the kernel with every parameter of the PrimFunc.
    func = _device_func(_lower(_double(16)))
    assert [p.name for p in func.params] == ["A", "unused", "count", "B"]


@pytest.mark.skipif(os.environ.get("TL_HEXAGON_DEVICE_TEST") != "1", reason="requires an authorized Hexagon device and SDK")
def test_unused_parameters_keep_the_others_in_place():
    import torch

    kernel = tilelang.compile(_double(16), out_idx=[3], target="hexagon")
    try:
        a = torch.randn(64)
        torch.testing.assert_close(kernel(a, torch.zeros(16, dtype=torch.int8), 5), 2 * a)
    finally:
        kernel.adapter.close()


@T.prim_func
def _deep(A: T.Tensor((4096,), "float32"), B: T.Tensor((4096,), "float32")):
    """B = 4 A, through 32 KB of stack: more than the FastRPC thread, which runs
    the kernel's first program, had by default."""
    with T.Kernel(1):
        local = T.alloc_local((8192,), "float32")
        for i in T.serial(8192):
            local[i] = A[i % 4096] * T.float32(2)
        for i in T.serial(4096):
            B[i] = local[i] + local[i + 4096]


@pytest.mark.skipif(os.environ.get("TL_HEXAGON_DEVICE_TEST") != "1", reason="requires an authorized Hexagon device and SDK")
def test_a_kernel_has_a_deep_stack():
    import torch

    kernel = tilelang.compile(_deep, out_idx=[1], target="hexagon")
    try:
        a = torch.randn(4096)
        torch.testing.assert_close(kernel(a), 4 * a)
    finally:
        kernel.adapter.close()


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
    assert "tl_parallel_for<tl_worker_scope, true>(4, 4, [&](int32_t bx) TL_HEXAGON_IEEE_ARITHMETIC -> int32_t {" in source
    assert source.count("if (int32_t tl_status = tl_grid_sync(bx)) return tl_status;") == 1


def test_sync_grid_needs_a_worker_per_program():
    with pytest.raises(Exception, match="the grid has 8 programs and num_workers=6"):
        _lower(_rotate(8, 3, 6))


def test_sync_grid_in_one_program_has_nothing_to_wait_for():
    source = _lower(_rotate(1, 3, None)).kernel_source
    assert "tl_grid_sync" not in source
    assert "tl_parallel_for" not in source


@pytest.fixture(scope="module")
def device_worker_limit():
    if os.environ.get("TL_HEXAGON_DEVICE_TEST") != "1":
        pytest.skip("requires an authorized Hexagon device and SDK")
    from tilelang.engine.param import KernelParam
    from tilelang.hexagon.runtime.session import Session

    # Query the device independently of worker.h. Architecture names do not
    # specify the number of 128-byte HVX contexts on a particular SoC.
    source = """#include <stdint.h>
#include <qurt.h>
int32_t worker_limits(int32_t* out) {
  qurt_sysenv_max_hthreads_t h = {};
  out[0] = qurt_sysenv_get_max_hw_threads(&h);
  out[1] = h.max_hthreads;
  out[2] = qurt_hvx_get_units();
  out[3] = QURT_MAX_HTHREAD_LIMIT;
  return 0;
}
"""
    with Session.of_lowered([("worker_limits", source, [KernelParam("int32", [4])], [0])]) as session:
        status, hardware, units, capacity = session.kernels[0]().tolist()
    contexts = (units >> 8) & 255 if units > 0 else 0
    assert capacity > 0
    return min(capacity, hardware, contexts) if status == 0 and hardware > 0 and contexts > 0 else 1


def _check_capped_grid(func, arguments, out_idx, expected, programs, worker_limit):
    import torch
    from tilelang.hexagon.runtime.session import Session, _KernelError

    # Resident arguments make every write observable even when the kernel
    # returns an error. Keep the original numeric assertion on supported grids.
    initial = [value.contiguous().numpy().tobytes() for value in arguments]
    guard = bytes([0xA5]) * 128
    with Session([(func, [out_idx])]) as session:
        buffers = [session.buffer(len(raw) + len(guard)) for raw in initial]
        for buffer, raw in zip(buffers, initial):
            buffer.write(raw + guard)
        if programs > worker_limit:
            with pytest.raises(_KernelError, match=r"^Hexagon session: .* failed: TL_ERR_GRID \(too few workers for the grid\)$"):
                session.kernels[0](*buffers)
            # No callback should start: output, scratch rows and input stay
            # byte-identical, including the extra bytes outside each parameter.
            assert [buffer.read() for buffer in buffers] == [raw + guard for raw in initial]
        else:
            session.kernels[0](*buffers)
            raw = [buffer.read() for buffer in buffers]
            assert all(value[-128:] == guard for value in raw)
            assert raw[0][:-128] == initial[0]
            result = torch.frombuffer(bytearray(raw[out_idx][:-128]), dtype=expected.dtype).reshape(expected.shape)
            torch.testing.assert_close(result, expected)


def _budgeted_worker_copy(sync):
    programs = 2 if sync else 8

    @T.prim_func
    def copy(A: T.Tensor((programs, 32768), "float32"), B: T.Tensor((programs, 32768), "float32")):
        with T.Kernel(programs, num_workers=8) as p:
            s = T.alloc_shared((32768,), "float32")  # 128 KiB per arena.
            T.copy(A[p, :], s)
            if sync:
                T.sync_grid()
            T.copy(s, B[p, :])

    return copy


@pytest.mark.parametrize("sync", [False, True])
def test_budgeted_worker_copy_uses_the_requested_upper_bound(sync):
    source = _lower(_budgeted_worker_copy(sync), {"kind": "hexagon", "mcpu": "hexagonv73", "vtcm-capacity": 512 << 10}).kernel_source
    assert f"({2 if sync else 8}, 8, tl_vtcm_worker_arenas(131072u)," in source
    assert ("tl_parallel_for<tl_worker_scope, true>" in source) == sync


@pytest.mark.skipif(os.environ.get("TL_HEXAGON_DEVICE_TEST") != "1", reason="requires an authorized Hexagon device and SDK")
@pytest.mark.parametrize("sync", [False, True])
def test_worker_request_can_exceed_the_arena_budget_on_device(sync, device_worker_limit):
    import torch
    from tilelang.hexagon.runtime.session import Session

    if sync and device_worker_limit < 2:
        pytest.skip("requires capacity for two simultaneous workers")
    programs = 2 if sync else 8
    data = (torch.arange(programs * 32768, dtype=torch.int32) % 257 - 128).float().reshape(programs, 32768)
    initial = data.numpy().tobytes()
    guard = bytes([0xA5]) * 128
    # The target budget fits four private arenas, while the request is eight.
    # Independent programs may batch; the synchronous grid needs just two.
    target = {"kind": "hexagon", "mcpu": f"hexagon{os.environ.get('HEXAGON_DSP_ARCH', 'v79')}", "vtcm-capacity": 512 << 10}
    with Session([(_budgeted_worker_copy(sync), [1])], target=target) as session:
        source = session.buffer(len(initial) + 128)
        output = session.buffer(len(initial) + 128)
        source.write(initial + guard)
        output.write(bytes([0xA5]) * (len(initial) + 128))
        session.kernels[0](source, output)
        assert source.read() == initial + guard
        assert output.read() == initial + guard


@pytest.mark.skipif(os.environ.get("TL_HEXAGON_DEVICE_TEST") != "1", reason="requires an authorized Hexagon device and SDK")
@pytest.mark.parametrize("programs,num_workers", [(2, 2), (4, 4), (4, 6), (4, 8), (6, 6), (8, 8)])
def test_programs_exchange_rows_across_sync_grid(programs, num_workers, device_worker_limit):
    import torch

    rounds = 61
    a = torch.randn(programs, 64)
    rows = torch.zeros(2, programs, 64)
    expected = torch.roll(a, -rounds, dims=0) + rounds
    _check_capped_grid(
        _rotate(programs, rounds, num_workers),
        [a, rows, torch.full_like(a, -123)],
        2,
        expected,
        programs,
        device_worker_limit,
    )


def _grid_rotate(programs, rounds):
    """_rotate with the rows in VTCM the programs share (global.vtcm)."""

    @T.prim_func
    def rotate(A: T.Tensor((programs, 64), "float32"), B: T.Tensor((programs, 64), "float32")):
        with T.Kernel(programs, num_workers=programs) as w:
            rows = T.alloc_global((2, programs, 64), "float32", scope="global.vtcm")
            for j in T.Parallel(64):
                rows[0, w, j] = A[w, j]
            for r in T.serial(rounds):
                T.sync_grid()
                for j in T.Parallel(64):
                    rows[(r + 1) % 2, w, j] = rows[r % 2, (w + 1) % programs, j] + T.float32(1)
            for j in T.Parallel(64):
                B[w, j] = rows[rounds % 2, w, j]

    return rotate


def test_programs_share_global_vtcm_buffers():
    source = _lower(_grid_rotate(4, 3)).kernel_source
    # One buffer at the VTCM base, bound outside the programs, which capture it.
    head = source[: source.index("tl_parallel_for")]
    assert "float* rows = (float*)tl_exclusive(tl_vtcm_arena() + 0);" in head
    assert "tl_vtcm_worker_arenas" not in source


def _grid_gemm(programs, M, N, K):
    """C[p] = A @ B[p] on HMX, A in VTCM the programs share: each program
    copies its rows of A in, and after the barrier multiplies all of them."""
    rows = M // programs

    @T.prim_func
    def gemm(A: T.Tensor((M, K), "float16"), B: T.Tensor((programs, K, N), "float16"), C: T.Tensor((programs, M, N), "float32")):
        with T.Kernel(programs, num_workers=programs) as p:
            a = T.alloc_global((M, K), "float16", scope="global.vtcm")
            b = T.alloc_shared((K, N), "float16")
            c = T.alloc_shared((M, N), "float32")
            T.copy(A[p * rows : (p + 1) * rows, :], a[p * rows : (p + 1) * rows, :])
            T.copy(B[p, :, :], b)
            T.sync_grid()
            T.gemm(a, b, c, clear_accum=True)
            T.copy(c, C[p, :, :])

    return gemm


def test_global_vtcm_buffers_come_before_the_program_arenas():
    source = _lower(_grid_gemm(4, 128, 64, 256)).kernel_source
    head = source[: source.index("tl_parallel_for")]
    # A, 64 KB, then the programs' arenas after it.
    assert "half* a = (half*)tl_exclusive(tl_vtcm_arena() + 0);" in head
    assert "tl_vtcm_worker_arenas(" in source and ", 65536u), [&](int32_t bx, uint8_t *tl_arena)" in source
    # A program takes the buffer and the arguments it reaches as objects of
    # their own.
    assert "half* a_w = (half*)tl_exclusive(a);" in source and "tl_escape(a_w);" in source
    assert "half* A_w = (half*)tl_exclusive(A);" in source
    # Each program packs its rows of the shared HMX operand.
    assert "tl_crouton_copy<true, false>((&(a_w[0])), 4, 8, (bx * 32), 0, 32, 256, (&(A_w[(bx * 8192)])), 256, 1);" in source
    # All of them multiply all of it, taking turns at HMX.
    assert "tl_hexagon_hmx_gemm<true, true, true, false>(128, 64, 256, (&(((float*)c)[0])), 64, (&(a_w[0])), 8192, " in source


@pytest.mark.parametrize("arch", ["v73", "v79"])
def test_global_vtcm_buffers_must_fit_with_the_arenas(arch):
    @T.prim_func
    def big(A: T.Tensor((4, 64), "float32"), B: T.Tensor((4, 64), "float32")):
        with T.Kernel(4, num_workers=4) as w:
            g = T.alloc_global((1024, 1024, 2), "float32", scope="global.vtcm")
            s = T.alloc_shared((64,), "float32")
            for j in T.Parallel(64):
                s[j] = A[w, j]
            for j in T.Parallel(64):
                g[w, j, 0] = s[j]
            T.sync_grid()
            for j in T.Parallel(64):
                B[w, j] = g[(w + 1) % 4, j, 0]

    with pytest.raises(Exception, match="leave after its 8388608 bytes of global.vtcm buffers"):
        _lower(big, {"kind": "hexagon", "mcpu": f"hexagon{arch}", "vtcm-capacity": 8 << 20})


@pytest.mark.skipif(os.environ.get("TL_HEXAGON_DEVICE_TEST") != "1", reason="requires an authorized Hexagon device and SDK")
@pytest.mark.parametrize("programs", [4, 6])
def test_programs_exchange_rows_through_global_vtcm(programs, device_worker_limit):
    import torch

    rounds = 61
    a = torch.randn(programs, 64)
    _check_capped_grid(
        _grid_rotate(programs, rounds),
        [a, torch.full_like(a, -123)],
        1,
        torch.roll(a, -rounds, dims=0) + rounds,
        programs,
        device_worker_limit,
    )


@pytest.mark.skipif(os.environ.get("TL_HEXAGON_DEVICE_TEST") != "1", reason="requires an authorized Hexagon device and SDK")
def test_hmx_workers_multiply_a_shared_operand():
    import torch

    programs, M, N, K = 4, 128, 64, 256
    kernel = tilelang.compile(_grid_gemm(programs, M, N, K), out_idx=[2], target="hexagon")
    try:
        a = torch.randn(M, K).half()
        b = torch.randn(programs, K, N).half()
        expected = a.float() @ b.float()
        torch.testing.assert_close(kernel(a, b), expected, rtol=2e-3, atol=2e-2)
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
@pytest.mark.parametrize("programs", [4, 6])
def test_dma_crosses_sync_grid(programs, device_worker_limit):
    import torch

    n = 4096
    a = torch.randn(programs, n)
    expected = a + ((torch.arange(programs) + 1) % programs).float()[:, None]
    _check_capped_grid(
        _prefetch(programs, n), [a, torch.zeros(programs, 64), torch.full_like(a, -123)], 2, expected, programs, device_worker_limit
    )


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
