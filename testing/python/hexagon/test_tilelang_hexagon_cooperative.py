"""Resource checkpoints keep a complete grid resident, including one worker."""

import re

import pytest

import tilelang
from tilelang import tvm
from tilelang.hexagon import language as T


def _lower(func, arch="v73"):
    target = tvm.target.Target({"kind": "hexagon", "mcpu": f"hexagon{arch}", "vtcm-capacity": 8 << 20})
    with target, tvm.transform.PassContext(opt_level=3):
        return tilelang.lower(func, target=target, enable_host_codegen=False, enable_device_compile=False)


def _exchange(workers, cooperative=True):
    @T.prim_func
    def exchange(A: T.Tensor((workers, 128), "float32"), B: T.Tensor((workers, 128), "float32")):
        with T.Kernel(workers, num_workers=workers, cooperative=cooperative) as w:
            shared = T.alloc_global((workers, 128), "float32", scope="global.vtcm")
            for j in T.Parallel(128):
                shared[w, j] = A[w, j]
            T.sync_grid()
            for j in T.Parallel(128):
                B[w, j] = shared[(w + 1) % workers, j]

    return exchange


@pytest.mark.parametrize("arch", ["v73", "v75", "v79"])
@pytest.mark.parametrize("workers", [1, 2, 4])
def test_cooperative_grid_keeps_all_programs(arch, workers):
    source = _lower(_exchange(workers), arch).kernel_source
    assert "#define TL_HEXAGON_USES_COOPERATIVE 1" in source
    assert f"tl_cooperative_parallel_for<false>({workers}, {workers}," in source
    assert "tl_cooperative_checkpoint(" in source
    assert "tl_grid_sync(" not in source


def test_ordinary_grid_keeps_its_existing_barrier():
    source = _lower(_exchange(2, cooperative=False)).kernel_source
    assert "tl_grid_sync(" in source
    assert "tl_cooperative" not in source


@pytest.mark.parametrize("enabled", [None, 0, 1, "true"])
def test_cooperative_keyword_requires_bool(enabled):
    with pytest.raises(ValueError, match="must be a bool"):
        T.Kernel(1, cooperative=enabled)


def test_cooperative_requires_explicit_grid():
    with pytest.raises(ValueError, match="requires an explicit grid"):
        T.Kernel(cooperative=True)


def test_cooperative_requires_checkpoint():
    @T.prim_func
    def copy(A: T.Tensor((32,), "float32"), B: T.Tensor((32,), "float32")):
        with T.Kernel(1, cooperative=True):
            for j in T.Parallel(32):
                B[j] = A[j]

    with pytest.raises(Exception, match=r"requires T.sync_grid\(\) checkpoints"):
        _lower(copy)


def test_cooperative_cannot_batch_programs():
    @T.prim_func
    def grid(A: T.Tensor((4,), "int32")):
        with T.Kernel(4, num_workers=2, cooperative=True) as w:
            A[w] = w
            T.sync_grid()

    with pytest.raises(Exception, match="one per worker"):
        _lower(grid)


def test_cooperative_rejects_worker_dependent_checkpoint():
    @T.prim_func
    def divergent(A: T.Tensor((4,), "int32")):
        with T.Kernel(4, num_workers=4, cooperative=True) as w:
            if w < 2:
                A[w] = w
                T.sync_grid()

    with pytest.raises(Exception, match="uniform conditions"):
        _lower(divergent)


def test_cooperative_rejects_worker_dependent_checkpoint_loop():
    @T.prim_func
    def divergent(A: T.Tensor((4,), "int32")):
        with T.Kernel(4, num_workers=4, cooperative=True) as w:
            for i in T.serial(w + 1):
                A[w] = i
                T.sync_grid()

    with pytest.raises(Exception, match="uniform loop bounds"):
        _lower(divergent)


@pytest.mark.parametrize("arch", ["v73", "v75", "v79"])
def test_dma_context_has_checked_cleanup_before_its_destructor(arch):
    @T.prim_func
    def load(A: T.Tensor((2, 128), "float32"), B: T.Tensor((2, 128), "float32")):
        with T.Kernel(2, num_workers=2, cooperative=True) as w:
            tile = T.alloc_shared((128,), "float32")
            T.dma_copy(A[w, :], tile)
            T.sync_grid()
            T.dma_wait(0)
            for j in T.Parallel(128):
                B[w, j] = tile[j]

    source = _lower(load, arch).kernel_source
    assert "tl_cooperative_dma_scope<decltype(tl_dma_async)>" in source
    assert "tl_cooperative_checkpoint(bx, tl_dma_async)" in source
    assert source.index("> tl_dma_async;") < source.index("tl_cooperative_dma_scope<")


def test_cooperative_accepts_common_scalar_loop_bound():
    @T.prim_func
    def repeated(A: T.Tensor((4,), "int32"), n: T.int32):
        with T.Kernel(4, num_workers=4, cooperative=True) as w:
            for i in T.serial(n):
                A[w] = i
                T.sync_grid()

    source = _lower(repeated).kernel_source
    assert "tl_cooperative_checkpoint(" in source
    assert "for (int32_t i" in source


def test_cooperative_rejects_buffer_loaded_checkpoint_condition():
    @T.prim_func
    def conditional(A: T.Tensor((4,), "int32")):
        with T.Kernel(4, num_workers=4, cooperative=True) as w:
            if A[0] > 0:
                A[w] = w
                T.sync_grid()

    with pytest.raises(Exception, match="uniform conditions"):
        _lower(conditional)


def test_cooperative_rejects_call_dependent_checkpoint_condition():
    @T.prim_func
    def conditional(A: T.Tensor((4,), "int32")):
        with T.Kernel(4, num_workers=4, cooperative=True) as w:
            if T.call_extern("int32", "worker_predicate") > 0:
                A[w] = w
                T.sync_grid()

    with pytest.raises(Exception, match="uniform conditions"):
        _lower(conditional)


def test_cooperative_rejects_while_checkpoints():
    @T.prim_func
    def repeated(A: T.Tensor((4,), "int32")):
        with T.Kernel(4, num_workers=4, cooperative=True) as w:
            i = T.alloc_local((1,), "int32")
            i[0] = 0
            while i[0] < 3:
                A[w] = i[0]
                T.sync_grid()
                i[0] += 1

    with pytest.raises(Exception, match="bounded for loops"):
        _lower(repeated)


@pytest.mark.parametrize("arch", ["v73", "v75", "v79"])
def test_cooperative_publishes_exclusive_live_scratch(arch):
    @T.prim_func
    def scratch(A: T.Tensor((2, 128), "float32"), B: T.Tensor((2, 128), "float32")):
        with T.Kernel(2, num_workers=2, cooperative=True) as w:
            left = T.alloc_shared((128,), "float32")
            right = T.alloc_shared((128,), "float32")
            for j in T.Parallel(128):
                left[j] = A[w, j]
                right[j] = A[w, j] + T.float32(1)
            T.sync_grid()
            for j in T.Parallel(128):
                B[w, j] = left[j] + right[j]

    source = _lower(scratch, arch).kernel_source
    worker = source[source.index("tl_cooperative_parallel_for<") :]
    bindings = re.findall(r"^\s*[^;=]+\b(\w+)\s*=\s*[^;]*\btl_exclusive\([^;]*;", worker, re.MULTILINE)
    scratch_bindings = [name for name in bindings if name.startswith(("left", "right"))]
    assert len(scratch_bindings) == 2, source
    for name in scratch_bindings:
        declaration = re.search(rf"^\s*[^;=]+\b{re.escape(name)}\s*=\s*[^;]*tl_exclusive\([^;]*;", worker, re.MULTILINE)
        assert declaration is not None
        following = worker[declaration.end() :].lstrip()
        assert following.startswith(f"tl_escape({name});"), source
