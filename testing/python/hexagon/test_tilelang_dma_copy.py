"""Hexagon async submission, completion, and VTCM allocation contracts."""
import importlib.util
from pathlib import Path
import re

import pytest
import tilelang
import tilelang.language as T
from tilelang import tvm
from tilelang.utils.target import determine_target
from tilelang.hexagon.hmx_intrin import HMXIntrinEmitter

ROOT = Path(__file__).resolve().parents[3]


def lower(func):
    target = determine_target("hexagon", return_object=True)
    with tvm.target.Target(target):
        return str(tilelang.lower(func, target=target, enable_host_codegen=False,
                                  enable_device_compile=False).kernel_source)


def example(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'examples/hexagon' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_async_manual_ping_pong():
    @T.prim_func
    def probe(A: T.Tensor((768,), 'float16')):
        T.func_attr({'hexagon.dma_queue_capacity': 2})
        with T.Kernel(1, threads=1):
            s = T.alloc_shared((2, 256), 'float16')
            T.dma_copy(A[0:256], s[0, 0:256])
            T.dma_copy(A[256:512], s[1, 0:256])
            T.dma_wait(1)
            T.dma_copy(A[512:768], s[0, 0:256])
            T.dma_wait()
    source = lower(probe)
    assert 'tl_hexagon_dma_async_context<2>' in source
    assert source.count('tl_dma_async.copy_1d(') == 3
    assert 'tl_dma_async.wait(1)' in source
    assert source.rindex('tl_dma_async.wait(0)') > source.rindex('tl_dma_async.copy_1d(')
    assert 'ptx_' not in source


def test_async_strided_load_store():
    @T.prim_func
    def probe(A: T.Tensor((32, 128), 'float16'), B: T.Tensor((32, 128), 'float16')):
        with T.Kernel(1, threads=1):
            s = T.alloc_shared((32, 64), 'float16')
            T.dma_copy(A[0:32, 32:96], s)
            T.dma_wait()
            T.dma_copy(s, B[0:32, 32:96])
    source = lower(probe)
    calls = [line for line in source.splitlines() if 'tl_dma_async.copy_2d(' in line]
    assert len(calls) == 2
    assert '(uint32_t)128, (uint32_t)256, (uint32_t)128, (uint32_t)32, 0' in calls[0]
    assert '(uint32_t)256, (uint32_t)128, (uint32_t)128, (uint32_t)32, 1' in calls[1]
    assert 'A[32]' in calls[0] and 'B[32]' in calls[1]
    assert source.count('tl_dma_async.wait(0)') == 2


def test_async_hmx_dsl_schedule():
    source = lower(example('example_dma_hmx_matmul').make_dma_hmx_matmul())
    assert 'tl_hexagon_dma_async_context<4>' in source
    assert 'tl_dma_async.wait(2)' in source
    assert 'tl_hexagon_hmx_pack_crouton' in source
    assert 'tl_hexagon_hmx_mma_atom' in source
    assert 'tl_hexagon_dma_copy_hmx_matmul_double_buffer' not in source


def make_capacity_probe(capacity, pending=0, workers=None):
    @T.prim_func
    def probe(A: T.Tensor((256,), 'float16'), B: T.Tensor((256,), 'float16')):
        T.func_attr({'hexagon.dma_queue_capacity': capacity})
        with T.Kernel(1, threads=1, num_workers=workers) as _:
            s = T.alloc_shared((256,), 'float16')
            T.dma_copy(A, s)
            T.dma_wait(pending)
            T.copy(s, B)
    return probe


@pytest.mark.parametrize('capacity', [0, -1, 257])
def test_reject_invalid_capacity(capacity):
    with pytest.raises(Exception, match='dma_queue_capacity'):
        lower(make_capacity_probe(capacity))


def test_reject_wait_above_capacity():
    with pytest.raises(Exception, match='within queue capacity'):
        lower(make_capacity_probe(2, pending=3))


@pytest.mark.parametrize('pending', [-1, True, 1.5])
def test_reject_invalid_wait_argument(pending):
    with pytest.raises(ValueError, match='nonnegative integer'):
        T.dma_wait(pending)


def test_reject_worker_pool():
    with pytest.raises(Exception, match='worker pools'):
        lower(make_capacity_probe(2, workers=2))


def test_reject_crouton_async():
    e = HMXIntrinEmitter(32, 32, 32)
    @T.prim_func
    def probe(A: T.Tensor((32, 32), 'float16')):
        with T.Kernel(1, threads=1):
            s = T.alloc_shared((32, 32), 'float16')
            T.annotate_layout({s: e.activation_layout(s)})
            T.dma_copy(A, s)
            T.dma_wait()
    with pytest.raises(Exception, match='no Crouton'):
        lower(probe)


def test_reject_out_of_bounds():
    @T.prim_func
    def probe(A: T.Tensor((256,), 'float16')):
        with T.Kernel(1, threads=1):
            s = T.alloc_shared((256,), 'float16')
            T.dma_copy(A[128:384], s)
            T.dma_wait()
    with pytest.raises(Exception, match='in-bounds'):
        lower(probe)


def test_reject_software_pipeline():
    @T.prim_func
    def probe(A: T.Tensor((4, 256), 'float16'), B: T.Tensor((4, 256), 'float16')):
        with T.Kernel(1, threads=1):
            s = T.alloc_shared((256,), 'float16')
            for i in T.Pipelined(4, num_stages=2):
                T.dma_copy(A[i, 0:256], s)
                T.dma_wait()
                T.copy(s, B[i, 0:256])
    with pytest.raises(Exception, match='manual schedule'):
        lower(probe)


def test_async_store_source_is_not_recycled_before_wait():
    @T.prim_func
    def probe(A: T.Tensor((256,), 'float16'), B: T.Tensor((256,), 'float16'),
              C: T.Tensor((256,), 'float16')):
        with T.Kernel(1, threads=1):
            s = T.alloc_shared((256,), 'float16')
            t = T.alloc_shared((256,), 'float16')
            T.copy(A, s)
            T.dma_copy(s, B)
            T.fill(t, 2)
            T.copy(t, C)
            T.dma_wait()
    source = lower(probe)
    offsets = re.findall(r'void\* [st] = .*buf_dyn_shmem \+ (\d+)', source)
    assert len(offsets) == 2, source
    assert offsets[0] != offsets[1], source


def test_normal_copy_does_not_create_dma_queue():
    @T.prim_func
    def probe(A: T.Tensor((256,), 'float16'), B: T.Tensor((256,), 'float16')):
        with T.Kernel(1, threads=1):
            s = T.alloc_shared((256,), 'float16')
            T.copy(A, s)
            T.copy(s, B)
    assert 'tl_dma_async' not in lower(probe)


def test_async_dsp_build_avoids_exception_runtime():
    from tilelang.hexagon import _fastrpc
    cmake = _fastrpc.gen_cmake('test_async', '/templates', async_dma=True)
    assert '-fno-exceptions' in cmake
    assert '-fno-exceptions' not in _fastrpc.gen_cmake('test_sync', '/templates')


def test_async_hmx_error_cleanup_is_armed_and_disarmed():
    source = lower(example('example_dma_hmx_matmul').make_dma_hmx_matmul())
    assert 'tl_dma_async.release_hmx = tl_hmx_unit_release;' in source
    assert 'tl_dma_async.release_hmx = nullptr;' in source


def test_async_scalar_access_is_not_silently_synchronous():
    @T.prim_func
    def probe(A: T.Tensor((256,), 'float16'), B: T.Tensor((256,), 'float16')):
        with T.Kernel(1, threads=1):
            s = T.alloc_shared((256,), 'float16')
            T.dma_copy(A[0], s[0])
            T.dma_wait()
            T.copy(s[0], B[0])
    source = lower(probe)
    assert 'tl_dma_async.copy_1d(' in source
    assert '(uint32_t)2, 0)' in source


@pytest.mark.parametrize('shape,dtype', [((2, 8, 8), 'float16'), ((256,), 'float32')])
def test_reject_unsupported_region_or_dtype(shape, dtype):
    @T.prim_func
    def probe(A: T.Tensor(shape, dtype)):
        with T.Kernel(1, threads=1):
            s = T.alloc_shared(shape, 'float16')
            T.dma_copy(A, s)
            T.dma_wait()
    with pytest.raises(Exception, match='Hexagon T.dma_copy'):
        lower(probe)


def test_reject_mixed_raw_dma_ownership():
    @T.prim_func
    def probe(A: T.Tensor((256,), 'float16')):
        with T.Kernel(1, threads=1):
            s = T.alloc_shared((256,), 'float16')
            T.dma_copy(A, s)
            T.call_extern('uint32', 'tl_hexagon_dma_poll')
            T.dma_wait()
    with pytest.raises(Exception, match='raw DMA calls'):
        lower(probe)


def test_async_copy_is_separate_from_dma_copy():
    @T.prim_func
    def probe(A: T.Tensor((256,), 'float16')):
        with T.Kernel(1, threads=1):
            s = T.alloc_shared((256,), 'float16')
            T.async_copy(A, s)
    assert 'T.async_copy(' in probe.script()
    with pytest.raises(Exception, match='requires T.dma_copy instead of T.async_copy'):
        lower(probe)


def test_dma_copy_rejects_cpu_target():
    target = determine_target('llvm', return_object=True)
    with tvm.target.Target(target):
        with pytest.raises(Exception, match='T.dma_copy is supported only on Hexagon'):
            tilelang.lower(make_capacity_probe(2), target=target,
                           enable_host_codegen=False, enable_device_compile=False)
