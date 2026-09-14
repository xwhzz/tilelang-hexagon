"""Verify the DSL benchmark retains DMA prefetch and the native HMX schedule."""
import importlib.util
from pathlib import Path

import pytest
import tilelang
from tilelang import tvm
from tilelang.utils.target import determine_target


@pytest.mark.parametrize('filename,factory,timers', [('benchmark_dma_matmul.py', 'make_bench', 2),
                                                  ('example_dma_hmx_matmul.py', 'make_dma_hmx_matmul', 0)])
@pytest.mark.parametrize('dims', [(32, 32, 64, 32, 32), (256, 512, 2048, 128, 128)])
def test_dma_benchmark_dsl_schedule(dims, filename, factory, timers):
    path = Path(__file__).resolve().parents[3] / 'examples/hexagon' / filename
    spec = importlib.util.spec_from_file_location('dma_benchmark_dsl', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    target = determine_target('hexagon', return_object=True)
    with tvm.target.Target(target):
        src = str(tilelang.lower(getattr(module, factory)(*dims), target=target,
                                enable_host_codegen=False,
                                enable_device_compile=False).kernel_source)
    assert 'bench_run(' not in src
    assert 'bench_once(' not in src
    assert 'tl_hexagon_dma_async_context<4>' in src
    assert src.count('HAP_perf_get_time_us()') == timers
    assert 'tl_hexagon_hmx_mma_atom(' in src
    assert 'tl_hexagon_hmx_unpack_crouton(' in src
    if dims[0] > dims[3]:
        assert 'tl_dma_async.wait(2)' in src
        # A refill follows packing and precedes the current output block's MMA.
        pack = src.index('tl_hexagon_hmx_pack_crouton(')
        refill = src.index('tl_dma_async.copy_1d(', pack)
        assert pack < refill < src.index('tl_hexagon_hmx_mma_atom(')
