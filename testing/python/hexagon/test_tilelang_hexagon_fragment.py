"""Fragments on Hexagon: the one thread's tiles, laid out row-major in the VTCM arena."""

import os

import pytest

import tilelang
from tilelang import tvm
from tilelang.backend.target import determine_target
from tilelang.hexagon import language as T

M, N = 64, 256


def _source(func):
    target = determine_target("hexagon", return_object=True)
    with tvm.target.Target(target), tvm.transform.PassContext(opt_level=3):
        return str(tilelang.lower(func, target=target, enable_host_codegen=False, enable_device_compile=False).kernel_source)


def _copy_map_fill():
    @T.prim_func
    def kernel(X: T.Tensor((M, N), "float32"), Y: T.Tensor((M, N), "float32"), Z: T.Tensor((M,), "float16")):
        with T.Kernel(1):
            x = T.alloc_fragment((M, N), "float32")
            z = T.alloc_fragment((M,), "float16")
            T.copy(X, x)
            for i, j in T.Parallel(M, N):
                x[i, j] = x[i, j] * 2.0 + 1.0
            T.fill(z, 3.5)
            T.copy(x, Y)
            T.copy(z, Z)

    return kernel


def test_fragments_live_in_the_vtcm_arena():
    source = _source(_copy_map_fill())
    # Both fragments share the tile program's arena, laid out row-major.
    assert "uint8_t* buf_dyn_shmem = (uint8_t*)tl_vtcm_arena();" in source
    assert "void* x = ((void*)((char*)buf_dyn_shmem + 0));" in source
    assert "void* z = ((void*)((char*)buf_dyn_shmem + 65536));" in source
    assert "float x[" not in source


def test_fragments_that_only_copy_or_fill_get_a_layout():
    @T.prim_func
    def copy_only(X: T.Tensor((M, N), "float32"), Y: T.Tensor((M, N), "float32")):
        with T.Kernel(1):
            x = T.alloc_fragment((M, N), "float32")
            T.copy(X, x)
            T.copy(x, Y)

    @T.prim_func
    def fill_only(Y: T.Tensor((M,), "float32")):
        with T.Kernel(1):
            y = T.alloc_fragment((M,), "float32")
            T.fill(y, 1.0)
            T.copy(y, Y)

    # No operator constrains these, so they get the single-thread layout, and
    # each is its program's whole arena.
    assert "float* x = (float*)tl_vtcm_arena();" in _source(copy_only)
    assert "float* y = (float*)tl_vtcm_arena();" in _source(fill_only)


@pytest.mark.skipif(os.environ.get("TL_HEXAGON_DEVICE_TEST") != "1", reason="requires an authorized Hexagon device and SDK")
def test_fragment_copy_map_fill_match_torch():
    import torch

    kernel = tilelang.compile(_copy_map_fill(), out_idx=[1, 2], target="hexagon")
    try:
        x = torch.randn(M, N)
        y, z = kernel(x)
        torch.testing.assert_close(y, x * 2 + 1, rtol=0, atol=0)
        torch.testing.assert_close(z, torch.full((M,), 3.5, dtype=torch.float16), rtol=0, atol=0)
    finally:
        kernel.adapter.close()
