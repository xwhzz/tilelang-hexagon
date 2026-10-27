"""Host lowering contracts for Hexagon demo operators and layout copies."""

import os
import pytest
import tilelang
from tilelang.backend.target import determine_target
from examples.hexagon.example_demo_ops import hexreduce, make_elementwise, make_gemm_f16_io, make_map_tiled, make_rmsnorm_fp32
from tilelang.hexagon import language as T
from tilelang.hexagon.intrinsics.hmx import HMXIntrinEmitter


def make_reduce_probe(rows, cols, dtype="float32"):
    @T.prim_func
    def reduce_probe(X: T.Tensor((rows, cols), dtype), O: T.Tensor((2, rows), "float32")):
        with T.Kernel(1):
            shared = T.alloc_shared((rows, cols), dtype)
            sums = T.alloc_shared((rows,), "float32")
            maxima = T.alloc_shared((rows,), "float32")
            T.copy(X, shared)
            hexreduce(shared, sums, "sum", 1)
            hexreduce(shared, maxima, "max", 1)
            for i in T.serial(rows):
                O[0, i] = sums[i]
                O[1, i] = maxima[i]

    return reduce_probe


def make_weight_copy_probe(cols, start=64, workers=1):
    """Exercise the generic transposed weight layout with strided endpoints."""
    # Two N tiles distinguish weight-transposed from the equivalent
    # activation-transposed layout of a single N tile.
    emitter = HMXIntrinEmitter(32, 64, cols, b_transposed=True)

    @T.prim_func
    def weight_copy(X: T.Tensor((64, cols + 64), "float16"), O: T.Tensor((64, cols + 4), "float16")):
        with T.Kernel(1):
            native = T.alloc_shared((64, cols), "float16", align=2048)
            T.annotate_layout({native: emitter.weight_layout(native)})
            for i, j in T.Parallel(64, cols + 4):
                O[i, j] = T.float16(8192)
            T.copy(X[:, start : start + cols], native, annotations={"hexagon.copy_workers": workers} if workers > 1 else {})
            T.copy(native, O[:, 2 : cols + 2])

    return weight_copy


def lower(func):
    target = determine_target("hexagon", return_object=True)
    with target:
        return str(tilelang.lower(func, target=target, enable_host_codegen=False, enable_device_compile=False).kernel_source)


@pytest.mark.parametrize("kind", ["add", "mul", "scale"])
def test_fp32_map_uses_native_vectors_and_own_workers(kind):
    source = lower(make_elementwise(32, 2048, kind))
    assert "float32*" in source  # 32 FP32 lanes, not a dtype string.
    assert "tl_parallel_for(" in source
    assert "own_row_f32" not in source
    assert "demo_" not in source


@pytest.mark.parametrize("dtype,suffix", [("float16", "_f16"), ("float32", "_f32")])
def test_reduce_keeps_source_precision(dtype, suffix):
    source = lower(make_reduce_probe(3, 65, dtype))
    assert f"tl_hvx_rowsum_mat{suffix}(" in source
    assert f"tl_hvx_rowmax_mat{suffix}(" in source
    if dtype == "float32":
        assert "half" not in source


def test_rmsnorm_stays_fp32_and_composes_reduction():
    source = lower(make_rmsnorm_fp32(32, 2048))
    assert "tl_hvx_rowsum_mat_f32(" in source
    assert "float32*" in source
    assert "sqrtf" in source
    assert "half" not in source
    assert "tl_parallel_for(" in source


def test_reduce_rejects_integer_sources():
    with pytest.raises(Exception, match="fp16/fp32 source only"):
        lower(make_reduce_probe(3, 65, "int32"))


@pytest.mark.parametrize("shape", [(64, 64, 64, 64), (128, 256, 256, 256), (256, 2048, 2048, 256)])
def test_gemm_preserves_interface_and_dma_completion(shape):
    source = lower(make_gemm_f16_io(*shape))
    assert "float* A, half* W, float* C" in source
    assert "tl_hexagon_hmx_mma_atom(" in source
    assert "tl_hexagon_hmx_pack_crouton(" in source
    assert "tl_hexagon_hmx_unpack_crouton(" in source
    assert "tl_hexagon_dma_async_context<" in source
    assert "tl_hexagon_dma_async_wait(tl_dma_async, 0)" in source
    assert "demo_parallel_pack" not in source
    assert "worker_pool_run_func" not in source
    assert "tl_bridge_enter" not in source


def test_tail_maps_do_not_require_full_vector_rows():
    for kind in ["add", "mul", "scale"]:
        source = lower(make_elementwise(3, 65, kind))
        assert "float32*" not in source
    assert "tl_hvx_rowsum_mat_f32" in lower(make_rmsnorm_fp32(3, 65))


def test_gemm_bad_block_is_rejected_before_lowering():
    with pytest.raises(ValueError, match="32-multiple"):
        make_gemm_f16_io(128, 256, 256, 96)


def test_transposed_weight_copy_preserves_source_and_destination_strides():
    source = lower(make_weight_copy_probe(128, 64))
    assert "(&(X[64])), 64, 128, 192, 1, 3)" in source
    assert "tl_hexagon_hmx_unpack_crouton((&(O[2]))" in source
    assert ", 64, 128, 132, 1, 3)" in source


def test_gemm_fused_parallel_layout_copies_remain_dsl():
    source = lower(make_gemm_f16_io(256, 512, 2048, 256, copy_workers=6, fused_copy=True))
    assert source.count("tl_hexagon_hmx_copy_crouton_parallel(") == 3
    assert "tl_hexagon_hmx_mma_atom(" in source
    assert "a_row" not in source and "c_row" not in source
    assert "demo_parallel_pack" not in source


@pytest.mark.skipif(os.environ.get("TL_HEXAGON_DEVICE_TEST") != "1", reason="requires an authorized Hexagon device and SDK")
@pytest.mark.parametrize("width", [32, 96, 128, 160, 2048])
def test_tiled_rmsnorm_vector_partials_cover_full_width(width):
    import torch
    from tilelang import tvm

    fn = make_map_tiled(8, width, "rmsnorm", 2, 2, num_workers=2)
    # The comparison harness supplies a common X/Y/O interface, but unary
    # kernels do not use Y. Match its adapter before direct FastRPC compilation.
    unused = fn.params[1]
    fn = tvm.tirx.PrimFunc(
        [fn.params[0], fn.params[2]],
        fn.body,
        fn.ret_type,
        {p: b for p, b in fn.buffer_map.items() if not p.same_as(unused)},
        fn.attrs,
        fn.span,
    )
    kernel = tilelang.compile(fn, out_idx=[1], target="hexagon")
    try:
        for seed in [1, 2, 3]:
            torch.manual_seed(seed)
            x = torch.randn(8, width)
            out = kernel(x)
            ref = x / torch.sqrt((x * x).mean(dim=1, keepdim=True) + 1e-5)
            assert torch.isfinite(out).all()
            torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-6)
    finally:
        kernel.adapter.close()


def test_wide_gemm_keeps_single_raw_slot_within_vtcm_budget():
    import re

    source = lower(make_gemm_f16_io(256, 2048, 2048, 512, copy_workers=6, fused_copy=True))
    high = max(map(int, re.findall(r"tl_vtcm_shared_high_water = (\d+)u", source)))
    assert high <= 8 * 1024 * 1024
    assert "tl_hexagon_dma_async_wait(tl_dma_async, 0)" in source
    assert "tl_hexagon_hmx_mma_atom(" in source


@pytest.mark.parametrize("workers", [0, 7])
def test_parallel_copy_rejects_bad_worker_count(workers):
    e = HMXIntrinEmitter(32, 32, 64)

    @T.prim_func
    def probe(A: T.Tensor((32, 64), "float16")):
        with T.Kernel(1):
            native = T.alloc_shared((32, 64), "float16")
            T.annotate_layout({native: e.activation_layout(native)})
            T.copy(A, native, annotations={"hexagon.copy_workers": workers})

    with pytest.raises(Exception, match="copy_workers must be"):
        lower(probe)


def test_parallel_copy_cannot_nest_worker_dispatch():
    e = HMXIntrinEmitter(32, 32, 64)

    @T.prim_func
    def probe(A: T.Tensor((32, 64), "float16")):
        with T.Kernel(1, num_workers=2):
            native = T.alloc_shared((32, 64), "float16")
            T.annotate_layout({native: e.activation_layout(native)})
            T.copy(A, native, annotations={"hexagon.copy_workers": 2})

    with pytest.raises(Exception, match="cannot nest"):
        lower(probe)
