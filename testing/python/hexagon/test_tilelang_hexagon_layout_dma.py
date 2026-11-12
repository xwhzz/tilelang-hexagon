"""Packed-layout slices retain async DMA semantics across descriptor formats."""

import os
import re

import pytest

import tilelang
from tilelang import tvm
from tilelang.backend.target import determine_target
from tilelang.hexagon import language as T
from tilelang.hexagon.intrinsics.crouton import make_hmx_activation_layout, make_hmx_weight_layout
from tilelang.hexagon.intrinsics.panel import make_hvx_panel_layout


def _lower(func, arch):
    target = determine_target({"kind": "hexagon", "mcpu": f"hexagon{arch}"}, return_object=True)
    with tvm.target.Target(target):
        return str(tilelang.lower(func, target=target, enable_host_codegen=False, enable_device_compile=False).kernel_source)


def _slice_copy(kind):
    dtype = kind if kind in ("int8", "uint4") else "float16"

    def layout(buffer):
        if dtype != "float16":
            return make_hvx_panel_layout(buffer)
        factory = make_hmx_weight_layout if kind.startswith("weight") else make_hmx_activation_layout
        return factory(buffer, transposed=kind.endswith("_t"))

    @T.prim_func
    def kernel(A: T.Tensor((2, 128, 192), dtype), B: T.Tensor((2, 128, 192), dtype)):
        with T.Kernel(1):
            stage = T.alloc_shared((2, 64, 64), dtype)
            T.annotate_layout({A: layout(A), B: layout(B), stage: layout(stage)})
            T.dma_copy(A[1, 32:96, 64:128], stage[1, :, :])
            T.dma_wait()
            T.dma_copy(stage[1, :, :], B[0, 64:128, 32:96])
            T.dma_wait()

    return kernel


@pytest.mark.parametrize("arch", ["v73", "v75", "v79"])
@pytest.mark.parametrize("kind", ["weight", "weight_t", "activation", "activation_t", "int8", "uint4"])
def test_strided_layout_load_store_keeps_one_group_per_copy(arch, kind):
    source = _lower(_slice_copy(kind), arch)
    calls = [line for line in source.splitlines() if "tl_hexagon_dma_async_submit(" in line]
    # Two physical runs in each direction, inside one logical copy group.
    # Input and output start at nonzero matrix/row/column offsets.
    run_bytes = {"int8": 2048, "uint4": 1024}.get(kind, 4096)
    column_runs = kind in ("weight", "activation_t")
    global_stride = run_bytes * (2 if column_runs else 3)
    assert len(calls) == 2
    assert "tl_hexagon_dma_async_context<1, 1>" in source
    assert f"(uint32_t){run_bytes}, (uint32_t){global_stride}, (uint32_t){run_bytes}, (uint32_t)2" in calls[0]
    assert f"(uint32_t){global_stride}, (uint32_t){run_bytes}, (uint32_t){run_bytes}, (uint32_t)2" in calls[-1]
    assert source.count("tl_hexagon_dma_async_commit(") == 2
    assert source.count("tl_hexagon_dma_async_wait(tl_dma_async, 0)") == 2
    assert all(re.search(r", 3\)\) return tl_status;", call) for call in calls[: len(calls) // 2])
    assert all(re.search(r", 1\)\) return tl_status;", call) for call in calls[len(calls) // 2 :])


def _many_panels(panels):
    rows = panels * 32

    @T.prim_func
    def kernel(A: T.Tensor((rows, 2048), "int8")):
        with T.Kernel(1):
            stage = T.alloc_shared((rows, 4), "int8")
            T.annotate_layout({A: make_hvx_panel_layout(A), stage: make_hvx_panel_layout(stage)})
            T.dma_copy(A[:, 4:8], stage)
            T.dma_wait()

    return kernel


def test_v73_layout_split_respects_descriptor_capacity():
    # Physical source panels are 65,536 bytes apart, beyond the original
    # 2D format. Each selected four-column block needs one 1D descriptor.
    source = _lower(_many_panels(256), "v73")
    assert "tl_hexagon_dma_compact_load_context<256, 1>" in source
    assert source.count("tl_hexagon_dma_compact_submit(") == 256
    assert source.count("tl_hexagon_dma_compact_commit(") == 1
    with pytest.raises(Exception, match="more than 256 DMA transfers in flight"):
        _lower(_many_panels(257), "v73")
    # A newer target still handles all panels with one 2D descriptor.
    assert "tl_hexagon_dma_compact_load_context<1, 1>" in _lower(_many_panels(257), "v75")


def _weight_slice_round_trip(transposed):
    output_shape = (64, 32) if transposed else (32, 64)

    @T.prim_func
    def kernel(W: T.Tensor((64, 64), "float16"), O: T.Tensor(output_shape, "float16")):
        with T.Kernel(1):
            stage = T.alloc_shared(output_shape, "float16")
            T.annotate_layout(
                {
                    W: make_hmx_weight_layout(W, transposed=transposed),
                    stage: make_hmx_weight_layout(stage, transposed=transposed),
                    O: make_hmx_weight_layout(O, transposed=transposed),
                }
            )
            if transposed:
                T.dma_copy(W[:, 32:64], stage)
            else:
                T.dma_copy(W[32:64, :], stage)
            T.dma_wait()
            T.dma_copy(stage, O)
            T.dma_wait()

    return kernel


@pytest.mark.skipif(os.environ.get("TL_HEXAGON_DEVICE_TEST") != "1", reason="requires an authorized Hexagon device and SDK")
@pytest.mark.parametrize("transposed", [False, True])
def test_weight_slice_round_trip_preserves_bits(transposed):
    import torch

    kernel = tilelang.compile(_weight_slice_round_trip(transposed), out_idx=[1], target="hexagon")
    try:
        # Raw packed bytes, including NaNs and signed zeros: no floating-point
        # arithmetic participates in this DMA-only regression.
        bits = (torch.arange(4096, dtype=torch.int32) * 17).to(torch.int16)
        bits[1024:1032] = torch.tensor([0, -32768, 31744, -1024, 31745, 32256, 1, -32767], dtype=torch.int16)
        out = kernel(bits.view(torch.float16).reshape(64, 64))
        # The second K tile from each of two column panels.
        expected = torch.cat((bits[1024:2048], bits[3072:4096]))
        assert torch.equal(out.view(torch.int16).flatten(), expected)
    finally:
        kernel.adapter.close()


@pytest.mark.skipif(os.environ.get("TL_HEXAGON_DEVICE_TEST") != "1", reason="requires an authorized Hexagon device and SDK")
@pytest.mark.parametrize("dtype", ["int8", "uint4"])
def test_panel_slice_round_trip_preserves_bits_and_guards(dtype):
    import torch

    from tilelang.hexagon.runtime.session import Session

    bits = 8 if dtype == "int8" else 4
    nbytes = 2 * 128 * 192 * bits // 8
    generator = torch.Generator().manual_seed(233)
    packed = torch.randint(0, 256, (nbytes,), dtype=torch.uint8, generator=generator)
    raw = bytes(packed.tolist())
    initial = bytes([0xA5]) * nbytes
    expected = bytearray(initial)
    # Panel storage is [matrix, row//32, column//lane, row%32, column%lane].
    # Select two source panels from matrix 1 at column 64, and place them in
    # matrix 0's last two panels at column 32. Every other byte is a guard.
    count = 32 * 64 * bits // 8
    for panel in range(2):
        src = (128 * 192 + (1 + panel) * 32 * 192 + 64 * 32) * bits // 8
        dst = ((2 + panel) * 32 * 192 + 32 * 32) * bits // 8
        expected[dst : dst + count] = raw[src : src + count]
    with Session([(_slice_copy(dtype), [])], target="hexagon") as session:
        source = session.buffer(nbytes)
        destination = session.buffer(nbytes)
        source.write(packed)
        destination.write(initial)
        session.kernels[0](source, destination)
        assert destination.read() == bytes(expected)
