"""Original v73 2D descriptors and bounded strided DMA copies."""

import os

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


STRIDED_LAYOUT_SHAPES = {"rows": (128, 128), "panel": (128, 2048), "activation": (128, 2048), "weight": (1024, 128)}


def make_strided_layout_copy(layout, wide_stride=True):
    rows, cols = STRIDED_LAYOUT_SHAPES[layout] if wide_stride else (128, 128)
    dtype = "uint8" if layout == "panel" else "float16"
    layout_fn = {
        "panel": make_hvx_panel_layout,
        "activation": make_hmx_activation_layout,
        "weight": make_hmx_weight_layout,
    }.get(layout)

    @T.prim_func
    def probe(A: T.Tensor((rows, cols), dtype), B: T.Tensor((rows, cols), dtype)):
        with T.Kernel(1):
            scratch = T.alloc_shared((64, 64), dtype)
            if layout_fn is not None:
                T.annotate_layout({A: layout_fn(A), B: layout_fn(B), scratch: layout_fn(scratch)})
            T.dma_copy(A[32:96, 32:96], scratch)
            T.dma_wait()
            T.dma_copy(scratch, B[32:96, 32:96])
            T.dma_wait()

    return probe


@pytest.mark.parametrize("layout", ["rows", "panel", "activation", "weight"])
@pytest.mark.parametrize("arch", ["v73", "v75", "v79"])
@pytest.mark.parametrize("wide_stride", [False, True])
def test_strided_layout_copies_use_target_descriptors(layout, arch, wide_stride):
    source = _lower(make_strided_layout_copy(layout, wide_stride), arch)
    submits = [line for line in source.splitlines() if "tl_hexagon_dma_async_submit(" in line]
    rows = 64 if layout == "rows" else 2
    split = arch == "v73" and wide_stride and layout != "rows"
    descriptors = rows if split else 1
    assert len(submits) == 2 * descriptors
    assert f"tl_hexagon_dma_async_context<{descriptors}, 1>" in source
    assert source.count("tl_hexagon_dma_async_commit(tl_dma_async)") == 2
    assert source.count("tl_hexagon_dma_async_wait(tl_dma_async, 0)") == 2
    encoded_rows = 1 if split else rows
    for call in submits[:descriptors]:
        assert f"(uint32_t){encoded_rows}, 3)" in call
    for call in submits[descriptors:]:
        assert f"(uint32_t){encoded_rows}, 1)" in call


@pytest.mark.skipif(os.environ.get("TL_HEXAGON_DEVICE_TEST") != "1", reason="requires an authorized Hexagon device and SDK")
@pytest.mark.parametrize("layout", ["rows", "panel", "activation", "weight"])
def test_strided_layout_copies_preserve_the_region_and_guards(layout):
    import torch
    from tilelang.hexagon.runtime.session import Session

    dtype = torch.uint8 if layout == "panel" else torch.float16
    # Inputs are the physical buffer bytes. Compute the selected region's
    # offsets independently of the layouts used by the compiler.
    rows, cols = STRIDED_LAYOUT_SHAPES[layout]
    source = (torch.arange(rows * cols) % 97 + 1).to(dtype)
    expected = torch.full_like(source, 199)
    for row in range(32, 96):
        for col in range(32, 96):
            if layout == "rows":
                offset = row * cols + col
            elif layout == "panel":
                offset = ((row // 32 * (cols // 4) + col // 4) * 32 + row % 32) * 4 + col % 4
            else:
                tile = (col // 32 * (rows // 32) + row // 32) if layout == "weight" else (row // 32 * (cols // 32) + col // 32)
                offset = tile * 1024 + row % 32 // 2 * 64 + col % 32 * 2 + row % 2
            expected[offset] = source[offset]
    with Session([(make_strided_layout_copy(layout), [])]) as session:
        a = session.buffer(source.numel() * source.element_size())
        b = session.buffer(source.numel() * source.element_size())
        a.write(source)
        b.write(torch.full_like(source, 199))
        session.kernels[0](a, b)
        actual = torch.frombuffer(bytearray(b.read()), dtype=dtype)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def make_embedding_gather():
    @T.prim_func
    def gather(A: T.Tensor((1024, 64), "float16"), B: T.Tensor((1024, 2), "float16")):
        with T.Kernel(1):
            scratch = T.alloc_shared((1024, 2), "float16")
            T.dma_copy(A[:, 34:36], scratch)
            T.dma_wait()
            T.copy(scratch, B)

    return gather


@pytest.mark.parametrize("arch", ["v73", "v75", "v79"])
def test_embedding_gather_uses_one_dma_group(arch):
    source = _lower(make_embedding_gather(), arch)
    assert source.count("tl_hexagon_dma_async_submit(") == 1
    assert "(uint32_t)4, (uint32_t)128, (uint32_t)4, (uint32_t)1024, 3)" in source
    assert "tl_hexagon_dma_async_context<1, 1>" in source
    assert source.count("tl_hexagon_dma_async_commit(tl_dma_async)") == 1


@pytest.mark.skipif(os.environ.get("TL_HEXAGON_DEVICE_TEST") != "1", reason="requires an authorized Hexagon device and SDK")
def test_embedding_gather_moves_all_1024_pairs():
    import torch

    source = (torch.arange(1024 * 64) % 997).reshape(1024, 64).half()
    kernel = tilelang.compile(make_embedding_gather(), out_idx=[1], target="hexagon")
    try:
        torch.testing.assert_close(kernel(source), source[:, 34:36], rtol=0, atol=0)
    finally:
        kernel.adapter.close()


def make_descriptor_encoding_probe():
    source = r"""
#include <tl_templates/hexagon/dma.h>
static void check_descriptors(int64_t *out) {
  const uint32_t cases[8][4] = {
    {128, 256, 64, 3},
    {65535, 65535, 65535, 2},
    {65536, 65535, 1, 2},
    {65535, 65536, 1, 2},
    {65536, 65536, 65536, 2},
    {1, 1, 1, 65535},
    {1, 1, 1, 65536},
    {0, 0, 0, 1},
  };
  const tl_hexagon_dma_options options{true, true, true};
  for (int i = 0; i < 9; ++i) {
    tl_hexagon_dma_descriptor descriptor{};
    int status;
    if (i == 8) {
      status = tl_hexagon_dma_descriptor_init_1d(
          &descriptor, (void *)0x200, (void *)0x100, 65536, options);
    } else {
      status = tl_hexagon_dma_descriptor_init_2d(
          &descriptor, (void *)0x200, (void *)0x100,
          cases[i][0], cases[i][1], cases[i][2], cases[i][3], options);
    }
    out[i * 9] = status;
    for (int word = 0; word < 8; ++word)
      out[i * 9 + 1 + word] = descriptor.word[word];
  }
  for (int i = 81; i < 90; ++i) out[i] = 0;
  out[81] = __HEXAGON_ARCH__;
}
"""

    @T.prim_func
    def probe(Out: T.Tensor((10, 9), "int64")):
        with T.Kernel(1):
            T.import_source(source)
            T.evaluate(T.call_extern("void", "check_descriptors", T.access_ptr(Out[0, 0], "w", 90)))

    return probe


@pytest.mark.skipif(os.environ.get("TL_HEXAGON_DEVICE_TEST") != "1", reason="requires an authorized Hexagon device and SDK")
def test_dma_descriptor_field_encoding_and_limits():
    """Inspect the actual target header's descriptors without submitting DMA."""
    probe = make_descriptor_encoding_probe()
    kernel = tilelang.compile(probe, out_idx=[0], target="hexagon")
    try:
        actual = kernel().tolist()
    finally:
        kernel.adapter.close()
    arch = actual[9][0]
    assert arch in (73, 75, 79, 81)
    cases = [
        (128, 256, 64, 3),
        (65535, 65535, 65535, 2),
        (65536, 65535, 1, 2),
        (65535, 65536, 1, 2),
        (65536, 65536, 65536, 2),
        (1, 1, 1, 65535),
        (1, 1, 1, 65536),
    ]
    for got, (dst_stride, src_stride, width, rows) in zip(actual, cases):
        if rows > 65535 or (arch < 75 and max(dst_stride, src_stride, width) > 65535):
            assert got == [-2] + [0] * 8
        elif arch < 75:
            assert got == [0, 0, 0x71000000, 0x100, 0x200, 0, width | rows << 16, src_stride | dst_stride << 16, 0]
        else:
            assert got == [0, 0, 0x71000000 | dst_stride, 0x100, 0x200, 9, width | (rows & 255) << 24, rows >> 8 | src_stride << 8, 0]
    assert actual[7] == [0, 0, 0x80000000, 0, 0, 0, 0, 0, 0]
    assert actual[8] == [0, 0, 0x70010000, 0x100, 0x200, 0, 0, 0, 0]
