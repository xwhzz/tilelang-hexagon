"""Integer HMX lowering and exact uint8-by-int8 accumulator readout."""

import os

import pytest
import torch

import tilelang
from tilelang import tvm
from tilelang.backend.target import determine_target
from tilelang.hexagon import language as T

DEVICE = pytest.mark.skipif(os.environ.get("TL_HEXAGON_DEVICE_TEST") != "1", reason="requires an authorized Hexagon device and SDK")


def _tile(kt, workers=1, *, permuted=False):
    @T.prim_func
    def gemm(
        A: T.Tensor((workers, kt, 64, 32), "uint8"),
        B: T.Tensor((workers, kt, 8, 32, 4), "int8"),
        C: T.Tensor((workers, 64, 32), "int32"),
    ):
        with T.Kernel(workers, num_workers=workers) as p:
            a = T.alloc_shared((kt + 1, 64, 32), "uint8")
            b = T.alloc_shared((2, kt, 8, 32, 4), "int8")
            c = T.alloc_shared((2, 64, 32), "int32")
            if permuted:
                T.annotate_layout({a: T.Layout((kt + 1, 64, 32), lambda t, m, k: [t, k, m])})
            T.copy(A[p, :, :, :], a[1:, :, :])
            T.copy(B[p, :, :, :, :], b[1, :, :, :, :])
            T.hmx_gemm_u8i8(a[1:, :, :], b[1, :, :, :, :], c[1, :, :])
            T.copy(c[1, :, :], C[p, :, :])

    return gemm


def _source(func, arch="v79"):
    target = determine_target({"kind": "hexagon", "mcpu": f"hexagon{arch}"}, return_object=True)
    with tvm.target.Target(target), tvm.transform.PassContext(opt_level=3):
        return tilelang.lower(func, target=target, enable_host_codegen=False, enable_device_compile=False).kernel_source


@pytest.mark.parametrize("arch", ["v73", "v79"])
def test_hmx_int_lowers_with_worker_resource_scope(arch):
    source = _source(_tile(3, workers=2), arch)
    assert "#include <tl_templates/hexagon/hmx_gemm_int.h>" in source
    assert "tl_parallel_for<tl_hmx_worker_scope" in source
    # The operands have no explicit alignment and start at nonzero offsets.
    assert "tl_hexagon_hmx_gemm_u8i8(3, " in source
    assert "[2048]" in source and "[3072]" in source


def test_hmx_int_rejects_layout_permutation():
    with pytest.raises(Exception, match="requires dense row-major layout for a"):
        _source(_tile(1, permuted=True))


@pytest.mark.parametrize("wrap", [True, False])
def test_hmx_int_checks_symbolic_batch_bounds(wrap):
    @T.prim_func
    def gemm(A: T.Tensor((2, 1, 64, 32), "uint8"), B: T.Tensor((1, 8, 32, 4), "int8"), C: T.Tensor((64, 32), "int32")):
        with T.Kernel(1):
            a = T.alloc_shared((2, 1, 64, 32), "uint8")
            b = T.alloc_shared((1, 8, 32, 4), "int8")
            c = T.alloc_shared((64, 32), "int32")
            T.copy(A, a)
            T.copy(B, b)
            for i in T.serial(3):
                if wrap:
                    T.hmx_gemm_u8i8(a[i % 2, :, :, :], b, c)
                else:
                    T.hmx_gemm_u8i8(a[i, :, :, :], b, c)
            T.copy(c, C)

    if wrap:
        assert "tl_hexagon_hmx_gemm_u8i8(1, " in _source(gemm)
    else:
        with pytest.raises(Exception, match="requires provably in-bounds regions; a axis 0"):
            _source(gemm)


@pytest.mark.parametrize(
    "batch,index,unsigned,match",
    [
        (2, 2147483647, False, "requires provably in-bounds regions; a axis 0"),
        (2, 4294967295, True, "requires provably in-bounds regions; a axis 0"),
        (2097153, 0, False, "fits signed 32-bit byte offsets"),
    ],
)
def test_hmx_int_rejects_overflow_in_batch_address(batch, index, unsigned, match):
    @T.prim_func
    def gemm(A: T.Tensor((1, 64, 32), "uint8"), B: T.Tensor((1, 8, 32, 4), "int8"), C: T.Tensor((64, 32), "int32")):
        with T.Kernel(1):
            a = T.alloc_shared((batch, 1, 64, 32), "uint8")
            b = T.alloc_shared((1, 8, 32, 4), "int8")
            c = T.alloc_shared((64, 32), "int32")
            T.copy(A, a[0, :, :, :])
            T.copy(B, b)
            if unsigned:
                T.hmx_gemm_u8i8(a[T.uint32(index), :, :, :], b, c)
            else:
                T.hmx_gemm_u8i8(a[index, :, :, :], b, c)
            T.copy(c, C)

    with pytest.raises(Exception, match=match):
        _source(gemm)


@pytest.mark.parametrize("first,last", [(1, 3), (0, 3)])
def test_hmx_int_rejects_k_slice_crossing_batch_boundary(first, last):
    @T.prim_func
    def gemm(A: T.Tensor((2, 2, 64, 32), "uint8"), B: T.Tensor((last - first, 8, 32, 4), "int8"), C: T.Tensor((64, 32), "int32")):
        with T.Kernel(1):
            a = T.alloc_shared((2, 2, 64, 32), "uint8")
            b = T.alloc_shared((last - first, 8, 32, 4), "int8")
            c = T.alloc_shared((64, 32), "int32")
            T.copy(A, a)
            T.copy(B, b)
            T.hmx_gemm_u8i8(a[0, first:last, :, :], b, c)
            T.copy(c, C)

    with pytest.raises(Exception, match="requires provably in-bounds regions; a axis 1"):
        _source(gemm)


def test_hmx_int_requires_a_region_of_the_backing_allocation():
    @T.prim_func
    def gemm():
        with T.Kernel(1):
            a = T.alloc_shared((1, 64, 64), "uint8")
            b = T.alloc_shared((1, 8, 32, 4), "int8")
            c = T.alloc_shared((64, 32), "int32")
            with T.sblock("activation_view"):
                # This view hides A's physical row stride and byte offset.
                view = T.match_buffer(a[:, :, 1:33], (1, 64, 32), dtype="uint8", scope="shared.dyn")
                T.hmx_gemm_u8i8(view, b, c)

    with pytest.raises(ValueError, match="T.hmx_gemm_u8i8 on a match_buffer view"):
        _source(gemm)


@pytest.mark.parametrize("target", [{"kind": "hexagon", "mcpu": "hexagonv69"}, "llvm"])
def test_hmx_int_checks_lowering_target(target):
    @T.prim_func
    def gemm():
        with T.Kernel(1):
            a = T.alloc_shared((1, 64, 32), "uint8")
            b = T.alloc_shared((1, 8, 32, 4), "int8")
            c = T.alloc_shared((64, 32), "int32")
            T.hmx_gemm_u8i8(a, b, c)

    mod = tvm.IRModule({"gemm": gemm.with_attr("target", tvm.target.Target(target))})
    with pytest.raises(Exception, match="requires a Hexagon v73, v75, v79 or v81 target"):
        tilelang.transform.LowerTileOp()(mod)


@pytest.mark.parametrize(
    "a_dtype,a_shape,b_tiles,match",
    [
        ("int8", (1, 64, 32), 1, "A in program-local or grid VTCM with dtype uint8"),
        ("uint8", (1, 64, 64), 1, "A to select a contiguous tile region"),
        ("uint8", (2, 64, 32), 1, "same number of K tiles"),
        ("uint8", (4096, 64, 32), 4096, "statically known K tiles"),
    ],
)
def test_hmx_int_checks_tile_contract(a_dtype, a_shape, b_tiles, match):
    with pytest.raises(Exception, match=match):

        @T.prim_func
        def invalid():
            with T.Kernel(1):
                a = T.alloc_shared(a_shape, a_dtype)
                b = T.alloc_shared((b_tiles, 8, 32, 4), "int8")
                c = T.alloc_shared((64, 32), "int32")
                T.hmx_gemm_u8i8(a, b, c)


def test_hmx_int_rejects_static_shared_scope():
    with pytest.raises(Exception, match="A in program-local or grid VTCM with dtype uint8"):

        @T.prim_func
        def invalid():
            with T.Kernel(1):
                a = T.alloc_shared((1, 64, 32), "uint8", scope="shared")
                b = T.alloc_shared((1, 8, 32, 4), "int8")
                c = T.alloc_shared((64, 32), "int32")
                T.hmx_gemm_u8i8(a, b, c)


@DEVICE
@pytest.mark.parametrize("kt,workers", [(1, 1), (4, 2), (21, 2), (33, 1)])
def test_hmx_int_matches_full_signed_int32_dots(kt, workers):
    torch.manual_seed(kt)
    k = kt * 32
    a = torch.randint(0, 256, (workers, 64, k), dtype=torch.int64)
    b = torch.randint(-128, 128, (workers, 32, k), dtype=torch.int64)
    a[:, 0] = 255
    a[:, 1] = 0
    b[:, 0] = 127
    b[:, 1] = -128
    b[:, 2] = 0
    # Exercise distinct low bytes beyond fp32's exact integer range, both
    # signs, and the transition to the second 32-tile packet.
    a[:, 0, -1] = 254
    b[:, 1, -1] = -127
    if kt == 4:
        # A logical K tail whose padded channels are zero even in the rows
        # and columns filled with extrema above.
        a[:, :, -13:] = 0
        b[:, :, -13:] = 0
    packed_a = a.reshape(workers, 64, kt, 32).permute(0, 2, 1, 3).to(torch.uint8).contiguous()
    packed_b = b.reshape(workers, 32, kt, 8, 4).permute(0, 2, 3, 1, 4).to(torch.int8).contiguous()
    expected = torch.stack([x @ w.T for x, w in zip(a, b)]).to(torch.int32)
    kernel = tilelang.compile(_tile(kt, workers), out_idx=[2], target="hexagon")
    try:
        torch.testing.assert_close(kernel(packed_a, packed_b), expected, rtol=0, atol=0)
    finally:
        kernel.adapter.close()
