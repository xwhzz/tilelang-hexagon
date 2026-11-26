"""The Hexagon shared-tile transpose uses the same layout mapping as a parallel loop."""

import os

import pytest
import torch

from tilelang.hexagon import language as T
from tilelang.hexagon.intrinsics.crouton import make_hmx_activation_layout
from tilelang.hexagon.runtime.session import Session, lower_kernels


def _source(fn, arch):
    return lower_kernels([(fn, [])], target={"kind": "hexagon", "mcpu": f"hexagon{arch}"})[0][1]


def _transpose(rows, cols, dtype, manual, crouton=False):
    @T.prim_func
    def transpose(A: T.Tensor((rows, cols), dtype), B: T.Tensor((cols, rows), dtype)):
        with T.Kernel(1):
            src = T.alloc_shared((rows, cols), dtype)
            dst = T.alloc_shared((cols, rows), dtype)
            if crouton:
                T.annotate_layout({src: make_hmx_activation_layout(src)})
            T.copy(A, src)
            if manual:
                for i, j in T.Parallel(rows, cols):
                    dst[j, i] = src[i, j]
            else:
                T.transpose(src, dst)
            T.copy(dst, B)

    return transpose


@pytest.mark.parametrize("arch", ["v73", "v79"])
@pytest.mark.parametrize(
    "rows,cols,dtype,crouton",
    [(32, 64, "float16", False), (8, 12, "int32", False), (1, 4, "int8", False), (1, 1, "float16", False), (32, 64, "float16", True)],
)
def test_shared_transpose_matches_parallel_loop(arch, rows, cols, dtype, crouton):
    assert _source(_transpose(rows, cols, dtype, False, crouton), arch) == _source(_transpose(rows, cols, dtype, True, crouton), arch)


def _batched_subregion(manual):
    @T.prim_func
    def transpose(A: T.Tensor((2, 8, 12), "int32"), B: T.Tensor((2, 12, 8), "int32")):
        with T.Kernel(1):
            src = T.alloc_shared((2, 8, 12), "int32")
            dst = T.alloc_shared((2, 12, 8), "int32")
            T.copy(A, src)
            T.clear(dst)
            if manual:
                for b, i, j in T.Parallel(2, 3, 4):
                    dst[b, 1 + j, 2 + i] = src[b, 3 + i, 6 + j]
            else:
                T.transpose(src[:, 3:6, 6:10], dst[:, 1:5, 2:5])
            T.copy(dst, B)

    return transpose


def test_shared_transpose_preserves_batch_and_region_offsets():
    assert _source(_batched_subregion(False), "v79") == _source(_batched_subregion(True), "v79")


@pytest.mark.skipif(os.environ.get("TL_HEXAGON_DEVICE_TEST") != "1", reason="requires an authorized Hexagon device and SDK")
def test_batched_shared_transpose_writes_only_the_requested_region():
    with Session([(_batched_subregion(False), [1])]) as session:
        data = torch.arange(2 * 8 * 12, dtype=torch.int32).reshape(2, 8, 12)
        expected = torch.zeros((2, 12, 8), dtype=torch.int32)
        expected[:, 1:5, 2:5] = data[:, 3:6, 6:10].transpose(-1, -2)
        assert torch.equal(session.kernels[0](data), expected)


@pytest.mark.parametrize(
    "case,message",
    [
        ("alias", "distinct shared buffers"),
        ("matched_alias", "distinct shared buffers"),
        ("shape", "region shapes"),
        ("global", "shared source and destination"),
    ],
)
def test_shared_transpose_checks_its_regions(case, message):
    @T.prim_func
    def transpose(A: T.Tensor((8, 8), "int32"), B: T.Tensor((8, 8), "int32")):
        with T.Kernel(1):
            src = T.alloc_shared((8, 8), "int32")
            dst = T.alloc_shared((8, 8), "int32")
            T.copy(A, src)
            if case == "alias":
                T.transpose(src, T.view(src, (8, 8)))
            elif case == "matched_alias":
                with T.sblock("matched"):
                    alias = T.match_buffer(src[0:8, 0:8], (8, 8), dtype="int32", scope="shared.dyn")
                    T.transpose(src, alias)
            elif case == "shape":
                T.transpose(src[0:4, 0:5], dst[1:5, 0:4])
            else:
                T.transpose(src, B)
            T.copy(dst, B)

    with pytest.raises(Exception, match=message):
        _source(transpose, "v79")
