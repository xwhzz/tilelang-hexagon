"""Hexagon inclusive prefix sums on shared tiles and their regions."""

import os

import pytest
import torch

import tilelang
from tilelang import tvm
from tilelang.hexagon import language as T
from tilelang.hexagon.runtime.session import Session, lower_kernels


def _scan(rows, cols, axis, reverse, inplace, dtype, manual=False):
    @T.prim_func
    def scan(A: T.Tensor((rows, cols), dtype), B: T.Tensor((rows, cols), dtype)):
        with T.Kernel(1):
            src = T.alloc_shared((rows, cols), dtype)
            dst = T.alloc_shared((rows, cols), dtype)
            T.copy(A, src)
            if manual:
                for r in T.serial(rows):
                    for k in T.serial(1, cols):
                        src[r, k] = src[r, k - 1] + src[r, k]
                T.copy(src, B)
            elif inplace:
                T.cumsum(src, dim=axis, reverse=reverse)
                T.copy(src, B)
            else:
                T.cumsum(src, dst, dim=axis, reverse=reverse)
                T.copy(dst, B)

    return scan


def _regions(case):
    @T.prim_func
    def scan(A: T.Tensor((3, 9), "int32"), B: T.Tensor((3, 9), "int32")):
        with T.Kernel(1):
            buf = T.alloc_shared((3, 9), "int32")
            T.copy(A, buf)
            if case == "disjoint":
                T.cumsum(buf[1:3, 1:4], buf[1:3, 5:8], dim=-1, reverse=True)
            elif case == "empty":
                T.cumsum(buf[:, 0:0], buf[:, 4:4], dim=1)
            elif case == "overlap":
                T.cumsum(buf[:, 1:4], buf[:, 3:6], dim=1)
            elif case == "bounds":
                T.cumsum(buf[:, 7:10], dim=1)
            elif case == "global":
                T.cumsum(A, buf)
            elif case == "strided_view":
                T.cumsum(T.view(buf, (3, 9), strides=(1, 3)))
            elif case == "view":
                T.cumsum(buf, T.view(buf, (3, 9)))
            else:
                with T.sblock("matched"):
                    alias = T.match_buffer(buf[:, :], (3, 9), dtype="int32", scope="shared.dyn")
                    T.cumsum(alias)
            T.copy(buf, B)

    return scan


@pytest.mark.parametrize("arch", ["v73", "v79"])
def test_cumsum_matches_serial_recurrence(arch):
    shape = (2, 32, -1, False, True, "float32")
    target = tvm.target.Target({"kind": "hexagon", "mcpu": f"hexagon{arch}"})

    def lowered(func):
        with target, tvm.transform.PassContext(opt_level=3):
            return tilelang.lower(func, target=target).device_mod

    tvm.ir.assert_structural_equal(lowered(_scan(*shape)), lowered(_scan(*shape, manual=True)))


@pytest.mark.parametrize(
    "case,message",
    [
        ("overlap", "identical or disjoint"),
        ("bounds", "in-bounds regions"),
        ("global", "shared source and destination"),
        ("view", "identical or disjoint"),
        ("strided_view", "compact shared buffers"),
        ("matched", "match_buffer view"),
    ],
)
def test_cumsum_checks_regions(case, message):
    with pytest.raises(Exception, match=message):
        lower_kernels([(_regions(case), [1])], target="hexagon")


@pytest.mark.skipif(os.environ.get("TL_HEXAGON_DEVICE_TEST") != "1", reason="requires an authorized Hexagon device and SDK")
def test_cumsum_device_values_and_region_boundaries():
    cases = [
        (2, 32, -1, False, True, "float32"),
        (3, 7, 1, True, False, "int32"),
        (3, 5, 0, False, True, "int32"),
        (1, 1, 0, True, False, "float16"),
    ]
    kernels = [(_scan(*case), [1]) for case in cases] + [(_regions(case), [1]) for case in ("disjoint", "empty")]
    with Session(kernels) as session:
        for i, (rows, cols, axis, reverse, _, dtype) in enumerate(cases):
            data = torch.arange(1, rows * cols + 1).reshape(rows, cols).to(getattr(torch, dtype))
            expected = data.flip([axis]).cumsum(axis).flip([axis]) if reverse else data.cumsum(axis)
            assert torch.equal(session.kernels[i](data), expected.to(data.dtype))
        data = torch.arange(27, dtype=torch.int32).reshape(3, 9) - 10
        expected = data.clone()
        expected[1:3, 5:8] = data[1:3, 1:4].flip([1]).cumsum(1).flip([1])
        assert torch.equal(session.kernels[len(cases)](data), expected)
        assert torch.equal(session.kernels[len(cases) + 1](data), data)
