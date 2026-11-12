"""Finite one-program device checks for the optional synchronous prologue."""

import os

import numpy as np
import pytest

from tilelang.hexagon import language as T
from tilelang.hexagon.runtime.session import Session

MODES = ("readonly", "write_through", "dma_only", "mixed")
SHAPE = (6, 256)
GUARD = bytes([0xA5]) * 128
DEVICE = pytest.mark.skipif(
    os.environ.get("TL_HEXAGON_DEVICE_TEST") != "1",
    reason="requires an authorized Hexagon device and SDK",
)


def make_case(mode, policy="sync_if_supported"):
    assert mode in MODES
    second = {"custom_copy_hint": 1} if mode == "mixed" else {}

    @T.prim_func
    def kernel(
        A: T.Tensor(SHAPE, "int32"),
        B: T.Tensor(SHAPE, "int32"),
        C: T.Tensor(SHAPE, "int32"),
    ):
        with T.Kernel(1):
            if mode == "dma_only":
                T.annotate_dma_only(A)
            if mode == "write_through":
                T.annotate_write_through(A)
                for j in T.Parallel(256):
                    A[1, j] = A[1, j] + 1
            sa = T.alloc_shared((256,), "int32")
            sb = T.alloc_shared((256,), "int32")
            for i in T.Pipelined(1, 5, num_stages=1, annotations={"hexagon.prologue_copy_policy": policy}):
                T.copy(A[i, 0:256], sa)
                T.copy(B[i, 0:256], sb, annotations=second)
                for j in T.Parallel(256):
                    C[i, j] = sa[j] + sb[j]

    return kernel.with_attr("global_symbol", "prologue_" + mode)


def inputs():
    positions = np.arange(np.prod(SHAPE), dtype=np.int32).reshape(SHAPE)
    return (
        positions * 31 - 17003,
        13007 - positions * 17,
        np.full(SHAPE, 0x5A5A5A5A, dtype=np.int32),
    )


def expected(mode):
    a, b, c = inputs()
    if mode == "write_through":
        a[1] += 1
    c[1:5] = a[1:5] + b[1:5]
    return a, b, c


def check_raw(mode, raw):
    assert len(raw) == 3
    for value, wanted in zip(raw, expected(mode)):
        assert value[:128] == GUARD and value[-128:] == GUARD
        np.testing.assert_array_equal(np.frombuffer(value[128:-128], dtype="<i4").reshape(SHAPE), wanted)


@DEVICE
@pytest.mark.parametrize("mode", MODES)
def test_sync_prologue_device(mode):
    # Every allocation, including both source arrays, has two 128-byte guards.
    with Session([(make_case(mode), [])]) as current:
        buffers = []
        for array in inputs():
            raw = GUARD + array.astype("<i4", copy=False).tobytes() + GUARD
            buffer = current.buffer(len(raw))
            buffer.write(raw)
            buffers.append(buffer)
        current.kernels[0](*(buffer.at(128, np.prod(SHAPE).item() * 4) for buffer in buffers))
        raw = [buffer.read() for buffer in buffers]
    check_raw(mode, raw)
