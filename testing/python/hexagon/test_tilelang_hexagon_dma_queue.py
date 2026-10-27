"""Hexagon DMA in TIR: submits and commit groups, and the queue sized for them."""

import re

import pytest

import tilelang
from tilelang import tvm
from tilelang.backend.target import determine_target
from tilelang.hexagon import language as T


def _lower(func):
    target = determine_target("hexagon", return_object=True)
    with tvm.target.Target(target), tvm.transform.PassContext(opt_level=3):
        return tilelang.lower(func, target=target, enable_host_codegen=False, enable_device_compile=False)


def _dma_calls(artifact):
    (func,) = artifact.device_mod.functions.values()
    calls = []

    def visit(node):
        if isinstance(node, tvm.tirx.Call) and node.op.name.startswith("tl.hexagon_dma_"):
            calls.append(node)

    tvm.tirx.stmt_functor.post_order_visit(func.body, visit)
    return calls


def _queue(artifact):
    """The (descriptors, groups) capacities of the program's queue."""
    (queue,) = [call for call in _dma_calls(artifact) if call.op.name == "tl.hexagon_dma_queue"]
    return tuple(int(arg) for arg in queue.args)


def _ops(artifact):
    return [call.op.name.removeprefix("tl.hexagon_dma_") for call in _dma_calls(artifact)]


def test_copy_is_one_group():
    @T.prim_func
    def one(A: T.Tensor((256,), "float16")):
        with T.Kernel(1):
            s = T.alloc_shared((256,), "float16")
            T.dma_copy(A, s)
            T.dma_wait()

    artifact = _lower(one)
    assert _ops(artifact) == ["queue", "submit", "commit", "wait"]
    assert _queue(artifact) == (1, 1)


def test_tall_copy_splits_into_one_group():
    rows = 70000

    @T.prim_func
    def column(A: T.Tensor((rows, 2), "float16")):
        with T.Kernel(1):
            s = T.alloc_shared((rows, 1), "float16")
            T.dma_copy(A[0:rows, 0:1], s)
            T.dma_wait()

    artifact = _lower(column)
    # A 2D descriptor has at most 65535 rows.
    assert _ops(artifact) == ["queue", "submit", "submit", "commit", "wait"]
    assert _queue(artifact) == (2, 1)
    submits = re.findall(r"tl_hexagon_dma_async_submit\(tl_dma_async, (.*)\)\) return", artifact.kernel_source)
    assert submits[0].endswith("(uint32_t)65535, 0")
    assert submits[1].endswith(f"(uint32_t){rows - 65535}, 0")
    assert "A[131070]" in submits[1]


def test_queue_is_sized_for_the_copies_in_flight():
    tiles = 8

    @T.prim_func
    def double_buffered(A: T.Tensor((tiles, 256), "float16"), B: T.Tensor((tiles, 256), "float16")):
        with T.Kernel(1):
            s = T.alloc_shared((2, 256), "float16")
            T.dma_copy(A[0, :], s[0, :])
            T.dma_copy(A[1, :], s[1, :])
            for i in T.serial(tiles):
                if i < tiles - 1:
                    T.dma_wait(1)
                else:
                    T.dma_wait()
                T.copy(s[i % 2, :], B[i, :])
                if i + 2 < tiles:
                    T.dma_copy(A[i + 2, :], s[i % 2, :])

    assert _queue(_lower(double_buffered)) == (2, 2)


def test_queue_bound_covers_both_branches():
    @T.prim_func
    def branches(A: T.Tensor((3, 256), "float16"), flag: T.int32):
        with T.Kernel(1):
            s = T.alloc_shared((3, 256), "float16")
            if flag > 0:
                T.dma_copy(A[0, :], s[0, :])
            else:
                for i in T.serial(3):
                    T.dma_copy(A[i, :], s[i, :])
            T.dma_wait()

    assert _queue(_lower(branches)) == (3, 3)


def test_loop_without_waits_is_rejected():
    @T.prim_func
    def unbounded(A: T.Tensor((256,), "float16"), n: T.int32):
        with T.Kernel(1):
            s = T.alloc_shared((256,), "float16")
            for _ in T.serial(n):
                T.dma_copy(A, s)
            T.dma_wait()

    with pytest.raises(Exception, match="has no bound"):
        _lower(unbounded)


def test_too_many_copies_in_flight_are_rejected():
    @T.prim_func
    def flood(A: T.Tensor((300, 64), "float16")):
        with T.Kernel(1):
            s = T.alloc_shared((300, 64), "float16")
            for i in T.serial(300):
                T.dma_copy(A[i, :], s[i, :])
            T.dma_wait()

    with pytest.raises(Exception, match="more than 256 DMA transfers in flight"):
        _lower(flood)
