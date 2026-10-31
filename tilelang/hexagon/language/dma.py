"""Explicit asynchronous DMA copies of the Hexagon dialect."""

from __future__ import annotations

from tvm import tirx
from tvm.tirx.script.builder.ir import sblock_attr

from tilelang._typing import BufferLikeType
from tilelang.language.copy_op import _normalize_copy_regions
from tilelang.utils.language import to_buffer_region

__all__ = ["annotate_dma_only", "annotate_write_through", "dma_copy", "dma_wait"]


def dma_copy(src: BufferLikeType, dst: BufferLikeType) -> tirx.PrimExpr:
    """Submit one asynchronous Hexagon DDR <-> VTCM DMA transfer.

    Supports in-bounds 1D/2D row-major regions with identical scalar dtypes.
    Each call is one group of the tile program's DMA queue: its descriptors
    complete together. Use ``T.dma_wait(n)`` before consuming destinations or
    overwriting sources; no separate commit is needed. Every tile program owns
    its queue, which the compiler sizes for the most transfers in flight, and
    drains it before the program ends. Raw DMA calls must not be mixed with
    this managed queue. Compiler-generated software pipelines remain
    unsupported; use explicit prefetch and wait scheduling.
    """
    src, dst = _normalize_copy_regions(src, dst)
    if isinstance(src, tirx.BufferLoad) and isinstance(dst, tirx.BufferLoad):
        src = to_buffer_region(src, access_type="r", extents=[1] * len(src.indices))
        dst = to_buffer_region(dst, access_type="w", extents=[1] * len(dst.indices))
    return tirx.call_intrin("handle", tirx.op.Op.get("tl.tileop.dma_copy"), src, dst)


def dma_wait(pending: int = 0) -> tirx.PrimExpr:
    """Wait until at most the newest ``pending`` DMA copies are in flight.

    The older copies complete and their queue entries are reclaimed. ``0`` waits
    for every submission, including DDR cache maintenance.
    """
    if isinstance(pending, bool) or not isinstance(pending, int) or pending < 0:
        raise ValueError("dma_wait pending must be a nonnegative integer")
    return tirx.call_intrin("void", tirx.op.Op.get("tl.hexagon_dma_wait"), pending)


def annotate_dma_only(*buffers: tirx.Buffer):
    """Declare DDR buffer parameters that only DMA moves.

    DMA reads and writes DDR past the DSP's caches, so a transfer's DDR side
    costs cache maintenance, about a cycle every 16 bytes: a load first writes
    the source's lines back, and a store writes the destination's lines back
    and then drops them, so that DMA and the DSP's own loads and stores see the
    same bytes. A buffer declared here is moved by ``T.dma_copy`` alone, in
    every kernel of the session that reaches it, so the DSP holds none of its
    lines and its transfers skip that maintenance: a kernel's scratch in DDR,
    such as rows it moves out of VTCM and back.

    The compiler checks this kernel: any other use of the buffer (a load, a
    store, a ``T.copy`` that is not a DMA) is an error. It cannot check the
    session's other kernels; do not declare a buffer that one of them reads or
    writes itself.
    """
    for buffer in buffers:
        if not isinstance(buffer, tirx.Buffer):
            raise TypeError(f"annotate_dma_only takes buffers, got {type(buffer).__name__}")
    if not buffers:
        return None
    return sblock_attr({"tl.hexagon_dma_only": [buffer.data for buffer in buffers]})


def annotate_write_through(*buffers: tirx.Buffer):
    """Declare DDR buffer parameters whose every write by the DSP is written
    back to DDR right away, so that DMA reads them with no writeback first.

    The DSP's stores to DDR stay in its caches, which DMA does not read, so a
    DMA load first writes its source's lines back, about a cycle every 16
    bytes it moves. For a buffer declared here the compiler writes back what
    each write wrote instead, right after it: after a ``T.copy`` into the
    buffer, a ``T.Parallel`` loop that stores into it, or another store (of a
    buffer in Crouton tiles, the whole tiles a write touches). Its DMA loads
    then skip their writeback. That pays for a buffer that DMA reads many
    times after each write, such as a KV cache.

    Every kernel of the session that writes such a buffer must declare it,
    which the compiler cannot check across kernels. Other tile ops that write
    the buffer are rejected.
    """
    for buffer in buffers:
        if not isinstance(buffer, tirx.Buffer):
            raise TypeError(f"annotate_write_through takes buffers, got {type(buffer).__name__}")
    if not buffers:
        return None
    return sblock_attr({"tl.hexagon_write_through": [buffer.data for buffer in buffers]})
