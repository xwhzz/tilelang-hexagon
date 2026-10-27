"""Explicit asynchronous DMA copies of the Hexagon dialect."""

from __future__ import annotations

from tvm import tirx

from tilelang._typing import BufferLikeType
from tilelang.language.copy_op import _normalize_copy_regions
from tilelang.utils.language import to_buffer_region

__all__ = ["dma_copy", "dma_wait"]


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
