"""Panel layouts: the layout of integer GEMM operands that HVX ``vrmpy`` reads.

``vrmpy`` multiplies the 4 bytes of each 32-bit lane of a register by 4 bytes
and adds them up. A register of a matrix in a panel layout holds 32 rows, each
row's lane ``lane`` consecutive columns of it: 4 columns of 8 bits, or 8 of 4
bits. A panel is 32 rows by every column, one column block after another, so
element ``(r, c)`` sits at::

    (r // 32) * 32 * cols + (c // lane) * 32 * lane + (r % 32) * lane + c % lane

Within a lane the columns keep their order; two 4-bit elements share a byte,
the first in its low nibble. Leading dimensions stack whole matrices.

``T.gemm`` with integer operands on HVX reads B, ``[N, K]`` with
``transpose_B``, in this layout. A parameter annotated with it arrives packed
(``pack_panel``), and copies between two panel buffers move whole panels by DMA.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import tvm_ffi
from tvm import DataType

from tilelang.layout import Layout

ROWS = 32

__all__ = ["PanelLayout", "make_hvx_panel_layout", "pack_panel", "unpack_panel"]


@tvm_ffi.register_object("tl.hexagon.PanelLayout")
class PanelLayout(Layout):
    """Panels over the last two dimensions of a buffer, ``lane`` columns per 32-bit lane."""

    def __init__(self, shape, lane: int):
        self.__init_handle_by_constructor__(tvm_ffi.get_global_func("tl.hexagon.PanelLayout"), list(shape), lane)


def _lane(dtype) -> int:
    bits = DataType(str(dtype)).bits
    if bits not in (4, 8):
        raise ValueError(f"a panel layout holds 4- or 8-bit integers, not {dtype}")
    return 32 // bits


def make_hvx_panel_layout(buffer_or_shape, dtype=None) -> PanelLayout:
    """The panel layout of a buffer, or of a shape of ``dtype`` elements."""
    shape = getattr(buffer_or_shape, "shape", buffer_or_shape)
    if not isinstance(shape, Sequence):
        raise TypeError("make_hvx_panel_layout expects a Buffer or a shape sequence")
    dtype = dtype if dtype is not None else buffer_or_shape.dtype
    return PanelLayout(list(shape), _lane(dtype))


def _panels(values: torch.Tensor, lane: int) -> torch.Tensor:
    """``values`` [..., rows, cols] reordered to [..., rows/32, cols/lane, 32, lane]."""
    *lead, rows, cols = values.shape
    if rows % ROWS or cols % lane:
        raise ValueError(f"a panel matrix has rows in multiples of {ROWS} and columns in multiples of {lane}, got {rows}x{cols}")
    v = values.reshape(*lead, rows // ROWS, ROWS, cols // lane, lane)
    return v.transpose(-3, -2).contiguous()


def pack_panel(values: torch.Tensor, bits: int) -> torch.Tensor:
    """The bytes of an integer matrix stored in the panel layout, as uint8.

    ``values`` [..., rows, cols] holds the integers (signed or unsigned); each
    keeps its low ``bits`` bits.
    """
    if bits not in (4, 8):
        raise ValueError(f"pack_panel packs 4- or 8-bit integers, not {bits}-bit")
    p = _panels(values.to(torch.int64) & ((1 << bits) - 1), 32 // bits).reshape(-1)
    if bits == 4:
        p = p[0::2] | (p[1::2] << 4)
    return p.to(torch.uint8)


def unpack_panel(packed: torch.Tensor, shape: Sequence[int], bits: int, signed: bool) -> torch.Tensor:
    """The integer matrix [..., rows, cols] whose panel-layout bytes are ``packed``."""
    lane = 32 // bits
    v = packed.reshape(-1).to(torch.int64)
    if bits == 4:
        v = torch.stack([v & 0xF, v >> 4], dim=-1).reshape(-1)
    if signed:
        v = v - ((v >> (bits - 1)) << bits)
    *lead, rows, cols = shape
    v = v.reshape(*lead, rows // ROWS, cols // lane, ROWS, lane).transpose(-3, -2)
    return v.reshape(*lead, rows, cols)
