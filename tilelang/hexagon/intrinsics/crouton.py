"""Crouton layouts: the native VTCM layouts of HMX operands.

A Crouton tile is a 32x32 FP16 block stored as interleaved row pairs: element
``(row, col)`` sits at ``(row // 2) * 64 + col * 2 + row % 2``. HMX operands
tile a matrix with these blocks in an operand-specific order:

* activation ``[M, K]``: physical ``[..., mt, kt, cpos]``;
* weight ``[K, N]``: physical ``[..., nt, kt, cpos]`` (K tiles contiguous);
* output ``[M, N]``: physical ``[..., mt, nt, cpos]``.
"""

from __future__ import annotations

from collections.abc import Sequence

from tilelang.hexagon import language as T

TILE = 32
TILE_ELEMS = TILE * TILE
TILE_BYTES = TILE_ELEMS * 2

__all__ = [
    "TILE",
    "TILE_ELEMS",
    "TILE_BYTES",
    "make_hmx_crouton_layout",
    "make_hmx_activation_layout",
    "make_hmx_weight_layout",
    "make_hmx_output_layout",
]


def _shape_of(buffer_or_shape) -> tuple:
    shape = getattr(buffer_or_shape, "shape", buffer_or_shape)
    if not isinstance(shape, Sequence):
        raise TypeError("HMX layout expects a Buffer or a shape sequence")
    return tuple(shape)


def _static_extent(value, name: str) -> int:
    value = getattr(value, "value", value)
    if not isinstance(value, int):
        raise ValueError(f"HMX {name} extent must be static; got {value}")
    return value


def _validate_matrix_shape(shape: tuple, name: str) -> tuple[int, int]:
    if len(shape) < 2:
        raise ValueError(f"HMX {name} layout requires at least two dimensions")
    rows = _static_extent(shape[-2], f"{name} rows")
    cols = _static_extent(shape[-1], f"{name} columns")
    if rows <= 0 or cols <= 0 or rows % TILE or cols % TILE:
        raise ValueError(f"HMX {name} dimensions must be positive multiples of {TILE}; got ({rows}, {cols})")
    for i, extent in enumerate(shape[:-2]):
        _static_extent(extent, f"{name} prefix dimension {i}")
    return rows, cols


def make_hmx_crouton_layout() -> T.Layout:
    """Return the native 32x32 FP16 HMX Crouton layout atom."""

    return T.Layout(
        (TILE, TILE),
        lambda row, col: [(row // 2) * (2 * TILE) + col * 2 + row % 2],
    )


def _expand_matrix_layout(shape: tuple, tile_order: tuple[int, int], name: str) -> T.Layout:
    rows, cols = _validate_matrix_shape(shape, name)
    factors = (rows // TILE, cols // TILE)
    layout = make_hmx_crouton_layout()
    for dim in tile_order:
        layout = layout.repeat(dim, factors[dim])
    return layout.expand(shape[:-2])


def make_hmx_activation_layout(buffer_or_shape, transposed: bool = False) -> T.Layout:
    """Map activation storage to physical ``[..., mt, kt, cpos]``.

    The normal logical shape is ``[..., M, K]``.  With ``transposed=True`` the
    input buffer is ``[..., K, M]``, but its physical tile order remains the HMX
    activation order for the mathematical ``[M, K]`` operand.
    """

    shape = _shape_of(buffer_or_shape)
    if not transposed:
        return _expand_matrix_layout(shape, (1, 0), "activation")
    base = _expand_matrix_layout(shape[:-2] + (shape[-1], shape[-2]), (1, 0), "activation")

    def forward(*indices):
        return base(*(indices[:-2] + (indices[-1], indices[-2])))

    return T.Layout(shape, forward)


def make_hmx_weight_layout(buffer_or_shape, transposed: bool = False) -> T.Layout:
    """Map weight storage to physical ``[..., nt, kt, cpos]``.

    The normal logical shape is ``[..., K, N]``.  With ``transposed=True`` the
    input buffer is ``[..., N, K]``, while HMX still sees mathematical ``[K, N]``.
    """

    shape = _shape_of(buffer_or_shape)
    if not transposed:
        return _expand_matrix_layout(shape, (0, 1), "weight")
    base = _expand_matrix_layout(shape[:-2] + (shape[-1], shape[-2]), (0, 1), "weight")

    def forward(*indices):
        return base(*(indices[:-2] + (indices[-1], indices[-2])))

    return T.Layout(shape, forward)


def make_hmx_output_layout(buffer_or_shape) -> T.Layout:
    """Map logical ``[..., M, N]`` to physical ``[..., mt, nt, cpos]``."""

    shape = _shape_of(buffer_or_shape)
    return _expand_matrix_layout(shape, (1, 0), "output")
