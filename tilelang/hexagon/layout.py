"""Crouton tile layout for the Hexagon HMX matrix engine, expressed as a tilelang
Layout so LayoutInference can push it onto the global→VTCM ``T.copy``.

HMX reads/writes operands in a tiled, row-pair-interleaved ("Crouton"/"deep")
layout: a 2D operand is cut into 32×32 tiles, and within a tile the operand's
natural element (row r, col c) lands at ``cpos(r, c) = (r & ~1) * 32 + c * 2 +
(r & 1)`` (== ``(r//2)*64 + c*2 + (r%2)``).  Tiles are ordered row-major for the
activation/output and col-major for the weight — matching ``pack_A`` / ``pack_B``
/ ``unpack_C`` in ``src/tl_templates/hexagon/hmx.h``.

The intra-tile ``cpos`` is ALWAYS over the operand's *natural* (row, col); a
transposed buffer (e.g. a weight stored ``[N, K]`` with ``transpose_B=True``)
just swaps which buffer axis is row vs col — it does NOT transpose the tile
interior.  Getting that wrong silently produces a transposed weight tile and a
wrong matmul, so the ``transposed`` flag is handled explicitly here.

Making this a layout (not a repack) lets the load write Crouton directly into
VTCM, and HMX consumes it with no intermediate copy.
"""

from __future__ import annotations

from tilelang import language as T

TILE = 32
TILE_ELMS = TILE * TILE  # 1024 elements per 32×32 tile


def make_crouton_layout(shape, role: str = "row", transposed: bool = False):
    """Crouton VTCM layout for a 2D HMX operand of buffer ``shape``.

    role        : "row" → tiles row-major over col-tiles (activation A, output C);
                  "col" → tiles col-major over row-tiles (weight B, pack_B).
    transposed  : the buffer stores the operand transposed, i.e. buffer index
                  (i, j) is the operand's (col, row).  ``cpos`` is always taken
                  over the operand's natural (row, col), so this swaps i/j.
    """
    assert role in ("row", "col"), f"role must be 'row' or 'col', got {role!r}"
    d0, d1 = int(shape[0]), int(shape[1])
    assert d0 % TILE == 0 and d1 % TILE == 0, f"Crouton needs 32-multiple dims, got {tuple(shape)}"
    # Operand-natural (rows, cols): reversed when the buffer is transposed.
    nrows, ncols = (d1, d0) if transposed else (d0, d1)
    nrt, nct = nrows // TILE, ncols // TILE

    def fwd(i, j):
        r, c = (j, i) if transposed else (i, j)  # operand-natural (row, col)
        rt, ct = r // TILE, c // TILE
        rr, cc = r % TILE, c % TILE
        tile = (rt * nct + ct) if role == "row" else (ct * nrt + rt)
        cpos = (rr // 2) * 64 + cc * 2 + (rr % 2)
        return [tile * TILE_ELMS + cpos]

    return T.Layout(shape, fwd)
