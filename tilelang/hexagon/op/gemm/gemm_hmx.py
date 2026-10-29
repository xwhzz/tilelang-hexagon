"""Hexagon HMX implementation of ``T.gemm`` and ``T.gemm_blockscaled``.

Layout inference gives A and B the HMX activation and weight layouts, Crouton
tiles in VTCM that the surrounding copies fill. A call is one function of its
own (``hmx_gemm.h``): each 32x32 output tile reduces the call's K in the HMX
accumulator, in deep packets of at most 32 K tiles, and converts to fp16. HMX
accumulates within a call, and HVX across calls.

An fp16 C keeps the HMX output layout, and a call that overwrites it stores
its tiles there directly. Every other C takes them through a staging tile,
which an HVX epilogue writes or adds into C (``tl_crouton_tile_store``): an
fp32 or fp16 C, row-major, or an fp16 C in the output layout that the call
accumulates into. Each tile's epilogue runs while HMX multiplies the next one.
A C of another layout takes the product from a row-major matrix of its own,
element by element.

In a worker pool, whose workers share HMX (``LowerKernelLaunch`` marks such
calls), each output tile is a critical section of its own instead, and its
epilogue runs outside it, while another worker multiplies.

``T.gemm_blockscaled`` of an fp16 A runs here too (:class:`GemmHMXBlockScaled`):
HVX dequantizes its 4-bit B into fp16 weight tiles, a column of output tiles
at a time, for HMX to multiply every row of A by.
"""

from __future__ import annotations

import tvm_ffi
from tvm import tirx

from tilelang.hexagon import language as T
from tilelang.hexagon.intrinsics.crouton import (
    TILE,
    TILE_ELEMS,
    CroutonLayout,
    make_hmx_activation_layout,
    make_hmx_output_layout,
    make_hmx_weight_layout,
)
from tilelang.hexagon.intrinsics.panel import PanelLayout, make_hvx_panel_layout
from tilelang.tileop.gemm.gemm_base import GemmBase
from tilelang.tileop.gemm_blockscaled.gemm_blockscaled_base import GemmBlockScaledMixin

from .gemm_hvx import _strides, scale_group

GEMM_INST_HMX = "hexagon.hmx"
GEMM_INST_HMX_BLOCK_SCALED = "hexagon.hmx.blockscaled"

# The worker count LowerKernelLaunch puts on the gemms of a worker pool.
_NUM_WORKERS = "hexagon.num_workers"
# The conversion config HMX reads: 32 scale words, then 32 bias words.
CONFIG_WORDS = 64

_row_major = tvm_ffi.get_global_func("tl.hexagon.RowMajor")


def _flag(value: bool) -> str:
    return "true" if value else "false"


def _origin(region) -> tuple:
    """The first index of a buffer region, constants as Python ints."""
    return tuple(int(r.min) if isinstance(r.min, tirx.IntImm) else r.min for r in region.region)


def _first(region):
    """The first element of a buffer region."""
    return region.buffer[_origin(region)]


def _tiles(extent) -> int:
    """The tiles of a static extent."""
    return int(extent) // TILE


class GemmHMX(GemmBase):
    """``T.gemm`` of fp16 operands on HMX: one ``tl_hexagon_hmx_gemm`` call."""

    def _check_shape(self, name: str) -> None:
        for dim, extent in (("M", self.M), ("N", self.N), ("K", self.K)):
            if extent <= 0 or extent % TILE:
                raise ValueError(f"{name}: {dim} = {extent} is not a positive multiple of {TILE}")

    def infer_layout(self, target, thread_nums: int):
        layouts = {
            self.A: make_hmx_activation_layout(self.A, self.trans_A),
            self.B: make_hmx_weight_layout(self.B, self.trans_B),
        }
        # An fp32 C stays row-major, where each register of the epilogue is
        # one row of a tile.
        if str(self.C.dtype) == "float16":
            layouts[self.C] = make_hmx_output_layout(self.C)
        return layouts

    def _check_operand_layouts(self, layout_map) -> None:
        """A and B must have the HMX layouts: another layout given first
        (T.annotate_layout, or another operator) stays on the buffer."""

        for name, buffer, column_tiles, transposed in (
            ("A", self.A, False, self.trans_A),
            ("B", self.B, True, self.trans_B),
        ):
            layout = layout_map.get(buffer, None)
            if not (
                isinstance(layout, CroutonLayout)
                and bool(layout.column_tiles) == column_tiles
                and bool(layout.transposed) == bool(transposed)
            ):
                role = "weight" if column_tiles else "activation"
                raise ValueError(
                    f"T.gemm on HMX reads {name} ({buffer.name}) in the HMX {role} layout"
                    f"{' (transposed)' if transposed else ''}, but it has another layout: {layout}"
                )

    def _c_kind(self, layout_map) -> str:
        """How the tiles reach C: "crouton" (C's own tiles), "rows"
        (row-major) or "elements" (through a row-major matrix)."""

        layout = layout_map.get(self.C, None)
        if str(self.C.dtype) == "float16" and isinstance(layout, CroutonLayout) and not layout.column_tiles and not layout.transposed:
            return "crouton"
        if _row_major(self.C, layout_map):
            return "rows"
        return "elements"

    def _shared(self) -> bool:
        """Whether the workers of a pool take turns at HMX for this call."""
        workers = self.gemm_node.annotations.get(_NUM_WORKERS, None)
        return workers is not None and int(workers) > 1

    def _a(self):
        """A's first tile, and the elements between its row tiles: the K
        tiles of a row tile are contiguous, along the buffer's K extent."""
        A_buf = self.ARegion.buffer
        row_tile = _tiles(A_buf.shape[-2] if self.trans_A else A_buf.shape[-1]) * TILE_ELEMS
        extent = (self.M // TILE - 1) * row_tile + self.K * TILE
        return T.access_ptr(_first(self.ARegion), "r", extent), row_tile

    def _c(self, kind: str):
        """C's first element, its leading dimension (elements, or tiles of
        Crouton rows), whether its rows are aligned, and how the call
        accesses it."""
        C_buf = self.CRegion.buffer
        clear = self.clear_accum
        mode = "w" if isinstance(clear, tirx.IntImm) and bool(clear.value) else "rw"
        if kind == "rows":
            ldc = C_buf.shape[-1]
            # Rows of a tile on register (fp32) or half-register (fp16)
            # boundaries: the region starts on a tile, and VTCM buffers on
            # registers.
            aligned = isinstance(ldc, tirx.IntImm) and ldc.value % TILE == 0
            extent = (self.M - 1) * ldc + self.N
        else:
            ldc = _tiles(C_buf.shape[-1])
            aligned = True
            extent = ((self.M // TILE - 1) * ldc + self.N // TILE) * TILE_ELEMS
        return T.access_ptr(_first(self.CRegion), mode, extent), ldc, aligned

    def lower(self, layout_map, target, thread_bounds, thread_var, mbar_phase_expr=None):
        name = f"T.gemm({self.A.name}, {self.B.name}, {self.C.name}) on HMX"
        self._check_shape(name)
        if str(self.a_dtype) != "float16" or str(self.b_dtype) != "float16":
            raise ValueError(f"{name} multiplies fp16 operands, not {self.a_dtype} by {self.b_dtype}")
        self._check_operand_layouts(layout_map)
        kind = self._c_kind(layout_map)
        clear = self.clear_accum
        static_clear = bool(clear.value) if isinstance(clear, tirx.IntImm) else None
        # An overwritten Crouton C takes HMX's tiles as they are.
        direct = kind == "crouton" and static_clear is True
        a, a_row_tile = self._a()
        B_buf = self.BRegion.buffer
        b_column_tile = _tiles(B_buf.shape[-1] if self.trans_B else B_buf.shape[-2]) * TILE_ELEMS
        b = T.access_ptr(_first(self.BRegion), "r", (self.N // TILE - 1) * b_column_tile + self.K * TILE)
        gemm = tirx.op.Op.get("tl.hexagon_hmx_gemm")
        shared = _flag(self._shared())
        M, N, K = self.M, self.N, self.K

        def call(c, ldc, rows: bool, aligned: bool, clear, stage, config):
            flags = [_flag(rows), _flag(aligned), shared, _flag(direct)]
            args = [M, N, K, c, ldc, a, a_row_tile, b, b_column_tile, T.Cast("bool", clear)]
            args += [T.access_ptr(stage[0], "rw", 2 * TILE_ELEMS), T.access_ptr(config[0], "rw", CONFIG_WORDS)]
            return tirx.call_intrin("void", gemm, *flags, *args)

        if kind != "elements":
            c, ldc, aligned = self._c(kind)

            @T.prim_func
            def _gemm_hmx() -> None:
                # Two staging tiles, HMX storing one while the epilogue reads
                # the other, and the conversion config.
                config = T.alloc_shared((CONFIG_WORDS,), "uint32", align=256)
                stage = T.alloc_shared((2 * TILE_ELEMS,), "float16", align=2048)
                T.evaluate(call(c, ldc, kind == "rows", aligned, clear, stage, config))

            return _gemm_hmx

        C_buf = self.CRegion.buffer
        c_origin = _origin(self.CRegion)

        def at(i, j):
            return c_origin[:-2] + (c_origin[-2] + i, c_origin[-1] + j)

        @T.macro
        def write(product, accumulate: bool):
            for i, j in T.grid(M, N):
                if accumulate:
                    C_buf[at(i, j)] = C_buf[at(i, j)] + product[i, j]
                else:
                    C_buf[at(i, j)] = product[i, j]

        @T.prim_func
        def _gemm_hmx_elements() -> None:
            config = T.alloc_shared((CONFIG_WORDS,), "uint32", align=256)
            stage = T.alloc_shared((2 * TILE_ELEMS,), "float16", align=2048)
            # The product, row-major, which C's elements then take.
            product = T.alloc_shared((M, N), C_buf.dtype)
            T.evaluate(call(T.access_ptr(product[0, 0], "w", M * N), N, True, True, True, stage, config))
            if static_clear is None:
                if clear:
                    write(product, False)
                else:
                    write(product, True)
            else:
                write(product, not static_clear)

        return _gemm_hmx_elements


class GemmHMXBlockScaled(GemmBlockScaledMixin, GemmHMX):
    """``T.gemm_blockscaled`` of an fp16 A on HMX: ``C (+)= (A * SFA) @ (B *
    SFB)^T`` for a 4-bit B in the panel layout, its rows along K
    (``transpose_B``), the decode GEMV's operand.

    A has the HMX activation layout; SFA is fp32, one factor per row across
    the call's K, ``[M, K / ga]`` with ga >= K; SFB is fp16 and group-major,
    ``[K / gb, N]``, gb a multiple of 8. C is an fp32 row-major or fp16
    Crouton matrix.

    The call is one ``tl_hexagon_hmx_gemm_blockscaled`` (``hmx_gemm.h``):
    HVX dequantizes B 32 rows at a time, one column of output tiles, into
    fp16 weight tiles (``tl_crouton_dequant4``), and HMX multiplies every row
    tile of A by them, so each weight serves all of A's rows. In one program a
    slice of the next column dequantizes while HMX multiplies each tile; in a
    worker pool each tile is a critical section, and the slice and the tile's
    epilogue run while other workers multiply.
    """

    def __post_init__(self) -> None:
        # B dequantizes to fp16 before HMX multiplies: an fp16 A by a 4-bit B
        # is not a mixed-dtype GEMM, whose operands the common rule checks.
        pass

    def infer_layout(self, target, thread_nums: int):
        layouts = {self.A: make_hmx_activation_layout(self.A, self.trans_A), self.B: make_hvx_panel_layout(self.B)}
        if str(self.C.dtype) == "float16":
            layouts[self.C] = make_hmx_output_layout(self.C)
        return layouts

    def lower(self, layout_map, target, thread_bounds, thread_var, mbar_phase_expr=None):
        name = f"T.gemm_blockscaled({self.A.name}, {self.B.name}, {self.C.name}) on HMX"
        self._check_shape(name)
        if self.trans_A:
            raise ValueError(f"{name} reads A as [M, K], not transposed")
        layout = layout_map.get(self.A, None)
        if not (isinstance(layout, CroutonLayout) and not layout.column_tiles and not layout.transposed):
            raise ValueError(f"{name} reads A in the HMX activation layout, but it has {layout}")
        if not isinstance(layout_map.get(self.B, None), PanelLayout):
            raise ValueError(f"{name} reads B in the panel layout, but it has {layout_map.get(self.B, None)}")
        kind = self._c_kind(layout_map)
        if kind == "elements":
            raise ValueError(f"{name} writes C row-major or in Crouton tiles, not {layout_map.get(self.C, None)}")
        M, N, K = self.M, self.N, self.K
        ga, first_a = scale_group(self, "A", self.sf_a_granularity_k)
        gb, first_b = scale_group(self, "B", self.sf_b_granularity_k)
        if ga != K:
            raise ValueError(f"{name} scales whole rows of A: one factor across the call's {K} columns, not one per {ga}")
        if gb % 8:
            raise ValueError(f"{name} dequantizes B in groups of 8 columns of K: its scale groups are a multiple of 8, not {gb}")

        a, a_row_tile = self._a()
        # B's region starts on a panel: a pointer at its logical start, which
        # LowerTileOp moves through the layout to the panel's bytes.
        b = T.access_ptr(_first(self.BRegion), "r", N * K)
        panel_bytes = TILE * int(self.B.shape[-1]) * 4 // 8
        signed = str(self.B.dtype).startswith("int")

        sfb_region = self.SFBRegion
        sfb_origin = _origin(sfb_region)
        if len(sfb_origin) < 2:
            raise ValueError(f"{name}: SFB {sfb_region.buffer.name} is [K / {gb}, N], group-major")
        ld_sfb = int(_strides(sfb_region.buffer)[-2])
        sfb_first = sfb_origin[:-2] + (sfb_origin[-2] + first_b, sfb_origin[-1])
        sfb = T.access_ptr(sfb_region.buffer[sfb_first], "r", (K // gb - 1) * ld_sfb + N)

        sfa_region = self.SFARegion
        sfa_origin = _origin(sfa_region)
        if len(sfa_origin) < 2:
            raise ValueError(f"{name}: SFA {sfa_region.buffer.name} is [M, K / {ga}], a factor per row of A")
        ld_sfa = int(_strides(sfa_region.buffer)[-2])
        sfa_first = sfa_origin[:-1] + (sfa_origin[-1] + first_a,)
        sfa = T.access_ptr(sfa_region.buffer[sfa_first], "r", (M - 1) * ld_sfa + 1)

        c, ldc, aligned = self._c(kind)
        clear = self.clear_accum
        gemm = tirx.op.Op.get("tl.hexagon_hmx_gemm_blockscaled")
        flags = [_flag(signed), _flag(kind == "rows"), _flag(aligned), _flag(True), _flag(self._shared())]

        @T.prim_func
        def _gemm_hmx_blockscaled() -> None:
            # Two columns of weight tiles, two staging tiles and the conversion
            # config: HMX reads one column while HVX fills the other.
            config = T.alloc_shared((CONFIG_WORDS,), "uint32", align=256)
            stage = T.alloc_shared((2 * TILE_ELEMS,), "float16", align=2048)
            w = T.alloc_shared((2 * K * TILE,), "float16", align=2048)
            T.evaluate(
                tirx.call_intrin(
                    "void",
                    gemm,
                    *flags,
                    M,
                    N,
                    K,
                    c,
                    ldc,
                    a,
                    a_row_tile,
                    b,
                    panel_bytes,
                    sfb,
                    ld_sfb,
                    gb,
                    sfa,
                    ld_sfa,
                    T.Cast("bool", clear),
                    T.access_ptr(w[0], "rw", 2 * K * TILE),
                    T.access_ptr(stage[0], "rw", 2 * TILE_ELEMS),
                    T.access_ptr(config[0], "rw", CONFIG_WORDS),
                )
            )

        return _gemm_hmx_blockscaled
