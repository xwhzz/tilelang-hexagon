"""Hexagon HVX implementation of ``T.gemm``, for the GEMMs HMX does not run.

A floating-point GEMM is one ``tl_hvx_gemm`` call (``hvx_gemm.h``) on
row-major operands: outer products when B's rows run along N, dot products
when A's and B's rows both run along K (``transpose_B``). Products accumulate
in registers over the call's K, and reach C once. Operands keep their layouts:
an fp16 B in the HMX weight layout (``make_hmx_weight_layout``, the copy of a
model's weights that HMX multiplies in its prefill) is one
``tl_hvx_gemm_weight`` call, a decode's GEMV; a GEMM whose operand another
operator gave another layout runs as the scalar loop.

An integer GEMM (``transpose_B``: an int8, uint8 or int16 A, an 8- or 4-bit B,
an int32 C) is one ``tl_hvx_gemm_int`` call, exact: ``vrmpy`` multiplies the
bytes of B's 32-bit lanes by A's, so B, in VTCM, takes the panel layout
(``tilelang.hexagon.intrinsics.panel``), and A and C stay row-major.
"""

from __future__ import annotations

import logging

import tvm_ffi
from tvm import DataType, arith, ir, tirx

from tilelang.cpu.op.gemm.gemm_scalar import GemmScalar
from tilelang.hexagon import language as T
from tilelang.hexagon.intrinsics.crouton import TILE, TILE_ELEMS, CroutonLayout
from tilelang.hexagon.intrinsics.panel import PanelLayout, make_hvx_panel_layout
from tilelang.language.eager.utils import construct_strides
from tilelang.tileop.gemm_blockscaled.gemm_blockscaled_base import GemmBlockScaledMixin
from tilelang.tileop.gemm.gemm_base import GemmBase

GEMM_INST_HVX = "hexagon.hvx"
GEMM_INST_HVX_BLOCK_SCALED = "hexagon.hvx.blockscaled"

logger = logging.getLogger(__name__)

_row_major = tvm_ffi.get_global_func("tl.hexagon.RowMajor")
_hmx_rejection = tvm_ffi.get_global_func("tl.hexagon.GemmHMXRejection")


def _strides(buffer) -> list:
    if buffer.strides:
        return list(buffer.strides)
    strides = [1]
    for extent in reversed(buffer.shape[1:]):
        strides.insert(0, strides[0] * extent)
    return strides


def scale_group(gemm, operand: str, g) -> tuple:
    """The columns per factor of `operand` ("A" or "B") of the block-scaled
    `gemm` across its K, and the index of its first factor: K when the call's
    columns lie in one group of g, else g, when they split into whole groups
    of g."""
    K, k0 = gemm.K, gemm.sf_k_start
    analyzer = arith.Analyzer()
    if analyzer.can_prove(tirx.floormod(k0, g) + K <= g):
        return K, k0 // g
    if K % g or not analyzer.can_prove(tirx.floormod(k0, g) == 0):
        raise ValueError(
            f"T.gemm_blockscaled: columns [{k0}, {k0} + {K}) neither lie in one group of {g} {operand} scale columns "
            "nor split into whole ones"
        )
    return g, k0 // g


def _external_strided_view(buf, layout_map):
    """Preserve the physical strides in an explicitly strided global ABI."""
    if buf.scope() != "global" or not buf.strides or buf not in layout_map:
        return buf
    analyzer = arith.Analyzer()
    if all(analyzer.can_prove_equal(s, d) for s, d in zip(buf.strides, construct_strides(buf.shape))):
        return buf
    if not _row_major(buf, layout_map):
        raise ValueError(f"Hexagon GEMM: {buf.name} combines noncompact global strides with a non-row-major layout")
    # LowerTileOp remaps annotated loads to compact storage. The global ABI
    # has its original strides, so access it through its unannotated view.
    return tirx.decl_buffer(buf.shape, buf.dtype, data=buf.data, strides=buf.strides, elem_offset=buf.elem_offset, name=buf.name)


def _sfb_row_stride(buf, layout_map, origin, groups, cols):
    """The physical row stride if this slice has contiguous columns."""
    analyzer = arith.Analyzer()
    layout = layout_map.get(buf, None)
    if layout is None:
        strides = _strides(buf)
        return strides[-2] if analyzer.can_prove_equal(strides[-1], 1) else None
    if _row_major(buf, layout_map):
        return buf.shape[-1]  # LowerTileOp remaps layouts to compact physical buffers.

    # A submatrix can be directly addressable even within a non-row-major
    # buffer, e.g. its batch axes have a different physical order.
    axes = layout.get_forward_vars()
    if len(axes) != len(origin):
        return None
    g, n = tirx.Var("g", origin[-2].dtype), tirx.Var("n", origin[-1].dtype)
    analyzer.bind(g, ir.Range.from_min_extent(0, groups))
    analyzer.bind(n, ir.Range.from_min_extent(0, cols))
    indices = (*origin[:-2], origin[-2] + g, origin[-1] + n)
    linear = tirx.stmt_functor.substitute(layout.get_linearized_forward_index(), dict(zip(axes, indices)))
    coeffs = arith.detect_linear_equation(analyzer.simplify(linear), [g, n])
    if coeffs and analyzer.can_prove_equal(coeffs[1], 1) and analyzer.can_prove(coeffs[0] >= 0):
        return coeffs[0]
    return None


def group_major_sfb(region, layout_map, first_group, groups, cols):
    """Return a reader and row stride for the GEMM's [group, column] scales."""
    buf = region.buffer
    origin = tuple(r.min for r in region.region)
    if len(origin) < 2:
        raise ValueError(f"T.gemm_blockscaled: SFB {buf.name} must have one row per scale group")
    origin = (*origin[:-2], origin[-2] + first_group, origin[-1])
    buf = _external_strided_view(buf, layout_map)
    ld = _sfb_row_stride(buf, layout_map, origin, groups, cols)
    if ld is not None:
        ptr = T.access_ptr(buf[origin], "r", (groups - 1) * ld + cols)
        return lambda: ptr, ld

    @T.macro
    def pack():
        packed_sfb = T.alloc_shared((groups, cols), buf.dtype)
        for g, n in T.Parallel(groups, cols):
            packed_sfb[g, n] = buf[(*origin[:-2], origin[-2] + g, origin[-1] + n)]
        return T.access_ptr(packed_sfb[0, 0], "r", groups * cols)

    return pack, cols


def _is_integer(dtype) -> bool:
    return str(dtype).startswith(("int", "uint"))


class GemmHVX(GemmBase):
    """Lower one logical GEMM into a ``tl_hvx_gemm`` or ``tl_hvx_gemm_int`` call."""

    def __post_init__(self) -> None:
        # vrmpy multiplies integers of different widths, such as int16 by
        # int4, exactly; other operands follow the common rule.
        if not self.integer:
            super().__post_init__()

    @property
    def integer(self) -> bool:
        return _is_integer(self.A.dtype) and _is_integer(self.B.dtype)

    def infer_layout(self, target, thread_nums: int):
        if self.integer:
            return {self.B: make_hvx_panel_layout(self.B)}
        return {}

    def _operand(self, region, rows: int, cols: int, access: str):
        """A pointer to the region's first element that covers `rows` rows of
        `cols` elements, the row stride, and whether the region starts on a
        register boundary."""

        buffer = region.buffer
        strides = _strides(buffer)
        ld = int(strides[-2])
        origin = [r.min for r in region.region]
        offset = sum((m * s for m, s in zip(origin, strides)), tirx.const(0, "int64"))
        # Buffers start on a register boundary: VTCM arena buffers and stack
        # arrays are 128-byte aligned.
        size = DataType(buffer.dtype).bits // 8
        aligned = arith.Analyzer().can_prove(tirx.floormod(offset * size, 128) == 0)
        pointer = T.access_ptr(buffer[tuple(origin)], access, (rows - 1) * ld + cols)
        return pointer, ld, aligned

    def _integer_operands(self, layout_map):
        """A, C and B of an integer GEMM: B's pointer, the bytes between its
        panels, its bits and its signedness."""
        layout = layout_map.get(self.B, None)
        if not isinstance(layout, PanelLayout):
            raise ValueError(
                f"T.gemm({self.A.name}, {self.B.name}, {self.C.name}): HVX reads an integer B in the panel layout, but {self.B.name} has {layout}"
            )
        others = [buffer.name for buffer in (self.A, self.C) if not _row_major(buffer, layout_map)]
        if others:
            raise ValueError(
                f"T.gemm({self.A.name}, {self.B.name}, {self.C.name}): HVX reads integer A and C row-major, not {', '.join(others)}"
            )
        M, N, K = self.M, self.N, self.K
        a, lda, _ = self._operand(self.ARegion, M, K, "r")
        c, ldc, align_c = self._operand(self.CRegion, M, N, "rw")
        # B's region starts on a panel: a pointer at its logical start, which
        # LowerTileOp moves through the layout to the panel's bytes.
        origin = [r.min for r in self.BRegion.region]
        bits = DataType(str(self.B.dtype)).bits
        b = T.access_ptr(self.B[tuple(origin)], "r", N * K)
        panel_bytes = 32 * int(self.B.shape[-1]) * bits // 8
        signed = str(self.B.dtype).startswith("int")
        return a, lda, c, ldc, align_c, b, panel_bytes, bits, signed

    def _lower_integer(self, layout_map):
        M, N, K = self.M, self.N, self.K
        a, lda, c, ldc, align_c, b, panel_bytes, bits, signed = self._integer_operands(layout_map)
        gemm = tirx.op.Op.get("tl.hexagon_hvx_gemm_int")
        flags = [tirx.const(bits, "int32"), tirx.const(signed, "bool"), lda, ldc, tirx.const(bool(align_c), "bool")]
        clear = self.clear_accum

        @T.prim_func
        def _gemm_hvx_int() -> None:
            T.evaluate(tirx.call_intrin("void", gemm, M, N, K, *flags, c, a, b, panel_bytes, T.Cast("bool", clear)))

        return _gemm_hvx_int

    def _weight_axes(self) -> tuple:
        """B's axes of K and N: B is [K, N], or [N, K] with ``transpose_B``."""
        return (-1, -2) if self.trans_B else (-2, -1)

    def _weight_rejection(self, layout_map, layout) -> str:
        """Why HVX cannot multiply by a B in the Crouton `layout`, or "": B
        the HMX weight operand (transposed as the GEMM is), of whole tiles,
        times an fp16 A [M, K] into a C, both row-major."""
        if not layout.column_tiles or bool(layout.transposed) != bool(self.trans_B):
            return f"{self.B.name} is in a Crouton layout, not the HMX weight operand's{' (transposed)' if self.trans_B else ''}"
        if str(self.A.dtype) != "float16" or self.trans_A:
            return "HVX multiplies a weight-layout B by an fp16 A [M, K], not transposed"
        others = [buffer.name for buffer in (self.A, self.C) if not _row_major(buffer, layout_map)]
        if others:
            return f"{', '.join(others)} not row-major"
        region = self.BRegion.region
        analyzer = arith.Analyzer()
        on_tiles = all(analyzer.can_prove(tirx.floormod(region[i].min, TILE) == 0) for i in self._weight_axes())
        if self.N % TILE or self.K % TILE or not on_tiles:
            return f"{self.B.name}'s region is not whole Crouton tiles"
        return ""

    def _lower_weight(self):
        M, N, K = self.M, self.N, self.K
        a, lda, align_a = self._operand(self.ARegion, M, K, "r")
        c, ldc, _ = self._operand(self.CRegion, M, N, "rw")
        # A panel's tiles span the matrix's K. B's pointer is at its region's
        # logical start, which LowerTileOp moves through the layout to its tile.
        k_axis, _ = self._weight_axes()
        panel_stride = int(self.B.shape[k_axis]) // TILE * TILE_ELEMS
        origin = tuple(r.min for r in self.BRegion.region)
        b = T.access_ptr(self.B[origin], "r", (N // TILE - 1) * panel_stride + K * TILE)
        gemm = tirx.op.Op.get("tl.hexagon_hvx_gemm_weight")
        flags = [lda, ldc, tirx.const(bool(align_a), "bool"), tirx.const(panel_stride, "int32")]
        clear = self.clear_accum

        @T.prim_func
        def _gemm_hvx_weight() -> None:
            T.evaluate(tirx.call_intrin("void", gemm, M, N, K, *flags, c, a, b, T.Cast("bool", clear)))

        return _gemm_hvx_weight

    def lower(self, layout_map, target, thread_bounds, thread_var, mbar_phase_expr=None):
        if self.integer:
            return self._lower_integer(layout_map)
        layout = layout_map.get(self.B, None)
        if isinstance(layout, CroutonLayout):
            why = self._weight_rejection(layout_map, layout)
        else:
            others = [buffer.name for buffer in (self.A, self.B, self.C) if not _row_major(buffer, layout_map)]
            why = f"{', '.join(others)} not row-major" if others else ""
        if why:
            logger.warning("T.gemm(%s, %s, %s) runs as a scalar loop, not on HVX: %s", self.A.name, self.B.name, self.C.name, why)
            return GemmScalar.lower(self, layout_map, target, thread_bounds, thread_var, mbar_phase_expr)
        if isinstance(layout, CroutonLayout):
            return self._lower_weight()
        logger.debug(
            "T.gemm(%s, %s, %s) runs on HVX, not on HMX: %s", self.A.name, self.B.name, self.C.name, _hmx_rejection(self.gemm_node)
        )

        M, N, K = self.M, self.N, self.K
        a, lda, align_a = self._operand(self.ARegion, *((K, M) if self.trans_A else (M, K)), "r")
        b, ldb, align_b = self._operand(self.BRegion, *((N, K) if self.trans_B else (K, N)), "r")
        c, ldc, align_c = self._operand(self.CRegion, M, N, "rw")
        gemm = tirx.op.Op.get("tl.hexagon_hvx_gemm")
        flags = [tirx.const(bool(x), "bool") for x in (self.trans_A, self.trans_B)]
        aligned = [tirx.const(bool(x), "bool") for x in (align_a, align_b, align_c)]
        clear = self.clear_accum

        @T.prim_func
        def _gemm_hvx() -> None:
            T.evaluate(tirx.call_intrin("void", gemm, M, N, K, *flags, lda, ldb, ldc, *aligned, c, a, b, T.Cast("bool", clear)))

        return _gemm_hvx


class GemmHVXBlockScaled(GemmBlockScaledMixin, GemmHVX):
    """``T.gemm_blockscaled``: ``C (+)= (A * SFA) @ (B * SFB)^T`` as one
    ``tl_hvx_gemm_int_scaled`` call, an integer GEMM whose groups of K leave
    the vrmpy accumulators into fp32 sums, weighed by their scales.

    A, B and C are those of an integer ``T.gemm`` with ``transpose_B``, except
    that C is fp32. SFA is fp32, one factor per ``sf_a_granularity_k`` columns
    of each row of A: ``[M, K / ga]``, or ``[K / ga]`` for one row. SFB is fp16
    and group-major: ``[K / gb, N]``, factor ``(g, j)`` for group g of row j of B.
    Columns ``[k_start, k_start + K)`` of the scales' K start whole groups of
    each, or lie in one: a factor for all of them weighs the sums once, at the
    end.
    """

    def lower(self, layout_map, target, thread_bounds, thread_var, mbar_phase_expr=None):
        M, N, K = self.M, self.N, self.K
        a, lda, c, ldc, align_c, b, panel_bytes, bits, signed = self._integer_operands(layout_map)
        ga, first_a = scale_group(self, "A", self.sf_a_granularity_k)
        gb, first_b = scale_group(self, "B", self.sf_b_granularity_k)
        # A signed B's groups take whole int16 registers of A for their sums.
        group, step = min(ga, gb), 64 if signed else 32 // bits
        if max(ga, gb) % group or group % step:
            raise ValueError(
                f"T.gemm_blockscaled on HVX: scale groups of {ga} (A) and {gb} (B) columns, one a multiple of the other "
                f"and the smaller one of {step}"
            )
        # SFA: rows of A by groups (the last axis); SFB: groups by rows of B.
        sfa_region, sfb_region = self.SFARegion, self.SFBRegion
        sfa_origin = [r.min for r in sfa_region.region]
        sfa_origin[-1] = sfa_origin[-1] + first_a
        if len(sfa_origin) < 2 and M > 1:
            raise ValueError(f"T.gemm_blockscaled: SFA {sfa_region.buffer.name} holds one row of factors for {M} rows of A")
        ld_sfa = int(_strides(sfa_region.buffer)[-2]) if len(sfa_origin) >= 2 else 0
        sfa = T.access_ptr(sfa_region.buffer[tuple(sfa_origin)], "r", (M - 1) * ld_sfa + K // ga)
        read_sfb, ld_sfb = group_major_sfb(sfb_region, layout_map, first_b, K // gb, N)
        gemm = tirx.op.Op.get("tl.hexagon_hvx_gemm_int_scaled")
        flags = [tirx.const(bits, "int32"), tirx.const(signed, "bool"), lda, ldc, ga, gb, tirx.const(bool(align_c), "bool")]
        clear = self.clear_accum

        @T.prim_func
        def _gemm_hvx_blockscaled() -> None:
            T.evaluate(
                tirx.call_intrin(
                    "void", gemm, M, N, K, *flags, c, a, b, panel_bytes, sfa, ld_sfa, read_sfb(), ld_sfb, T.Cast("bool", clear)
                )
            )

        return _gemm_hvx_blockscaled
