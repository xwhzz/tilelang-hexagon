"""Hexagon HMX implementation of ``T.gemm``.

Layout inference gives A and B the HMX activation and weight layouts, Crouton
tiles in VTCM that the surrounding copies fill. Each 32x32 output tile reduces
the call's K in the HMX accumulator, in deep packets of at most 32 K tiles,
and converts to fp16: HMX accumulates within a call, and HVX across calls.

An fp16 C keeps the HMX output layout, and a call that overwrites it stores
its tiles there directly. Every other C takes them through a staging tile,
which an HVX epilogue writes or adds into C (``tl_crouton_tile_store``): an
fp32 C, which stays row-major, an fp16 C that the call accumulates into, or
element by element a C of another layout. Each tile's epilogue runs while HMX
multiplies the next tile; the HMX accumulator is held for the whole call.

In a worker pool, whose workers share HMX (``LowerKernelLaunch`` marks such
calls), each staged output tile is a critical section of its own instead, and
its epilogue runs outside it, while another worker multiplies.

``T.gemm_blockscaled`` of an fp16 A runs here too (:class:`GemmHMXBlockScaled`):
HVX dequantizes its 4-bit B into fp16 weight tiles, a column of output tiles
at a time, for HMX to multiply every row of A by.
"""

from __future__ import annotations

import tvm_ffi
from tvm import tirx

from tilelang.hexagon import language as T
from tilelang.hexagon.intrinsics.crouton import TILE, TILE_ELEMS, CroutonLayout
from tilelang.hexagon.intrinsics.hmx import BIAS_WORDS, HMXIntrinEmitter
from tilelang.hexagon.intrinsics.panel import PanelLayout, make_hvx_panel_layout
from tilelang.tileop.gemm.gemm_base import GemmBase
from tilelang.tileop.gemm_blockscaled.gemm_blockscaled_base import GemmBlockScaledMixin

from .gemm_hvx import _strides, scale_group

GEMM_INST_HMX = "hexagon.hmx"
GEMM_INST_HMX_BLOCK_SCALED = "hexagon.hmx.blockscaled"

# The worker count LowerKernelLaunch puts on the gemms of a worker pool.
_NUM_WORKERS = "hexagon.num_workers"

_row_major = tvm_ffi.get_global_func("tl.hexagon.RowMajor")


def _flag(value: bool) -> str:
    return "true" if value else "false"


def _origin(region) -> tuple:
    """The first index of a buffer region, constants as Python ints."""
    return tuple(int(r.min) if isinstance(r.min, tirx.IntImm) else r.min for r in region.region)


class GemmHMX(GemmBase):
    """Lower one logical GEMM into HMX 32x32x32 instruction atoms."""

    def _make_emitter(self) -> HMXIntrinEmitter:
        return HMXIntrinEmitter(
            self.M,
            self.N,
            self.K,
            a_dtype=str(self.a_dtype),
            b_dtype=str(self.b_dtype),
            a_transposed=self.trans_A,
            b_transposed=self.trans_B,
        )

    def infer_layout(self, target, thread_nums: int):
        emitter = self._make_emitter()
        layouts = {
            self.A: emitter.activation_layout(self.A),
            self.B: emitter.weight_layout(self.B),
        }
        # An fp32 C stays row-major, where each register of the epilogue is
        # one row of a tile.
        if str(self.C.dtype) == "float16":
            layouts[self.C] = emitter.output_layout(self.C)
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

    def _epilogue_kind(self, layout_map) -> str:
        """How the staged tiles reach C: "crouton" (C's own tiles), "rows"
        (row-major) or "elements"."""

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

    def _tile_epilogue(self, layout_map, coords, scale=None):
        """A macro that writes, or adds per ``clear_accum``, output tile t from
        its staging tile, slot t % 2 of `stage`, into C: tile (mt, nt) =
        coords(t). `scale`, a factor per row of C, multiplies the rows first:
        the buffer, the index of row 0's factor, its last two indices those of
        a matrix, and the elements between the factors of two rows."""

        C_buf = self.CRegion.buffer
        c_origin = _origin(self.CRegion)
        kind = self._epilogue_kind(layout_map)
        ldc = C_buf.shape[-1]
        # Rows of a tile on register (fp32) or half-register (fp16) boundaries:
        # the region starts on a tile, and VTCM buffers on registers.
        aligned = isinstance(ldc, tirx.IntImm) and ldc.value % TILE == 0
        tile_store = tirx.op.Op.get("tl.hexagon_crouton_tile_store")
        rows_extent = 31 * ldc + TILE if kind == "rows" else TILE_ELEMS
        clear = self.clear_accum
        static_clear = bool(clear.value) if isinstance(clear, tirx.IntImm) else None

        def c_tile(t):
            """The first element of C's output tile t."""
            mt, nt = coords(t)
            return C_buf[c_origin[:-2] + (c_origin[-2] + mt * TILE, c_origin[-1] + nt * TILE)]

        def c_element(t, i, j):
            mt, nt = coords(t)
            return c_origin[:-2] + (c_origin[-2] + mt * TILE + i, c_origin[-1] + nt * TILE + j)

        def factor(t, i):
            """The factor of row i of output tile t."""
            buffer, row0, _ = scale
            return buffer[row0[:-2] + (row0[-2] + coords(t)[0] * TILE + i, row0[-1])]

        # Python assignments in a macro bind TIR variables: the tile's indices
        # stay expressions of t, which the layouts then simplify.
        @T.macro
        def write(stage, t, accumulate: bool):
            """Write (or add) the staged tile t into C."""
            if kind == "elements":
                for i, j in T.grid(TILE, TILE):
                    value = stage[(t % 2) * TILE_ELEMS + (i // 2) * 64 + j * 2 + i % 2]
                    if scale is not None:
                        value = T.cast(value, "float32") * factor(t, i)
                    if accumulate:
                        C_buf[c_element(t, i, j)] = C_buf[c_element(t, i, j)] + T.cast(value, C_buf.dtype)
                    else:
                        C_buf[c_element(t, i, j)] = T.cast(value, C_buf.dtype)
            else:
                args = [
                    _flag(accumulate),
                    _flag(kind == "rows"),
                    _flag(aligned),
                    _flag(scale is not None),
                    T.access_ptr(c_tile(t), "rw" if accumulate else "w", rows_extent),
                    ldc,
                    T.access_ptr(stage[(t % 2) * TILE_ELEMS], "r", TILE_ELEMS),
                ]
                if scale is not None:
                    args += [T.access_ptr(factor(t, 0), "r", 31 * scale[2] + 1), scale[2]]
                T.evaluate(tirx.call_intrin("void", tile_store, *args))

        @T.macro
        def epilogue(stage, t):
            if static_clear is None:
                if clear:
                    write(stage, t, False)
                else:
                    write(stage, t, True)
            else:
                write(stage, t, not static_clear)

        return epilogue

    def lower(self, layout_map, target, thread_bounds, thread_var, mbar_phase_expr=None):
        self._check_operand_layouts(layout_map)
        emitter = self._make_emitter()
        A_buf = self.ARegion.buffer
        B_buf = self.BRegion.buffer
        C_buf = self.CRegion.buffer
        a_origin = _origin(self.ARegion)
        b_origin = _origin(self.BRegion)
        c_origin = _origin(self.CRegion)
        NT = emitter.num_n_tiles
        tiles = emitter.num_m_tiles * NT

        clear = self.clear_accum
        static_clear = bool(clear.value) if isinstance(clear, tirx.IntImm) else None
        staged = not (self._epilogue_kind(layout_map) == "crouton" and static_clear is True)
        shared = self._shared()
        epilogue = self._tile_epilogue(layout_map, lambda t: (t // NT, t % NT))

        @T.prim_func
        def _gemm_hmx() -> None:
            # HMX conversion reads 32 u32 scale words followed by 32 bias words;
            # each scale word carries FP16 1.0 in its low half.
            bias_vtcm = T.alloc_shared((BIAS_WORDS,), "uint32", align=256)
            # Two staging tiles: HMX stores one while the epilogue reads the
            # other. The planner drops them when the call stores into C.
            stage = T.alloc_shared((2 * TILE_ELEMS,), "float16", align=2048)
            acc = T.hmx.alloc_accumulator()
            cvt = T.hmx.alloc_convert_state()
            bias = T.hmx.alloc_bias_state()

            for i in T.vectorized(BIAS_WORDS // 2):
                bias_vtcm[i] = T.Cast("uint32", 0x3C00)
            for i in T.vectorized(BIAS_WORDS // 2):
                bias_vtcm[BIAS_WORDS // 2 + i] = T.Cast("uint32", 0)

            if shared and staged:
                # Other workers multiply while this one runs a tile's epilogue.
                # Each holder sets the scale/bias configuration again.
                for t in T.serial(tiles):
                    emitter.acquire(acc)
                    emitter.load_bias(bias, bias_vtcm)
                    emitter.clear(acc)
                    emitter.mma_tile(acc, A_buf, B_buf, t // NT, t % NT, a_origin, b_origin)
                    emitter.convert(cvt, acc, bias, bias_vtcm)
                    emitter.store_at(cvt, acc, stage[(t % 2) * TILE_ELEMS], bias, bias_vtcm)
                    emitter.release(acc)
                    epilogue(stage, t)
            else:
                emitter.acquire(acc)
                # All output tiles share this scale/bias configuration.
                emitter.load_bias(bias, bias_vtcm)
                for t in T.serial(tiles):
                    emitter.clear(acc)
                    emitter.mma_tile(acc, A_buf, B_buf, t // NT, t % NT, a_origin, b_origin)
                    # `staged` is decided here, `t > 0` in the kernel.
                    if staged:  # noqa: SIM102
                        # The previous tile's epilogue overlaps these MACs.
                        if t > 0:
                            epilogue(stage, t - 1)
                    emitter.convert(cvt, acc, bias, bias_vtcm)
                    if staged:
                        emitter.store_at(cvt, acc, stage[(t % 2) * TILE_ELEMS], bias, bias_vtcm)
                    else:
                        emitter.store(cvt, acc, C_buf, bias, bias_vtcm, t // NT, t % NT, c_origin)
                emitter.release(acc)
                if staged:
                    epilogue(stage, tiles - 1)

        return _gemm_hmx


class GemmHMXBlockScaled(GemmBlockScaledMixin, GemmHMX):
    """``T.gemm_blockscaled`` of an fp16 A on HMX: ``C (+)= (A * SFA) @ (B *
    SFB)^T`` for a 4-bit B in the panel layout, its rows along K
    (``transpose_B``), the decode GEMV's operand.

    A has the HMX activation layout; SFA is fp32, one factor per row across
    the call's K, ``[M, K / ga]`` with ga >= K; SFB is fp16 and group-major,
    ``[K / gb, N]``, gb a multiple of 8. C is an fp32 row-major or fp16
    Crouton matrix.

    The call is one ``tl_hexagon_hmx_gemm_blockscaled`` (``hmx_gemm.h``), a
    function of its own: HVX dequantizes B 32 rows at a time, one column of
    output tiles, into fp16 weight tiles (``tl_crouton_dequant4``), and HMX
    multiplies every row tile of A by them, so each weight serves all of A's
    rows. In one program a slice of the next column dequantizes while HMX
    multiplies each tile; in a worker pool each tile is a critical section,
    and the slice and the tile's epilogue run while other workers multiply.
    """

    def __post_init__(self) -> None:
        # B dequantizes to fp16 before HMX multiplies: an fp16 A by a 4-bit B
        # is not a mixed-dtype GEMM, whose operands the common rule checks.
        pass

    def _make_emitter(self) -> HMXIntrinEmitter:
        # HMX multiplies A by B's dequantized fp16 weight tiles.
        return HMXIntrinEmitter(self.M, self.N, self.K, a_transposed=self.trans_A)

    def infer_layout(self, target, thread_nums: int):
        layouts = {self.A: self._make_emitter().activation_layout(self.A), self.B: make_hvx_panel_layout(self.B)}
        if str(self.C.dtype) == "float16":
            layouts[self.C] = HMXIntrinEmitter.output_layout(self.C)
        return layouts

    def lower(self, layout_map, target, thread_bounds, thread_var, mbar_phase_expr=None):
        name = f"T.gemm_blockscaled({self.A.name}, {self.B.name}, {self.C.name}) on HMX"
        layout = layout_map.get(self.A, None)
        if not (isinstance(layout, CroutonLayout) and not layout.column_tiles and not layout.transposed):
            raise ValueError(f"{name} reads A in the HMX activation layout, but it has {layout}")
        if not isinstance(layout_map.get(self.B, None), PanelLayout):
            raise ValueError(f"{name} reads B in the panel layout, but it has {layout_map.get(self.B, None)}")
        kind = self._epilogue_kind(layout_map)
        if kind == "elements":
            raise ValueError(f"{name} writes C row-major or in Crouton tiles, not {layout_map.get(self.C, None)}")
        M, N, K = self.M, self.N, self.K
        ga, first_a = scale_group(self, "A", self.sf_a_granularity_k)
        gb, first_b = scale_group(self, "B", self.sf_b_granularity_k)
        if ga != K:
            raise ValueError(f"{name} scales whole rows of A: one factor across the call's {K} columns, not one per {ga}")
        if gb % 8:
            raise ValueError(f"{name} dequantizes B in groups of 8 columns of K: its scale groups are a multiple of 8, not {gb}")

        def first(region):
            return region.buffer[_origin(region)]

        A_buf, C_buf = self.ARegion.buffer, self.CRegion.buffer
        # A's row tiles are its buffer's column tiles of 1024 elements apart.
        a_row_tile = int(A_buf.shape[-1]) // TILE * TILE_ELEMS
        a = T.access_ptr(first(self.ARegion), "r", (M // TILE - 1) * a_row_tile + K * TILE)
        # B's region starts on a panel: a pointer at its logical start, which
        # LowerTileOp moves through the layout to the panel's bytes.
        b = T.access_ptr(first(self.BRegion), "r", N * K)
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

        rows = kind == "rows"
        if rows:
            ldc = C_buf.shape[-1]
            aligned = isinstance(ldc, tirx.IntImm) and ldc.value % TILE == 0
            c_extent = (M - 1) * ldc + N
        else:
            ldc = int(C_buf.shape[-1]) // TILE
            aligned = True
            c_extent = M * N
        clear = self.clear_accum
        c = T.access_ptr(first(self.CRegion), "w" if clear is True else "rw", c_extent)
        gemm = tirx.op.Op.get("tl.hexagon_hmx_gemm_blockscaled")
        flags = [_flag(signed), _flag(rows), _flag(aligned), _flag(True), _flag(self._shared())]

        @T.prim_func
        def _gemm_hmx_blockscaled() -> None:
            # Two columns of weight tiles, two staging tiles and the conversion
            # config: HMX reads one column while HVX fills the other.
            config = T.alloc_shared((BIAS_WORDS,), "uint32", align=256)
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
                    T.access_ptr(config[0], "rw", BIAS_WORDS),
                )
            )

        return _gemm_hmx_blockscaled
