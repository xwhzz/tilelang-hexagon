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
"""

from __future__ import annotations

import tvm_ffi
from tvm import tirx

from tilelang.hexagon import language as T
from tilelang.hexagon.intrinsics.crouton import TILE, TILE_ELEMS, CroutonLayout
from tilelang.hexagon.intrinsics.hmx import BIAS_WORDS, HMXIntrinEmitter
from tilelang.tileop.gemm.gemm_base import GemmBase

GEMM_INST_HMX = "hexagon.hmx"

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
        kind = self._epilogue_kind(layout_map)
        staged = not (kind == "crouton" and static_clear is True)

        ldc = C_buf.shape[-1]
        # Rows of a tile on register (fp32) or half-register (fp16) boundaries:
        # the region starts on a tile, and VTCM buffers on registers.
        aligned = isinstance(ldc, tirx.IntImm) and ldc.value % TILE == 0
        tile_store = tirx.op.Op.get("tl.hexagon_crouton_tile_store")
        rows_extent = 31 * ldc + TILE if kind == "rows" else TILE_ELEMS

        def c_tile(t):
            """The first element of C's output tile t."""
            return C_buf[c_origin[:-2] + (c_origin[-2] + (t // NT) * TILE, c_origin[-1] + (t % NT) * TILE)]

        def c_element(t, i, j):
            return c_origin[:-2] + (c_origin[-2] + (t // NT) * TILE + i, c_origin[-1] + (t % NT) * TILE + j)

        @T.macro
        def write(stage, t, accumulate: bool):
            """Write (or add) the staged tile t into C."""
            if kind == "elements":
                for i, j in T.grid(TILE, TILE):
                    value = T.cast(stage[(t % 2) * TILE_ELEMS + (i // 2) * 64 + j * 2 + i % 2], C_buf.dtype)
                    if accumulate:
                        C_buf[c_element(t, i, j)] = C_buf[c_element(t, i, j)] + value
                    else:
                        C_buf[c_element(t, i, j)] = value
            else:
                T.evaluate(
                    tirx.call_intrin(
                        "void",
                        tile_store,
                        _flag(accumulate),
                        _flag(kind == "rows"),
                        _flag(aligned),
                        T.access_ptr(c_tile(t), "rw" if accumulate else "w", rows_extent),
                        ldc,
                        T.access_ptr(stage[(t % 2) * TILE_ELEMS], "r", TILE_ELEMS),
                    )
                )

        @T.macro
        def epilogue(stage, t):
            if static_clear is None:
                if clear:
                    write(stage, t, False)
                else:
                    write(stage, t, True)
            else:
                write(stage, t, not static_clear)

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
