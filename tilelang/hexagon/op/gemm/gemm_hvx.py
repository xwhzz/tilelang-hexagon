"""Hexagon HVX implementation of ``T.gemm``, for the GEMMs HMX does not run.

The whole GEMM is one ``tl_hvx_gemm`` call (``hvx_gemm.h``) on row-major
operands: outer products when B's rows run along N, dot products when A's and
B's rows both run along K (``transpose_B``). Products accumulate in registers
over the call's K, and reach C once.

Operands keep their layouts: a GEMM whose operand another operator gave a
layout that is not row-major runs as the scalar loop.
"""

from __future__ import annotations

import logging

import tvm_ffi
from tvm import DataType, arith, tirx

from tilelang.cpu.op.gemm.gemm_scalar import GemmScalar
from tilelang.hexagon import language as T
from tilelang.tileop.gemm.gemm_base import GemmBase

GEMM_INST_HVX = "hexagon.hvx"

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


class GemmHVX(GemmBase):
    """Lower one logical GEMM into a ``tl_hvx_gemm`` call."""

    def infer_layout(self, target, thread_nums: int):
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

    def lower(self, layout_map, target, thread_bounds, thread_var, mbar_phase_expr=None):
        others = [buffer.name for buffer in (self.A, self.B, self.C) if not _row_major(buffer, layout_map)]
        if others:
            logger.warning(
                "T.gemm(%s, %s, %s) runs as a scalar loop, not on HVX: %s not row-major",
                self.A.name,
                self.B.name,
                self.C.name,
                ", ".join(others),
            )
            return GemmScalar.lower(self, layout_map, target, thread_bounds, thread_var, mbar_phase_expr)
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
