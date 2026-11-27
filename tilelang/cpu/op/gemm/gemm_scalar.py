from __future__ import annotations

from tilelang.tileop.gemm.gemm_base import GemmBase
from tilelang import language as T
from tvm.target import Target
from tvm.ir import Range
from tvm import tirx


GEMM_INST_SCALAR = "cpu.scalar"


class GemmScalar(GemmBase):
    """CPU scalar fallback: triple nested loop gemm."""

    def infer_layout(self, target: Target, thread_nums: int):
        return {}

    def lower(
        self,
        layout_map: dict,
        target: Target,
        thread_bounds: Range,
        thread_index: tirx.PrimExpr,
        mbar_phase_expr: tirx.PrimExpr | None = None,
    ):
        M, N, K = self.M, self.N, self.K
        A_buf = self.ARegion.buffer
        B_buf = self.BRegion.buffer
        C_region = self.CRegion
        C_buf = C_region.buffer
        trans_A = self.trans_A
        trans_B = self.trans_B
        clear_accum = self.clear_accum
        accum_dtype = self.accum_dtype

        a_origin = tuple(r.min for r in self.ARegion.region)
        b_origin = tuple(r.min for r in self.BRegion.region)
        c_origin = tuple(r.min for r in C_region.region)

        def at(origin, row, col):
            return origin[:-2] + (origin[-2] + row, origin[-1] + col)

        @T.prim_func
        def _gemm_scalar() -> None:
            if clear_accum:
                T.clear(C_region)
            for i, j, k in T.grid(M, N, K):
                C_buf[at(c_origin, i, j)] += T.cast(
                    A_buf[at(a_origin, k if trans_A else i, i if trans_A else k)],
                    accum_dtype,
                ) * T.cast(
                    B_buf[at(b_origin, j if trans_B else k, k if trans_B else j)],
                    accum_dtype,
                )

        return _gemm_scalar
