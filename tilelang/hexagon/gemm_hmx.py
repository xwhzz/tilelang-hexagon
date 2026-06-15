"""Hexagon HMX matrix-engine implementation of ``T.gemm``.

The operands live in VTCM in the Crouton tile layout (declared by ``infer_layout``
so the global→VTCM ``T.copy`` writes Crouton directly), and the gemm lowers to the
``tl_hexagon_hmx_mac_f16`` runtime entry, which does one full-K HMX accumulator
sequence per output tile and stores fp16 — no pack/unpack.

This is the SS variant (shared A, shared B → shared/VTCM C), clear_accum=True
(overwrite).  Accumulate (clear_accum=False) and a fragment A-operand (RS, for
attention's second gemm) come with the attention phase.
"""

from __future__ import annotations

from tilelang.tileop.gemm.gemm_base import GemmBase
from tilelang.hexagon.layout import make_crouton_layout
from tilelang import language as T

GEMM_INST_HMX = "hexagon.hmx"


class GemmHMX(GemmBase):

    def infer_layout(self, target, thread_nums: int):
        # Crouton VTCM layout per operand: A and the output C are row-tile-major,
        # the weight B is col-tile-major (matching pack_A / pack_B / unpack_C in
        # tl_templates/hexagon/hmx.h).  cpos is over the operand's *natural*
        # (row, col); trans_A/trans_B only flip which buffer axis is row vs col.
        A, B, C = self.A, self.B, self.C
        return {
            A: make_crouton_layout(A.shape, role="row", transposed=self.trans_A),
            B: make_crouton_layout(B.shape, role="col", transposed=self.trans_B),
            C: make_crouton_layout(C.shape, role="row", transposed=False),
        }

    def lower(self, layout_map, target, thread_bounds, thread_var, mbar_phase_expr=None):
        # SelectInst (src/hexagon/op/gemm.cc) gates HMX on clear_accum=const-true,
        # SS operands and static 32-multiple 2D shapes — everything else falls back
        # to the scalar impl, so these are defensive asserts, not the routing.
        if not self.clear_accum:
            raise NotImplementedError(
                "GemmHMX: clear_accum=False (accumulate) is not supported yet — "
                "the HMX accumulator can't be preloaded, so accumulate must add the "
                "fp16 tile into VTCM via HVX (attention phase).")
        M, N, K = self.M, self.N, self.K
        A_buf, B_buf, C_buf = self.ARegion.buffer, self.BRegion.buffer, self.CRegion.buffer
        a0, a1 = self.ARegion.region[0].min, self.ARegion.region[1].min
        b0, b1 = self.BRegion.region[0].min, self.BRegion.region[1].min
        c0, c1 = self.CRegion.region[0].min, self.CRegion.region[1].min
        # tl_hexagon_hmx_mac_f16 walks tiles with the gemm's M/N/K strides, which
        # only matches the buffer's Crouton tiling when each region spans its whole
        # buffer.  A sub-region gemm (A_sh[0:32, :]) would address the wrong tiles,
        # so reject it loudly rather than silently corrupt.
        def _full(region, buf):
            return all(int(r.min) == 0 for r in region.region) and \
                int(region.region[0].extent) == int(buf.shape[0]) and \
                int(region.region[1].extent) == int(buf.shape[1])
        if not (_full(self.ARegion, A_buf) and _full(self.BRegion, B_buf) and _full(self.CRegion, C_buf)):
            raise NotImplementedError(
                "GemmHMX: sub-region gemm (operands that don't span their whole "
                "shared buffer) is not supported yet — make A_shared/B_shared/"
                "C_shared exactly the gemm tile.")

        @T.prim_func
        def _gemm_hmx() -> None:
            T.call_extern(
                "int32", "tl_hexagon_hmx_mac_f16",
                T.address_of(C_buf[c0, c1]),
                T.address_of(A_buf[a0, a1]),
                T.address_of(B_buf[b0, b1]),
                M, N, K)

        return _gemm_hmx
