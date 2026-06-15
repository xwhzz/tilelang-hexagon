"""Hexagon HMX implementation of ``T.gemm`` (Level-1).

The operands stay **row-major** in VTCM — so the global→VTCM ``T.copy`` that fills
them auto-vectorizes onto HVX (a contiguous half8 copy) — and the gemm lowers to
``tl_hexagon_hmx_gemm``, which HVX-packs them to a Crouton scratch (top of VTCM),
runs the HMX MAC, and HVX-unpacks the result back to row-major C.  Keeping the
operands row-major also means any surrounding elementwise/softmax loops stay
vectorizable (vs the permuted Crouton layout, which forces scalar code).

SS variant, clear_accum=True (overwrite).  trans_a/trans_b are handled in the
pack (scalar transposed pack for the rarer transposed operand).  Accumulate
(clear_accum=False) and fragment/RS operands stay on the scalar fallback.
"""

from __future__ import annotations

from tilelang.tileop.gemm.gemm_base import GemmBase
from tilelang import language as T

GEMM_INST_HMX = "hexagon.hmx"


class GemmHMX(GemmBase):

    def infer_layout(self, target, thread_nums: int):
        # No Crouton layout: operands stay row-major (fast vectorized loads); the
        # runtime packs to Crouton with HVX inside tl_hexagon_hmx_gemm.
        return {}

    def lower(self, layout_map, target, thread_bounds, thread_var, mbar_phase_expr=None):
        # SelectInst (src/hexagon/op/gemm.cc) gates HMX on clear_accum=const-true,
        # SS operands and static 32-multiple 2D shapes — these are defensive.
        if not self.clear_accum:
            raise NotImplementedError(
                "GemmHMX: clear_accum=False (accumulate) is not supported yet — "
                "the HMX accumulator can't be preloaded, so accumulate must add the "
                "fp16 tile into VTCM via HVX (decompose into an overwrite gemm + add).")
        M, N, K = self.M, self.N, self.K
        A_buf, B_buf, C_buf = self.ARegion.buffer, self.BRegion.buffer, self.CRegion.buffer
        a0, a1 = self.ARegion.region[0].min, self.ARegion.region[1].min
        b0, b1 = self.BRegion.region[0].min, self.BRegion.region[1].min
        c0, c1 = self.CRegion.region[0].min, self.CRegion.region[1].min
        # The gemm reads each operand row-major as its full M/N/K, which only
        # matches the buffer when the region spans the whole buffer — reject a
        # sub-region gemm loudly rather than silently corrupt.
        def _full(region, buf):
            # A symbolic region min/extent (e.g. a strided sub-region A_sh[ko*64:...]
            # inside a serial loop) is by definition not a static full-buffer span;
            # int() would raise TypeError, so treat non-const as "not full" and let
            # the NotImplementedError below report it cleanly.
            try:
                return all(int(r.min) == 0 for r in region.region) and \
                    int(region.region[0].extent) == int(buf.shape[0]) and \
                    int(region.region[1].extent) == int(buf.shape[1])
            except (TypeError, ValueError):
                return False
        if not (_full(self.ARegion, A_buf) and _full(self.BRegion, B_buf) and _full(self.CRegion, C_buf)):
            raise NotImplementedError(
                "GemmHMX: sub-region gemm (operands that don't span their whole "
                "shared buffer) is not supported yet — make the shared buffers "
                "exactly the gemm tile.")
        ta, tb = (1 if self.trans_A else 0), (1 if self.trans_B else 0)

        @T.prim_func
        def _gemm_hmx() -> None:
            T.call_extern(
                "int32", "tl_hexagon_hmx_gemm",
                T.address_of(C_buf[c0, c1]),
                T.address_of(A_buf[a0, a1]),
                T.address_of(B_buf[b0, b1]),
                M, N, K, ta, tb)

        return _gemm_hmx
