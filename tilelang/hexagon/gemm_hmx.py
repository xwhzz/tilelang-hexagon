"""Hexagon HMX implementation of ``T.gemm`` over native Crouton VTCM tiles.

This implementation owns only the GEMM instruction sequence.  Layout inference
assigns the HMX activation, weight, and output layouts to the three shared
buffers; surrounding producers and ``T.copy`` operations are responsible for
placing logical values into those layouts.

The currently supported surface is a full-buffer FP16 SS GEMM (including logical
operand transposes) with ``clear_accum=True``.  Other variants stay on the scalar
selector fallback.
"""

from __future__ import annotations

from tilelang import language as T
from tilelang.hexagon.hmx_intrin import BIAS_WORDS, HMXIntrinEmitter
from tilelang.tileop.gemm.gemm_base import GemmBase


GEMM_INST_HMX = "hexagon.hmx"


class GemmHMX(GemmBase):
    """Lower one logical GEMM into explicit HMX 32x32x32 instruction atoms."""

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
        return {
            self.A: emitter.activation_layout(self.A),
            self.B: emitter.weight_layout(self.B),
            self.C: emitter.output_layout(self.C),
        }

    @staticmethod
    def _is_full_region(region, buffer) -> bool:
        """Return whether a static 2-D region covers its complete buffer."""

        try:
            return (
                len(region.region) == 2
                and all(int(dim.min) == 0 for dim in region.region)
                and int(region.region[0].extent) == int(buffer.shape[0])
                and int(region.region[1].extent) == int(buffer.shape[1])
            )
        except (TypeError, ValueError):
            return False

    def lower(self, layout_map, target, thread_bounds, thread_var, mbar_phase_expr=None):
        # SelectInst applies the same restrictions.  Keep these checks here so a
        # direct construction cannot silently emit an invalid HMX sequence.
        if not self.clear_accum:
            raise NotImplementedError("GemmHMX requires clear_accum=True")
        if str(self.C.dtype) != "float16":
            raise NotImplementedError("GemmHMX currently stores FP16 output tiles")

        A_buf = self.ARegion.buffer
        B_buf = self.BRegion.buffer
        C_buf = self.CRegion.buffer
        if not (
            self._is_full_region(self.ARegion, A_buf)
            and self._is_full_region(self.BRegion, B_buf)
            and self._is_full_region(self.CRegion, C_buf)
        ):
            raise NotImplementedError(
                "GemmHMX requires each GEMM region to span its complete shared buffer"
            )

        emitter = self._make_emitter()

        @T.prim_func
        def _gemm_hmx() -> None:
            # HMX conversion reads 32 u32 scale words followed by 32 bias words;
            # each scale word carries FP16 1.0 in its low half.
            # Keep this block internal for now so T.gemm's public contract remains
            # exactly A/B/C in VTCM.
            bias_vtcm = T.alloc_shared((BIAS_WORDS,), "uint32", align=256)
            acc = T.alloc_hmx_accumulator()
            cvt = T.alloc_hmx_convert_state()
            bias = T.alloc_hmx_bias_state()

            for i in T.serial(BIAS_WORDS // 2):
                bias_vtcm[i] = T.Cast("uint32", 0x3C00)
                bias_vtcm[BIAS_WORDS // 2 + i] = T.Cast("uint32", 0)

            emitter.acquire(acc)
            for inst_m_idx in T.serial(emitter.num_m_tiles):
                for inst_n_idx in T.serial(emitter.num_n_tiles):
                    emitter.clear(acc)
                    emitter.load_bias(bias, bias_vtcm)
                    emitter.mma_tile(
                        acc,
                        A_buf,
                        B_buf,
                        inst_m_idx=inst_m_idx,
                        inst_n_idx=inst_n_idx,
                    )
                    emitter.convert(cvt, acc, bias, bias_vtcm)
                    emitter.store(
                        cvt,
                        acc,
                        C_buf,
                        bias,
                        bias_vtcm,
                        A_buf,
                        B_buf,
                        inst_m_idx=inst_m_idx,
                        inst_n_idx=inst_n_idx,
                    )
            emitter.release(acc)

        return _gemm_hmx
