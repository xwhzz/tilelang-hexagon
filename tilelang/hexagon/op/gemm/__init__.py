"""Hexagon ``T.gemm`` implementations.

The native GemmImpl (``src/hexagon/op/gemm.cc``) selects ``hexagon.hmx`` for
GEMMs whose operands fit the matrix unit (-> :class:`GemmHMX`: Crouton VTCM
operands and HMX MACs), ``hexagon.hvx`` for the other GEMMs of fp16 or fp32
row-major matrices (-> :class:`GemmHVX`), and ``cpu.scalar`` otherwise (-> a
loop nest, which logs why neither unit applies). A block-scaled GEMM of an
fp16 A runs on HMX (:class:`GemmHMXBlockScaled`), one of integers on HVX
(:class:`GemmHVXBlockScaled`).
"""

from __future__ import annotations

import logging

import tvm_ffi

from tilelang.cpu.op.gemm.gemm_scalar import GEMM_INST_SCALAR, GemmScalar
from tilelang.tileop.gemm.registry import register_gemm_impl

from .gemm_hmx import GEMM_INST_HMX, GEMM_INST_HMX_BLOCK_SCALED, GemmHMX, GemmHMXBlockScaled
from .gemm_hvx import GEMM_INST_HVX, GEMM_INST_HVX_BLOCK_SCALED, GemmHVX, GemmHVXBlockScaled

logger = logging.getLogger(__name__)

_hmx_rejection = tvm_ffi.get_global_func("tl.hexagon.GemmHMXRejection")
_hvx_rejection = tvm_ffi.get_global_func("tl.hexagon.GemmHVXRejection")


class GemmScalarHexagon(GemmScalar):
    """The scalar loop, for a GEMM that neither HMX nor HVX can run; far slower
    than either."""

    def __post_init__(self) -> None:
        # 4-bit elements are not addressable one at a time: only HVX runs them.
        if any(str(buffer.dtype) in ("int4", "uint4") for buffer in (self.A, self.B)):
            raise ValueError(
                f"T.gemm({self.A.name}, {self.B.name}, {self.C.name}) has 4-bit operands, which only HVX multiplies, but: {_hvx_rejection(self.gemm_node)}"
            )
        super().__post_init__()

    def lower(self, *args, **kwargs):
        logger.warning(
            "T.gemm(%s, %s, %s) runs as a scalar loop, not on HMX: %s; not on HVX: %s",
            self.A.name,
            self.B.name,
            self.C.name,
            _hmx_rejection(self.gemm_node),
            _hvx_rejection(self.gemm_node),
        )
        return super().lower(*args, **kwargs)


def _match_hexagon(target) -> bool:
    return target.kind.name == "hexagon"


register_gemm_impl("hexagon.scalar", GEMM_INST_SCALAR, _match_hexagon, GemmScalarHexagon)
register_gemm_impl("hexagon.hmx", GEMM_INST_HMX, _match_hexagon, GemmHMX)
register_gemm_impl("hexagon.hvx", GEMM_INST_HVX, _match_hexagon, GemmHVX)
register_gemm_impl("hexagon.hvx_blockscaled", GEMM_INST_HVX_BLOCK_SCALED, _match_hexagon, GemmHVXBlockScaled)
register_gemm_impl("hexagon.hmx_blockscaled", GEMM_INST_HMX_BLOCK_SCALED, _match_hexagon, GemmHMXBlockScaled)
