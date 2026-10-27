"""Hexagon ``T.gemm`` implementations.

The native GemmImpl (``src/hexagon/op/gemm.cc``) selects ``hexagon.hmx`` for
GEMMs whose operands fit the matrix unit (-> :class:`GemmHMX`: Crouton VTCM
operands and HMX MACs) and ``cpu.scalar`` otherwise (-> a loop nest that
hexagon-clang vectorizes onto HVX, which logs why HMX does not apply).
"""

from __future__ import annotations

import logging

import tvm_ffi

from tilelang.cpu.op.gemm.gemm_scalar import GEMM_INST_SCALAR, GemmScalar
from tilelang.tileop.gemm.registry import register_gemm_impl

from .gemm_hmx import GEMM_INST_HMX, GemmHMX

logger = logging.getLogger(__name__)

_hmx_rejection = tvm_ffi.get_global_func("tl.hexagon.GemmHMXRejection")


class GemmScalarHexagon(GemmScalar):
    """The scalar loop, for a GEMM that HMX cannot run; far slower than HMX."""

    def lower(self, *args, **kwargs):
        logger.warning(
            "T.gemm(%s, %s, %s) runs as a scalar loop, not on HMX: %s",
            self.A.name,
            self.B.name,
            self.C.name,
            _hmx_rejection(self.gemm_node),
        )
        return super().lower(*args, **kwargs)


def _match_hexagon(target) -> bool:
    return target.kind.name == "hexagon"


register_gemm_impl("hexagon.scalar", GEMM_INST_SCALAR, _match_hexagon, GemmScalarHexagon)
register_gemm_impl("hexagon.hmx", GEMM_INST_HMX, _match_hexagon, GemmHMX)
