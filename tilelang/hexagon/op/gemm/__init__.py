"""Hexagon ``T.gemm`` implementations.

The native GemmImpl (``src/hexagon/op/gemm.cc``) selects ``hexagon.hmx`` for
GEMMs on the validated HMX surface (-> :class:`GemmHMX`: Crouton VTCM operands
and HMX MACs) and ``cpu.scalar`` otherwise (-> ``GemmScalar``: a loop nest that
hexagon-clang vectorizes onto HVX).
"""

from __future__ import annotations

from tilelang.cpu.op.gemm.gemm_scalar import GEMM_INST_SCALAR, GemmScalar
from tilelang.tileop.gemm.registry import register_gemm_impl

from .gemm_hmx import GEMM_INST_HMX, GemmHMX


def _match_hexagon(target) -> bool:
    return target.kind.name == "hexagon"


register_gemm_impl("hexagon.scalar", GEMM_INST_SCALAR, _match_hexagon, GemmScalar)
register_gemm_impl("hexagon.hmx", GEMM_INST_HMX, _match_hexagon, GemmHMX)
