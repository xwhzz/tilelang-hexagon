"""Lowering pipeline registration for the Hexagon backend.

A Hexagon device kernel is sequential C with HVX/HMX intrinsics — the GPU-style
``T.Kernel`` grid/threads are lowered to ordinary loops exactly as for the CPU
``"c"`` backend.  So for now we reuse :func:`CPUPassPipelineBody` verbatim under
the ``"hexagon"`` target kind.  Crouton-layout inference and HMX tile-op lowering
(M2) will extend this with Hexagon-specific passes.

This module is imported by :mod:`tilelang` at init time (and *not* by
``tilelang.hexagon.__init__``), so the lean build/deploy harness in that package
stays importable without the tilelang native library.
"""

from __future__ import annotations

from tilelang.backend.pass_pipeline import PassPipeline, register_pipeline
from tilelang.cpu.pipeline import CPUPassPipelineBody
from tilelang.tileop.gemm.registry import register_gemm_impl
from tilelang.cpu.op.gemm.gemm_scalar import GEMM_INST_SCALAR, GemmScalar

register_pipeline(PassPipeline("hexagon", CPUPassPipelineBody))

# Lower T.gemm on Hexagon via the generic scalar triple-loop for now (the C++
# hexagon GemmImpl selects "cpu.scalar"); hexagon-clang auto-vectorizes it onto
# HVX.  Routing T.gemm onto the HMX matrix engine (Crouton layout + HMX template)
# is the next step.
register_gemm_impl(
    "hexagon.scalar", GEMM_INST_SCALAR, lambda t: t.kind.name == "hexagon", GemmScalar
)
