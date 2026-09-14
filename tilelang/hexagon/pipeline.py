"""Lowering pipeline registration for the Hexagon backend.

A Hexagon device kernel is sequential C with HVX/HMX intrinsics — the GPU-style
``T.Kernel`` grid/threads are lowered to ordinary loops exactly as for the CPU
``"c"`` backend. The Hexagon wrapper rejects automatic/parallel async DMA
schedules before reusing the CPU pass sequence. Target copy/GEMM hooks lower
Crouton operations and explicit async-copy submissions.

This module is imported by :mod:`tilelang` at init time (and *not* by
``tilelang.hexagon.__init__``), so the lean build/deploy harness in that package
stays importable without the tilelang native library.
"""

from __future__ import annotations

from tilelang.backend.pass_pipeline import PassPipeline, register_pipeline
from tilelang.cpu.pipeline import CPUPassPipelineBody
from tilelang.tileop.gemm.registry import register_gemm_impl
from tilelang.cpu.op.gemm.gemm_scalar import GEMM_INST_SCALAR, GemmScalar
from tilelang.hexagon.gemm_hmx import GEMM_INST_HMX, GemmHMX

def HexagonPassPipelineBody(mod, target):
    from tvm import tirx

    for func in mod.functions.values():
        if not isinstance(func, tirx.PrimFunc):
            continue
        calls = []
        loops = []

        def collect(node):
            if isinstance(node, tirx.Call) and getattr(node.op, "name", "") == "tl.tileop.dma_copy":
                calls.append(node)
            if isinstance(node, tirx.For):
                loops.append(node)

        tirx.stmt_functor.post_order_visit(func.body, collect)
        if calls:
            for loop in loops:
                if any(str(key) == "num_stages" or str(key).startswith(("software_pipeline", "tl_pipeline"))
                       for key in loop.annotations):
                    raise ValueError("Hexagon T.dma_copy requires a manual schedule; T.Pipelined is unsupported")
                if loop.kind in (tirx.ForKind.PARALLEL, tirx.ForKind.VECTORIZED):
                    # Ordinary T.Parallel compute loops are allowed; only reject
                    # async submission under a parallel/vectorized loop.
                    nested = []
                    tirx.stmt_functor.post_order_visit(
                        loop.body,
                        lambda node: nested.append(node) if isinstance(node, tirx.Call)
                        and getattr(node.op, "name", "") == "tl.tileop.dma_copy" else None,
                    )
                    if nested:
                        raise ValueError("Hexagon T.dma_copy must be submitted from serial loops")
    return CPUPassPipelineBody(mod, target)


register_pipeline(PassPipeline("hexagon", HexagonPassPipelineBody))

# The C++ hexagon GemmImpl (src/hexagon/op/gemm.cc) selects "hexagon.hmx" for fp16
# 32-multiple gemms (-> GemmHMX: Crouton VTCM operands + HMX MAC), else "cpu.scalar"
# (-> GemmScalar: a triple loop hexagon-clang auto-vectorizes onto HVX).
register_gemm_impl(
    "hexagon.scalar", GEMM_INST_SCALAR, lambda t: t.kind.name == "hexagon", GemmScalar
)
register_gemm_impl(
    "hexagon.hmx", GEMM_INST_HMX, lambda t: t.kind.name == "hexagon", GemmHMX
)
