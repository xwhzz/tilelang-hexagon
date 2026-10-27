"""Lowering pipeline for the Hexagon backend.

A Hexagon device kernel is sequential C with HVX/HMX intrinsics. The launch
stays GPU-shaped until codegen: ``MaterializeKernelLaunch`` keeps the grid as
``blockIdx`` thread_extent scopes (one thread unless ``threads=`` says
otherwise), and the Hexagon codegen serializes them into loops or fans the
outer loop across the worker pool. The sequence below is the CPU pass order
this backend was validated against, plus Hexagon's VTCM arena planning
(``MergeSharedMemoryAllocations``). Target copy/GEMM hooks lower Crouton
operations and explicit async-copy submissions.
"""

from __future__ import annotations

from tvm import IRModule, s_tir, tirx
from tvm.target import Target

import tilelang
from tilelang.backend.pass_pipeline.pipeline_utils import (
    LayoutVisual,
    allow_vectorize,
    should_disable_shared_memory_reuse,
    should_enable_aggressive_merge,
    should_enable_race_check,
    should_force_let_inline,
)


def _check_dma_schedule(mod: IRModule) -> None:
    """Reject automatic or parallel schedules around ``T.dma_copy``."""
    for func in mod.functions.values():
        if not isinstance(func, tirx.PrimFunc):
            continue
        calls = []
        loops = []

        def collect(node, calls=calls, loops=loops):
            if isinstance(node, tirx.Call) and getattr(node.op, "name", "") == "tl.tileop.dma_copy":
                calls.append(node)
            if isinstance(node, tirx.For):
                loops.append(node)

        tirx.stmt_functor.post_order_visit(func.body, collect)
        if calls:
            for loop in loops:
                if any(str(key) == "num_stages" or str(key).startswith(("software_pipeline", "tl_pipeline")) for key in loop.annotations):
                    raise ValueError("Hexagon T.dma_copy requires a manual schedule; T.Pipelined is unsupported")
                if loop.kind in (tirx.ForKind.PARALLEL, tirx.ForKind.VECTORIZED):
                    # Ordinary T.Parallel compute loops are allowed; only reject
                    # async submission under a parallel/vectorized loop.
                    nested = []
                    tirx.stmt_functor.post_order_visit(
                        loop.body,
                        lambda node, nested=nested: (
                            nested.append(node)
                            if isinstance(node, tirx.Call) and getattr(node.op, "name", "") == "tl.tileop.dma_copy"
                            else None
                        ),
                    )
                    if nested:
                        raise ValueError("Hexagon T.dma_copy must be submitted from serial loops")


def HexagonPassPipelineBody(mod: IRModule, target: Target) -> IRModule:
    _check_dma_schedule(mod)
    mod = tirx.transform.BindTarget(target)(mod)
    mod = tilelang.transform.MaterializeKernelLaunch(
        lower_thread_binding=True,
        default_threads=1,
        unsupported_annotations=["cluster_dims"],
    )(mod)
    pass_ctx = tilelang.transform.get_pass_context()

    if should_force_let_inline():
        mod = tilelang.transform.LetInline()(mod)
    mod = tilelang.transform.AddWrapperForSingleBufStore()(mod)
    mod = tilelang.transform.LegalizeNegativeIndex()(mod)
    if should_enable_race_check():
        mod = tilelang.transform.VerifyParallelLoop()(mod)
    mod = tilelang.transform.InjectAssumes()(mod)
    mod = tilelang.transform.Simplify()(mod)
    mod = tilelang.transform.CanonicalizeLegacyReducer()(mod)
    mod = tilelang.transform.VerifyReducerEpoch()(mod)

    mod = tilelang.transform.IfStmtBinding()(mod)
    mod = tilelang.transform.PipelinePlanning()(mod)
    mod = tilelang.transform.InjectSoftwarePipeline()(mod)
    mod = tilelang.transform.Simplify()(mod)

    mod = tilelang.transform.LayoutInference()(mod)
    mod = tilelang.transform.ReducerPlanAndMaterialize()(mod)
    LayoutVisual(mod)
    mod = tilelang.transform.LowerTileOp()(mod)
    mod = tilelang.transform.VerifyReducerConsumed()(mod)

    mod = tilelang.transform.DecoupleTypeCast()(mod)
    mod = tilelang.transform.LegalizeVectorizedLoop()(mod)
    mod = tilelang.transform.LegalizeSafeMemoryAccess()(mod)
    mod = tilelang.transform.LowerAccessPtr()(mod)
    mod = tilelang.transform.Simplify()(mod)
    mod = tilelang.transform.HoistNonRestrictParams()(mod)

    mod = tilelang.transform.PlanAndUpdateBufferAllocationLocation()(mod)
    mod = tilelang.transform.HoistGlobalBufferAllocations()(mod)
    mod = tilelang.transform.LowerOpaqueBlock()(mod)
    mod = tilelang.transform.Simplify()(mod)
    mod = tirx.transform.NarrowDataType(32)(mod)
    mod = tilelang.transform.FlattenBuffer()(mod)
    mod = tilelang.transform.ConfigIndexBitwidth()(mod)
    mod = tirx.transform.Simplify()(mod)
    mod = tilelang.transform.VectorizeLoop(enable_vectorize=allow_vectorize(pass_ctx=pass_ctx))(mod)
    mod = tilelang.transform.StorageRewrite()(mod)
    mod = tilelang.transform.LoopUnswitching()(mod)
    mod = tilelang.transform.UnrollLoop()(mod)
    mod = s_tir.transform.RenormalizeSplitPattern()(mod)
    mod = tirx.transform.Simplify()(mod)
    mod = tirx.transform.RemoveNoOp()(mod)
    mod = s_tir.transform.HoistIfThenElse()(mod)

    mod = tirx.transform.VerifyMemory()(mod)
    mod = tirx.transform.AnnotateEntryFunc()(mod)
    mod = s_tir.transform.InferFragment()(mod)
    mod = tilelang.transform.LowerThreadAllreduce()(mod)

    mod = tilelang.transform.AnnotateDeviceRegions()(mod)
    mod = tilelang.transform.SplitHostDevice()(mod)
    mod = tilelang.transform.AnnotateReadOnlyParams()(mod)

    # Shared buffers live in one VTCM arena; this pass also applies the HMX
    # operand alignments and keeps managed-DMA buffers from overlapping.
    enable_aggressive_merge = should_enable_aggressive_merge(pass_ctx=pass_ctx, target=target)
    disable_reuse = should_disable_shared_memory_reuse(pass_ctx=pass_ctx)
    mod = tilelang.transform.MergeSharedMemoryAllocations(enable_aggressive_merge=enable_aggressive_merge, disable_reuse=disable_reuse)(mod)

    mod = tilelang.transform.ThreadSync("shared")(mod)
    mod = tilelang.transform.ThreadSync("shared.dyn")(mod)
    mod = tilelang.transform.MergeIfStmt()(mod)
    mod = tilelang.transform.MakePackedAPI()(mod)
    mod = tilelang.transform.Simplify()(mod)
    mod = tilelang.transform.LowerDeviceKernelLaunch()(mod)
    return mod
