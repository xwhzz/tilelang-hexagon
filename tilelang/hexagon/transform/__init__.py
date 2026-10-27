"""Hexagon-specific transformation frontends."""

from .. import _ffi_api


def LowerKernelLaunch():
    """Mark the launch grid before MaterializeKernelLaunch.

    Each grid loop records its axis (``hexagon.grid_dim``), and the
    ``T.Kernel(num_workers=...)`` annotation moves from the launch block to the
    outermost grid loop.
    """
    return _ffi_api.LowerKernelLaunch()  # type: ignore


def MaterializeGrid():
    """Give the launch grid its final loops, after the loop transformations.

    Sinks the allocations the tile program owns back into its body, wraps the
    body in ``tl.tile_program_scope`` (where the VTCM arena is merged), turns a
    ``num_workers > 1`` grid into one parallel loop over the linearized grid
    index, and marks the function body as device code.
    """
    return _ffi_api.MaterializeGrid()  # type: ignore


def PlanDMAQueue():
    """Size and place the DMA queue of every tile program.

    Bounds the descriptors and commit groups in flight at any point of a tile
    program, declares the program's queue with those capacities
    (``hexagon_dma_queue``) at the start of its ``tl.tile_program_scope``, and
    drains it at the end.
    """
    return _ffi_api.PlanDMAQueue()  # type: ignore


def LowerMath():
    """Lower the fp16/fp32 math intrinsics to the HVX math builtins.

    Runs before layout inference so their loops vectorize; the codegen prints the
    builtins as the scalar or vector ``hvx_math.h`` functions.
    """
    return _ffi_api.LowerMath()  # type: ignore
