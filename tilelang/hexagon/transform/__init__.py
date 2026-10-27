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


def VerifyVTCMArena():
    """Check that the VTCM arena of every tile program fits the target.

    Runs after ``MergeSharedMemoryAllocations``. ``T.Kernel(num_workers=N)``
    needs N arenas, one per worker.
    """
    return _ffi_api.VerifyVTCMArena()  # type: ignore


def LowerMath():
    """Lower the fp16/fp32 math intrinsics to the HVX math builtins.

    Runs before layout inference so their loops vectorize; the codegen prints the
    builtins as the scalar or vector ``hvx_math.h`` functions.
    """
    return _ffi_api.LowerMath()  # type: ignore


def BroadcastFromRegisters():
    """Broadcast per-row scalars of vectorized maps from registers.

    A row loop that broadcasts ``b[i]`` into the vectors of row ``i`` runs in
    strips of one register's worth of rows: each strip loads its rows'
    scalars into a register once, and each row broadcasts its lane from it.
    The loop then reads only the tile, so it does not stall on a scalar load
    per row, and LLVM can software-pipeline it. Runs after ``VectorizeLoop``.
    """
    return _ffi_api.BroadcastFromRegisters()  # type: ignore


def LowerWindowLoads():
    """Lower vector loads of small windows to a load and a lane shuffle.

    In the physical order of a Crouton tile, a vector of the rows' values reads
    ``[r0, r1, r0, r1, ...]`` and one of the columns' values ``[c0, c0, c1, c1,
    ...]``. The vectorizer leaves these as per-lane gathers; this pass loads
    the window once and replicates it, or repeats its elements, in registers.
    Runs after ``VectorizeLoop``.
    """
    return _ffi_api.LowerWindowLoads()  # type: ignore


def MarkExclusiveBuffers():
    """Give the VTCM buffers that share no memory pointers of their own.

    Runs after ``MergeSharedMemoryAllocations``. LLVM sees every buffer of an
    arena as one object, so its software pipeliner serializes maps that read
    one buffer and write another. A buffer no other buffer overlaps gets its
    pointer through ``tl_vtcm_exclusive``, which LLVM treats as a separate
    allocation.
    """
    return _ffi_api.MarkExclusiveBuffers()  # type: ignore
