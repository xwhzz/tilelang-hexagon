"""Hexagon dialect of ``T.Kernel``: the common launch plus Hexagon launch annotations."""

from __future__ import annotations

from tvm import tirx

from tilelang.language.kernel import KernelLaunchFrame, kernel_launch_factory, launch_kernel

__all__ = ["Kernel"]


@kernel_launch_factory
def Kernel(
    *blocks: int | tirx.PrimExpr,
    prelude: str | None = None,
    num_workers: int | None = None,
) -> KernelLaunchFrame:
    """Construct a kernel launch frame for Hexagon: a grid of tile programs.

    The grid becomes a loop nest over the tile programs, run by one DSP
    hardware thread, or by a worker pool when ``num_workers`` is given. There
    are no SIMT threads, so this dialect has no ``threads`` and
    ``T.get_thread_binding()`` is rejected at compile time.

    Parameters
    ----------
    *blocks : int | PrimExpr
        Grid extent along each axis (1-3 dimensions). The launch yields one
        program index per axis.
    prelude : str, optional
        C source injected before the generated kernel, e.g. ``#include``
        lines or helper functions.
    num_workers : int, optional
        Run the tile programs on up to this many HW threads. Each worker runs
        whole programs with its own buffers and VTCM arena, so the programs
        must be independent, except at ``T.sync_grid()``: programs that call
        it run all at once, one per worker, and wait there for each other, so
        their grid may have at most ``num_workers`` programs. ``None`` runs the
        grid on one thread.

    Examples
    --------
    .. code-block:: python

        with T.Kernel(T.ceildiv(M, 32), num_workers=6) as bx:
            ...

        # Phases: every program writes its slice of `h`, then reads all of it.
        with T.Kernel(6, num_workers=6) as w:
            ...
            T.sync_grid()
            ...
    """
    annotations = {}
    # Validate here so a typo raises a clean ValueError instead of a later
    # compiler error (bool is an int subclass, so reject it explicitly).
    if num_workers is not None:
        if isinstance(num_workers, bool) or not isinstance(num_workers, int) or num_workers < 1:
            raise ValueError(f"T.Kernel(num_workers=...) must be a positive int, got {num_workers!r}")
        annotations["hexagon.num_workers"] = num_workers
    return launch_kernel(blocks, prelude=prelude, **annotations)
