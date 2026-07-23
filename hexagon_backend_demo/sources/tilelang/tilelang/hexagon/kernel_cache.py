"""In-memory kernel cache for the Hexagon backend.

A Hexagon "compiled artifact" is an on-device FastRPC deployment plus a live
persistent agent session (sockets + a spawned ``adb`` process) — none of which
is disk-serializable the way an in-process ``.so`` is.  So this cache keeps the
base :class:`KernelCache`'s in-memory reuse (don't rebuild + redeploy the same
kernel twice within a process) but skips the disk persistence that the base does
for in-process backends.  A cold process recompiles, which also re-deploys — the
intended behavior, since the device state doesn't outlive the process either.
"""

from __future__ import annotations

from tilelang.cache.kernel_cache import KernelCache


class HexagonKernelCache(KernelCache):
    # Cosmetic: the base always receives the backend explicitly from
    # ``_resolve_cache_dispatch``; this just documents the dispatch key.
    execution_backend = "hexagon"

    def _save_kernel_to_disk(self, key, kernel, func=None, verbose=False):
        # No-op: an on-device deployment + live agent session isn't persistable.
        return

    def _load_kernel_from_disk(
        self,
        key,
        target="auto",
        target_host=None,
        out_idx=None,
        execution_backend="hexagon",
        pass_configs=None,
        compile_flags=None,
        func=None,
        verbose=False,
    ):
        # Always a miss: rely on the in-memory cache (or a fresh compile+deploy).
        return None
