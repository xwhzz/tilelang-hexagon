from __future__ import annotations

from tilelang.backend.device_codegen import global_func_device_codegen

# The native build only emits the kernel C source; the Hexagon execution
# backend compiles it into a FastRPC skel with the Hexagon SDK.
build_hexagon = global_func_device_codegen("target.build.tilelang_hexagon")


class _GeneratedSource(str):
    """Unmodified Hexagon codegen output. String edits produce ordinary text."""
