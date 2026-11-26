from __future__ import annotations

from tilelang.backend.device_codegen import global_func_device_codegen

# The native build only emits the kernel C source; the Hexagon execution
# backend compiles it into a FastRPC skel with the Hexagon SDK.
build_hexagon = global_func_device_codegen("target.build.tilelang_hexagon")


class _GeneratedSource(str):
    """Unmodified Hexagon codegen output. String edits produce ordinary text."""


def kernel_entry_name(device_mod) -> str:
    """Return the symbol that Hexagon codegen emits for a single device kernel."""
    if len(device_mod.functions) != 1:
        raise ValueError(f"Hexagon dispatch requires exactly one device kernel, got {len(device_mod.functions)}")
    (func,) = device_mod.functions.values()
    return str(func.attrs["global_symbol"])
