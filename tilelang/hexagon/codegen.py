from __future__ import annotations

from tilelang.backend.device_codegen import global_func_device_codegen

# The native build only emits the kernel C source; the Hexagon execution
# backend compiles it into a FastRPC skel with the Hexagon SDK.
build_hexagon = global_func_device_codegen("target.build.tilelang_hexagon")


def single_device_kernel(device_mod):
    """Return the lowered function whose attributes describe the kernel ABI."""
    if len(device_mod.functions) != 1:
        raise ValueError(f"Hexagon dispatch requires exactly one device kernel, got {len(device_mod.functions)}")
    (func,) = device_mod.functions.values()
    return func
