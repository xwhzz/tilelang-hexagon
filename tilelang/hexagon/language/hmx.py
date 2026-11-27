"""Hexagon matrix accelerator tile instructions."""

from __future__ import annotations

from tvm import arith, tirx

from tilelang._typing import BufferLikeType
from tilelang.language.allocate import alloc_shared
from tilelang.language.builtin import access_ptr
from tilelang.utils.language import to_buffer_region

__all__ = ["hmx_gemm_u8i8"]

# A 32-channel tile's largest possible dot magnitude is 32 * 255 * 128.
_MAX_K_TILES = (2**31 - 1) // (32 * 255 * 128)


def _operand(value, name, dtype, inner_shape, *, has_k_tiles=False, allow_grid=False):
    region = to_buffer_region(value)
    buffer = region.buffer
    local = buffer.scope() == "shared.dyn"
    grid = allow_grid and buffer.scope() == "global.vtcm"
    if not (local or grid) or str(buffer.dtype) != dtype:
        scope = "program-local or grid VTCM" if allow_grid else "program-local VTCM"
        raise ValueError(f"T.hmx_gemm_u8i8 requires {name} in {scope} with dtype {dtype}")

    analyzer = arith.Analyzer()
    inner_rank = len(inner_shape)
    outer_rank = len(buffer.shape) - inner_rank - int(has_k_tiles)
    expected_shape = f"[K//32, {', '.join(map(str, inner_shape))}]" if has_k_tiles else str(list(inner_shape))
    if outer_rank < 0 or any(not analyzer.can_prove_equal(r.extent, 1) for r in region.region[:outer_rank]):
        raise ValueError(f"T.hmx_gemm_u8i8 requires {name} to select a contiguous tile region shaped {expected_shape}")

    for dim, axis, expected in zip(buffer.shape[-inner_rank:], region.region[-inner_rank:], inner_shape):
        if not (
            analyzer.can_prove_equal(dim, expected)
            and analyzer.can_prove_equal(axis.min, 0)
            and analyzer.can_prove_equal(axis.extent, expected)
        ):
            raise ValueError(f"T.hmx_gemm_u8i8 requires {name} to select a contiguous tile region shaped {expected_shape}")

    if len(buffer.strides) or not analyzer.can_prove_equal(buffer.elem_offset, 0):
        raise ValueError(f"T.hmx_gemm_u8i8 requires a compact, zero-offset {name} buffer")
    return region, analyzer


def hmx_gemm_u8i8(A: BufferLikeType, B: BufferLikeType, C: BufferLikeType) -> tirx.PrimExpr:
    """Overwrite one 64-by-32 int32 tile with an exact uint8-by-int8 HMX dot.

    Stage all operands in program-local VTCM (the default ``T.alloc_shared``
    scope, ``shared.dyn``). A can also live in grid VTCM (``global.vtcm``)
    when the caller synchronizes writes across programs.
    Pack logical ``A[m, k]`` as ``A[k // 32, m, k % 32]`` with shape
    ``[K // 32, 64, 32]``. Pack logical ``B[n, k]`` as
    ``B[k // 32, (k % 32) // 4, n, k % 4]`` with shape
    ``[K // 32, 8, 32, 4]``. The output is row-major ``C[64, 32]``. These
    physical layouts use compact buffers without a layout permutation.

    ``K`` is a positive compile-time multiple of 32 within the signed int32
    worst-case accumulation bound. The call accepts consecutive K tiles from
    larger buffers and an optional single index for each leading batch axis.
    Pass the allocated buffers or their directly indexed, in-bounds regions.
    For partial tiles, zero-pad the operands and use only the valid output.
    Accumulation across calls, scales and zero points are applied by the caller.
    """
    return _hmx_gemm_u8i8(A, B, C, *_alloc_hmx_u8i8_scratch())


def _alloc_hmx_u8i8_scratch():
    # Four accumulator byte planes and one conversion config per plane.
    planes = alloc_shared((4 * 64 * 32,), "uint8", align=2048)
    config = alloc_shared((4 * 256,), "uint32", align=256)
    return planes, config


def _hmx_gemm_u8i8(A, B, C, planes, config):
    """Issue the tile multiply with reusable, program-local scratch."""
    a, analyzer = _operand(A, "A", "uint8", (64, 32), has_k_tiles=True, allow_grid=True)
    b, _ = _operand(B, "B", "int8", (8, 32, 4), has_k_tiles=True)
    c, _ = _operand(C, "C", "int32", (64, 32))
    k_tiles = analyzer.simplify(a.region[-3].extent)
    if not isinstance(k_tiles, tirx.IntImm) or not 1 <= k_tiles.value <= _MAX_K_TILES:
        raise ValueError(f"T.hmx_gemm_u8i8 requires between 1 and {_MAX_K_TILES} statically known K tiles of 32 channels")
    if not analyzer.can_prove_equal(k_tiles, b.region[-4].extent):
        raise ValueError("T.hmx_gemm_u8i8 requires A and B to have the same number of K tiles")

    # The runtime fully initializes both scratch arrays, regardless of reuse.
    return tirx.call_intrin(
        "void",
        tirx.op.Op.get("tl.hexagon_hmx_gemm_u8i8"),
        k_tiles,
        access_ptr(a, "r"),
        access_ptr(b, "r"),
        access_ptr(c, "w"),
        access_ptr(planes, "w"),
        access_ptr(config, "w"),
    )
