"""Tile-level HMX accumulator ops for tilelang Hexagon kernels.

HMX has no `mma` instruction and an **invisible, non-addressable, single-per-core
accumulator**: issuing the activation+weight tile loads (`mxmem`) is what triggers the
MAC into the accumulator; `cvt` reads it out.  So a matmul is authored as tile ops
around that accumulator — `begin → clear → (mac per K-tile) → store` — NOT as a
monolithic `T.gemm`.  The point: a dequant (or any elementwise) can be interleaved
between the MACs (the K-streaming / dequant-hiding pattern), because the K-loop is
written in the DSL.

These are `@T.macro`s so they **inline into the kernel body** — required, because the
buffer uses (`T.address_of`) must be visible to the eager builder's VTCM liveness/arena
analysis, or the Crouton scratch aliases other `alloc_shared` buffers.  The scratch
itself is alloc'd by the caller in the body (macros can't return handles); `acc_scratch`
gives the sizes.  They lower to the `tl_hexagon_hmx_*` primitives in `hmx.h`.

    import tilelang.language as T
    from tilelang.hexagon import hmx_tile as hmx

    a_t = T.alloc_shared((hmx.a_tiles(M, K),), "float16")   # Crouton scratch
    b_t = T.alloc_shared((hmx.TILE,), "float16")            # one streamed K-tile of B
    c_t = T.alloc_shared((hmx.TILE,), "float16")
    hmx.pack_a(a_t, A_sh, M, K)                             # pack A once
    hmx.begin(); hmx.clear()
    for kt in range(K // 32):
        <dequant K-tile kt into B_tile [32][N]>            # DSL HVX dequant
        hmx.mac(a_t, b_t, kt, B_tile, N)                   # pack + MAC (dequant interleaves)
    hmx.store(c_t, C_sh, M, N)                             # cvt -> unpack -> C_sh; release

Single output tile (MT = NT = 1) for now; NT>1 loops clear/mac/store per N-tile.
"""
import tilelang.language as T

TILE = 1024  # fp16 per 32x32 Crouton tile


def a_tiles(M, K):
    """Element count for the A Crouton scratch (all M/K tiles). MT=1 -> (K//32) tiles."""
    return (M // 32) * (K // 32) * TILE


@T.macro
def pack_a(a_t, A_sh, M, K):
    """Pack the full row-major A [M][K] into its Crouton scratch (once)."""
    T.call_extern("int32", "tl_hexagon_hmx_pack_a", T.address_of(a_t[0]),
                  T.address_of(A_sh[0, 0]), M, K)


@T.macro
def begin():
    """Acquire the HMX unit and set the session's (unit) output scales."""
    T.call_extern("int32", "tl_hexagon_hmx_open")


@T.macro
def clear():
    """Clear the invisible accumulator (mxclracc)."""
    T.call_extern("int32", "tl_hexagon_hmx_clear")


@T.macro
def mac(a_t, b_t, kt, B_tile, N):
    """Pack one K-tile of B (row-major [32][N], freshly dequantized) to Crouton and MAC
    it into the accumulator. Placing the dequant right before this call interleaves the
    HVX dequant with the HMX MACs."""
    T.call_extern("int32", "tl_hexagon_hmx_pack_b", T.address_of(b_t[0]),
                  T.address_of(B_tile[0, 0]), 32, N)
    T.call_extern("int32", "tl_hexagon_hmx_mac", T.address_of(a_t[kt * TILE]),
                  T.address_of(b_t[0]))


@T.macro
def store(c_t, C_sh, M, N):
    """cvt the accumulator to fp16 and unpack to row-major C_sh [M][N]. Call once per
    output N-tile (each N-tile is its own clear -> mac-K -> store on the one accumulator).
    Does NOT release the unit — call `end()` once after the last store."""
    T.call_extern("int32", "tl_hexagon_hmx_store", T.address_of(c_t[0]))
    T.call_extern("int32", "tl_hexagon_hmx_unpack_c", T.address_of(C_sh[0, 0]),
                  T.address_of(c_t[0]), M, N)


@T.macro
def end():
    """Release the HMX unit (once, after the last store)."""
    T.call_extern("int32", "tl_hexagon_hmx_close")
