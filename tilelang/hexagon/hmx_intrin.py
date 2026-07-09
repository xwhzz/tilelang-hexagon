"""HMX instruction-atom emitter — the Hexagon analog of
``tilelang.cuda.intrinsics.macro.mma_macro_generator.TensorCoreIntrinEmitter``.

Like the tensor-core emitter, this turns a matmul's tiling config into
**instruction-atom** primitives that emit ``@T.macro`` TIR, so a kernel drives the
K-loop itself (and can interleave a dequant between the loads and the MAC) instead of
calling a monolithic ``T.gemm``.

Atom mapping (tensor core -> HMX):
  * ``ldmatrix_a/ldmatrix_b`` (shared -> register fragment)  -> ``pack_a/pack_b``
    (row-major VTCM -> **Crouton** VTCM, via HVX vshuff).
  * ``mma`` / ``mma_atom`` (fragment x fragment -> acc)       -> ``mma`` (the ``mxmem``
    activation+weight load pair, which IS the MAC).
  * ``stmatrix`` (acc fragment -> out)                        -> ``store`` (``cvt`` acc
    -> Crouton -> unpack row-major).
  * ``T.clear(C_local)``                                      -> ``clear`` (``mxclracc``).

**Honest divergence from tensor cores.** HMX has no ``mma`` instruction and NO
addressable accumulator fragment: loading the a/b Crouton tiles (``mxmem``) is what
triggers the MAC, into a *single, invisible, per-core* accumulator. So ``clear``/``mma``/
``store`` operate on that implicit accumulator (no ``C_local`` is passed), and the A/B
"fragments" are the Crouton VTCM tiles that ``pack_a``/``pack_b`` produce.

The methods build a nested ``@T.macro`` and return its call (the tensor-core pattern) —
required so the Crouton-scratch buffer uses are visible to the VTCM liveness/arena pass.
Single M-tile (M<=32); loop N-tiles outside for N>32 (one physical accumulator). Lowers
to the ``tl_hexagon_hmx_*`` primitives in ``src/tl_templates/hexagon/hmx.h``.
"""
import tilelang.language as T

TILE = 32           # HMX Crouton tile is 32x32
TILE_ELEMS = 1024   # fp16 per tile


class HMXIntrinEmitter:

    def __init__(self, M, N, K, a_dtype="float16", b_dtype="float16", accum_dtype="float16"):
        assert M % 32 == 0 and N % 32 == 0 and K % 32 == 0, "HMX works in 32x32x32 tiles"
        self.M, self.N, self.K = M, N, K
        self.MT, self.NT, self.KT = M // 32, N // 32, K // 32
        self.a_dtype, self.b_dtype, self.accum_dtype = a_dtype, b_dtype, accum_dtype

    # ---- Crouton "fragment" shapes the caller allocs (== the register frags on CUDA) ----
    @property
    def a_frag_shape(self):
        return (self.MT * self.KT * TILE_ELEMS,)   # whole A, packed once

    @property
    def b_frag_shape(self):
        return (TILE_ELEMS,)                        # one streamed K-tile of B

    @property
    def c_frag_shape(self):
        return (TILE_ELEMS,)                        # one output tile

    # ---- HMX unit session (no tensor-core analog) ----
    def begin(self):
        @T.macro
        def _begin():
            T.call_extern("int32", "tl_hexagon_hmx_open")
        return _begin()

    def end(self):
        @T.macro
        def _end():
            T.call_extern("int32", "tl_hexagon_hmx_close")
        return _end()

    # ---- clear the (implicit) accumulator  (== T.clear(C_local)) ----
    def clear(self):
        @T.macro
        def _clear():
            T.call_extern("int32", "tl_hexagon_hmx_clear")
        return _clear()

    # ---- pack row-major VTCM -> Crouton VTCM  (== ldmatrix) ----
    def pack_a(self, a_frag, A_shared):
        M, K = self.M, self.K
        @T.macro
        def _pack_a(a_frag, A_shared):
            T.call_extern("int32", "tl_hexagon_hmx_pack_a", T.address_of(a_frag[0]),
                          T.address_of(A_shared[0, 0]), M, K)
        return _pack_a(a_frag, A_shared)

    def pack_b(self, b_frag, B_tile):
        # B_tile: one row-major [32][32] tile, freshly dequantized
        @T.macro
        def _pack_b(b_frag, B_tile):
            T.call_extern("int32", "tl_hexagon_hmx_pack_b", T.address_of(b_frag[0]),
                          T.address_of(B_tile[0, 0]), 32, 32)
        return _pack_b(b_frag, B_tile)

    # ---- the MAC atom: mxmem(a_tile) + mxmem(b_tile) -> acc  (== mma_atom) ----
    def mma(self, a_frag, b_frag, kt):
        @T.macro
        def _mma(a_frag, b_frag):
            T.call_extern("int32", "tl_hexagon_hmx_mac",
                          T.address_of(a_frag[kt * TILE_ELEMS]), T.address_of(b_frag[0]))
        return _mma(a_frag, b_frag)

    # ---- cvt acc -> Crouton -> row-major out  (== stmatrix) ----
    def store(self, C_tile, c_frag):
        M = self.M
        @T.macro
        def _store(C_tile, c_frag):
            T.call_extern("int32", "tl_hexagon_hmx_store", T.address_of(c_frag[0]))
            T.call_extern("int32", "tl_hexagon_hmx_unpack_c", T.address_of(C_tile[0, 0]),
                          T.address_of(c_frag[0]), M, 32)
        return _store(C_tile, c_frag)
