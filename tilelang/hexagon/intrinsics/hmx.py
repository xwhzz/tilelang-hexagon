"""HMX instruction emitter for hand-scheduled kernels.

HMX operands are memory-resident Crouton tiles in VTCM
(:mod:`tilelang.hexagon.intrinsics.crouton`). The accumulator, convert state
and bias state are implicit hardware resources rather than addressable
register fragments, so this module keeps three concerns separate:

* :class:`~tilelang.layout.Layout` describes physical Crouton addresses;
* dependency-only state tokens (``T.hmx.alloc_*``) describe implicit HMX
  state ordering;
* :class:`HMXIntrinEmitter` emits one hardware protocol operation at a time.

Session, power, and VTCM ownership belong to the embedding runtime.  The
emitter only brackets exclusive accumulator use and never acquires a session.
"""

from __future__ import annotations

from tilelang.hexagon import language as T

from .crouton import (
    TILE,
    TILE_ELEMS,
    make_hmx_activation_layout,
    make_hmx_output_layout,
    make_hmx_weight_layout,
)

BIAS_WORDS = 64


def _buffer_prefix(buffer, prefix, name: str) -> tuple:
    prefix = () if prefix is None else tuple(prefix)
    rank = len(getattr(buffer, "shape", ()))
    expected = max(rank - 2, 0)
    if len(prefix) != expected:
        raise ValueError(f"HMX {name} expects {expected} prefix indices for a rank-{rank} buffer; got {len(prefix)}")
    return prefix


def _validate_atom_index(index, extent: int, name: str):
    if isinstance(index, int) and not 0 <= index < extent:
        raise ValueError(f"HMX {name} must be in [0, {extent}); got {index}")


class HMXIntrinEmitter:
    """Compose explicit HMX state and 32x32x32 instruction atoms.

    Unlike CUDA tensor cores, HMX exposes one implicit accumulator rather than
    an addressable accumulator fragment array.  ``mma_tile()`` therefore reduces K
    for exactly one ``(inst_m_idx, inst_n_idx)`` output tile.  The caller must
    bracket each output tile with ``clear -> mma_tile -> convert -> store``.
    For a multi-tile output, ``store`` addresses the requested native Crouton
    tile directly. TileLang's shared-memory planner propagates the native VTCM
    alignment contracts per operand: 2 KiB for activation/output, 128 B for
    weight, and 256 B for the scale/bias config block.

    Operands must already reside in caller-owned VTCM with the layouts returned
    by this emitter.  Session/power setup and the cold-session accumulator-read
    configuration remain runtime responsibilities; ``load_bias`` only loads
    the conversion scale/bias block used by this instruction sequence.
    """

    def __init__(
        self,
        M: int = TILE,
        N: int = TILE,
        K: int = TILE,
        a_dtype: str = "float16",
        b_dtype: str = "float16",
        a_transposed: bool = False,
        b_transposed: bool = False,
    ):
        if M <= 0 or N <= 0 or K <= 0 or M % TILE or N % TILE or K % TILE:
            raise ValueError("HMX dimensions must be positive multiples of 32")
        if a_dtype != "float16" or b_dtype != "float16":
            raise ValueError("The validated HMX path accepts FP16 operands")
        self.M, self.N, self.K = M, N, K
        self.MT, self.NT, self.KT = M // TILE, N // TILE, K // TILE
        self.a_dtype, self.b_dtype = a_dtype, b_dtype
        self.a_transposed, self.b_transposed = a_transposed, b_transposed

    @property
    def num_m_tiles(self) -> int:
        """Number of output tiles along M; scheduled outside ``mma_tile``."""

        return self.MT

    @property
    def num_n_tiles(self) -> int:
        """Number of output tiles along N; scheduled outside ``mma_tile``."""

        return self.NT

    @property
    def num_k_atoms(self) -> int:
        """Number of 32-wide K atoms reduced by ``mma_tile``."""

        return self.KT

    def activation_layout(self, buffer_or_shape) -> T.Layout:
        return make_hmx_activation_layout(buffer_or_shape, self.a_transposed)

    def weight_layout(self, buffer_or_shape) -> T.Layout:
        return make_hmx_weight_layout(buffer_or_shape, self.b_transposed)

    @staticmethod
    def output_layout(buffer_or_shape) -> T.Layout:
        return make_hmx_output_layout(buffer_or_shape)

    def acquire(self, acc_state):
        @T.macro
        def _acquire(acc_state):
            T.hmx.acquire(T.access_ptr(acc_state[0], "rw"))

        return _acquire(acc_state)

    def release(self, acc_state):
        @T.macro
        def _release(acc_state):
            T.hmx.release(T.access_ptr(acc_state[0], "rw"))

        return _release(acc_state)

    def load_bias(self, bias_state, bias_vtcm):
        @T.macro
        def _load_bias(bias_state, bias_vtcm):
            T.hmx.load_bias(
                T.access_ptr(bias_state[0], "w"),
                T.access_ptr(bias_vtcm[0], "r", BIAS_WORDS),
            )

        return _load_bias(bias_state, bias_vtcm)

    def clear(self, acc_state):
        @T.macro
        def _clear(acc_state):
            T.hmx.clear(T.access_ptr(acc_state[0], "w"))

        return _clear(acc_state)

    def mma_atom(
        self,
        acc_state,
        activation,
        weight,
        inst_m_idx=0,
        inst_n_idx=0,
        k_inner=0,
        a_prefix=(),
        b_prefix=(),
        num_tiles=1,
    ):
        """Emit one HMX packet reducing 1..32 contiguous K tiles for an output tile.

        ``a_prefix`` and ``b_prefix`` address independent staged matrices.  The
        final dimensions follow the operand's logical storage shape
        (``[M,K]``/``[K,N]`` or their transposes) and are rewritten by
        ``T.Layout`` to the same native mathematical Crouton tiles.
        """

        _validate_atom_index(inst_m_idx, self.MT, "inst_m_idx")
        _validate_atom_index(inst_n_idx, self.NT, "inst_n_idx")
        _validate_atom_index(k_inner, self.KT, "k_inner")
        if isinstance(num_tiles, int):
            if not 1 <= num_tiles <= 32:
                raise ValueError("HMX deep num_tiles must be in [1, 32]")
            if isinstance(k_inner, int) and k_inner + num_tiles > self.KT:
                raise ValueError("HMX deep packet exceeds operand K tiles")
        a_prefix = _buffer_prefix(activation, a_prefix, "activation")
        b_prefix = _buffer_prefix(weight, b_prefix, "weight")
        if self.a_transposed:
            a_index = a_prefix + (k_inner * TILE, inst_m_idx * TILE)
        else:
            a_index = a_prefix + (inst_m_idx * TILE, k_inner * TILE)
        if self.b_transposed:
            b_index = b_prefix + (inst_n_idx * TILE, k_inner * TILE)
        else:
            b_index = b_prefix + (k_inner * TILE, inst_n_idx * TILE)

        @T.macro
        def _mma(acc_state, activation, weight):
            T.hmx.mma(
                T.access_ptr(acc_state[0], "rw"),
                T.access_ptr(activation[a_index], "r", num_tiles * TILE_ELEMS),
                T.access_ptr(weight[b_index], "r", num_tiles * TILE_ELEMS),
                num_tiles,
            )

        return _mma(acc_state, activation, weight)

    def mma_tile(
        self,
        acc_state,
        activation,
        weight,
        inst_m_idx=0,
        inst_n_idx=0,
        a_prefix=(),
        b_prefix=(),
    ):
        """Reduce one output tile with caller-side chunks of at most 32 K tiles.

        Each chunk issues one deep packet; the final packet reads only the
        remaining tiles. Native operand layouts keep K tiles contiguous.
        """

        _validate_atom_index(inst_m_idx, self.MT, "inst_m_idx")
        _validate_atom_index(inst_n_idx, self.NT, "inst_n_idx")
        a_prefix = _buffer_prefix(activation, a_prefix, "activation")
        b_prefix = _buffer_prefix(weight, b_prefix, "weight")

        @T.macro
        def _mma(acc_state, activation, weight):
            for k_chunk in T.serial(T.ceildiv(self.KT, 32)):
                self.mma_atom(
                    acc_state,
                    activation,
                    weight,
                    inst_m_idx,
                    inst_n_idx,
                    k_chunk * 32,
                    a_prefix,
                    b_prefix,
                    T.min(self.KT - k_chunk * 32, 32),
                )

        return _mma(acc_state, activation, weight)

    def convert(self, cvt_state, acc_state, bias_state, bias_vtcm, mode=2):
        @T.macro
        def _convert(cvt_state, acc_state, bias_state, bias_vtcm):
            T.hmx.convert(
                T.access_ptr(cvt_state[0], "w"),
                T.access_ptr(acc_state[0], "r"),
                T.access_ptr(bias_state[0], "r"),
                T.access_ptr(bias_vtcm[0], "r", BIAS_WORDS),
                mode,
            )

        return _convert(cvt_state, acc_state, bias_state, bias_vtcm)

    def store(
        self,
        cvt_state,
        acc_state,
        output,
        bias_state,
        bias_vtcm,
        inst_m_idx=0,
        inst_n_idx=0,
        output_prefix=(),
    ):
        _validate_atom_index(inst_m_idx, self.MT, "inst_m_idx")
        _validate_atom_index(inst_n_idx, self.NT, "inst_n_idx")
        output_prefix = _buffer_prefix(output, output_prefix, "output")
        output_index = output_prefix + (inst_m_idx * TILE, inst_n_idx * TILE)

        @T.macro
        def _store(
            cvt_state,
            acc_state,
            output,
            bias_state,
            bias_vtcm,
        ):
            T.hmx.store(
                T.access_ptr(cvt_state[0], "r"),
                T.access_ptr(output[output_index], "w", TILE_ELEMS),
                T.access_ptr(acc_state[0], "rw"),
                T.access_ptr(bias_state[0], "r"),
                T.access_ptr(bias_vtcm[0], "r", BIAS_WORDS),
            )

        return _store(
            cvt_state,
            acc_state,
            output,
            bias_state,
            bias_vtcm,
        )


__all__ = [
    "BIAS_WORDS",
    "HMXIntrinEmitter",
]
