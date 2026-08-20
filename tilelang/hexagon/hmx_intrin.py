"""Instruction-level HMX primitives and Crouton layouts.

HMX operands are memory-resident Crouton tiles in VTCM.  The accumulator,
convert state, and bias state are implicit hardware resources rather than
addressable register fragments.  This module therefore keeps three concerns
separate:

* :class:`~tilelang.layout.Layout` describes physical Crouton addresses;
* dependency-only state tokens describe implicit HMX state ordering;
* :class:`HMXIntrinEmitter` emits one hardware protocol operation at a time.

Session, power, and VTCM ownership belong to the embedding runtime.  The
emitter only brackets exclusive accumulator use and never acquires a session.
"""

from __future__ import annotations

from collections.abc import Sequence

import tilelang.language as T

TILE = 32
TILE_ELEMS = TILE * TILE
TILE_BYTES = TILE_ELEMS * 2
BIAS_WORDS = 64


def _shape_of(buffer_or_shape) -> tuple:
    shape = getattr(buffer_or_shape, "shape", buffer_or_shape)
    if not isinstance(shape, Sequence):
        raise TypeError("HMX layout expects a Buffer or a shape sequence")
    return tuple(shape)


def _static_extent(value, name: str) -> int:
    value = getattr(value, "value", value)
    if not isinstance(value, int):
        raise ValueError(f"HMX {name} extent must be static; got {value}")
    return value


def _validate_matrix_shape(shape: tuple, name: str) -> tuple[int, int]:
    if len(shape) < 2:
        raise ValueError(f"HMX {name} layout requires at least two dimensions")
    rows = _static_extent(shape[-2], f"{name} rows")
    cols = _static_extent(shape[-1], f"{name} columns")
    if rows <= 0 or cols <= 0 or rows % TILE or cols % TILE:
        raise ValueError(
            f"HMX {name} dimensions must be positive multiples of {TILE}; "
            f"got ({rows}, {cols})"
        )
    for i, extent in enumerate(shape[:-2]):
        _static_extent(extent, f"{name} prefix dimension {i}")
    return rows, cols


def make_hmx_crouton_layout() -> T.Layout:
    """Return the native 32x32 FP16 HMX Crouton layout atom."""

    return T.Layout(
        (TILE, TILE),
        lambda row, col: [(row // 2) * (2 * TILE) + col * 2 + row % 2],
    )


def _expand_matrix_layout(
    shape: tuple, tile_order: tuple[int, int], name: str
) -> T.Layout:
    rows, cols = _validate_matrix_shape(shape, name)
    factors = (rows // TILE, cols // TILE)
    layout = make_hmx_crouton_layout()
    for dim in tile_order:
        layout = layout.repeat(dim, factors[dim])
    return layout.expand(shape[:-2])


def make_hmx_activation_layout(buffer_or_shape, transposed: bool = False) -> T.Layout:
    """Map activation storage to physical ``[..., mt, kt, cpos]``.

    The normal logical shape is ``[..., M, K]``.  With ``transposed=True`` the
    input buffer is ``[..., K, M]``, but its physical tile order remains the HMX
    activation order for the mathematical ``[M, K]`` operand.
    """

    shape = _shape_of(buffer_or_shape)
    if not transposed:
        return _expand_matrix_layout(shape, (1, 0), "activation")
    base = _expand_matrix_layout(shape[:-2] + (shape[-1], shape[-2]), (1, 0), "activation")

    def forward(*indices):
        return base(*(indices[:-2] + (indices[-1], indices[-2])))

    return T.Layout(shape, forward)


def make_hmx_weight_layout(buffer_or_shape, transposed: bool = False) -> T.Layout:
    """Map weight storage to physical ``[..., nt, kt, cpos]``.

    The normal logical shape is ``[..., K, N]``.  With ``transposed=True`` the
    input buffer is ``[..., N, K]``, while HMX still sees mathematical ``[K, N]``.
    """

    shape = _shape_of(buffer_or_shape)
    if not transposed:
        return _expand_matrix_layout(shape, (0, 1), "weight")
    base = _expand_matrix_layout(shape[:-2] + (shape[-1], shape[-2]), (0, 1), "weight")

    def forward(*indices):
        return base(*(indices[:-2] + (indices[-1], indices[-2])))

    return T.Layout(shape, forward)


def make_hmx_output_layout(buffer_or_shape) -> T.Layout:
    """Map logical ``[..., M, N]`` to physical ``[..., mt, nt, cpos]``."""

    shape = _shape_of(buffer_or_shape)
    return _expand_matrix_layout(shape, (1, 0), "output")


def _buffer_prefix(buffer, prefix, name: str) -> tuple:
    prefix = () if prefix is None else tuple(prefix)
    rank = len(getattr(buffer, "shape", ()))
    expected = max(rank - 2, 0)
    if len(prefix) != expected:
        raise ValueError(
            f"HMX {name} expects {expected} prefix indices for a rank-{rank} buffer; "
            f"got {len(prefix)}"
        )
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
            T.hexagon_hmx_acquire(T.access_ptr(acc_state[0], "rw"))

        return _acquire(acc_state)

    def release(self, acc_state):
        @T.macro
        def _release(acc_state):
            T.hexagon_hmx_release(T.access_ptr(acc_state[0], "rw"))

        return _release(acc_state)

    def load_bias(self, bias_state, bias_vtcm):
        @T.macro
        def _load_bias(bias_state, bias_vtcm):
            T.hexagon_hmx_load_bias(
                T.access_ptr(bias_state[0], "w"),
                T.access_ptr(bias_vtcm[0], "r", BIAS_WORDS),
            )

        return _load_bias(bias_state, bias_vtcm)

    def clear(self, acc_state):
        @T.macro
        def _clear(acc_state):
            T.hexagon_hmx_clear(T.access_ptr(acc_state[0], "w"))

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
    ):
        """Emit one HMX MAC atom for output tile ``(inst_m_idx, inst_n_idx)``.

        ``a_prefix`` and ``b_prefix`` address independent staged matrices.  The
        The final dimensions follow the operand's logical storage shape
        (``[M,K]``/``[K,N]`` or their transposes) and are rewritten by
        ``T.Layout`` to the same native mathematical Crouton tiles.
        """

        _validate_atom_index(inst_m_idx, self.MT, "inst_m_idx")
        _validate_atom_index(inst_n_idx, self.NT, "inst_n_idx")
        _validate_atom_index(k_inner, self.KT, "k_inner")
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
            T.hexagon_hmx_mma(
                T.access_ptr(acc_state[0], "rw"),
                T.access_ptr(activation[a_index], "r", TILE_ELEMS),
                T.access_ptr(weight[b_index], "r", TILE_ELEMS),
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
        """Reduce all K atoms for one output tile into the implicit accumulator."""

        _validate_atom_index(inst_m_idx, self.MT, "inst_m_idx")
        _validate_atom_index(inst_n_idx, self.NT, "inst_n_idx")
        a_prefix = _buffer_prefix(activation, a_prefix, "activation")
        b_prefix = _buffer_prefix(weight, b_prefix, "weight")

        @T.macro
        def _mma(acc_state, activation, weight):
            for k_inner in T.serial(self.KT):
                self.mma_atom(
                    acc_state,
                    activation,
                    weight,
                    inst_m_idx,
                    inst_n_idx,
                    k_inner,
                    a_prefix,
                    b_prefix,
                )

        return _mma(acc_state, activation, weight)

    def convert(self, cvt_state, acc_state, bias_state, bias_vtcm, mode=2):
        @T.macro
        def _convert(cvt_state, acc_state, bias_state, bias_vtcm):
            T.hexagon_hmx_convert(
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
        activation,
        weight,
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
            activation,
            weight,
        ):
            T.hexagon_hmx_store(
                T.access_ptr(cvt_state[0], "r"),
                T.access_ptr(output[output_index], "w", TILE_ELEMS),
                T.access_ptr(acc_state[0], "rw"),
                T.access_ptr(bias_state[0], "r"),
                T.access_ptr(bias_vtcm[0], "r", BIAS_WORDS),
                # These are whole-buffer lifetime dependencies, not additional
                # hardware operands. Passing the Buffer itself keeps store()
                # valid for both 2-D Croutons and staged 3-D K-tile arrays.
                T.access_ptr(activation, "r"),
                T.access_ptr(weight, "r"),
            )

        return _store(
            cvt_state,
            acc_state,
            output,
            bias_state,
            bias_vtcm,
            activation,
            weight,
        )


__all__ = [
    "TILE",
    "TILE_ELEMS",
    "TILE_BYTES",
    "BIAS_WORDS",
    "make_hmx_crouton_layout",
    "make_hmx_activation_layout",
    "make_hmx_weight_layout",
    "make_hmx_output_layout",
    "HMXIntrinEmitter",
]
