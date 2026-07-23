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


def _prefix_offset(indices, shape: tuple, matrix_elems: int):
    prefix = 0
    for index, extent in zip(indices[:-2], shape[:-2]):
        prefix = prefix * _static_extent(extent, "prefix") + index
    return prefix * matrix_elems


def make_hmx_activation_layout(buffer_or_shape) -> T.Layout:
    """Map logical ``[..., M, K]`` activation coordinates to Crouton VTCM."""

    shape = _shape_of(buffer_or_shape)
    rows, cols = _validate_matrix_shape(shape, "activation")
    tiles_k = cols // TILE

    def forward(*indices):
        m, k = indices[-2:]
        tile = (m // TILE) * tiles_k + k // TILE
        intra = ((m % TILE) // 2) * 64 + (k % TILE) * 2 + m % 2
        return [_prefix_offset(indices, shape, rows * cols) + tile * TILE_ELEMS + intra]

    return T.Layout(shape, forward)


def make_hmx_weight_layout(buffer_or_shape) -> T.Layout:
    """Map logical ``[..., K, N]`` weight coordinates to Crouton VTCM."""

    shape = _shape_of(buffer_or_shape)
    rows, cols = _validate_matrix_shape(shape, "weight")
    tiles_k = rows // TILE

    def forward(*indices):
        k, n = indices[-2:]
        tile = (n // TILE) * tiles_k + k // TILE
        intra = ((k % TILE) // 2) * 64 + (n % TILE) * 2 + k % 2
        return [_prefix_offset(indices, shape, rows * cols) + tile * TILE_ELEMS + intra]

    return T.Layout(shape, forward)


def make_hmx_output_layout(buffer_or_shape) -> T.Layout:
    """Map logical ``[..., M, N]`` output coordinates to Crouton VTCM."""

    shape = _shape_of(buffer_or_shape)
    rows, cols = _validate_matrix_shape(shape, "output")
    tiles_n = cols // TILE

    def forward(*indices):
        m, n = indices[-2:]
        tile = (m // TILE) * tiles_n + n // TILE
        intra = ((m % TILE) // 2) * 64 + (n % TILE) * 2 + m % 2
        return [_prefix_offset(indices, shape, rows * cols) + tile * TILE_ELEMS + intra]

    return T.Layout(shape, forward)


class HMXIntrinEmitter:
    """Compose explicit HMX state and 32x32x32 instruction atoms."""

    def __init__(
        self,
        M: int = TILE,
        N: int = TILE,
        K: int = TILE,
        a_dtype: str = "float16",
        b_dtype: str = "float16",
    ):
        if M <= 0 or N <= 0 or K <= 0 or M % TILE or N % TILE or K % TILE:
            raise ValueError("HMX dimensions must be positive multiples of 32")
        if a_dtype != "float16" or b_dtype != "float16":
            raise ValueError("The validated HMX path accepts FP16 operands")
        self.M, self.N, self.K = M, N, K
        self.MT, self.NT, self.KT = M // TILE, N // TILE, K // TILE
        self.a_dtype, self.b_dtype = a_dtype, b_dtype

    @staticmethod
    def activation_layout(buffer_or_shape) -> T.Layout:
        return make_hmx_activation_layout(buffer_or_shape)

    @staticmethod
    def weight_layout(buffer_or_shape) -> T.Layout:
        return make_hmx_weight_layout(buffer_or_shape)

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
        a_m=0,
        a_k=0,
        b_k=0,
        b_n=0,
        weight_tile=None,
    ):
        """Emit one paired activation/weight HMX multiply operation."""

        if weight_tile is None:
            @T.macro
            def _mma(acc_state, activation, weight):
                T.hexagon_hmx_mma(
                    T.access_ptr(acc_state[0], "rw"),
                    T.access_ptr(
                        activation[a_m, a_k], "r", TILE_ELEMS
                    ),
                    T.access_ptr(weight[b_k, b_n], "r", TILE_ELEMS),
                )
        else:
            @T.macro
            def _mma(acc_state, activation, weight):
                T.hexagon_hmx_mma(
                    T.access_ptr(acc_state[0], "rw"),
                    T.access_ptr(
                        activation[a_m, a_k], "r", TILE_ELEMS
                    ),
                    T.access_ptr(
                        weight[weight_tile, b_k, b_n], "r", TILE_ELEMS
                    ),
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
        out_m=0,
        out_n=0,
    ):
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
                T.access_ptr(output[out_m, out_n], "w", TILE_ELEMS),
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
    "make_hmx_activation_layout",
    "make_hmx_weight_layout",
    "make_hmx_output_layout",
    "HMXIntrinEmitter",
]
