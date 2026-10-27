"""Expert HMX instruction atoms of the Hexagon dialect (``T.hmx``).

HMX keeps its accumulator, convert result and bias configuration in implicit
hardware state. ``alloc_accumulator``, ``alloc_convert_state`` and
``alloc_bias_state`` return one-byte dependency tokens for that state, so TIR
analyses see the order between atoms. The codegen passes each token through
and the runtime ignores its address. Memory operands address native Crouton
tiles in VTCM (:mod:`tilelang.hexagon.intrinsics.crouton`).

Most kernels use ``T.gemm``. :class:`tilelang.hexagon.intrinsics.hmx.HMXIntrinEmitter`
composes these atoms for hand-scheduled kernels.
"""

from __future__ import annotations

from typing import Literal

from tvm import tirx
from tvm.script import tirx as T
from tvm.tirx.buffer import Buffer

from tilelang.language import dtypes as _dtypes
from tilelang.language.allocate import _with_span

__all__ = [
    "StateKind",
    "alloc_state",
    "alloc_accumulator",
    "alloc_convert_state",
    "alloc_bias_state",
    "acquire",
    "release",
    "load_bias",
    "clear",
    "mma",
    "convert",
    "store",
]

StateKind = Literal["accumulator", "convert", "bias"]


def alloc_state(kind: StateKind) -> Buffer:
    """Allocate an IR token for one kind of implicit HMX state.

    HMX accumulator, convert and bias state are not addressable storage. The
    one-byte buffer only carries dependencies through TIR.
    """

    if kind not in ("accumulator", "convert", "bias"):
        raise ValueError(f"Unsupported HMX state kind: {kind}")
    return _with_span(T.sblock_alloc_buffer([1], _dtypes.uint8, scope=f"local.hmx_state.{kind}"))


def alloc_accumulator() -> Buffer:
    return alloc_state("accumulator")


def alloc_convert_state() -> Buffer:
    return alloc_state("convert")


def alloc_bias_state() -> Buffer:
    return alloc_state("bias")


def acquire(acc_state):
    """Acquire exclusive access to the implicit HMX accumulator."""

    return tirx.call_intrin("void", tirx.op.Op.get("tl.hexagon_hmx_acquire"), acc_state)


def release(acc_state):
    """Release exclusive access to the implicit HMX accumulator."""

    return tirx.call_intrin("void", tirx.op.Op.get("tl.hexagon_hmx_release"), acc_state)


def load_bias(bias_state, bias_ptr):
    """Load HMX bias/scale state from a VTCM memory operand."""

    return tirx.call_intrin("void", tirx.op.Op.get("tl.hexagon_hmx_load_bias"), bias_state, bias_ptr)


def clear(acc_state):
    """Clear the implicit HMX accumulator."""

    return tirx.call_intrin("void", tirx.op.Op.get("tl.hexagon_hmx_clear"), acc_state)


def mma(acc_state, activation_ptr, weight_ptr, num_tiles=1):
    """Multiply 1..32 contiguous native K tiles in one HMX deep packet.

    Dynamic counts must satisfy this range at runtime. Both pointers must
    expose ``num_tiles * 1024`` FP16 elements in native Crouton order.
    """
    if isinstance(num_tiles, (int, tirx.IntImm)) and not 1 <= int(num_tiles) <= 32:
        raise ValueError("HMX deep num_tiles must be in [1, 32]")

    return tirx.call_intrin(
        "void",
        tirx.op.Op.get("tl.hexagon_hmx_mma"),
        acc_state,
        activation_ptr,
        weight_ptr,
        num_tiles,
    )


def convert(cvt_state, acc_state, bias_state, bias_ptr, mode=2):
    """Convert the HMX accumulator while retaining its asynchronous bias source."""

    return tirx.call_intrin(
        "void",
        tirx.op.Op.get("tl.hexagon_hmx_convert"),
        cvt_state,
        acc_state,
        bias_state,
        bias_ptr,
        mode,
    )


def store(cvt_state, output_ptr, acc_state, bias_state, bias_ptr):
    """Store HMX convert state to its VTCM output tile."""

    return tirx.call_intrin(
        "void",
        tirx.op.Op.get("tl.hexagon_hmx_store"),
        cvt_state,
        output_ptr,
        acc_state,
        bias_state,
        bias_ptr,
    )
