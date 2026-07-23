"""Hexagon Q8_0 GEMV layout contract and HVX instruction atom.

This module intentionally models the *compute atom*, not a complete DDR-to-VTCM
pipeline.  A full decode GEMV also needs a target-owned DMA queue, activation
quantization, and worker-pool scheduling.  Keeping those concerns outside the
atom lets an embedding runtime (for example ggml-hexagon) retain its mature
streaming pipeline while replacing or tuning the dot implementation.

The staged layout matches ggml-hexagon's Q8_0 tiled kernel:

* 32 output rows x 32 K values per logical weight tile;
* 1024 signed quant bytes followed by 32 fp16 scales (1088 source bytes);
* each VTCM row is padded to 1152 bytes, or nine 128-byte HVX vectors;
* the activation tile uses eight replicated int8 vectors plus one replicated
  fp16-scale vector, also 1152 bytes.

``Q8GemvIntrinEmitter.dot_32x1`` is analogous to an MMA instruction atom: it
lowers to the target intrinsic in ``tl_templates/hexagon/qgemv.h``.  The caller
owns tiling, DMA, and parallelism.
"""

from dataclasses import dataclass

from tilelang import language as T


@dataclass(frozen=True)
class Q8_0TiledLayout:
    """Physical layout consumed by the Hexagon Q8_0 32x1 dot atom."""

    block_k: int = 32
    block_n: int = 32
    quant_bytes: int = 1024
    scale_bytes: int = 64
    source_tile_bytes: int = 1088
    staged_tile_bytes: int = 1152
    hvx_bytes: int = 128

    def validate_k(self, k: int) -> None:
        if k <= 0 or k % self.block_k != 0:
            raise ValueError(f"Q8_0 tiled GEMV requires K > 0 and K % 32 == 0; got K={k}")

    def k_tiles(self, k: int) -> int:
        self.validate_k(k)
        return k // self.block_k

    def staged_weight_bytes(self, k: int) -> int:
        return self.k_tiles(k) * self.staged_tile_bytes

    def staged_activation_bytes(self, k: int) -> int:
        return self.k_tiles(k) * self.staged_tile_bytes


Q8_0_TILED_LAYOUT = Q8_0TiledLayout()


class Q8GemvIntrinEmitter:
    """Emit the v79 HVX Q8_0 dot atom for one 32-row output tile.

    Parameters are byte-flat buffers in :class:`Q8_0TiledLayout`.  ``weight``
    and ``activation`` are expected to be in VTCM in a performance kernel, but
    the intrinsic uses unaligned-safe loads so the same atom is testable through
    the standalone FastRPC harness.
    """

    def __init__(self, k: int, layout: Q8_0TiledLayout = Q8_0_TILED_LAYOUT):
        layout.validate_k(k)
        self.k = k
        self.layout = layout

    def dot_32x1(self, dst, weight, activation, bias=None, valid_rows=32):
        """Accumulate one ``[32, K] x [K]`` tile into fp32 ``dst[32]``.

        ``bias`` is optional and represents the fused residual/bias used by
        ggml's ``MUL_MAT+ADD`` path.  ``valid_rows`` permits the final padded
        output tile, although model shapes normally use all 32 rows.
        """

        if isinstance(valid_rows, int) and not 0 <= valid_rows <= self.layout.block_n:
            raise ValueError(f"valid_rows must be in [0, 32]; got {valid_rows}")

        k = self.k
        if bias is None:

            @T.macro
            def _dot(dst, weight, activation):
                T.call_extern(
                    "int32",
                    "tl_hexagon_q8_0_dot_32x1_nobias",
                    k,
                    T.address_of(dst[0]),
                    T.address_of(weight[0]),
                    T.address_of(activation[0]),
                    valid_rows,
                )

            return _dot(dst, weight, activation)

        @T.macro
        def _dot_bias(dst, weight, activation, bias):
            T.call_extern(
                "int32",
                "tl_hexagon_q8_0_dot_32x1",
                k,
                T.address_of(dst[0]),
                T.address_of(weight[0]),
                T.address_of(activation[0]),
                valid_rows,
                T.address_of(bias[0]),
            )

        return _dot_bias(dst, weight, activation, bias)


def make_q8_0_dot_prim_func(
    k: int, symbol: str | None = None, *, with_bias: bool = True
):
    """Build an embeddable fixed-K PrimFunc for the Q8_0 dot atom.

    ``with_bias`` selects the C ABI.  Embedding runtimes should call the no-bias
    entry instead of passing a null tensor pointer to a bias-bearing PrimFunc.
    """

    layout = Q8_0_TILED_LAYOUT
    emitter = Q8GemvIntrinEmitter(k, layout)
    weight_bytes = layout.staged_weight_bytes(k)
    activation_bytes = layout.staged_activation_bytes(k)

    if with_bias:

        @T.prim_func
        def qgemv_q8_0_dot(
            weight: T.Tensor((weight_bytes,), "uint8"),
            activation: T.Tensor((activation_bytes,), "uint8"),
            bias: T.Tensor((layout.block_n,), "float32"),
            dst: T.Tensor((layout.block_n,), "float32"),
        ):
            with T.Kernel(1, threads=1) as _:
                emitter.dot_32x1(dst, weight, activation, bias=bias)

    else:

        @T.prim_func
        def qgemv_q8_0_dot(
            weight: T.Tensor((weight_bytes,), "uint8"),
            activation: T.Tensor((activation_bytes,), "uint8"),
            dst: T.Tensor((layout.block_n,), "float32"),
        ):
            with T.Kernel(1, threads=1) as _:
                emitter.dot_32x1(dst, weight, activation)

    if symbol is not None:
        qgemv_q8_0_dot = qgemv_q8_0_dot.with_attr("global_symbol", symbol)
    return qgemv_q8_0_dot


__all__ = [
    "Q8_0TiledLayout",
    "Q8_0_TILED_LAYOUT",
    "Q8GemvIntrinEmitter",
    "make_q8_0_dot_prim_func",
]
