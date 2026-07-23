"""One-shape TileLang Q4/HMX kernel used by the llama.cpp walkthrough.

The example deliberately fixes M=32, N=256, K=2048 so the integration path is
visible without a multi-shape dispatch table. It produces three embeddable
stages: activation pack, parallel-range Q4 dequant, and explicit HMX compute.
"""

import tilelang.language as T
from tilelang.hexagon.hmx_intrin import HMXIntrinEmitter
from tilelang.hexagon.qmatmul import Q4HMXIntrinEmitter


M = 32
N = 256
K = 2048
NT = N // 32
KT = K // 32
PADDED_Q4_TILE_BYTES = 640


def make_pack_stage():
    """Pack llama.cpp FP32 activation into the HMX A Crouton once."""

    hmx = HMXIntrinEmitter(M, 32, K)
    q4 = Q4HMXIntrinEmitter()

    @T.prim_func
    def pack(
        activation: T.Tensor((M, K), "float32"),
        activation_hmx: T.Tensor((M, K), "float16"),
        bias_vtcm: T.Tensor((64,), "uint32"),
    ):
        with T.Kernel(1, threads=1):
            T.annotate_layout(
                {activation_hmx: hmx.activation_layout(activation_hmx)}
            )
            for lane in T.serial(32):
                bias_vtcm[lane] = T.Cast("uint32", 0x3C00)
                bias_vtcm[32 + lane] = T.Cast("uint32", 0)
            for kt in T.serial(KT):
                for row_pair in T.serial(16):
                    row = 2 * row_pair
                    q4.pack_activation_pair(
                        activation_hmx,
                        activation,
                        (row, kt * 32),
                        (row, kt * 32),
                        (row + 1, kt * 32),
                    )

    return pack


def make_dequant_stage():
    """Decode a caller-selected tile range into the final HMX B layout."""

    hmx = HMXIntrinEmitter(32, 32, K)
    q4 = Q4HMXIntrinEmitter()

    @T.prim_func
    def dequant(
        staged_weight: T.Tensor((NT, KT, PADDED_Q4_TILE_BYTES), "uint8"),
        weight_hmx: T.Tensor((NT, K, 32), "float16"),
        tile_begin: T.int32,
        tile_end: T.int32,
    ):
        with T.Kernel(1, threads=1):
            T.annotate_layout({weight_hmx: hmx.weight_layout(weight_hmx)})
            for tile in T.serial(tile_begin, tile_end):
                nt = tile // KT
                kt = tile % KT
                q4.dequant_tile(
                    weight_hmx,
                    staged_weight,
                    (nt, kt * 32, 0),
                    (nt, kt, 0),
                )

    return dequant


def make_compute_stage():
    """Run the explicit HMX protocol and write FP32 with a runtime stride."""

    hmx = HMXIntrinEmitter(M, N, K)
    q4 = Q4HMXIntrinEmitter()
    dst_stride = T.symbolic("dst_stride", dtype=T.int32)

    @T.prim_func
    def compute(
        activation_hmx: T.Tensor((M, K), "float16"),
        weight_hmx: T.Tensor((NT, K, 32), "float16"),
        bias_vtcm: T.Tensor((64,), "uint32"),
        output_hmx: T.Tensor((M, 32), "float16"),
        output: T.Tensor((M, dst_stride), "float32"),
    ):
        with T.Kernel(1, threads=1):
            acc = T.alloc_hmx_accumulator()
            cvt = T.alloc_hmx_convert_state()
            bias = T.alloc_hmx_bias_state()
            T.annotate_layout(
                {
                    activation_hmx: hmx.activation_layout(activation_hmx),
                    weight_hmx: hmx.weight_layout(weight_hmx),
                    output_hmx: hmx.output_layout(output_hmx),
                }
            )

            hmx.acquire(acc)
            for nt in T.serial(NT):
                hmx.clear(acc)
                hmx.load_bias(bias, bias_vtcm)
                for kt in T.serial(KT):
                    hmx.mma_atom(
                        acc,
                        activation_hmx,
                        weight_hmx,
                        a_m=0,
                        a_k=kt * 32,
                        b_k=kt * 32,
                        weight_tile=nt,
                    )
                hmx.convert(cvt, acc, bias, bias_vtcm)
                hmx.store(
                    cvt,
                    acc,
                    output_hmx,
                    bias,
                    bias_vtcm,
                    activation_hmx,
                    weight_hmx,
                )
                for row_pair in T.serial(16):
                    row = 2 * row_pair
                    q4.store_output_pair_f32(
                        output,
                        output_hmx,
                        (row, nt * 32),
                        (row + 1, nt * 32),
                        (2 * row_pair, 0),
                    )
            hmx.release(acc)

    return compute
