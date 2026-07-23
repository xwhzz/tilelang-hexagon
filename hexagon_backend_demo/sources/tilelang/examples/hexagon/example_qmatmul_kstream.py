"""Q4_0 matmul composed from explicit HMX atoms and Crouton layouts.

The weight input uses the same 576-byte 32x32 tile contract as ggml-hexagon:
512 quant bytes ``[k_pair, output_row]`` followed by 32 FP16 scales.  TileLang
owns the K-loop and dequantizes each weight tile directly into its final HMX
weight layout.  There is no ``T.gemm``, row-major FP16 weight tile, or pack_b
helper in this path.  Dequantization is exposed as a native 576-byte Q4 tile
intrinsic, analogous to the explicit HMX MMA intrinsic: TileLang owns where and
when it runs, while the intrinsic fixes the v79 HVX instruction sequence.

    python example_qmatmul_kstream.py --n 128 --k 512 --io-dtype float32
"""

import argparse

import numpy as np
import torch

import tilelang
import tilelang.language as T
from tilelang.hexagon.hmx_intrin import HMXIntrinEmitter
from tilelang.hexagon.qmatmul import Q4HMXIntrinEmitter


Q4_TILE_BYTES = 576
Q4_QUANT_BYTES = 512


def make(M, N, K, input_dtype="float16", output_dtype="float16"):
    if M <= 0 or M % 32 or N % 32 or K % 32:
        raise ValueError("The HMX atom example requires positive M/N/K multiples of 32")
    if input_dtype not in ("float16", "float32"):
        raise ValueError(f"unsupported activation dtype: {input_dtype}")
    if output_dtype not in ("float16", "float32"):
        raise ValueError(f"unsupported output dtype: {output_dtype}")

    MT, NT, KT = M // 32, N // 32, K // 32
    E = HMXIntrinEmitter(M, N, K)
    Q = Q4HMXIntrinEmitter()

    @T.prim_func
    def qmatmul_hmx_atoms(
        A: T.Tensor((M, K), input_dtype),
        W: T.Tensor((NT, KT, Q4_TILE_BYTES), "uint8"),
        C: T.Tensor((M, N), output_dtype),
    ):
        with T.Kernel(1, threads=1) as _:
            A_hmx = T.alloc_shared((M, K), "float16", align=2048)
            B_hmx = T.alloc_shared((KT, 32, 32), "float16", align=2048)
            C_hmx = T.alloc_shared((M, 32), "float16", align=2048)
            bias_vtcm = T.alloc_shared((64,), "uint32", align=256)
            acc = T.alloc_hmx_accumulator()
            cvt = T.alloc_hmx_convert_state()
            bias = T.alloc_hmx_bias_state()

            T.annotate_layout(
                {
                    A_hmx: E.activation_layout(A_hmx),
                    B_hmx: E.weight_layout(B_hmx),
                    C_hmx: E.output_layout(C_hmx),
                }
            )

            # HMX bias state: 32 unit FP16 output scales followed by zero bias.
            for i in T.serial(32):
                bias_vtcm[i] = T.Cast("uint32", 0x3C00)
                bias_vtcm[32 + i] = T.Cast("uint32", 0)

            if input_dtype == "float32":
                # One atom consumes two contiguous FP32 row vectors and writes
                # their final interleaved activation-Crouton vector.
                for mt in T.serial(MT):
                    for kt in T.serial(KT):
                        for mpair in T.serial(16):
                            m = mt * 32 + 2 * mpair
                            Q.pack_activation_pair(
                                A_hmx,
                                A,
                                (m, kt * 32),
                                (m, kt * 32),
                                (m + 1, kt * 32),
                            )
            else:
                # FP16 reference path: walk the same physical Crouton vectors
                # explicitly when no FP32 conversion atom is needed.
                for mt in T.serial(MT):
                    for kt in T.serial(KT):
                        for mpair in T.serial(16):
                            for lane_group in T.serial(2):
                                for lane in T.vectorized(32):
                                    packed_lane = lane_group * 32 + lane
                                    m = mt * 32 + 2 * mpair + packed_lane % 2
                                    k = kt * 32 + packed_lane // 2
                                    A_hmx[m, k] = A[m, k]

            E.acquire(acc)

            for nt in T.serial(NT):
                # Keep every in-flight K tile distinct until the completed
                # accumulator store. Reusing one tile here races HMX mxmem.
                for kt in T.serial(KT):
                    # One native Q4 tile atom keeps four independent HVX
                    # register chains in flight. T.Layout still supplies the
                    # final weight-Crouton address and TileLang owns K order.
                    Q.dequant_tile(B_hmx, W, (kt, 0, 0), (nt, kt, 0))

                # Reuse this N-tile's dequantized weights for every M=32 HMX
                # accumulator pass before advancing to the next N tile.
                for mt in T.serial(MT):
                    E.clear(acc)
                    E.load_bias(bias, bias_vtcm)

                    for kt in T.serial(KT):
                        E.mma_atom(
                            acc,
                            A_hmx,
                            B_hmx,
                            a_m=mt * 32,
                            a_k=kt * 32,
                            weight_tile=kt,
                        )

                    E.convert(cvt, acc, bias, bias_vtcm)
                    E.store(
                        cvt,
                        acc,
                        C_hmx,
                        bias,
                        bias_vtcm,
                        A_hmx,
                        B_hmx,
                    )

                    if output_dtype == "float32":
                        for mpair in T.serial(16):
                            m = mt * 32 + 2 * mpair
                            Q.store_output_pair_f32(
                                C,
                                C_hmx,
                                (m, nt * 32),
                                (m + 1, nt * 32),
                                (2 * mpair, 0),
                            )
                    else:
                        for mpair in T.serial(16):
                            for lane_group in T.serial(2):
                                for lane in T.vectorized(32):
                                    packed_lane = lane_group * 32 + lane
                                    m = mt * 32 + 2 * mpair + packed_lane % 2
                                    n = packed_lane // 2
                                    C[m, nt * 32 + n] = C_hmx[
                                        2 * mpair + packed_lane % 2, n
                                    ]

            E.release(acc)

    return qmatmul_hmx_atoms


def make_staged_pack(M, K):
    """Pack FP32 activation once into a caller-owned activation Crouton."""

    if M <= 0 or M % 32 or K <= 0 or K % 32:
        raise ValueError("staged pack requires positive M/K multiples of 32")
    MT, KT = M // 32, K // 32
    E = HMXIntrinEmitter(M, 32, K)
    Q = Q4HMXIntrinEmitter()

    @T.prim_func
    def qmatmul_hmx_pack_stage(
        A: T.Tensor((M, K), "float32"),
        A_hmx: T.Tensor((M, K), "float16"),
        bias_vtcm: T.Tensor((64,), "uint32"),
    ):
        with T.Kernel(1, threads=1) as _:
            T.annotate_layout({A_hmx: E.activation_layout(A_hmx)})
            for i in T.serial(32):
                bias_vtcm[i] = T.Cast("uint32", 0x3C00)
                bias_vtcm[32 + i] = T.Cast("uint32", 0)
            for mt in T.serial(MT):
                for kt in T.serial(KT):
                    for mpair in T.serial(16):
                        m = mt * 32 + 2 * mpair
                        Q.pack_activation_pair(
                            A_hmx,
                            A,
                            (m, kt * 32),
                            (m, kt * 32),
                            (m + 1, kt * 32),
                        )

    return qmatmul_hmx_pack_stage


def make_staged_dequant(N, K, padded_tile_bytes=640):
    """Dequantize a caller-selected linear tile range into chunk Croutons."""

    if N <= 0 or N % 32 or K <= 0 or K % 32:
        raise ValueError("staged dequant requires positive N/K multiples of 32")
    if padded_tile_bytes < Q4_TILE_BYTES or padded_tile_bytes % 128:
        raise ValueError("padded Q4 tile stride must be a >=576 multiple of 128")
    NT, KT = N // 32, K // 32
    E = HMXIntrinEmitter(32, 32, K)
    Q = Q4HMXIntrinEmitter()

    @T.prim_func
    def qmatmul_hmx_dequant_stage(
        W: T.Tensor((NT, KT, padded_tile_bytes), "uint8"),
        B_hmx: T.Tensor((NT, K, 32), "float16"),
        tile_begin: T.int32,
        tile_end: T.int32,
    ):
        with T.Kernel(1, threads=1) as _:
            T.annotate_layout({B_hmx: E.weight_layout(B_hmx)})
            for tile in T.serial(tile_begin, tile_end):
                nt = tile // KT
                kt = tile % KT
                Q.dequant_tile(
                    B_hmx,
                    W,
                    (nt, kt * 32, 0),
                    (nt, kt, 0),
                )

    return qmatmul_hmx_dequant_stage


def make_staged_hmx(M, N, K):
    """Consume prepacked A/B Croutons and store FP32 with runtime stride."""

    if M <= 0 or M % 32 or N <= 0 or N % 32 or K <= 0 or K % 32:
        raise ValueError("staged HMX requires positive M/N/K multiples of 32")
    MT, NT, KT = M // 32, N // 32, K // 32
    E = HMXIntrinEmitter(M, N, K)
    Q = Q4HMXIntrinEmitter()
    dst_stride = T.symbolic("dst_stride", dtype=T.int32)

    @T.prim_func
    def qmatmul_hmx_compute_stage(
        A_hmx: T.Tensor((M, K), "float16"),
        B_hmx: T.Tensor((NT, K, 32), "float16"),
        bias_vtcm: T.Tensor((64,), "uint32"),
        C_hmx: T.Tensor((M, 32), "float16"),
        C: T.Tensor((M, dst_stride), "float32"),
    ):
        with T.Kernel(1, threads=1) as _:
            acc = T.alloc_hmx_accumulator()
            cvt = T.alloc_hmx_convert_state()
            bias = T.alloc_hmx_bias_state()
            T.annotate_layout(
                {
                    A_hmx: E.activation_layout(A_hmx),
                    B_hmx: E.weight_layout(B_hmx),
                    C_hmx: E.output_layout(C_hmx),
                }
            )

            E.acquire(acc)
            for nt in T.serial(NT):
                for mt in T.serial(MT):
                    E.clear(acc)
                    E.load_bias(bias, bias_vtcm)
                    for kt in T.serial(KT):
                        E.mma_atom(
                            acc,
                            A_hmx,
                            B_hmx,
                            a_m=mt * 32,
                            a_k=kt * 32,
                            b_k=kt * 32,
                            weight_tile=nt,
                        )
                    E.convert(cvt, acc, bias, bias_vtcm)
                    E.store(
                        cvt,
                        acc,
                        C_hmx,
                        bias,
                        bias_vtcm,
                        A_hmx,
                        B_hmx,
                    )
                    for mpair in T.serial(16):
                        m = mt * 32 + 2 * mpair
                        Q.store_output_pair_f32(
                            C,
                            C_hmx,
                            (m, nt * 32),
                            (m + 1, nt * 32),
                            (2 * mpair, 0),
                        )
            E.release(acc)

    return qmatmul_hmx_compute_stage


def make_dequant_tile(output_dtype="float32"):
    """Build an isolated raw-Q4-to-weight-Crouton atom test kernel."""

    E = HMXIntrinEmitter()

    @T.prim_func
    def dequant_q4_0_tile(
        W: T.Tensor((1, 1, Q4_TILE_BYTES), "uint8"),
        O: T.Tensor((32, 32), output_dtype),
    ):
        with T.Kernel(1, threads=1) as _:
            B_hmx = T.alloc_shared((32, 32), "float16", align=2048)
            q4_scales = T.alloc_local((32,), "float16")
            q4_low = T.alloc_local((32,), "float16")
            q4_high = T.alloc_local((32,), "float16")
            q4_values = T.alloc_local((64,), "float16")
            T.annotate_layout({B_hmx: E.weight_layout(B_hmx)})
            for n in T.vectorized(32):
                scale_bits = T.Cast(
                    "uint16", W[0, 0, Q4_QUANT_BYTES + 2 * n]
                ) | (
                    T.Cast(
                        "uint16", W[0, 0, Q4_QUANT_BYTES + 2 * n + 1]
                    )
                    << 8
                )
                q4_scales[n] = T.reinterpret(scale_bits, T.float16)
            for kpair in T.serial(16):
                for n in T.vectorized(32):
                    packed = T.Cast("uint16", W[0, 0, kpair * 32 + n])
                    low = T.Cast("int16", packed & 15) - T.Cast("int16", 8)
                    q4_low[n] = T.Cast("float16", low) * q4_scales[n]
                for n in T.vectorized(32):
                    packed = T.Cast("uint16", W[0, 0, kpair * 32 + n])
                    high = T.Cast("int16", (packed >> 4) & 15) - T.Cast(
                        "int16", 8
                    )
                    q4_high[n] = T.Cast("float16", high) * q4_scales[n]
                for n in T.vectorized(32):
                    q4_values[2 * n] = q4_low[n]
                    q4_values[2 * n + 1] = q4_high[n]
                for lane_group in T.serial(2):
                    for lane in T.vectorized(32):
                        packed_lane = lane_group * 32 + lane
                        n = packed_lane // 2
                        nibble = packed_lane % 2
                        B_hmx[2 * kpair + nibble, n] = q4_values[packed_lane]
            for k, n in T.Parallel(32, 32):
                O[k, n] = T.Cast(output_dtype, B_hmx[k, n])

    return dequant_q4_0_tile


def pack_q4_tiles(nibbles: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Pack logical ``[N,K]`` Q4 values into ggml-compatible 576-byte tiles."""

    N, K = nibbles.shape
    NT, KT = N // 32, K // 32
    packed = np.zeros((NT, KT, Q4_TILE_BYTES), dtype=np.uint8)
    for nt in range(NT):
        for kt in range(KT):
            q = nibbles[nt * 32 : (nt + 1) * 32, kt * 32 : (kt + 1) * 32]
            for kp in range(16):
                packed[nt, kt, kp * 32 : (kp + 1) * 32] = q[:, 2 * kp] | (
                    q[:, 2 * kp + 1] << 4
                )
            packed[nt, kt, Q4_QUANT_BYTES:Q4_TILE_BYTES] = np.ascontiguousarray(
                scales[nt * 32 : (nt + 1) * 32, kt]
            ).view(np.uint8)
    return packed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=32)
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--k", type=int, default=512)
    parser.add_argument(
        "--io-dtype", choices=("float16", "float32"), default="float32"
    )
    args = parser.parse_args()
    M, N, K = args.m, args.n, args.k

    rng = np.random.default_rng(0)
    nibbles = rng.integers(0, 16, size=(N, K), dtype=np.uint8)
    scales = (rng.standard_normal((N, K // 32)) * 0.05).astype(np.float16)
    weights = (
        (nibbles.astype(np.float32) - 8.0)
        * np.repeat(scales.astype(np.float32), 32, axis=1)
    ).astype(np.float16)
    packed = pack_q4_tiles(nibbles, scales)
    numpy_io_dtype = np.float16 if args.io_dtype == "float16" else np.float32
    activation = (rng.standard_normal((M, K)) * 0.1).astype(numpy_io_dtype)
    reference = activation.astype(np.float32) @ weights.astype(np.float32).T

    kernel = tilelang.compile(
        make(M, N, K, input_dtype=args.io_dtype, output_dtype=args.io_dtype),
        out_idx=[2],
        target="hexagon",
    )
    print(kernel.get_kernel_source())  # for debugging
    output = kernel(torch.from_numpy(activation), torch.from_numpy(packed))
    output = output.cpu().numpy().astype(np.float32)
    rel = np.abs(output - reference).max() / (np.abs(reference).max() + 1e-6)
    print(
        f"HMX atom q4_0 matmul {M}x{N}x{K}: rel err = {rel:.4g} "
        f"({'PASS' if rel < 5e-2 else 'FAIL'})"
    )


if __name__ == "__main__":
    main()
