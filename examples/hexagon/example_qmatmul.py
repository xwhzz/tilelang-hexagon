"""Fully-DSL q4_0 matmul on Hexagon — dequant + HMX gemm in ONE tilelang kernel.

You write the whole quantized matmul in the tilelang DSL: the q4_0 nibble decode is
plain DSL arithmetic that the Hexagon codegen lowers to full-width HVX (no hand-written
dequant intrinsics), and `T.gemm` lowers onto the HMX matrix engine.  The dequantized
weight stays in VTCM (`alloc_shared`) — it never round-trips through DDR.

Two things make the DSL dequant vectorize to full HVX (see docs/hexagon_dsl_kernels.md):
  * `T.vectorized(128)`: HVX has no sub-register ops, so a uint8 vector must fill a whole
    128-byte register (128 lanes).  Below that the codegen safely scalarizes.
  * widen to int16 BEFORE the nibble mask/shift (uint8 bitwise below full-register faults),
    and keep the constants int16 so the width isn't capped by an int32 promotion.
The weight is pre-packed column-major (qcm[K/2][N]) so the dequant store is contiguous.

    python example_qmatmul.py --n 128 --k 2048

Requires a Hexagon device over adb + the Hexagon SDK. See README.md.
"""
import argparse
import numpy as np
import torch
import tilelang
import tilelang.language as T


def make_qmatmul(M, N, K, W=128):
    KH = K // 2

    @T.prim_func
    def qmatmul(A: T.Tensor((M, K), "float16"),      # activation
                qcm: T.Tensor((KH, N), "uint8"),      # q4_0 weight, column-major packed
                scb: T.Tensor((K, N), "float16"),     # per-(k,n) scale
                C: T.Tensor((M, N), "float16")):
        with T.Kernel(1, threads=1) as _:
            A_sh = T.alloc_shared((M, K), "float16")
            B_sh = T.alloc_shared((K, N), "float16")  # dequantized weight, stays in VTCM
            C_sh = T.alloc_shared((M, N), "float16")
            T.copy(A, A_sh)
            for j in T.serial(KH):                     # DSL q4_0 dequant -> full-width HVX
                for no in T.serial(N // W):
                    for ni in T.vectorized(W):
                        n = no * W + ni
                        q = T.Cast("int16", qcm[j, n])                       # widen first
                        lo = (q & T.Cast("int16", 0xF)) - T.Cast("int16", 8)  # low nibble
                        hi = (q >> T.Cast("int16", 4)) - T.Cast("int16", 8)   # high nibble
                        B_sh[2 * j, n] = T.Cast("float16", lo) * scb[2 * j, n]
                        B_sh[2 * j + 1, n] = T.Cast("float16", hi) * scb[2 * j + 1, n]
            T.gemm(A_sh, B_sh, C_sh, clear_accum=True)  # -> HMX matrix engine
            T.copy(C_sh, C)

    return qmatmul


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--m", type=int, default=32)
    p.add_argument("--n", type=int, default=128, help="output features (>=128, multiple of 128)")
    p.add_argument("--k", type=int, default=2048)
    args = p.parse_args()
    M, N, K = args.m, args.n, args.k
    KH = K // 2

    rng = np.random.default_rng(0)
    nib = rng.integers(0, 16, size=(N, K), dtype=np.uint8)                 # weight nibbles [N][K]
    scales = (rng.standard_normal((N, K // 32)) * 0.05).astype(np.float16)  # per 32-K block
    Wq = ((nib.astype(np.float32) - 8.0) * np.repeat(scales.astype(np.float32), 32, axis=1)).astype(np.float16)
    # column-major pack: qcm[j][n] low nibble = W[n][2j], high = W[n][2j+1]
    qcm = (nib[:, 0::2].T | (nib[:, 1::2].T << 4)).astype(np.uint8)         # [K/2][N]
    scb = np.repeat(scales, 32, axis=1).T.astype(np.float16)               # [K][N]
    A = (rng.standard_normal((M, K)) * 0.1).astype(np.float16)
    ref = A.astype(np.float32) @ Wq.astype(np.float32).T                   # [M][N]

    kernel = tilelang.compile(make_qmatmul(M, N, K), out_idx=[3], target="hexagon")
    C = kernel(torch.from_numpy(A), torch.from_numpy(qcm), torch.from_numpy(scb)).cpu().numpy().astype(np.float32)
    rel = np.abs(C - ref).max() / (np.abs(ref).max() + 1e-6)
    print(f"fully-DSL q4_0 matmul {M}x{N}x{K}: rel err = {rel:.4g}  ({'PASS' if rel < 5e-2 else 'FAIL'})")


if __name__ == "__main__":
    main()
