"""The tiled GEMM with an fp32 accumulator and a K loop, on the Hexagon HMX matrix engine.

Each tile program computes a BM x BN block of C. Its K loop is a software pipeline
(`T.Pipelined`): the copies of a BM x BK block of A and a BK x BN block of B become
DMA transfers into VTCM, issued ahead of the blocks' use, and in the block's stage
an HVX pack puts them into Crouton tiles. `T.gemm` multiplies them on HMX and adds
the product into the fp32 fragment `acc` on HVX; the first call overwrites `acc`
instead (`clear_accum=(k == 0)`). Each call reduces its K in the HMX accumulator,
so a larger BK means fewer HVX passes over `acc`.

Requires a Hexagon device + SDK over adb (see README.md).

    python example_gemm.py --m 256 --n 256 --k 1024 --block-k 256
"""

import argparse

import torch

import tilelang
from tilelang.hexagon import language as T


def make_gemm(M, N, K, BM, BN, BK):
    @T.prim_func
    def gemm(A: T.Tensor((M, K), "float16"), B: T.Tensor((K, N), "float16"), C: T.Tensor((M, N), "float16")):
        with T.Kernel(T.ceildiv(N, BN), T.ceildiv(M, BM)) as (bx, by):
            a = T.alloc_shared((BM, BK), "float16")
            b = T.alloc_shared((BK, BN), "float16")
            acc = T.alloc_fragment((BM, BN), "float32")
            for k in T.Pipelined(T.ceildiv(K, BK), num_stages=2):
                T.copy(A[by * BM, k * BK], a)
                T.copy(B[k * BK, bx * BN], b)
                T.gemm(a, b, acc, clear_accum=(k == 0))
            T.copy(acc, C[by * BM, bx * BN])

    return gemm


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--m", type=int, default=256)
    p.add_argument("--n", type=int, default=256)
    p.add_argument("--k", type=int, default=1024)
    p.add_argument("--block", type=int, default=128, help="BM = BN (a multiple of 32)")
    p.add_argument("--block-k", type=int, default=256, help="BK (a multiple of 32)")
    args = p.parse_args()

    kernel = tilelang.compile(make_gemm(args.m, args.n, args.k, args.block, args.block, args.block_k), out_idx=[2], target="hexagon")
    a = (torch.randn(args.m, args.k) * 0.1).half()
    b = (torch.randn(args.k, args.n) * 0.1).half()
    c = kernel(a, b).float()
    ref = a.float() @ b.float()
    err = ((c - ref).abs().max() / ref.abs().max()).item()
    print(
        f"gemm {args.m}x{args.n}x{args.k} (blocks {args.block}x{args.block}x{args.block_k}, fp32 accumulator) on HMX: "
        f"rel err = {err:.4g}  ({'PASS' if err < 1e-2 else 'FAIL'})"
    )


if __name__ == "__main__":
    main()
