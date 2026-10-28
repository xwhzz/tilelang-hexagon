"""A streaming GEMV on Hexagon: y = x @ W.T for an nn.Linear weight W of shape (N, K).

Decode-time projections read every weight once, so they run at the speed of
memory. The loop over blocks of BN output features is a software pipeline
(`T.Pipelined`): the copy of each block's BN rows of W, contiguous in DDR,
becomes a DMA transfer issued ahead of the block's `T.gemm`, which runs on HVX
(a single row is no HMX tile) as BN dot products with x while DMA brings the
next blocks into VTCM.

Requires a Hexagon device + SDK over adb (see README.md).

    python example_gemv.py --k 4096 --n 1024
"""

import argparse

import torch

import tilelang
from tilelang.hexagon import language as T


def make_gemv(K, N, BN, stages):
    @T.prim_func
    def gemv(x: T.Tensor((1, K), "float16"), W: T.Tensor((N, K), "float16"), y: T.Tensor((1, N), "float32")):
        with T.Kernel(1):
            xs = T.alloc_shared((1, K), "float16")
            w = T.alloc_shared((BN, K), "float16")
            acc = T.alloc_fragment((1, N), "float32")
            T.copy(x, xs)
            for n in T.Pipelined(N // BN, num_stages=stages):
                T.copy(W[n * BN, 0], w)
                T.gemm(xs, w, acc[0:1, n * BN : (n + 1) * BN], transpose_B=True, clear_accum=True)
            T.copy(acc, y)

    return gemv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, default=4096)
    p.add_argument("--n", type=int, default=1024)
    p.add_argument("--block-n", type=int, default=64, help="rows of W per pipeline stage")
    p.add_argument("--stages", type=int, default=3)
    args = p.parse_args()

    kernel = tilelang.compile(make_gemv(args.k, args.n, args.block_n, args.stages), out_idx=[2], target="hexagon")
    x = torch.randn(1, args.k).half()
    w = (torch.randn(args.n, args.k) * 0.05).half()
    y = kernel(x, w)
    ref = x.float() @ w.float().t()
    err = ((y - ref).abs().max() / ref.abs().max()).item()
    print(
        f"gemv {args.k} -> {args.n} (W streamed in blocks of {args.block_n} rows): rel err = {err:.4g}  ({'PASS' if err < 1e-3 else 'FAIL'})"
    )


if __name__ == "__main__":
    main()
