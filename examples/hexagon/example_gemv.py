"""A streaming GEMV on Hexagon: y = x @ W, with W streamed from DDR by DMA.

Decode-time projections read every weight once, so they run at the speed of
memory. The K loop is a software pipeline (`T.Pipelined`): its copy of each
block of W rows becomes a DMA transfer issued ahead of the block's HVX
multiply-add, which multiplies the block by x while DMA brings the next ones
into VTCM.

Requires a Hexagon device + SDK over adb (see README.md).

    python example_gemv.py --k 4096 --n 1024
"""

import argparse

import torch

import tilelang
from tilelang.hexagon import language as T


def make_gemv(K, N, BK, stages):
    @T.prim_func
    def gemv(x: T.Tensor((K,), "float16"), W: T.Tensor((K, N), "float16"), y: T.Tensor((N,), "float32")):
        with T.Kernel(1):
            xs = T.alloc_shared((K,), "float16")
            w = T.alloc_shared((BK, N), "float16")
            acc = T.alloc_fragment((N,), "float32")
            T.copy(x, xs)
            T.clear(acc)
            for k in T.Pipelined(K // BK, num_stages=stages):
                T.copy(W[k * BK, 0], w)
                for i in T.serial(BK):
                    for n in T.Parallel(N):
                        acc[n] += T.cast(xs[k * BK + i], "float32") * T.cast(w[i, n], "float32")
            T.copy(acc, y)

    return gemv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, default=4096)
    p.add_argument("--n", type=int, default=1024)
    p.add_argument("--block-k", type=int, default=32, help="rows of W per pipeline stage")
    p.add_argument("--stages", type=int, default=2)
    args = p.parse_args()

    kernel = tilelang.compile(make_gemv(args.k, args.n, args.block_k, args.stages), out_idx=[2], target="hexagon")
    x = torch.randn(args.k).half()
    w = (torch.randn(args.k, args.n) * 0.05).half()
    y = kernel(x, w)
    ref = x.float() @ w.float()
    err = ((y - ref).abs().max() / ref.abs().max()).item()
    print(
        f"gemv {args.k} -> {args.n} (W streamed in blocks of {args.block_k} rows): rel err = {err:.4g}  ({'PASS' if err < 1e-3 else 'FAIL'})"
    )


if __name__ == "__main__":
    main()
