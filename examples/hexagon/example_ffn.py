"""A gated FFN block on Hexagon: Y = (silu(X @ W1) * (X @ W3)) @ W2.

Three HMX gemms and one HVX map. `h1` and `h3` are gemm outputs in the native
Crouton layout, and the gate `silu(h1) * h3` is a map over those tiles: it runs
in their physical order, a register at a time, and writes the gate into `h1`,
which the last gemm reads as its activation. Nothing converts layouts between
the gemms.

Requires a Hexagon device + SDK over adb (see README.md).

    python example_ffn.py
"""

import argparse

import torch

import tilelang
from tilelang.hexagon import language as T


def make_ffn(M, D, H, N):
    @T.prim_func
    def ffn(
        X: T.Tensor((M, D), "float16"),
        W1: T.Tensor((D, H), "float16"),
        W3: T.Tensor((D, H), "float16"),
        W2: T.Tensor((H, N), "float16"),
        Y: T.Tensor((M, N), "float16"),
    ):
        with T.Kernel(1) as _:
            x = T.alloc_shared((M, D), "float16")
            w1 = T.alloc_shared((D, H), "float16")
            w3 = T.alloc_shared((D, H), "float16")
            w2 = T.alloc_shared((H, N), "float16")
            h1 = T.alloc_shared((M, H), "float16")
            h3 = T.alloc_shared((M, H), "float16")
            y = T.alloc_shared((M, N), "float16")
            T.copy(X, x)
            T.copy(W1, w1)
            T.copy(W3, w3)
            T.copy(W2, w2)
            T.gemm(x, w1, h1, clear_accum=True)  # h1 = X @ W1 (HMX)
            T.gemm(x, w3, h3, clear_accum=True)  # h3 = X @ W3 (HMX)
            for i, j in T.Parallel(M, H):  # h1 = silu(h1) * h3 (HVX, on the tiles)
                a = T.cast(h1[i, j], "float32")
                h1[i, j] = T.cast(a / (T.float32(1) + T.exp(-a)) * T.cast(h3[i, j], "float32"), "float16")
            T.gemm(h1, w2, y, clear_accum=True)  # Y = gate @ W2 (HMX)
            T.copy(y, Y)

    return ffn


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--m", type=int, default=64, help="tokens")
    p.add_argument("--d", type=int, default=128, help="model width")
    p.add_argument("--h", type=int, default=256, help="hidden width")
    p.add_argument("--n", type=int, default=128, help="output width")
    args = p.parse_args()
    M, D, H, N = args.m, args.d, args.h, args.n

    kernel = tilelang.compile(make_ffn(M, D, H, N), out_idx=[4], target="hexagon")
    gen = torch.Generator().manual_seed(0)
    x = (torch.randn(M, D, generator=gen) * 0.5).half()
    w1 = (torch.randn(D, H, generator=gen) * 0.1).half()
    w3 = (torch.randn(D, H, generator=gen) * 0.1).half()
    w2 = (torch.randn(H, N, generator=gen) * 0.1).half()
    y = kernel(x, w1, w3, w2).float()
    h1 = (x.float() @ w1.float()).half().float()
    h3 = (x.float() @ w3.float()).half().float()
    gate = (torch.nn.functional.silu(h1) * h3).half().float()
    ref = gate @ w2.float()
    err = ((y - ref).abs().max() / ref.abs().max()).item()
    print(f"gated FFN ({M}x{D} -> {H} -> {N}) on HMX+HVX: rel err = {err:.4g}  ({'PASS' if err < 1e-2 else 'FAIL'})")


if __name__ == "__main__":
    main()
