"""RMSNorm on Hexagon — the {gemm, copy, map, reduce} basis beyond softmax.

No HMX here: RMSNorm is pure HVX `map` + `reduce`, which is the point — the same
primitives that vectorize the flash-attention softmax also cover normalization, so
the backend isn't softmax-shaped.  Per row:

    ssq[i]   = Σ_j X[i,j]²                 (square: HVX map; rowsum: HVX reduce)
    inv[i]   = rsqrt(ssq[i]/N + eps)       (per-row scalar — M of them)
    O[i,j]   = X[i,j] · inv[i] · gamma[j]  (normalize: HVX map)

Every elementwise `T.Parallel` loop vectorizes onto whole HVX registers through the
standard lowering (fp16 loads, fp32 arithmetic, fp16 stores); `T.rsqrt` becomes the HVX
`rsqrt`, so the per-row `inv[i]` map vectorizes too. `T.reduce_sum` reduces the rows of
the shared tile in place, with vector accumulators and an HVX lane tree.

Requires a Hexagon device + SDK over adb (see README.md).

    python example_rmsnorm.py --m 64 --n 256
"""

import argparse
import torch
import tilelang
from tilelang.hexagon import language as T

EPS = 1e-5


def make_rmsnorm(M, N):
    @T.prim_func
    def rmsnorm(X: T.Tensor((M, N), "float16"), G: T.Tensor((N,), "float16"), O: T.Tensor((M, N), "float16")):
        with T.Kernel(1) as _:
            X_sh = T.alloc_shared((M, N), "float16")
            Xsq = T.alloc_shared((M, N), "float16")
            G_sh = T.alloc_shared((N,), "float16")
            O_sh = T.alloc_shared((M, N), "float16")
            ssq = T.alloc_shared((M,), "float32")
            inv = T.alloc_shared((M,), "float32")
            T.copy(X, X_sh)
            T.copy(G, G_sh)
            for i, j in T.Parallel(M, N):  # X²
                Xsq[i, j] = T.cast(T.cast(X_sh[i, j], "float32") * T.cast(X_sh[i, j], "float32"), "float16")
            T.reduce_sum(Xsq, ssq, dim=1)  # Σ_j X² (HVX reduce, fp32 accum)
            for i in T.Parallel(M):  # per-row inverse RMS
                inv[i] = T.rsqrt(ssq[i] / T.float32(N) + T.float32(EPS))
            for i, j in T.Parallel(M, N):  # O = X · inv · gamma
                O_sh[i, j] = T.cast(T.cast(X_sh[i, j], "float32") * inv[i] * T.cast(G_sh[j], "float32"), "float16")
            T.copy(O_sh, O)

    return rmsnorm


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--m", type=int, default=64, help="rows (tokens)")
    p.add_argument("--n", type=int, default=256, help="norm dim (multiple of 64)")
    args = p.parse_args()
    M, N = args.m, args.n

    kernel = tilelang.compile(make_rmsnorm(M, N), out_idx=[2], target="hexagon")
    x = (torch.randn(M, N)).half()
    g = (torch.randn(N) * 0.3 + 1.0).half()
    o = kernel(x, g).float()
    ms = (x.float() ** 2).mean(dim=-1, keepdim=True)
    ref = x.float() / torch.sqrt(ms + EPS) * g.float()
    err = (o - ref).abs().max().item()
    print(f"RMSNorm (M={M}, N={N}) on HVX: max abs err = {err:.4g}  ({'PASS' if err < 0.05 else 'FAIL'})")


if __name__ == "__main__":
    main()
