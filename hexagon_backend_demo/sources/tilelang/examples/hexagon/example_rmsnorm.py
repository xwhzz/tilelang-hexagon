"""RMSNorm on Hexagon — the {gemm, copy, map, reduce} basis beyond softmax.

No HMX here: RMSNorm is pure HVX `map` + `reduce`, which is the point — the same
primitives that vectorize the flash-attention softmax also cover normalization, so
the backend isn't softmax-shaped.  Per row:

    ssq[i]   = Σ_j X[i,j]²                 (square: HVX map; rowsum: HVX reduce)
    inv[i]   = rsqrt(ssq[i]/N + eps)       (per-row scalar — M of them)
    O[i,j]   = X[i,j] · inv[i] · gamma[j]  (normalize: HVX map)

Every elementwise `T.serial` loop is emitted as 64-lane HVX by the Hexagon codegen
(widen fp16→fp32, compute, narrow); the row reduction goes through the reduce tile
op (`hexreduce`).  `inv[i]` is per-row, so it stays a small scalar `T.rsqrt` (which
lowers to `1/sqrtf`).  The codegen ALSO vectorizes a *per-element* `T.rsqrt`
(recognizing `1/sqrtf(x)` → the HVX `rsqrt` lane primitive) for kernels that need
it — see the wiring in codegen_hexagon.cc.

`hexreduce` invokes the reduce tile op directly on shared VTCM tiles (the stock
`T.reduce_sum` macro routes through register fragments Hexagon can't infer).

Requires a Hexagon device + SDK over adb (see README.md).

    python example_rmsnorm.py --m 64 --n 256
"""
import argparse
import torch
import tilelang
import tilelang.language as T
from tvm import tirx
from tilelang.language import macro
from tilelang.utils.language import to_tile_region

EPS = 1e-5


@macro
def hexreduce(src, dst, rtype, dim):
    """Hexagon row reduction on shared VTCM tiles (max/sum over the last axis)."""
    tirx.call_intrin("handle", tirx.op.Op.get("tl.tileop.reduce"),
                     to_tile_region(src, "r"), to_tile_region(dst, "w"), rtype, dim, True)


def make_rmsnorm(M, N):
    @T.prim_func
    def rmsnorm(X: T.Tensor((M, N), "float16"), G: T.Tensor((N,), "float16"),
                O: T.Tensor((M, N), "float16")):
        with T.Kernel(1, threads=1) as _:
            X_sh = T.alloc_shared((M, N), "float16")
            Xsq = T.alloc_shared((M, N), "float16")
            G_sh = T.alloc_shared((N,), "float16")
            O_sh = T.alloc_shared((M, N), "float16")
            ssq = T.alloc_shared((M,), "float32")
            inv = T.alloc_shared((M,), "float32")
            T.copy(X, X_sh)
            T.copy(G, G_sh)
            for i in T.serial(M):
                for j in T.serial(N):                                  # X² (HVX map)
                    Xsq[i, j] = T.cast(
                        T.cast(X_sh[i, j], "float32") * T.cast(X_sh[i, j], "float32"), "float16")
            hexreduce(Xsq, ssq, "sum", 1)                              # Σ_j X² (HVX reduce, fp32 accum)
            for i in T.serial(M):                                      # per-row inverse-RMS (scalar)
                inv[i] = T.rsqrt(ssq[i] / T.float32(N) + T.float32(EPS))
            for i in T.serial(M):
                for j in T.serial(N):                                  # O = X · inv · gamma (HVX map)
                    O_sh[i, j] = T.cast(
                        T.cast(X_sh[i, j], "float32") * inv[i] * T.cast(G_sh[j], "float32"), "float16")
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
    print(f"RMSNorm (M={M}, N={N}) on HVX: max abs err = {err:.4g}  "
          f"({'PASS' if err < 0.05 else 'FAIL'})")


if __name__ == "__main__":
    main()
