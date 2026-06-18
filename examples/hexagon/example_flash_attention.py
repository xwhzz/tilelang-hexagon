"""Flash attention on Hexagon — two HMX gemms with an online softmax between them.

This is the showcase for how the backend *composes*: `S = Q@Kᵀ` and `O = P@V` both lower
to the HMX matrix engine (`T.gemm`), and the online-softmax rescale runs on HVX, all
on-chip in VTCM — `S` is gemm-1's output *and* gemm-2's input with no relayout, which is
the Hexagon advantage. No new runtime code is needed; it's built from the same
`T.gemm` + `T.copy` primitives as the matmul example.

Note the softmax is written as explicit `T.serial` loops rather than `T.reduce`: the
Hexagon backend has no single-thread fragment-layout inference yet, so reductions are
spelled out (they still vectorize since `S` is row-major). The HMX accumulator can't be
preloaded, so the `clear_accum=False` accumulation of `O` is decomposed into an
overwrite-gemm-to-`temp` plus an HVX rescale-and-add into `acc_o`.

Requires a Hexagon device + SDK over adb (see README.md).

    python example_flash_attention.py --seq 256
"""
import argparse
import torch
import tilelang
import tilelang.language as T

NEG = -3.0e38


def make_flash(M, SEQ, BN, D):
    @T.prim_func
    def flash(Q: T.Tensor((M, D), "float16"), K: T.Tensor((SEQ, D), "float16"),
              V: T.Tensor((SEQ, D), "float16"), O: T.Tensor((M, D), "float16")):
        with T.Kernel(1, threads=1) as _:
            Q_sh = T.alloc_shared((M, D), "float16")
            K_sh = T.alloc_shared((BN, D), "float16")
            V_sh = T.alloc_shared((BN, D), "float16")
            S = T.alloc_shared((M, BN), "float16")
            temp = T.alloc_shared((M, D), "float16")
            acc_o = T.alloc_shared((M, D), "float16")
            m = T.alloc_local((M,), "float32")      # running max
            l = T.alloc_local((M,), "float32")      # running denominator
            rmax = T.alloc_local((M,), "float32")
            rsum = T.alloc_local((M,), "float32")
            scale = T.alloc_local((M,), "float32")
            T.copy(Q, Q_sh)
            for i, j in T.grid(M, D):
                acc_o[i, j] = T.float16(0)
            for i in T.serial(M):
                m[i] = T.float32(NEG)
                l[i] = T.float32(0)
            for kv in T.serial(SEQ // BN):
                T.copy(K[kv * BN, 0], K_sh)
                T.copy(V[kv * BN, 0], V_sh)
                T.gemm(Q_sh, K_sh, S, transpose_B=True, clear_accum=True)  # S = Q @ Kᵀ (HMX)
                for i in T.serial(M):                                       # running max + rescale
                    rmax[i] = m[i]
                    for j in T.serial(BN):
                        rmax[i] = T.max(rmax[i], T.cast(S[i, j], "float32"))
                    scale[i] = T.exp(m[i] - rmax[i])
                for i in T.serial(M):                                       # P = exp(S - max) (fp32 exp)
                    rsum[i] = T.float32(0)
                    for j in T.serial(BN):
                        p = T.exp(T.cast(S[i, j], "float32") - rmax[i])
                        S[i, j] = T.cast(p, "float16")
                        rsum[i] = rsum[i] + p
                for i in T.serial(M):
                    l[i] = l[i] * scale[i] + rsum[i]
                    m[i] = rmax[i]
                for i, j in T.grid(M, D):                                   # rescale acc_o (HVX)
                    acc_o[i, j] = T.cast(T.cast(acc_o[i, j], "float32") * scale[i], "float16")
                T.gemm(S, V_sh, temp, clear_accum=True)                     # temp = P @ V (HMX)
                for i, j in T.grid(M, D):                                   # acc_o += temp (HVX)
                    acc_o[i, j] = T.cast(
                        T.cast(acc_o[i, j], "float32") + T.cast(temp[i, j], "float32"), "float16")
            for i, j in T.grid(M, D):                                       # O = acc_o / l
                acc_o[i, j] = T.cast(T.cast(acc_o[i, j], "float32") / l[i], "float16")
            T.copy(acc_o, O)
    return flash


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--m", type=int, default=64, help="query length")
    p.add_argument("--seq", type=int, default=128, help="kv length")
    p.add_argument("--bn", type=int, default=64, help="kv block (multiple of 32)")
    p.add_argument("--d", type=int, default=64, help="head dim")
    args = p.parse_args()
    M, SEQ, BN, D = args.m, args.seq, args.bn, args.d

    kernel = tilelang.compile(make_flash(M, SEQ, BN, D), out_idx=[3], target="hexagon")
    q = (torch.randn(M, D) * 0.3).half()
    k = (torch.randn(SEQ, D) * 0.3).half()
    v = (torch.randn(SEQ, D) * 0.3).half()
    o = kernel(q, k, v).float()
    ref = torch.softmax(q.float() @ k.float().T, dim=-1) @ v.float()
    err = (o - ref).abs().max().item()
    print(f"flash attention (M={M}, SEQ={SEQ}, BN={BN}, D={D}) on HMX: max abs err = {err:.4g}  "
          f"({'PASS' if err < 0.05 else 'FAIL'})")


if __name__ == "__main__":
    main()
