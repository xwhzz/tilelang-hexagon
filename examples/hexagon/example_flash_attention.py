"""Flash attention on Hexagon — two HMX gemms with an HVX online softmax between them.

This is the showcase for how the backend *composes*: `S = Q@Kᵀ` and `O = P@V` both
lower to the HMX matrix engine (`T.gemm`), and the online softmax between them runs
entirely on **HVX** — no scalar fp16 libcalls.  The softmax is built from the two
non-gemm primitives of the backend's {gemm, copy, map, reduce} basis:

  * **reduce** — `T.reduce_max(S, rmax, dim=1)` / `T.reduce_sum` reduce the rows of
    `S` register by register: HVX transpose trees reduce each tile's row pairs into
    whole registers of results.
  * **map** — the elementwise `T.Parallel` loops (`exp(S − m)`, the `acc_o` rescale,
    the `+ temp` accumulate, the final `/l`) vectorize onto whole HVX registers;
    `T.exp` and the division become the HVX math functions.

Every buffer an HMX gemm touches keeps the gemm's native Crouton layout, and the
softmax works on those tiles directly: a map over a Crouton buffer runs in its
physical order, a register (a row pair of a tile) at a time, and reads the per-row
`m`, `l` and `scale` as two-lane windows of registers loaded once per tile. `S` is
the first gemm's output and the second gemm's input; `temp` and `acc_o` share the
output layout. So nothing converts layouts between the gemms. K and V stage through
row-major VTCM before their Crouton packs.

The running max/denominator (`m`, `l`, `scale`) are per-row (M-length) maps. The
HMX accumulator can't be preloaded, so the `clear_accum=False` accumulation of `O`
is decomposed into an overwrite gemm to `temp` plus the HVX rescale-and-add into
`acc_o`.

Requires a Hexagon device + SDK over adb (see README.md).

    python example_flash_attention.py --seq 256
"""

import argparse
import torch
import tilelang
from tilelang.hexagon import language as T
from tilelang.hexagon.intrinsics.crouton import make_hmx_output_layout

NEG = -3.0e38


def make_flash(M, SEQ, BN, D):
    @T.prim_func
    def flash(
        Q: T.Tensor((M, D), "float16"), K: T.Tensor((SEQ, D), "float16"), V: T.Tensor((SEQ, D), "float16"), O: T.Tensor((M, D), "float16")
    ):
        with T.Kernel(1) as _:
            K_row = T.alloc_shared((BN, D), "float16")
            V_row = T.alloc_shared((BN, D), "float16")
            Q_sh = T.alloc_shared((M, D), "float16")
            K_sh = T.alloc_shared((BN, D), "float16")
            V_sh = T.alloc_shared((BN, D), "float16")
            S = T.alloc_shared((M, BN), "float16")  # Crouton: gemm output, then input
            temp = T.alloc_shared((M, D), "float16")  # Crouton: gemm output
            acc_o = T.alloc_shared((M, D), "float16")
            m = T.alloc_shared((M,), "float32")  # running max
            l = T.alloc_shared((M,), "float32")  # running denominator
            rmax = T.alloc_shared((M,), "float32")  # reduce dst (also the new max)
            rsum = T.alloc_shared((M,), "float32")  # reduce dst
            scale = T.alloc_shared((M,), "float32")
            # acc_o meets temp in maps, so it takes temp's layout.
            T.annotate_layout({acc_o: make_hmx_output_layout(acc_o)})
            T.copy(Q, Q_sh)
            for i, j in T.Parallel(M, D):
                acc_o[i, j] = T.float16(0)
            for i in T.Parallel(M):
                m[i] = T.float32(NEG)
                l[i] = T.float32(0)
            for kv in T.serial(SEQ // BN):
                T.copy(K[kv * BN, 0], K_row)
                T.copy(V[kv * BN, 0], V_row)
                T.copy(K_row, K_sh)
                T.copy(V_row, V_sh)
                T.gemm(Q_sh, K_sh, S, transpose_B=True, clear_accum=True)  # S = Q@Kᵀ (HMX)
                T.reduce_max(S, rmax, dim=1)  # rmax = rowmax(S) (HVX)
                for i in T.Parallel(M):  # running max + rescale
                    rmax[i] = T.max(m[i], rmax[i])
                    scale[i] = T.exp(m[i] - rmax[i])
                for i, j in T.Parallel(M, BN):  # P = exp(S − m)
                    S[i, j] = T.cast(T.exp(T.cast(S[i, j], "float32") - rmax[i]), "float16")
                T.reduce_sum(S, rsum, dim=1)  # rsum = rowsum(P) (HVX)
                for i in T.Parallel(M):  # denominator update
                    l[i] = l[i] * scale[i] + rsum[i]
                    m[i] = rmax[i]
                for i, j in T.Parallel(M, D):  # rescale acc_o
                    acc_o[i, j] = T.cast(T.cast(acc_o[i, j], "float32") * scale[i], "float16")
                T.gemm(S, V_sh, temp, clear_accum=True)  # temp = P@V (HMX)
                for i, j in T.Parallel(M, D):  # acc_o += temp
                    acc_o[i, j] = T.cast(T.cast(acc_o[i, j], "float32") + T.cast(temp[i, j], "float32"), "float16")
            for i, j in T.Parallel(M, D):  # O = acc_o / l
                acc_o[i, j] = T.cast(T.cast(acc_o[i, j], "float32") / l[i], "float16")
            T.copy(acc_o, O)

    return flash


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--m", type=int, default=64, help="query length")
    p.add_argument("--seq", type=int, default=256, help="kv length")
    p.add_argument("--bn", type=int, default=64, help="kv block (multiple of 64)")
    p.add_argument("--d", type=int, default=64, help="head dim (multiple of 64)")
    args = p.parse_args()
    M, SEQ, BN, D = args.m, args.seq, args.bn, args.d

    kernel = tilelang.compile(make_flash(M, SEQ, BN, D), out_idx=[3], target="hexagon")
    q = (torch.randn(M, D) * 0.3).half()
    k = (torch.randn(SEQ, D) * 0.3).half()
    v = (torch.randn(SEQ, D) * 0.3).half()
    o = kernel(q, k, v).float()
    ref = torch.softmax(q.float() @ k.float().T, dim=-1) @ v.float()
    err = (o - ref).abs().max().item()
    print(f"flash attention (M={M}, SEQ={SEQ}, BN={BN}, D={D}) on HMX+HVX: max abs err = {err:.4g}  ({'PASS' if err < 0.05 else 'FAIL'})")


if __name__ == "__main__":
    main()
