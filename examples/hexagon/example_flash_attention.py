"""Flash attention on Hexagon — two HMX gemms with an HVX online softmax between them.

This is the showcase for how the backend *composes*: `S = Q@Kᵀ` and `O = P@V` both
lower to the HMX matrix engine (`T.gemm`), and the online softmax between them runs
entirely on **HVX** — no scalar fp16 libcalls.  The softmax is built from the two
non-gemm primitives of the backend's {gemm, copy, map, reduce} basis:

  * **reduce** — `hexreduce(S, rmax, "max", 1)` / `"sum"` lower the structured reduce
    tile op straight to the HVX row-reduction primitives.
  * **map** — the elementwise `T.serial` loops (`exp(S − m)`, the `acc_o` rescale, the
    `+ temp` accumulate, the final `/l`) are recognised by the Hexagon codegen and
    emitted as full-width (64-lane) HVX: widen fp16→fp32, compute, narrow back.  The
    kernel stays plain `T.serial` + `T.exp`; the codegen does the vectorization.

The running max/denominator (`m`, `l`, `scale`) are tiny per-row (M-length) updates
left as scalar. HMX GEMM buffers use native Crouton layout, while the HVX row-reduce
extern requires contiguous row-major input. `S_hmx -> S -> S_hmx` and
`temp_hmx -> temp` make those layout boundaries explicit; Hexagon `T.copy` recognizes
the exact HMX layout and lowers these VTCM-to-VTCM transforms to Crouton pack/unpack.
DDR inputs likewise pass through row-major VTCM staging before that transform.
The HMX accumulator can't be preloaded, so the `clear_accum=False` accumulation of `O`
is decomposed into an overwrite gemm to `temp` plus the HVX rescale-and-add into `acc_o`.

`hexreduce` invokes the reduce tile op directly on shared VTCM buffers — the stock
`T.reduce_max` macro routes through register fragments, which Hexagon has no layout
inference for; a cleaner `T.reduce`-shaped wrapper is a tracked follow-up.

Requires a Hexagon device + SDK over adb (see README.md).

    python example_flash_attention.py --seq 256
"""
import argparse
import torch
import tilelang
import tilelang.language as T
from tvm import tirx
from tilelang.language import macro
from tilelang.utils.language import to_tile_region

NEG = -3.0e38


@macro
def hexreduce(src, dst, rtype, dim):
    """Hexagon row reduction: invoke the reduce tile op directly on shared VTCM
    tiles (max/sum over the last axis), bypassing the fragment-based macro."""
    tirx.call_intrin("handle", tirx.op.Op.get("tl.tileop.reduce"),
                     to_tile_region(src, "r"), to_tile_region(dst, "w"), rtype, dim, True)


def make_flash(M, SEQ, BN, D):
    @T.prim_func
    def flash(Q: T.Tensor((M, D), "float16"), K: T.Tensor((SEQ, D), "float16"),
              V: T.Tensor((SEQ, D), "float16"), O: T.Tensor((M, D), "float16")):
        with T.Kernel(1, threads=1) as _:
            Q_row = T.alloc_shared((M, D), "float16")
            K_row = T.alloc_shared((BN, D), "float16")
            V_row = T.alloc_shared((BN, D), "float16")
            Q_sh = T.alloc_shared((M, D), "float16")
            K_sh = T.alloc_shared((BN, D), "float16")
            V_sh = T.alloc_shared((BN, D), "float16")
            S_hmx = T.alloc_shared((M, BN), "float16")
            S = T.alloc_shared((M, BN), "float16")
            temp_hmx = T.alloc_shared((M, D), "float16")
            temp = T.alloc_shared((M, D), "float16")
            acc_o = T.alloc_shared((M, D), "float16")
            m = T.alloc_shared((M,), "float32")       # running max
            l = T.alloc_shared((M,), "float32")       # running denominator
            rmax = T.alloc_shared((M,), "float32")    # reduce dst (also the new max)
            rsum = T.alloc_shared((M,), "float32")    # reduce dst
            scale = T.alloc_shared((M,), "float32")
            T.copy(Q, Q_row)
            T.copy(Q_row, Q_sh)
            for i in T.serial(M):
                for j in T.serial(D):
                    acc_o[i, j] = T.float16(0)                                  # map (HVX)
            for i in T.serial(M):
                m[i] = T.float32(NEG)
                l[i] = T.float32(0)
            for kv in T.serial(SEQ // BN):
                T.copy(K[kv * BN, 0], K_row)
                T.copy(V[kv * BN, 0], V_row)
                T.copy(K_row, K_sh)
                T.copy(V_row, V_sh)
                T.gemm(Q_sh, K_sh, S_hmx, transpose_B=True, clear_accum=True)   # S = Q@Kᵀ (HMX)
                T.copy(S_hmx, S)                                                # Crouton -> row-major
                hexreduce(S, rmax, "max", 1)                                    # rmax = rowmax(S) (HVX)
                for i in T.serial(M):                                           # running max + rescale (scalar, M)
                    rmax[i] = T.max(m[i], rmax[i])
                    scale[i] = T.exp(m[i] - rmax[i])
                for i in T.serial(M):
                    for j in T.serial(BN):                                      # P = exp(S − m) (HVX map)
                        S[i, j] = T.cast(T.exp(T.cast(S[i, j], "float32") - rmax[i]), "float16")
                hexreduce(S, rsum, "sum", 1)                                    # rsum = rowsum(P) (HVX)
                for i in T.serial(M):                                           # denominator update (scalar, M)
                    l[i] = l[i] * scale[i] + rsum[i]
                    m[i] = rmax[i]
                for i in T.serial(M):
                    for j in T.serial(D):                                       # rescale acc_o (HVX map)
                        acc_o[i, j] = T.cast(T.cast(acc_o[i, j], "float32") * scale[i], "float16")
                T.copy(S, S_hmx)                                                # row-major -> Crouton
                T.gemm(S_hmx, V_sh, temp_hmx, clear_accum=True)                 # temp = P@V (HMX)
                T.copy(temp_hmx, temp)                                          # Crouton -> row-major
                for i in T.serial(M):
                    for j in T.serial(D):                                       # acc_o += temp (HVX map)
                        acc_o[i, j] = T.cast(
                            T.cast(acc_o[i, j], "float32") + T.cast(temp[i, j], "float32"), "float16")
            for i in T.serial(M):
                for j in T.serial(D):                                           # O = acc_o / l (HVX map)
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
    print(f"flash attention (M={M}, SEQ={SEQ}, BN={BN}, D={D}) on HMX+HVX: max abs err = {err:.4g}  "
          f"({'PASS' if err < 0.05 else 'FAIL'})")


if __name__ == "__main__":
    main()
