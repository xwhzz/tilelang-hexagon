"""Tiled FP16 matmul on the Qualcomm Hexagon HMX matrix engine.

Each `T.copy` crosses the DDR/native-Crouton boundary directly.  The Hexagon copy
lowering preserves the DDR matrix stride and emits a fused Crouton pack/unpack
helper (HVX `vshuff`/`vdeal` for the common 64-wide case).  `T.gemm` itself receives
only caller-owned native-Crouton A/B/C buffers and lowers to explicit HMX atoms.

Requires a Hexagon device (e.g. OnePlus 13 / Snapdragon 8 Gen 4) reachable over adb,
the Hexagon SDK, and an authorized device. See README.md.

    python example_matmul.py --m 256 --n 256 --k 256 --block 64
"""
import argparse
import torch
import tilelang
import tilelang.language as T


def make_matmul(M, N, K, BM, BN):
    @T.prim_func
    def matmul(A: T.Tensor((M, K), "float16"), B: T.Tensor((K, N), "float16"),
               C: T.Tensor((M, N), "float16")):
        # 2-D grid of BM x BN output tiles; one HW thread per block (for now).
        with T.Kernel(T.ceildiv(N, BN), T.ceildiv(M, BM), threads=1) as (bx, by):
            A_hmx = T.alloc_shared((BM, K), "float16")  # native Crouton VTCM
            B_hmx = T.alloc_shared((K, BN), "float16")
            C_hmx = T.alloc_shared((BM, BN), "float16")
            T.copy(A[by * BM, 0], A_hmx)                 # DDR -> Crouton VTCM
            T.copy(B[0, bx * BN], B_hmx)
            T.gemm(A_hmx, B_hmx, C_hmx, clear_accum=True)
            T.copy(C_hmx, C[by * BM, bx * BN])           # Crouton VTCM -> DDR
    return matmul


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--m", type=int, default=256)
    p.add_argument("--n", type=int, default=256)
    p.add_argument("--k", type=int, default=256)
    p.add_argument("--block", type=int, default=64, help="BM = BN tile size (multiple of 32)")
    args = p.parse_args()
    M, N, K, BL = args.m, args.n, args.k, args.block

    # tilelang.compile builds the FastRPC skel, deploys it to the device, and returns
    # a torch-callable kernel. out_idx=[2] marks C as the output.
    kernel = tilelang.compile(make_matmul(M, N, K, BL, BL), out_idx=[2], target="hexagon")
    print("code source: \n")
    print(kernel.get_kernel_source())  # for debugging

    a = (torch.randn(M, K) * 0.25).half()
    b = (torch.randn(K, N) * 0.25).half()
    c = kernel(a, b).float()
    ref = a.float() @ b.float()
    err = (c - ref).abs().max().item()
    print(f"matmul {M}x{N}x{K} (block {BL}) on HMX: max abs err = {err:.4g}  "
          f"({'PASS' if err < 0.1 else 'FAIL'})")


if __name__ == "__main__":
    main()
