"""Generate a self-contained FastRPC project for a Hexagon HMX matmul — no device.

Lowers the SAME tilelang matmul you'd write for a GPU to cDSP C, then emits the
full FastRPC project (IDL + skel + host driver + agent + CMakeLists) under
``./project/``.  Also writes golden inputs (A.bin, B.bin) and the fp32 reference
(ref.npy) for the on-device comparison, and a readable snapshot of the generated
device kernel (generated_matmul_kernel.c).

This is the offline counterpart to ``example_matmul.py``: that one calls
``tilelang.compile`` (build + deploy + run in one shot); here we stop at codegen
so ``reproduce.sh`` can build it with the bare Hexagon SDK and run it by hand.

    python generate.py          # -> ./project/ + A.bin/B.bin/ref.npy
"""
import os

import numpy as np
import tilelang
import tilelang.language as T
from tilelang import tvm
from tilelang.hexagon import _fastrpc

M = N = K = 256
BM = BN = 64
HERE = os.path.dirname(os.path.abspath(__file__))


def make_matmul():
    @T.prim_func
    def matmul(A: T.Tensor((M, K), "float16"), B: T.Tensor((K, N), "float16"),
               C: T.Tensor((M, N), "float16")):
        with T.Kernel(T.ceildiv(N, BN), T.ceildiv(M, BM), threads=1) as (bx, by):
            A_sh = T.alloc_shared((BM, K), "float16")   # -> VTCM (on-chip)
            B_sh = T.alloc_shared((K, BN), "float16")
            C_sh = T.alloc_shared((BM, BN), "float16")
            T.copy(A[by * BM, 0], A_sh)                  # global -> VTCM (HVX half8)
            T.copy(B[0, bx * BN], B_sh)
            T.gemm(A_sh, B_sh, C_sh, clear_accum=True)   # -> HMX matrix engine
            T.copy(C_sh, C[by * BM, bx * BN])            # VTCM -> global
    return matmul


def main():
    with tvm.target.Target("hexagon"):                  # Layers 1-3: TIR -> cDSP C
        res = tilelang.lower(make_matmul(), target="hexagon")

    proj, iface = _fastrpc.write_project(               # Layer 5: C -> FastRPC project
        os.path.join(HERE, "project"), "matmul_kernel",
        res.kernel_source, res.params, result_idx=[2])

    # Golden: fixed-seed fp16 inputs + the fp32 reference (host side).
    rng = np.random.default_rng(0)
    A = (rng.standard_normal((M, K), dtype=np.float32) * 0.25).astype(np.float16)
    B = (rng.standard_normal((K, N), dtype=np.float32) * 0.25).astype(np.float16)
    A.tofile(os.path.join(proj, "A.bin"))
    B.tofile(os.path.join(proj, "B.bin"))
    np.save(os.path.join(proj, "ref.npy"), A.astype(np.float32) @ B.astype(np.float32))

    with open(os.path.join(HERE, "generated_matmul_kernel.c"), "w") as f:
        f.write(res.kernel_source)

    print(f"project : {proj}  (iface = {iface})")
    print(f"files   : {sorted(os.listdir(proj))}")
    print(f"golden  : A.bin B.bin ({M}x{K}, {K}x{N} fp16)  ref.npy ({M}x{N} fp32)")
    print(f"kernel  : generated_matmul_kernel.c")


if __name__ == "__main__":
    main()
