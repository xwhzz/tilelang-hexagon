"""Pre-generate the FastRPC project — the ONE step that needs tilelang.

Run this in advance (where tilelang is installed); its output is committed, so the
offline build/run needs only the Hexagon SDK + adb + numpy, never tilelang.  Re-run
it only when you change the kernel.

Emits, under ./project/ :  <iface>.idl, <iface>_dsp.cc (skel + kernel),
<iface>_host.c, <iface>_agent.c, CMakeLists.txt  — plus a readable
generated_matmul_kernel.c.  The absolute tl_templates include that write_project
bakes in (this checkout's path) is rewritten to a repo-relative one so the
committed CMakeLists builds on any checkout.

    python generate.py
"""
import os

import tilelang
import tilelang.language as T
from tilelang import tvm
from tilelang.env import TILELANG_TEMPLATE_PATH
from tilelang.hexagon import _fastrpc

M = N = K = 256
BM = BN = 64
HERE = os.path.dirname(os.path.abspath(__file__))
# project/CMakeLists.txt sits at examples/hexagon/offline_matmul/project — four
# levels under the repo root, whose `src/` is the parent of `tl_templates/`.
REL_SRC = "${CMAKE_CURRENT_SOURCE_DIR}/../../../../src"


def make_matmul():
    @T.prim_func
    def matmul(A: T.Tensor((M, K), "float16"), B: T.Tensor((K, N), "float16"),
               C: T.Tensor((M, N), "float16")):
        with T.Kernel(T.ceildiv(N, BN), T.ceildiv(M, BM), threads=1) as (bx, by):
            A_hmx = T.alloc_shared((BM, K), "float16")
            B_hmx = T.alloc_shared((K, BN), "float16")
            C_hmx = T.alloc_shared((BM, BN), "float16")
            T.copy(A[by * BM, 0], A_hmx)                 # DDR -> Crouton VTCM
            T.copy(B[0, bx * BN], B_hmx)
            T.gemm(A_hmx, B_hmx, C_hmx, clear_accum=True)
            T.copy(C_hmx, C[by * BM, bx * BN])           # Crouton VTCM -> DDR
    return matmul


def main():
    with tvm.target.Target("hexagon"):                  # Layers 1-3: TIR -> cDSP C
        res = tilelang.lower(make_matmul(), target="hexagon")

    proj, iface = _fastrpc.write_project(               # Layer 5: C -> FastRPC project
        os.path.join(HERE, "project"), "matmul_kernel",
        res.kernel_source, res.params, result_idx=[2])

    # Rewrite the baked-in absolute template include to a repo-relative path so the
    # committed CMakeLists is portable across checkouts/machines.
    cml = os.path.join(proj, "CMakeLists.txt")
    with open(cml) as f:
        txt = f.read()
    txt = txt.replace(TILELANG_TEMPLATE_PATH, REL_SRC)
    with open(cml, "w") as f:
        f.write(txt)

    with open(os.path.join(HERE, "generated_matmul_kernel.c"), "w") as f:
        f.write(res.kernel_source)

    print(f"generated project : {proj}  (iface = {iface})")
    print(f"committed C       : CMakeLists.txt, {iface}.idl, {iface}_dsp.cc, "
          f"{iface}_host.c, {iface}_agent.c, generated_matmul_kernel.c")
    print(f"portable include  : {REL_SRC}")


if __name__ == "__main__":
    main()
