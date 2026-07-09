"""K-streaming q4_0 matmul on Hexagon, authored at HMX instruction-atom granularity.

Unlike `example_qmatmul.py` (which materializes the whole fp16 weight then calls the
monolithic `T.gemm`), this drives the HMX accumulator's K-loop from the DSL with the
`HMXIntrinEmitter` — the Hexagon analog of tilelang's `TensorCoreIntrinEmitter`
(`tilelang/cuda/intrinsics/macro/mma_macro_generator.py`).  Each B tile is dequantized
just before its MAC, so the dequant interleaves with the matmul (the pattern
ggml-hexagon uses to hide the dequant under the MAC).

The emitter exposes instruction atoms so you compose the K-loop yourself, instead of a
black-box gemm:
  * `pack_a` / `pack_b`  — row-major VTCM -> Crouton VTCM  (== `ldmatrix` on tensor cores)
  * `mma`                — the `mxmem` activation+weight load pair, which IS the MAC atom
  * `store`              — `cvt` acc -> Crouton -> unpack   (== `stmatrix`)
  * `clear`              — `mxclracc`                       (== `T.clear(C_local)`)

HMX diverges from tensor cores in one honest way: there is no addressable accumulator
fragment — the accumulator is a single invisible per-core register, so `clear`/`mma`/
`store` operate on it implicitly (no `C_local` is threaded through).  N>32 loops
`clear -> mac-K -> store` per N-tile (one physical accumulator).  Weight is q4_0,
pre-packed column-major (`qcm[K/2][N]`, `scb[K][N]`).

Note: the per-32x32-tile dequant here is scalar (a 32-feature tile is sub-register for
HVX); the whole-register-dequant perf optimization is a follow-up (see
docs/hexagon_dsl_kernels.md).  This example is about the atom-granular authoring.

    python example_qmatmul_kstream.py --n 128 --k 512

Requires a Hexagon device over adb + the Hexagon SDK. See README.md.
"""
import argparse
import numpy as np
import torch
import tilelang
import tilelang.language as T
from tilelang.hexagon.hmx_intrin import HMXIntrinEmitter


def make(M, N, K):
    NT, KT, KH = N // 32, K // 32, K // 2
    E = HMXIntrinEmitter(M, N, K)

    @T.prim_func
    def qms(A: T.Tensor((M, K), "float16"), qcm: T.Tensor((KH, N), "uint8"),
            scb: T.Tensor((K, N), "float16"), C: T.Tensor((M, N), "float16")):
        with T.Kernel(1, threads=1) as _:
            A_sh = T.alloc_shared((M, K), "float16")
            a_frag = T.alloc_shared(E.a_frag_shape, "float16")   # Crouton "fragments"
            b_frag = T.alloc_shared(E.b_frag_shape, "float16")
            c_frag = T.alloc_shared(E.c_frag_shape, "float16")
            Bt = T.alloc_shared((32, 32), "float16")             # dequant staging (row-major)
            Ctile = T.alloc_shared((M, 32), "float16")
            T.copy(A, A_sh)
            E.begin()
            E.pack_a(a_frag, A_sh)                                # ldmatrix_a analog (once)
            for nt in range(NT):                    # each output N-tile (own accumulator pass)
                E.clear()
                for kt in T.serial(KT):
                    for kk in range(32):                          # dequant the (nt,kt) 32x32 tile
                        for nn in T.serial(32):
                            n = nt * 32 + nn
                            j = kt * 16 + kk // 2
                            nib = (qcm[j, n] & 0xF) if (kk % 2 == 0) else (qcm[j, n] >> 4)
                            Bt[kk, nn] = T.Cast("float16", T.Cast("int16", nib) - T.Cast("int16", 8)) * scb[kt * 32 + kk, n]
                    E.pack_b(b_frag, Bt)                          # ldmatrix_b analog
                    E.mma(a_frag, b_frag, kt)                     # mxmem load + MAC (atom)
                E.store(Ctile, c_frag)                            # stmatrix analog
                T.copy(Ctile, C[:, nt * 32:(nt + 1) * 32])
            E.end()
    return qms


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--m", type=int, default=32)
    p.add_argument("--n", type=int, default=128, help="output features (multiple of 32)")
    p.add_argument("--k", type=int, default=512)
    args = p.parse_args()
    M, N, K = args.m, args.n, args.k

    rng = np.random.default_rng(0)
    nib = rng.integers(0, 16, size=(N, K), dtype=np.uint8)
    scales = (rng.standard_normal((N, K // 32)) * 0.05).astype(np.float16)
    Wq = ((nib.astype(np.float32) - 8.0) * np.repeat(scales.astype(np.float32), 32, axis=1)).astype(np.float16)
    qcm = (nib[:, 0::2].T | (nib[:, 1::2].T << 4)).astype(np.uint8)     # [K/2][N]
    scb = np.repeat(scales, 32, axis=1).T.astype(np.float16)           # [K][N]
    A = (rng.standard_normal((M, K)) * 0.1).astype(np.float16)
    ref = A.astype(np.float32) @ Wq.astype(np.float32).T               # [M][N]

    kernel = tilelang.compile(make(M, N, K), out_idx=[3], target="hexagon")
    C = kernel(torch.from_numpy(A), torch.from_numpy(qcm), torch.from_numpy(scb)).cpu().numpy().astype(np.float32)
    rel = np.abs(C - ref).max() / (np.abs(ref).max() + 1e-6)
    print(f"K-streaming (HMXIntrinEmitter) q4_0 matmul {M}x{N}x{K}: rel err = {rel:.4g}  ({'PASS' if rel < 5e-2 else 'FAIL'})")


if __name__ == "__main__":
    main()
