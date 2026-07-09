"""Emit an EMBEDDABLE tilelang kernel for llama.cpp (ggml-hexagon), closing gap #1
of docs/llama_cpp_integration.md — the op body becomes a `tilelang.compile` artifact
instead of hand-C.

Key facts this encodes:
  * `k.get_kernel_source()` is ALREADY the embeddable body: `extern "C" int32_t
    qmatmul_kernel(half* A, uint8_t* qcm, half* sc, half* C)`, using `tl_vtcm_base()`
    (which the bridge sets) and calling the reused `tl_hexagon_hmx_gemm` runtime — no
    FastRPC skel / session wrapper. So embedding it needs only the source + the bridge.
  * The kernel takes COMPACT per-block scales `sc[K/32][N]` (resident-friendly), not the
    expanded `scb[K][N]` (4x the weight, can't be resident). It expands the block scale
    in-kernel with the whole-register HVX dequant.

Emits: kernel_qmatmul_compact_<M>x<N>x<K>.c  +  .manifest.json (entry sig, op/dtype/
shape, VTCM footprint, weight format) — the host embeds the .c and dispatches by the
manifest. Fixed-shape (codegen emits one shape); a real model needs one per matmul
shape family (see the doc's shape-family gap #4).

    python emit_embeddable.py --n 128 --k 2048
"""
import argparse
import json
import re
import tilelang
import tilelang.language as T


def make(M, N, K):
    KH, NB = K // 2, K // 32

    @T.prim_func
    def qmatmul_compact(A: T.Tensor((M, K), "float16"), qcm: T.Tensor((KH, N), "uint8"),
                        sc: T.Tensor((NB, N), "float16"), C: T.Tensor((M, N), "float16")):
        with T.Kernel(1, threads=1) as _:
            A_sh = T.alloc_shared((M, K), "float16")
            B_sh = T.alloc_shared((K, N), "float16")
            C_sh = T.alloc_shared((M, N), "float16")
            T.copy(A, A_sh)
            for j in T.serial(KH):                    # K-pair j -> K=2j (lo), 2j+1 (hi)
                blk = j // 16                          # 32-K scale block
                for no in T.serial(N // 128):
                    for ni in T.vectorized(128):
                        n = no * 128 + ni
                        q = T.Cast("int16", qcm[j, n])
                        lo = (q & T.Cast("int16", 0xF)) - T.Cast("int16", 8)
                        hi = (q >> T.Cast("int16", 4)) - T.Cast("int16", 8)
                        B_sh[2 * j, n] = T.Cast("float16", lo) * sc[blk, n]
                        B_sh[2 * j + 1, n] = T.Cast("float16", hi) * sc[blk, n]
            T.gemm(A_sh, B_sh, C_sh, clear_accum=True)
            T.copy(C_sh, C)
    return qmatmul_compact


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--m", type=int, default=32)
    p.add_argument("--n", type=int, default=128)
    p.add_argument("--k", type=int, default=2048)
    args = p.parse_args()
    M, N, K = args.m, args.n, args.k

    k = tilelang.compile(make(M, N, K), out_idx=[3], target="hexagon")
    src = k.get_kernel_source()
    entry = re.search(r"(int32_t \w+\([^)]*\))", src).group(1)
    hw = int(re.search(r"tl_vtcm_shared_high_water = (\d+)", src).group(1))

    base = f"kernel_qmatmul_compact_{M}x{N}x{K}"
    manifest = {
        "op": "matmul", "weight_dtype": "q4_0", "M": M, "N": N, "K": K,
        "entry": entry,
        "vtcm_bytes": hw + 2048,           # arena starts at base+2048; +high-water
        "weight_format": {
            "qcm": "[K/2][N] uint8, column-major packed nibbles (lo->K=2j, hi->K=2j+1)",
            "sc": "[K/32][N] fp16, per-32K-block scales (compact, resident-friendly)",
        },
        "call": "under tl_bridge_enter(vtcm_base, vtcm_size); qmatmul_kernel(act, qcm, sc, out)",
        "note": "fixed shape; one emit per matmul shape family",
    }
    with open(base + ".c", "w") as f:
        f.write(src)
    with open(base + ".manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"emitted {base}.c ({len(src.splitlines())} lines)")
    print(f"  entry : {entry}")
    print(f"  vtcm  : {manifest['vtcm_bytes']} bytes")
    print(f"  weight: qcm[{K//2}][{N}] u8 + sc[{K//32}][{N}] f16 (compact)")


if __name__ == "__main__":
    main()
