"""A w4a16 GEMV on Hexagon: y = W x for 4-bit weights with fp16 group scales.

Decode-time projections read every weight once, so they run at the speed of
memory, and 4-bit weights move a quarter of fp16's bytes. W holds unsigned
levels u = q + 8 of the signed codes q, with one fp16 scale per G columns of
a row: W(n, k) = (u(n, k) - 8) * S(n, k / G).

Each worker owns a slice of the outputs. It quantizes x to int16 with one
dynamic scale, sx = max|x| / 32767, then streams its rows of W, packed in the
panel layout (``pack_panel``), into VTCM by DMA (``T.Pipelined``). Per group
of G columns, an integer ``T.gemm`` multiplies the int16 x by the 4-bit block
on HVX, exactly in int32 (vrmpy); the group's scales then weigh the sums:

    y(n) = sx * sum_g S(n, g) * (sum_{k in g} u(n, k) xq(k) - 8 sum_{k in g} xq(k))

Requires a Hexagon device + SDK over adb (see README.md).

    python example_gemv_w4a16.py --k 2048 --n 8192
"""

import argparse

import torch

import tilelang
from tilelang.hexagon import language as T
from tilelang.hexagon.intrinsics.panel import make_hvx_panel_layout, pack_panel


def make_gemv(N, K, G, BN, workers, stages):
    rows = N // workers
    blocks = N // BN

    @T.prim_func
    def gemv(
        x: T.Tensor((1, K), "float32"),
        W: T.Tensor((N, K), "uint4"),
        S: T.Tensor((blocks, K // G, BN), "float16"),
        y: T.Tensor((1, N), "float32"),
    ):
        with T.Kernel(workers, num_workers=workers) as w:
            xf = T.alloc_fragment((1, K), "float32")
            amax = T.alloc_fragment((1,), "float32")
            xq = T.alloc_fragment((1, K), "int16")
            xg = T.alloc_fragment((K // G, G), "int32")
            xs = T.alloc_fragment((K // G,), "int32")
            wb = T.alloc_shared((BN, K), "uint4")
            sb = T.alloc_shared((K // G, BN), "float16")
            part = T.alloc_fragment((1, BN), "int32")
            acc = T.alloc_fragment((1, BN), "float32")
            # W arrives packed in the panel layout; T.gemm gives wb, which the
            # pipeline multi-buffers, the same layout.
            T.annotate_layout({W: make_hvx_panel_layout(W)})

            # x as int16 with one scale, and the sum of each of its groups.
            T.copy(x, xf)
            T.reduce_absmax(xf, amax, dim=1)
            for i, k in T.Parallel(1, K):
                v = xf[i, k] * (T.float32(32767) / T.max(amax[0], T.float32(1e-30)))
                xq[i, k] = T.Cast("int16", v + T.if_then_else(v >= T.float32(0), T.float32(0.5), T.float32(-0.5)))
            for g, k in T.Parallel(K // G, G):
                xg[g, k] = T.Cast("int32", xq[0, g * G + k])
            T.reduce_sum(xg, xs, dim=1)

            for n in T.Pipelined(rows // BN, num_stages=stages):
                block = w * (rows // BN) + n
                T.copy(W[block * BN : (block + 1) * BN, :], wb)
                T.copy(S[block, :, :], sb)
                T.clear(acc)
                for g in T.serial(K // G):
                    T.gemm(xq[:, g * G : (g + 1) * G], wb[:, g * G : (g + 1) * G], part, transpose_B=True, clear_accum=True)
                    for j in T.Parallel(BN):
                        acc[0, j] += T.Cast("float32", part[0, j] - 8 * xs[g]) * T.Cast("float32", sb[g, j])
                for j in T.Parallel(BN):
                    y[0, block * BN + j] = acc[0, j] * (T.max(amax[0], T.float32(1e-30)) / T.float32(32767))

    return gemv


def quantize(weight, G):
    """Symmetric 4-bit codes of `weight` [N, K], one fp16 scale per G columns."""
    N, K = weight.shape
    groups = weight.float().view(N, K // G, G)
    scale = (groups.abs().amax(-1, keepdim=True) / 7).clamp_min(1e-8)
    q = (groups / scale).round().clamp(-8, 7)
    return (q + 8).to(torch.int64).view(N, K), scale.squeeze(-1).half()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, default=2048)
    p.add_argument("--n", type=int, default=8192)
    p.add_argument("--group", type=int, default=128, help="columns per fp16 scale")
    p.add_argument("--block-n", type=int, default=128, help="rows of W per pipeline stage")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--stages", type=int, default=3)
    args = p.parse_args()
    N, K, G, BN = args.n, args.k, args.group, args.block_n

    kernel = tilelang.compile(make_gemv(N, K, G, BN, args.workers, args.stages), out_idx=[3], target="hexagon")
    torch.manual_seed(0)
    x = torch.randn(1, K)
    u, scale = quantize(torch.randn(N, K) * 0.05, G)
    # Scales per block of BN rows, [group, row] within the block.
    blocked = scale.view(N // BN, BN, K // G).transpose(1, 2).contiguous()
    y = kernel(x, pack_panel(u, 4), blocked)
    weight = (u - 8).float() * scale.float().repeat_interleave(G, dim=1)
    ref = x @ weight.t()
    err = ((y - ref).abs().max() / ref.abs().max()).item()
    print(f"w4a16 gemv {K} -> {N} (group {G}, {args.workers} workers): rel err = {err:.4g}  ({'PASS' if err < 1e-3 else 'FAIL'})")


if __name__ == "__main__":
    main()
