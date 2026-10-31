"""The fp16 decode on Hexagon: LFM2 or LFM3 from the fp16 bundle (convert.py),
every token of a generation in one kernel call.

As in the 4-bit decode (decode.py), NW worker programs meet at
``T.sync_grid()`` between the phases of a token, each streaming its share of
every projection by DMA, a panel of 32 output channels a block (``Weights``),
and multiplying it on HVX: ``T.gemm`` of an fp16 input by the block in the HMX
weight layout, the bundle's one copy of the weights (``tl_hvx_gemm_weight``).
Each worker owns matching shares of every layer:

- a slice of the model dimension: its rows of the projections into the
  residual (out_proj, w2), and its conv channels, whose B, C and x panels of
  in_proj follow each other in the bundle;
- whole KV heads: their keys and values, their query heads and (LFM3) the
  panel of those heads' gates, so attention needs no other worker;
- a slice of the FFN's hidden units, whose w1 and w3 panels alternate;
- a slice of the vocabulary.

Every worker keeps the residual stream itself, in fp32, and adds each
branch's whole output to it, read from the workers' slices in DDR. LFM3 norms
a branch's output over all of it before the add (sandwich norm), so a worker
writes its slice's sum of squares beside the slice. A layer takes four
barriers: after the operator's output, its projection, the FFN's hidden
units and their projection.

The KV cache is fp16, as the weights, in HMX's tiles (``kv_layouts``), which
the prefill's attention reads by DMA into its operands; the decode's attention
multiplies them on HVX with fp32 sums. The LM head proposes, from each block of 32 logits as it arrives, the
running argmax (or the sample, at a temperature) and the running sum of
exponentials, writing the logits to DDR.
"""

import os
import sys

import torch

from tilelang.hexagon import language as T
from tilelang.hexagon.intrinsics.crouton import make_hmx_weight_layout

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convert import TILE, qkv_group  # noqa: E402
from decode import NOISE_KEYS  # noqa: E402
from model import CONV, SLIDING, Config, Model, rope_angles  # noqa: E402
from stream import Phase, Stream, Weights  # noqa: E402

NW = 4  # workers
DEPTH = 4  # blocks in flight per worker
POS_BLOCK = 64  # KV positions per attention step
SLICE_TILES = 64  # a block's K tiles at most: 128 KB of weights


def kinds(c: Config) -> list:
    """The attention kinds of config c, in its RoPE tables' order."""
    return list(c.rope)


def rope_tables(c: Config, max_pos: int) -> tuple:
    """cos and sin [kinds, max_pos, head_dim / 2] of each attention kind's RoPE
    angles, as model.rope_angles computes them."""
    positions = torch.arange(max_pos)
    angles = torch.stack([rope_angles(positions, c.head_dim, *c.rope[k]) for k in kinds(c)])
    return angles.cos().float().contiguous(), angles.sin().float().contiguous()


def kv_layouts(kv_k, kv_v) -> dict:
    """The KV cache's layouts, [..., positions, head_dim] each: keys as the
    weight operand of q K^T (transposed: a block of positions is its tiles of
    every head dimension, contiguous), values as that of P V (a block of
    positions is a run of tiles for each 32 head dimensions)."""
    return {kv_k: make_hmx_weight_layout(kv_k, transposed=True), kv_v: make_hmx_weight_layout(kv_v)}


def slice_cols(K: int) -> int:
    """The largest slice of K, in whole tiles, of at most SLICE_TILES tiles."""
    tiles = K // TILE
    return TILE * max(t for t in range(1, SLICE_TILES + 1) if tiles % t == 0)


@T.macro
def rms_norm(src, weight, dst, sq, s, K: int, eps: float):
    """dst = src * rsqrt(mean(src^2) + eps) * weight(k), rows [1, K], in dst's
    dtype."""
    for i, k in T.Parallel(1, K):
        sq[i, k] = src[i, k] * src[i, k]
    T.reduce_sum(sq, s, dim=1)
    for i, k in T.Parallel(1, K):
        dst[i, k] = T.Cast(dst.dtype, src[i, k] * T.rsqrt(s[0] / T.float32(K) + T.float32(eps)) * weight(k))


@T.macro
def gemv(a, ring, k, acc, k0, cols: int, clear):
    """acc (+)= a[0, k0 : k0 + cols] times the block in slot k of the ring."""
    T.gemm(a[:, k0 : k0 + cols], ring[k, :, :], acc, clear_accum=clear)


@T.macro
def norm_rope(v, o, weight, cos, sin, kind: int, pos, t, sq, s, HD: int, eps: float):
    """The head at v[0, o : o + HD]: RMS norm by weight(d), then RoPE at pos,
    rotating its halves."""
    half = HD // 2
    for i, d in T.Parallel(1, HD):
        sq[i, d] = v[0, o + d] * v[0, o + d]
    T.reduce_sum(sq, s, dim=1)
    for i, d in T.Parallel(1, HD):
        t[i, d] = v[0, o + d] * T.rsqrt(s[0] / T.float32(HD) + T.float32(eps)) * weight(d)
    for i, d in T.Parallel(1, HD):
        sign = T.if_then_else(d < half, T.float32(-1), T.float32(1))
        v[0, o + d] = t[i, d] * cos[kind, pos, d % half] + sign * t[i, (d + half) % HD] * sin[kind, pos, d % half]


@T.macro
def attend(qkv, q0, kv_k, kv_v, ai, kvh, pos, first, out, out0, gate, group: int, HD: int, qg, kb, vb, sc, p, p16, o, m, mb, ls, lb, alpha):
    """out[0, out0 + g * HD + d] = gate(g) softmax(q_g K^T / sqrt(HD)) V over
    positions [first, pos] of KV head kvh, for the `group` query heads at
    qkv[0, q0 + ...]: blocks of POS_BLOCK positions with an online softmax."""
    for g, d in T.Parallel(group, HD):
        qg[g, d] = T.Cast("float16", qkv[0, q0 + g * HD + d] * T.float32(HD**-0.5))
    for g in T.Parallel(group):
        m[g] = T.float32(-3.0e38)
        ls[g] = T.float32(0)
    for g, d in T.Parallel(group, HD):
        o[g, d] = T.float32(0)
    for pb in T.serial(first // POS_BLOCK, T.ceildiv(pos + 1, POS_BLOCK)):
        T.copy(kv_k[ai, kvh, pb * POS_BLOCK : (pb + 1) * POS_BLOCK, :], kb)
        T.copy(kv_v[ai, kvh, pb * POS_BLOCK : (pb + 1) * POS_BLOCK, :], vb)
        T.gemm(qg, kb, sc, transpose_B=True, clear_accum=True)
        for g, j in T.Parallel(group, POS_BLOCK):
            key = pb * POS_BLOCK + j
            sc[g, j] = T.if_then_else(key <= pos and key >= first, sc[g, j], T.float32(-3.0e38))
        T.reduce_max(sc, mb, dim=1)
        for g in T.Parallel(group):
            alpha[g] = T.exp(m[g] - T.max(m[g], mb[g]))
            m[g] = T.max(m[g], mb[g])
        for g, j in T.Parallel(group, POS_BLOCK):
            p[g, j] = T.exp(sc[g, j] - m[g])
            p16[g, j] = T.Cast("float16", p[g, j])
        T.reduce_sum(p, lb, dim=1)
        for g in T.Parallel(group):
            ls[g] = ls[g] * alpha[g] + lb[g]
        for g, d in T.Parallel(group, HD):
            o[g, d] = o[g, d] * alpha[g]
        T.gemm(p16, vb, o, clear_accum=False)
    for g, d in T.Parallel(group, HD):
        out[0, out0 + g * HD + d] = T.Cast("float16", o[g, d] / ls[g] * gate(g))


@T.macro
def add_gumbel(logit, scale, seed, pos, first, hx, hy):
    """logit = logit * scale + Gumbel noise for vocabulary entries first + j at
    position pos (decode.add_gumbel)."""
    for i, j in T.Parallel(1, logit.shape[1]):
        hx[i, j] = T.Cast("uint32", first + j)
        hy[i, j] = T.Cast("uint32", pos) ^ (T.Cast("uint32", seed) << T.uint32(16))
    for r in list(range(len(NOISE_KEYS))):
        for i, j in T.Parallel(1, logit.shape[1]):
            hx[i, j] = (((hx[i, j] >> T.uint32(8)) | (hx[i, j] << T.uint32(24))) + hy[i, j]) ^ T.uint32(NOISE_KEYS[r])
        for i, j in T.Parallel(1, logit.shape[1]):
            hy[i, j] = ((hy[i, j] << T.uint32(3)) | (hy[i, j] >> T.uint32(29))) ^ hx[i, j]
    for i, j in T.Parallel(1, logit.shape[1]):
        u = (T.Cast("float32", T.Cast("int32", hy[i, j] >> T.uint32(8))) + T.float32(0.5)) * T.float32(1.0 / (1 << 24))
        logit[i, j] = logit[i, j] * scale + -T.log(-T.log(u))


@T.macro
def store_block(b, ring, k, a, acc, K: int, dst, off):
    """The block's 32 outputs into dst[0, off + 32 b ...]."""
    gemv(a, ring, k, acc, 0, K, True)
    for j in T.Parallel(TILE):
        dst[0, off + b * TILE + j] = acc[0, j]


@T.macro
def conv_block(b, ring, k, a, acc, D: int, bcx, conv_w, state, mix, ci, ch0, taps: int):
    """B, C or x (b % 3) of channel panel b // 3 from ch0; after x, the conv of
    those channels, its state and the gated output."""
    gemv(a, ring, k, acc, 0, D, True)
    for j in T.Parallel(TILE):
        bcx[b % 3, j] = acc[0, j]
    if b % 3 == 2:
        c0 = ch0 + b // 3 * TILE
        for j in T.Parallel(TILE):
            bx = bcx[0, j] * bcx[2, j]
            y = conv_w[ci, taps - 1, c0 + j] * bx
            for tap in list(range(taps - 1)):
                y = y + conv_w[ci, tap, c0 + j] * state[ci, tap, c0 + j]
            for tap in list(range(taps - 2)):
                state[ci, tap, c0 + j] = state[ci, tap + 1, c0 + j]
            state[ci, taps - 2, c0 + j] = bx
            mix[0, c0 + j] = T.Cast("float16", bcx[1, j] * y)


@T.macro
def glu_block(b, ring, k, a, acc, D: int, ga, out):
    """w1 (even) or w3 (odd) of hidden panel b // 2; after w3, silu(w1 x) * (w3 x)."""
    gemv(a, ring, k, acc, 0, D, True)
    if b % 2 == 0:
        for j in T.Parallel(TILE):
            ga[0, j] = acc[0, j]
    else:
        for j in T.Parallel(TILE):
            out[0, b // 2 * TILE + j] = T.Cast("float16", ga[0, j] * T.sigmoid(ga[0, j]) * acc[0, j])


@T.macro
def sliced_block(b, ring, k, a, acc, cols: int, slices: int, dst):
    """Slice b % slices of the K of panel b // slices; after the last, the
    panel's 32 outputs into dst."""
    gemv(a, ring, k, acc, (b % slices) * cols, cols, b % slices == 0)
    if b % slices == slices - 1:
        for j in T.Parallel(TILE):
            dst[0, b // slices * TILE + j] = acc[0, j]


@T.macro
def head_block(b, ring, k, a, acc, D: int, logits, first, pos, sampling, run_m, run_s, best, besti, bm, bs, bi, ex, idx, hx, hy):
    """The logits of vocabulary block b from `first` into DDR, and the running
    maximum, sum of exponentials and proposal: the argmax of the logits, or
    at a temperature of the logits over it plus Gumbel noise."""
    gemv(a, ring, k, acc, 0, D, True)
    v0 = first + b * TILE
    for j in T.Parallel(TILE):
        logits[0, v0 + j] = acc[0, j]
    T.reduce_max(acc, bm, dim=1)
    for i, j in T.Parallel(1, TILE):
        ex[i, j] = T.exp(acc[i, j] - T.max(run_m[0], bm[0]))
    T.reduce_sum(ex, bs, dim=1)
    run_s[0] = run_s[0] * T.exp(run_m[0] - T.max(run_m[0], bm[0])) + bs[0]
    run_m[0] = T.max(run_m[0], bm[0])
    if sampling[0] > T.float32(0):
        add_gumbel(acc, sampling[0], T.Cast("int32", sampling[1]), pos, v0, hx, hy)
        T.reduce_max(acc, bm, dim=1)
    for i, j in T.Parallel(1, TILE):
        idx[i, j] = T.if_then_else(acc[i, j] == bm[0], T.Cast("float32", j), T.float32(3.0e38))
    T.reduce_min(idx, bi, dim=1)
    # The first of the largest: a later block only when larger.
    if bm[0] > best[0]:
        best[0] = bm[0]
        besti[0] = bi[0] + T.Cast("float32", v0)


@T.macro
def publish(own, branch, w, DW: int, sums, sq, s, sandwich: bool):
    """Worker w's slice of a branch's output into DDR, with its sum of squares."""
    T.copy(own, branch[0:1, w * DW : (w + 1) * DW])
    if sandwich:
        for i, k in T.Parallel(1, DW):
            sq[i, k] = own[i, k] * own[i, k]
        T.reduce_sum(sq, s, dim=1)
        sums[w] = s[0]


@T.macro
def add_branch(xf, branch, bf, sums, s, post, D: int, nw: int, sandwich: bool, eps: float):
    """The residual xf plus the branch's whole output, read from the workers'
    slices: normed over all of it first, by post(k), with sandwich norms."""
    T.copy(branch, bf)
    if sandwich:
        s[0] = sums[0]
        for v in list(range(1, nw)):
            s[0] = s[0] + sums[v]
        for i, k in T.Parallel(1, D):
            xf[i, k] = xf[i, k] + bf[i, k] * T.rsqrt(s[0] / T.float32(D) + T.float32(eps)) * post(k)
    else:
        for i, k in T.Parallel(1, D):
            xf[i, k] = xf[i, k] + bf[i, k]


def make_decode(c: Config, max_pos: int, max_tokens: int):
    """The fp16 decode kernel of config c (convert.py's bundle)."""
    D, F, V, HD = c.dim, c.ff, c.vocab, c.head_dim
    H, KVH, taps = c.heads, c.kv_heads, c.conv_taps
    conv_layers = [i for i in range(c.layers) if not c.is_attention(i)]
    attn_layers = list(c.attention_layers)
    Lc, La, L = len(conv_layers), len(attn_layers), c.layers
    g = qkv_group(c)
    group = H // KVH
    assert H * HD == D and KVH % NW == 0 and HD % TILE == 0
    assert D % (NW * TILE) == 0 and F % (NW * TILE) == 0 and V % (NW * TILE) == 0
    DW, FW, VW = D // NW, F // NW, V // NW  # a worker's share
    KH = KVH // NW  # its KV heads
    QW = KH * g.width  # its columns of attn.qkv
    S2 = F // slice_cols(F)  # slices of w2's K
    NK = len(c.rope)
    kind = {k: i for i, k in enumerate(kinds(c))}
    eps = c.eps

    @T.prim_func
    def decode(
        in_w: T.Tensor((Lc, D, 3 * D), "float16"),
        qkv_w: T.Tensor((La, D, KVH * g.width), "float16"),
        out_w: T.Tensor((L, D, D), "float16"),
        w13_w: T.Tensor((L, D, 2 * F), "float16"),
        w2_w: T.Tensor((L, F, D), "float16"),
        head: T.Tensor((D, V), "float16"),
        # The head's bytes again, row-major: a token's embedding is D / 2 of
        # its pairs, a register of 32 tokens' apart (convert.embedding).
        head_rows: T.Tensor((V // TILE * D // 2, 2 * TILE), "float16"),
        op_norm: T.Tensor((L, D), "float32"),
        ffn_norm: T.Tensor((L, D), "float32"),
        post_op_norm: T.Tensor((L, D), "float32"),
        post_ffn_norm: T.Tensor((L, D), "float32"),
        q_norm: T.Tensor((La, HD), "float32"),
        k_norm: T.Tensor((La, HD), "float32"),
        conv_w: T.Tensor((Lc, taps, D), "float32"),
        final_norm: T.Tensor((D,), "float32"),
        rope_cos: T.Tensor((NK, max_pos, HD // 2), "float32"),
        rope_sin: T.Tensor((NK, max_pos, HD // 2), "float32"),
        kv_k: T.Tensor((La, KVH, max_pos, HD), "float16"),
        kv_v: T.Tensor((La, KVH, max_pos, HD), "float16"),
        conv_state: T.Tensor((Lc, taps - 1, D), "float32"),
        mix: T.Tensor((1, D), "float16"),
        branch: T.Tensor((1, D), "float32"),
        hidden: T.Tensor((1, F), "float16"),
        sums: T.Tensor((NW,), "float32"),
        cand: T.Tensor((NW, 4), "float32"),
        tokens: T.Tensor((max_tokens,), "int32"),
        logits: T.Tensor((1, V), "float32"),
        scores: T.Tensor((max_tokens,), "float32"),
        sampling: T.Tensor((2,), "float32"),
        pos0: T.int32,
        n: T.int32,
        n_forced: T.int32,
    ):
        with T.Kernel(NW, num_workers=NW) as w:
            p_in = Weights(in_w, TILE, slice_cols(D))
            p_qkv = Weights(qkv_w, TILE, slice_cols(D))
            p_out = Weights(out_w, TILE, slice_cols(D))
            p_13 = Weights(w13_w, TILE, slice_cols(D))
            p_2 = Weights(w2_w, TILE, slice_cols(F))
            p_head = Weights(head, TILE, slice_cols(D))

            xf = T.alloc_fragment((1, D), "float32")
            sq = T.alloc_fragment((1, D), "float32")
            h16 = T.alloc_fragment((1, D), "float16")
            bf = T.alloc_fragment((1, D), "float32")
            f16 = T.alloc_fragment((1, F), "float16")
            s1 = T.alloc_fragment((1,), "float32")
            acc = T.alloc_fragment((1, TILE), "float32")
            own = T.alloc_fragment((1, DW), "float32")
            sq_own = T.alloc_fragment((1, DW), "float32")
            ga = T.alloc_fragment((1, TILE), "float32")
            ffo = T.alloc_fragment((1, FW), "float16")
            bcx = T.alloc_shared((3, TILE), "float32")
            erow = T.alloc_shared((D // 2, 2), "float16")
            qkv = T.alloc_shared((1, QW), "float32")
            att = T.alloc_fragment((1, KH * group * HD), "float16")
            t_hd = T.alloc_shared((1, HD), "float32")
            sq_hd = T.alloc_fragment((1, HD), "float32")
            qg = T.alloc_fragment((group, HD), "float16")
            kb = T.alloc_shared((POS_BLOCK, HD), "float16")
            vb = T.alloc_shared((POS_BLOCK, HD), "float16")
            sc = T.alloc_fragment((group, POS_BLOCK), "float32")
            pr = T.alloc_fragment((group, POS_BLOCK), "float32")
            pr16 = T.alloc_fragment((group, POS_BLOCK), "float16")
            o = T.alloc_fragment((group, HD), "float32")
            m = T.alloc_fragment((group,), "float32")
            mb = T.alloc_fragment((group,), "float32")
            ls = T.alloc_fragment((group,), "float32")
            lb = T.alloc_fragment((group,), "float32")
            alpha = T.alloc_fragment((group,), "float32")
            ex = T.alloc_fragment((1, TILE), "float32")
            hx = T.alloc_fragment((1, TILE), "uint32")
            hy = T.alloc_fragment((1, TILE), "uint32")
            idx = T.alloc_fragment((1, TILE), "float32")
            bm = T.alloc_fragment((1,), "float32")
            bs = T.alloc_fragment((1,), "float32")
            bi = T.alloc_fragment((1,), "float32")
            # The head's running maximum and sum of exponentials, and its
            # running proposal (logit or sample, and index).
            run_m = T.alloc_fragment((1,), "float32")
            run_s = T.alloc_fragment((1,), "float32")
            best = T.alloc_fragment((1,), "float32")
            besti = T.alloc_fragment((1,), "float32")
            nxt = T.alloc_fragment((1,), "int32")
            nxt[0] = 0

            # A token's phases, worker w's blocks of each, in order: one round
            # of the stream's cycle.
            phases = []
            ci = ai = 0
            for layer in list(range(L)):
                if c.is_attention(layer):
                    phases.append(Phase.rows(p_qkv, ai, w * QW, QW // TILE))
                    ai += 1
                else:
                    # B, C and x of each of the worker's channel panels.
                    phases.append(Phase.rows(p_in, ci, w * 3 * DW, 3 * DW // TILE))
                    ci += 1
                phases.append(Phase.rows(p_out, layer, w * DW, DW // TILE))
                phases.append(Phase.rows(p_13, layer, w * 2 * FW, 2 * FW // TILE))
                phases.append(Phase.rows(p_2, layer, w * DW, DW // TILE))
            phases.append(Phase.rows(p_head, 0, w * VW, VW // TILE))
            st = Stream(phases, DEPTH)
            T.annotate_layout({**st.layouts(), **kv_layouts(kv_k, kv_v), **kv_layouts(kb, vb)})
            # The prefill reads the KV cache by DMA, so each key and value
            # written here goes back to DDR at once (prefill_fp16).
            T.annotate_write_through(kv_k, kv_v)
            st.prologue()

            for t in T.serial(n):
                pos = pos0 + t
                st.begin(t)

                # The token's embedding, which every worker reads whole: its
                # pairs from the head's tiles.
                token = T.max(T.min(T.if_then_else(t < n_forced, tokens[t], nxt[0]), V - 1), 0)
                T.dma_copy(
                    head_rows[token // TILE * (D // 2) : (token // TILE + 1) * (D // 2), token % TILE * 2 : token % TILE * 2 + 2], erow
                )
                T.dma_wait(0)
                for i, k in T.Parallel(1, D):
                    xf[i, k] = T.Cast("float32", erow[k // 2, k % 2])

                i_ph = 0
                ci = ai = 0
                for layer in list(range(L)):
                    rms_norm(xf, lambda k, layer=layer: op_norm[layer, k], h16, sq, s1, D, eps)
                    if c.is_attention(layer):
                        ki = kind[c.layer_types[layer]]
                        first = T.max(pos - c.window + 1, 0) if c.layer_types[layer] == SLIDING else 0
                        st.run(i_ph, lambda b, ring, k: store_block(b, ring, k, h16, acc, D, qkv, 0))
                        i_ph += 1
                        rope = (rope_cos, rope_sin, ki, pos, t_hd, sq_hd, s1, HD, eps)
                        for h in list(range(KH)):
                            base = h * g.width
                            kvh = w * KH + h
                            for q in list(range(group)):
                                norm_rope(qkv, base + q * HD, lambda d, ai=ai: q_norm[ai, d], *rope)
                            norm_rope(qkv, base + g.k, lambda d, ai=ai: k_norm[ai, d], *rope)
                            for d in T.Parallel(HD):
                                kv_k[ai, kvh, pos, d] = T.Cast("float16", qkv[0, base + g.k + d])
                                kv_v[ai, kvh, pos, d] = T.Cast("float16", qkv[0, base + g.v + d])
                            if c.head_gate:
                                gate = lambda q, base=base: T.sigmoid(qkv[0, base + g.gate + q])  # noqa: E731
                            else:
                                gate = lambda q: T.float32(1)  # noqa: E731
                            attend(
                                qkv, base, kv_k, kv_v, ai, kvh, pos, first, att, h * group * HD, gate, group, HD,
                                qg, kb, vb, sc, pr, pr16, o, m, mb, ls, lb, alpha,
                            )  # fmt: skip
                        T.copy(att, mix[0:1, w * DW : (w + 1) * DW])
                        ai += 1
                    else:
                        st.run(
                            i_ph,
                            lambda b, ring, k, ci=ci: conv_block(b, ring, k, h16, acc, D, bcx, conv_w, conv_state, mix, ci, w * DW, taps),
                        )
                        i_ph += 1
                        ci += 1
                    T.sync_grid()
                    # The operator's output projection: worker w's slice.
                    T.copy(mix, h16)
                    st.run(i_ph, lambda b, ring, k: store_block(b, ring, k, h16, acc, D, own, 0))
                    i_ph += 1
                    publish(own, branch, w, DW, sums, sq_own, s1, c.sandwich_norm)
                    T.sync_grid()
                    add_branch(xf, branch, bf, sums, s1, lambda k, layer=layer: post_op_norm[layer, k], D, NW, c.sandwich_norm, eps)
                    # The FFN: worker w's hidden units, then its slice of their
                    # projection, K in S2 slices.
                    rms_norm(xf, lambda k, layer=layer: ffn_norm[layer, k], h16, sq, s1, D, eps)
                    st.run(i_ph, lambda b, ring, k: glu_block(b, ring, k, h16, acc, D, ga, ffo))
                    i_ph += 1
                    T.copy(ffo, hidden[0:1, w * FW : (w + 1) * FW])
                    T.sync_grid()
                    T.copy(hidden, f16)
                    st.run(i_ph, lambda b, ring, k: sliced_block(b, ring, k, f16, acc, p_2.cols, S2, own))
                    i_ph += 1
                    publish(own, branch, w, DW, sums, sq_own, s1, c.sandwich_norm)
                    T.sync_grid()
                    add_branch(xf, branch, bf, sums, s1, lambda k, layer=layer: post_ffn_norm[layer, k], D, NW, c.sandwich_norm, eps)

                # The LM head: worker w's logits, their maximum and sum of
                # exponentials (for the scores), and its proposal.
                rms_norm(xf, lambda k: final_norm[k], h16, sq, s1, D, eps)
                run_m[0] = T.float32(-3.0e38)
                run_s[0] = T.float32(0)
                best[0] = T.float32(-3.0e38)
                besti[0] = T.float32(0)
                st.run(
                    i_ph,
                    lambda b, ring, k, pos=pos: head_block(
                        b, ring, k, h16, acc, D, logits, w * VW, pos, sampling, run_m, run_s, best, besti, bm, bs, bi, ex, idx, hx, hy
                    ),
                )
                cand[w, 0] = best[0]
                cand[w, 1] = besti[0]
                cand[w, 2] = run_m[0]
                cand[w, 3] = run_s[0]
                T.sync_grid()
                # Every worker takes the first of the largest proposals.
                best[0] = cand[0, 0]
                besti[0] = cand[0, 1]
                for v in list(range(1, NW)):
                    better = cand[v, 0] > best[0]
                    besti[0] = T.if_then_else(better, cand[v, 1], besti[0])
                    best[0] = T.if_then_else(better, cand[v, 0], best[0])
                nxt[0] = T.Cast("int32", besti[0])
                if w == 0 and t + 1 >= n_forced and t + 1 < max_tokens:
                    tokens[t + 1] = nxt[0]
                # The log-probability of a forced next token.
                if w == 0 and t + 1 < n_forced:
                    top = cand[0, 2]
                    for v in list(range(1, NW)):
                        top = T.max(top, cand[v, 2])
                    total = cand[0, 3] * T.exp(cand[0, 2] - top)
                    for v in list(range(1, NW)):
                        total = total + cand[v, 3] * T.exp(cand[v, 2] - top)
                    scores[t] = logits[0, tokens[t + 1]] - top - T.log(total)

    return decode


# The kernel's parameters from the bundle, in order. Without sandwich norms,
# the post norms are the pre norms, unread.
WEIGHT_ARGS = ("conv.in_proj", "attn.qkv", "out_proj", "ffn.w13", "ffn.w2", "head", "head")
NORM_ARGS = ("op_norm", "ffn_norm", "post_op_norm", "post_ffn_norm", "q_norm", "k_norm", "conv", "final_norm")


def bundle_args(c: Config, weights: dict) -> list:
    """The kernel's bundle parameters from bundle.load's buffers."""
    alias = {} if c.sandwich_norm else {"post_op_norm": "op_norm", "post_ffn_norm": "ffn_norm"}
    return [weights[n] for n in WEIGHT_ARGS] + [weights[alias.get(n, n)] for n in NORM_ARGS]


class DeviceModel(Model):
    """The reference on the fp16 bundle's weights with the device's rounding:
    each projection's input in fp16 (convert.read_weights gives the
    weights)."""

    def linear(self, name: str, x: torch.Tensor) -> torch.Tensor:
        return x.half().float() @ self.w[name].t()


__all__ = ["make_decode", "bundle_args", "rope_tables", "DeviceModel", "NW", "CONV"]
