"""LFM2 decode on Hexagon: every token of a generation in one kernel call.

The kernel runs on NW worker programs that meet at ``T.sync_grid()`` between
the phases of a token. Each worker owns matching shares of every layer:

- a slice of the model dimension, whose rows of every projection that writes
  the residual it multiplies, and whose conv channels it convolves;
- its query heads and the KV heads they read, so attention needs no other
  worker;
- a slice of the FFN's hidden units, both of whose projections (w1, w3) it
  multiplies, so SwiGLU needs no other worker;
- a slice of the vocabulary, whose argmax it proposes.

A token has two phases a layer (the operator, then the FFN, each ending in
its residual add), plus the LM head: the weights of all of them stream by DMA
through one cycle of the worker's ``Stream``, the next token's first blocks
arriving while the head's last ones are multiplied. Activations pass between
workers through small DDR buffers; each worker quantizes a phase's input to
int16 itself. The KV cache and the conv state stay on the device between
calls, with the weights.

``tokens[t]`` feeds step t; from step ``n_forced - 1`` on, each step's argmax
is written to ``tokens[t + 1]`` and feeds the next.
"""

import os
import sys

import torch

from tilelang.hexagon import language as T

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bundle  # noqa: E402
from model import Config  # noqa: E402
from prefill import make_prefill  # noqa: E402
from stream import Phase, Projection, Stream, gemv_block, quantize  # noqa: E402

NW = 4  # workers
DEPTH = 3  # blocks in flight per worker
BLOCK = 128  # rows per block of a 4-bit projection
HEAD_BLOCK = 64  # rows per block of the 8-bit head
POS_BLOCK = 64  # KV positions per attention step


@T.macro
def rms_norm(src, weight, dst, sq, s, K: int, eps: float):
    """dst = src * rsqrt(mean(src^2) + eps) * weight(k), rows [1, K]."""
    for i, k in T.Parallel(1, K):
        sq[i, k] = src[i, k] * src[i, k]
    T.reduce_sum(sq, s, dim=1)
    for i, k in T.Parallel(1, K):
        dst[i, k] = src[i, k] * T.rsqrt(s[0] / T.float32(K) + T.float32(eps)) * weight(k)


@T.macro
def conv_block(b, rw, rs, k, xq, sa, acc, D: int, bcx, conv_w, state, mix, ci, ch0):
    """in_proj block b: B, C or x (b % 3) of channel block b // 3 from ch0; after
    x, the conv of those channels, its state and the gated output."""
    gemv_block(xq, sa, rw, rs, k, acc, 0, D, 64, True)
    for j in T.Parallel(BLOCK):
        bcx[b % 3, j] = acc[0, j]
    if b % 3 == 2:
        c0 = ch0 + b // 3 * BLOCK
        for j in T.Parallel(BLOCK):
            bx = bcx[0, j] * bcx[2, j]
            y = conv_w[ci, 0, c0 + j] * state[ci, 0, c0 + j] + conv_w[ci, 1, c0 + j] * state[ci, 1, c0 + j] + conv_w[ci, 2, c0 + j] * bx
            state[ci, 0, c0 + j] = state[ci, 1, c0 + j]
            state[ci, 1, c0 + j] = bx
            mix[0, c0 + j] = bcx[1, j] * y


@T.macro
def store_block(b, rw, rs, k, xq, sa, acc, D: int, dst, off):
    gemv_block(xq, sa, rw, rs, k, acc, 0, D, 64, True)
    for j in T.Parallel(BLOCK):
        dst[0, off + b * BLOCK + j] = acc[0, j]


@T.macro
def residual_block(b, rw, rs, k, xq, sa, acc, cols: int, slices: int, x, row0):
    """Block b: rows row0 + (b // slices) * BLOCK, slice b % slices of the input;
    after the last slice, the rows' sums are added to the residual x."""
    gemv_block(xq, sa, rw, rs, k, acc, (b % slices) * cols, cols, 64, b % slices == 0)
    if b % slices == slices - 1:
        r0 = row0 + b // slices * BLOCK
        for j in T.Parallel(BLOCK):
            x[0, r0 + j] = x[0, r0 + j] + acc[0, j]


@T.macro
def glu_block(b, rw, rs, k, xq, sa, acc, D: int, ga, ff, row0):
    """w13 block b: w1 (even) or w3 (odd) of hidden block b // 2 from row0;
    after w3, silu(w1 x) * (w3 x)."""
    gemv_block(xq, sa, rw, rs, k, acc, 0, D, 64, True)
    if b % 2 == 0:
        for j in T.Parallel(BLOCK):
            ga[0, j] = acc[0, j]
    else:
        r0 = row0 + b // 2 * BLOCK
        for j in T.Parallel(BLOCK):
            ff[0, r0 + j] = ga[0, j] * T.sigmoid(ga[0, j]) * acc[0, j]


@T.macro
def head_block(b, rw, rs, k, xq, sa, acc, D: int, logit):
    gemv_block(xq, sa, rw, rs, k, acc, 0, D, D, True)
    for j in T.Parallel(HEAD_BLOCK):
        logit[0, b * HEAD_BLOCK + j] = acc[0, j]


@T.macro
def norm_rope(v, o, weight, cos, sin, pos, t64, sq, s, eps: float):
    """The head at v[0, o : o + 64]: RMS norm by weight(d), then RoPE at pos,
    rotating its halves."""
    for i, d in T.Parallel(1, 64):
        sq[i, d] = v[0, o + d] * v[0, o + d]
    T.reduce_sum(sq, s, dim=1)
    for i, d in T.Parallel(1, 64):
        t64[i, d] = v[0, o + d] * T.rsqrt(s[0] / T.float32(64) + T.float32(eps)) * weight(d)
    for i, d in T.Parallel(1, 64):
        sign = T.if_then_else(d < 32, T.float32(-1), T.float32(1))
        v[0, o + d] = t64[i, d] * cos[pos, d % 32] + sign * t64[i, (d + 32) % 64] * sin[pos, d % 32]


@T.macro
def attend(qkv, q0, kv_k, kv_v, ai, kvh, pos, mix, out0, group: int, qg, kb, vb, sc, p, o, m, mb, l, lb, alpha):
    """mix[0, out0 + (g * 64 + d)] = softmax(q_g K^T / 8) V over positions
    [0, pos] of KV head kvh, for the `group` query heads at qkv[0, q0 + ...]:
    blocks of POS_BLOCK positions with an online softmax."""
    for g, d in T.Parallel(group, 64):
        qg[g, d] = qkv[0, q0 + g * 64 + d] * T.float32(0.125)
    for g in T.Parallel(group):
        m[g] = T.float32(-3.0e38)
        l[g] = T.float32(0)
    for g, d in T.Parallel(group, 64):
        o[g, d] = T.float32(0)
    for pb in T.serial(T.ceildiv(pos + 1, POS_BLOCK)):
        T.copy(kv_k[ai, kvh, pb * POS_BLOCK : (pb + 1) * POS_BLOCK, :], kb)
        T.copy(kv_v[ai, kvh, pb * POS_BLOCK : (pb + 1) * POS_BLOCK, :], vb)
        T.gemm(qg, kb, sc, transpose_B=True, clear_accum=True)
        for g, j in T.Parallel(group, POS_BLOCK):
            sc[g, j] = T.if_then_else(pb * POS_BLOCK + j <= pos, sc[g, j], T.float32(-3.0e38))
        T.reduce_max(sc, mb, dim=1)
        for g in T.Parallel(group):
            alpha[g] = T.exp(m[g] - T.max(m[g], mb[g]))
            m[g] = T.max(m[g], mb[g])
        for g, j in T.Parallel(group, POS_BLOCK):
            p[g, j] = T.exp(sc[g, j] - m[g])
        T.reduce_sum(p, lb, dim=1)
        for g in T.Parallel(group):
            l[g] = l[g] * alpha[g] + lb[g]
        for g, d in T.Parallel(group, 64):
            o[g, d] = o[g, d] * alpha[g]
        T.gemm(p, vb, o, clear_accum=False)
    for g, d in T.Parallel(group, 64):
        mix[0, out0 + g * 64 + d] = o[g, d] / l[g]


def make_decode(c: Config, shapes: dict, max_pos: int, max_tokens: int):
    """The decode kernel for config c and the bundle's tensor `shapes`."""
    D, F, V, HD = c.dim, c.ff, c.vocab, c.head_dim
    H, KVH = c.heads, c.kv_heads
    conv_layers = [i for i in range(c.layers) if not c.is_attention(i)]
    attn_layers = list(c.attention_layers)
    Lc, La, L = len(conv_layers), len(attn_layers), c.layers
    assert HD == 64 and D % (NW * BLOCK) == 0 and F % (NW * BLOCK) == 0 and V % (NW * HEAD_BLOCK) == 0
    assert H % NW == 0 and KVH % NW == 0 and (KVH // NW * HD) % BLOCK == 0
    DW, FW, VW = D // NW, F // NW, V // NW  # a worker's share
    QW, KW = H // NW * HD, KVH // NW * HD  # its query and key rows
    group = H // KVH
    eps = c.eps
    s_in, s_qkv, s_out, s_13, s_2, s_head = (shapes[n] for n in ("conv.in_proj", "attn.qkv", "out_proj", "ffn.w13", "ffn.w2", "head"))

    def scale_shape(w, block, group_):
        L_, S_, N_, C_ = w
        return (L_, S_, N_ // block, C_ // group_, block)

    @T.prim_func
    def decode(
        in_w: T.Tensor(s_in, "int4"),
        in_s: T.Tensor(scale_shape(s_in, BLOCK, 64), "float16"),
        qkv_w: T.Tensor(s_qkv, "int4"),
        qkv_s: T.Tensor(scale_shape(s_qkv, BLOCK, 64), "float16"),
        out_w: T.Tensor(s_out, "int4"),
        out_s: T.Tensor(scale_shape(s_out, BLOCK, 64), "float16"),
        w13_w: T.Tensor(s_13, "int4"),
        w13_s: T.Tensor(scale_shape(s_13, BLOCK, 64), "float16"),
        w2_w: T.Tensor(s_2, "int4"),
        w2_s: T.Tensor(scale_shape(s_2, BLOCK, 64), "float16"),
        head_w: T.Tensor(s_head, "int8"),
        head_s: T.Tensor(scale_shape(s_head, HEAD_BLOCK, D), "float16"),
        embed: T.Tensor((V, D), "int8"),
        embed_s: T.Tensor((V,), "float32"),
        op_norm: T.Tensor((L, D), "float32"),
        ffn_norm: T.Tensor((L, D), "float32"),
        q_norm: T.Tensor((La, HD), "float32"),
        k_norm: T.Tensor((La, HD), "float32"),
        conv_w: T.Tensor((Lc, 3, D), "float32"),
        final_norm: T.Tensor((D,), "float32"),
        rope_cos: T.Tensor((max_pos, HD // 2), "float32"),
        rope_sin: T.Tensor((max_pos, HD // 2), "float32"),
        kv_k: T.Tensor((La, KVH, max_pos, HD), "float32"),
        kv_v: T.Tensor((La, KVH, max_pos, HD), "float32"),
        conv_state: T.Tensor((Lc, 2, D), "float32"),
        x: T.Tensor((1, D), "float32"),
        mix: T.Tensor((1, D), "float32"),
        ff: T.Tensor((1, F), "float32"),
        cand: T.Tensor((NW, 2), "float32"),
        tokens: T.Tensor((max_tokens,), "int32"),
        logits: T.Tensor((1, V), "float32"),
        pos0: T.int32,
        n: T.int32,
        n_forced: T.int32,
    ):
        with T.Kernel(NW, num_workers=NW) as w:
            p_in = Projection(in_w, in_s, 64, BLOCK)
            p_qkv = Projection(qkv_w, qkv_s, 64, BLOCK)
            p_out = Projection(out_w, out_s, 64, BLOCK)
            p_13 = Projection(w13_w, w13_s, 64, BLOCK)
            p_2 = Projection(w2_w, w2_s, 64, BLOCK)
            p_head = Projection(head_w, head_s, D, HEAD_BLOCK, bits=8)

            xf = T.alloc_fragment((1, D), "float32")
            hf = T.alloc_fragment((1, D), "float32")
            sq = T.alloc_fragment((1, D), "float32")
            xq = T.alloc_fragment((1, D), "int16")
            ffx = T.alloc_fragment((1, F), "float32")
            ffq = T.alloc_fragment((1, F), "int16")
            s1 = T.alloc_fragment((1,), "float32")
            sa = T.alloc_fragment((1,), "float32")
            amax = T.alloc_fragment((1,), "float32")
            acc = T.alloc_fragment((1, BLOCK), "float32")
            acc_h = T.alloc_fragment((1, HEAD_BLOCK), "float32")
            ga = T.alloc_fragment((1, BLOCK), "float32")
            erow = T.alloc_fragment((1, D), "int8")
            bcx = T.alloc_shared((3, BLOCK), "float32")
            qkv = T.alloc_shared((1, QW + 2 * KW), "float32")
            t64 = T.alloc_shared((1, HD), "float32")
            sq64 = T.alloc_fragment((1, HD), "float32")
            qg = T.alloc_fragment((group, HD), "float32")
            kb = T.alloc_fragment((POS_BLOCK, HD), "float32")
            vb = T.alloc_fragment((POS_BLOCK, HD), "float32")
            sc = T.alloc_fragment((group, POS_BLOCK), "float32")
            pr = T.alloc_fragment((group, POS_BLOCK), "float32")
            o = T.alloc_fragment((group, HD), "float32")
            m = T.alloc_fragment((group,), "float32")
            mb = T.alloc_fragment((group,), "float32")
            ls = T.alloc_fragment((group,), "float32")
            lb = T.alloc_fragment((group,), "float32")
            alpha = T.alloc_fragment((group,), "float32")
            logit = T.alloc_fragment((1, VW), "float32")
            idx = T.alloc_fragment((1, VW), "float32")
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
                    phases += [
                        Phase.rows(p_qkv, ai, w * QW, QW // BLOCK),
                        Phase.rows(p_qkv, ai, D + w * KW, KW // BLOCK),
                        Phase.rows(p_qkv, ai, D + KVH * HD + w * KW, KW // BLOCK),
                    ]
                    ai += 1
                else:
                    # B, C and x of each of the worker's channel blocks.
                    phases.append(Phase(p_in, ci, 3 * DW // BLOCK, lambda b: b % 3 * D + w * DW + b // 3 * BLOCK))
                    ci += 1
                phases.append(Phase.rows(p_out, layer, w * DW, DW // BLOCK))
                phases.append(Phase(p_13, layer, 2 * FW // BLOCK, lambda b: b % 2 * F + w * FW + b // 2 * BLOCK))
                phases.append(Phase.rows(p_2, layer, w * DW, DW // BLOCK))
            phases.append(Phase.rows(p_head, 0, w * VW, VW // HEAD_BLOCK))
            st = Stream(phases, DEPTH)
            T.annotate_layout(st.layouts())
            st.prologue()

            for t in T.serial(n):
                pos = pos0 + t
                st.begin(t)

                # The token's embedding: every worker the whole row, its slice
                # of the residual.
                token = T.if_then_else(t < n_forced, tokens[t], nxt[0])
                T.copy(embed[token : token + 1, :], erow)
                for i, k in T.Parallel(1, D):
                    xf[i, k] = T.Cast("float32", erow[i, k]) * embed_s[token]
                for k in T.Parallel(DW):
                    x[0, w * DW + k] = xf[0, w * DW + k]

                i_ph = 0
                ci = ai = 0
                for layer in list(range(L)):
                    if layer > 0:
                        T.copy(x, xf)
                    rms_norm(xf, lambda k, layer=layer: op_norm[layer, k], hf, sq, s1, D, eps)
                    quantize(hf, xq, sa, amax, D)
                    if c.is_attention(layer):
                        for part, off in ((0, 0), (1, QW), (2, QW + KW)):
                            st.run(i_ph + part, lambda b, rw, rs, k, off=off: store_block(b, rw, rs, k, xq, sa, acc, D, qkv, off))
                        i_ph += 3
                        for h in list(range(H // NW)):
                            norm_rope(qkv, h * HD, lambda d, ai=ai: q_norm[ai, d], rope_cos, rope_sin, pos, t64, sq64, s1, eps)
                        for h in list(range(KVH // NW)):
                            norm_rope(qkv, QW + h * HD, lambda d, ai=ai: k_norm[ai, d], rope_cos, rope_sin, pos, t64, sq64, s1, eps)
                            for d in T.Parallel(HD):
                                kv_k[ai, w * (KVH // NW) + h, pos, d] = qkv[0, QW + h * HD + d]
                                kv_v[ai, w * (KVH // NW) + h, pos, d] = qkv[0, QW + KW + h * HD + d]
                        for h in list(range(KVH // NW)):
                            attend(
                                qkv,
                                h * group * HD,
                                kv_k,
                                kv_v,
                                ai,
                                w * (KVH // NW) + h,
                                pos,
                                mix,
                                w * QW + h * group * HD,
                                group,
                                qg,
                                kb,
                                vb,
                                sc,
                                pr,
                                o,
                                m,
                                mb,
                                ls,
                                lb,
                                alpha,
                            )
                        ai += 1
                    else:
                        st.run(
                            i_ph,
                            lambda b, rw, rs, k, ci=ci: conv_block(b, rw, rs, k, xq, sa, acc, D, bcx, conv_w, conv_state, mix, ci, w * DW),
                        )
                        i_ph += 1
                        ci += 1
                    T.sync_grid()
                    # The operator's output projection, into the residual.
                    T.copy(mix, hf)
                    quantize(hf, xq, sa, amax, D)
                    st.run(i_ph, lambda b, rw, rs, k: residual_block(b, rw, rs, k, xq, sa, acc, D, 1, x, w * DW))
                    i_ph += 1
                    T.sync_grid()
                    # The FFN.
                    T.copy(x, xf)
                    rms_norm(xf, lambda k, layer=layer: ffn_norm[layer, k], hf, sq, s1, D, eps)
                    quantize(hf, xq, sa, amax, D)
                    st.run(i_ph, lambda b, rw, rs, k: glu_block(b, rw, rs, k, xq, sa, acc, D, ga, ff, w * FW))
                    i_ph += 1
                    T.sync_grid()
                    T.copy(ff, ffx)
                    quantize(ffx, ffq, sa, amax, F)
                    S2 = F // p_2.cols
                    st.run(i_ph, lambda b, rw, rs, k, S2=S2: residual_block(b, rw, rs, k, ffq, sa, acc, p_2.cols, S2, x, w * DW))
                    i_ph += 1
                    T.sync_grid()

                # The LM head and the worker's argmax.
                T.copy(x, xf)
                rms_norm(xf, lambda k: final_norm[k], hf, sq, s1, D, eps)
                quantize(hf, xq, sa, amax, D)
                st.run(i_ph, lambda b, rw, rs, k: head_block(b, rw, rs, k, xq, sa, acc_h, D, logit))
                T.reduce_max(logit, best, dim=1)
                for i, j in T.Parallel(1, VW):
                    idx[i, j] = T.if_then_else(logit[i, j] == best[0], T.Cast("float32", j), T.float32(3.0e38))
                T.reduce_min(idx, besti, dim=1)
                cand[w, 0] = best[0]
                cand[w, 1] = besti[0] + T.Cast("float32", w * VW)
                for j in T.Parallel(VW):
                    logits[0, w * VW + j] = logit[0, j]
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

    return decode


# The kernels' parameters from the bundle, in order.
PREFILL_ARGS = (
    "conv.in_proj", "conv.in_proj.scale", "attn.qkv", "attn.qkv.scale", "out_proj", "out_proj.scale",
    "ffn.w13", "ffn.w13.scale", "ffn.w2", "ffn.w2.scale", "embed", "embed.scale",
    "op_norm", "ffn_norm", "q_norm", "k_norm", "conv",
)  # fmt: skip
BUNDLE_ARGS = (
    "conv.in_proj", "conv.in_proj.scale", "attn.qkv", "attn.qkv.scale", "out_proj", "out_proj.scale",
    "ffn.w13", "ffn.w13.scale", "ffn.w2", "ffn.w2.scale", "head", "head.scale", "embed", "embed.scale",
    "op_norm", "ffn_norm", "q_norm", "k_norm", "conv", "final_norm",
)  # fmt: skip


def rope_tables(c: Config, max_pos: int) -> tuple:
    """cos and sin [max_pos, head_dim / 2] of RoPE's angles, as model.rope computes them."""
    d = c.head_dim
    inv = 1.0 / (c.rope_theta ** (torch.arange(0, d, 2, dtype=torch.float64) / d))
    angles = torch.arange(max_pos, dtype=torch.float64)[:, None] * inv[None, :]
    return angles.cos().float().contiguous(), angles.sin().float().contiguous()


class Decoder:
    """LFM2 on the device, in one session: the prefill and the decode kernels,
    with the bundle, the KV cache and the conv state resident across calls.

    The bundle is read from `device_path` on the phone when given (push it there
    once), else written from the host. `prefill_rows` is the prefill's chunk
    (None: prompts run through the decode, a token a step)."""

    def __init__(
        self, bundle_path: str, device_path: str | None = None, max_pos: int = 4096, max_tokens: int = 4096, prefill_rows: int | None = 128
    ):
        from tilelang.hexagon.runtime.session import Session

        header = bundle.read_header(bundle_path)
        c = Config(**{k: tuple(v) if isinstance(v, list) else v for k, v in header["config"].items()})
        self.c, self.header, self.max_pos, self.max_tokens = c, header, max_pos, max_tokens
        self.prefill_rows = prefill_rows
        assert max_pos % POS_BLOCK == 0
        shapes = {k: tuple(v["shape"]) for k, v in header["tensors"].items()}
        kernels = [(make_decode(c, shapes, max_pos, max_tokens), [])]
        if prefill_rows:
            kernels.append((make_prefill(c, shapes, prefill_rows, max_pos, max_tokens), []))
        self.session = Session(kernels)
        size = os.path.getsize(bundle_path)
        self.weights = self.session.buffer(size)
        if device_path is not None:
            self.weights.load(device_path)
        else:
            with open(bundle_path, "rb") as f:
                self.weights.write(f.read())
        start = header["data_offset"]
        tensors = header["tensors"]
        self.args = [self.weights.at(start + tensors[n]["offset"], tensors[n]["nbytes"]) for n in BUNDLE_ARGS]
        self.prefill_args = [self.weights.at(start + tensors[n]["offset"], tensors[n]["nbytes"]) for n in PREFILL_ARGS]
        cos, sin = rope_tables(c, max_pos)
        La, Lc = len(c.attention_layers), c.layers - len(c.attention_layers)
        kv = La * c.kv_heads * max_pos * c.head_dim * 4

        def resident(data=None, nbytes=None):
            buf = self.session.buffer(nbytes if data is None else data.numel() * data.element_size())
            if data is not None:
                buf.write(data)
            return buf

        self.rope = [resident(cos), resident(sin)]
        self.kv_k, self.kv_v = resident(nbytes=kv), resident(nbytes=kv)
        self.conv_state = resident(nbytes=Lc * 2 * c.dim * 4)
        self.scratch = [resident(nbytes=c.dim * 4), resident(nbytes=c.dim * 4), resident(nbytes=c.ff * 4), resident(nbytes=NW * 2 * 4)]
        self.tokens = resident(nbytes=max_tokens * 4)
        # The prefill's table of layers: whether each is attention, and its
        # index among the layers of its kind.
        attn = [int(c.is_attention(i)) for i in range(c.layers)]
        kinds = [[a, sum(attn[:i]) if a else i - sum(attn[:i])] for i, a in enumerate(attn)]
        self.kinds = resident(torch.tensor(kinds, dtype=torch.int32))
        self.logits = resident(nbytes=c.vocab * 4)
        # A step writes its position of the KV cache before it attends, and the
        # positions past it are masked: the cache needs zeros only once, so that
        # no position ever holds a NaN.
        for buf in (self.kv_k, self.kv_v):
            buf.write(bytes(buf.nbytes))
        self.reset()

    def reset(self) -> None:
        """Forget the context: the conv state starts from zeros."""
        self.conv_state.write(bytes(self.conv_state.nbytes))

    def run(self, tokens: list, pos0: int, n: int, n_forced: int) -> tuple:
        """Steps pos0 .. pos0 + n - 1 from tokens[0 .. n_forced): the tokens of all
        n + 1 positions, and the last step's logits."""
        assert 1 <= n_forced <= n + 1 <= self.max_tokens and pos0 + n <= self.max_pos
        self.tokens.write(torch.tensor(list(tokens[:n_forced]), dtype=torch.int32))
        self.session.kernels[0](
            *self.args, *self.rope, self.kv_k, self.kv_v, self.conv_state, *self.scratch, self.tokens, self.logits, pos0, n, n_forced
        )
        out = torch.frombuffer(bytearray(self.tokens.read((n + 1) * 4)), dtype=torch.int32).tolist()
        logits = torch.frombuffer(bytearray(self.logits.read()), dtype=torch.float32)
        return out, logits

    def prefill(self, tokens: list, pos0: int = 0) -> None:
        """Positions pos0 .. pos0 + len(tokens) - 1 from `tokens`: their KV cache
        entries and the conv state they leave, in chunks of prefill_rows."""
        n, R = len(tokens), self.prefill_rows
        assert R and 1 <= n <= self.max_tokens and pos0 + -(-n // R) * R <= self.max_pos
        self.tokens.write(torch.tensor(list(tokens), dtype=torch.int32))
        self.session.kernels[1](*self.prefill_args, *self.rope, self.kv_k, self.kv_v, self.conv_state, self.kinds, self.tokens, pos0, n)

    def generate(self, prompt: list, n_new: int) -> list:
        """Greedy: the `n_new` tokens after `prompt`, from an empty context. The
        prefill runs the prompt but its last token, which the decode's first
        step runs and the LM head reads."""
        self.reset()
        if self.prefill_rows and len(prompt) > 1:
            self.prefill(prompt[:-1])
            tokens, _ = self.run(prompt[-1:], len(prompt) - 1, n_new, 1)
            return tokens[1:]
        tokens, _ = self.run(prompt, 0, len(prompt) + n_new - 1, len(prompt))
        return tokens[len(prompt) :]

    def close(self) -> None:
        self.session.close()
