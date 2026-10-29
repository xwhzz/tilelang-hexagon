"""LFM2 prefill on Hexagon: a prompt in chunks of R rows on HMX, in one call.

The kernel shares the decode's bundle, KV cache and conv state (``Decoder``),
and runs on the decode's NW worker programs, which meet at ``T.sync_grid()``
between the phases of a layer. A chunk's activations stay in VTCM the
workers share (``global.vtcm``):

- its residual x;
- the normalized input a that a layer's GEMMs multiply;
- the operator's output, or a part of the FFN's hidden units.

The layers are one loop at run time, which reads each layer's kind and its
index among the layers of that kind from a table (``kinds``), so the code does
not grow with the depth. Each worker multiplies the decode's share of every
projection's output channels, its blocks streamed by DMA through one ring
(``Stream``): a layer's operator (the short conv or attention), then its
output projection and its FFN. Each block is one ``T.gemm_blockscaled`` on HMX
by the whole chunk: HVX dequantizes the 4-bit weights once for all R rows, and
the workers take turns at HMX, each dequantizing and running its epilogues
while another multiplies. The epilogues fuse the rest:

- the short conv's gates and its causal conv over time;
- SwiGLU;
- q/k norm and RoPE;
- the residual adds, into x itself.

Attention runs per KV head, against the KV cache in blocks of 128 positions
with an online softmax, on HMX: a block packs once for the four query heads
the KV head serves. The chunk's keys and values go into the cache first.

The FFN's hidden units run in parts, one per slice of w2 (the bundle stores
w2 in slices of 2048 columns, so 8192 hidden units are four parts): a part's
w1/w3 blocks fill the shared hidden buffer, and its w2 slice adds the part's
share into x.

The prompt's last token is left to the decode, whose first step runs it and
the LM head (``Decoder.generate``).
"""

import os
import sys
from types import SimpleNamespace

from tilelang.hexagon import language as T

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import Config  # noqa: E402
from stream import Phase, Projection, Stream  # noqa: E402

NW = 4  # workers, as the decode's
DEPTH = 2  # blocks in flight per worker: a block multiplies far longer than it streams
BLOCK = 128  # rows per block of a 4-bit projection
KEYS = 128  # KV positions per attention step
NEG = -3.0e38


@T.macro
def gemm(A, K: int, rw, rs, k, C, ones, clear):
    """C (+)= A @ (block k of the ring)^T on HMX, A [R, K] fp16."""
    T.gemm_blockscaled(
        A, rw[k, :, :], C, ones, rs[k, :, :], transpose_B=True, clear_accum=clear, k_start=0, sf_a_granularity_k=K, sf_b_granularity_k=64
    )


def make_prefill(c: Config, shapes: dict, R: int, max_pos: int, max_tokens: int):
    """The prefill kernel for config c, the bundle's tensor `shapes`, and
    chunks of R rows."""
    D, F, HD = c.dim, c.ff, c.head_dim
    H, KVH = c.heads, c.kv_heads
    conv_layers = [i for i in range(c.layers) if not c.is_attention(i)]
    attn_layers = list(c.attention_layers)
    Lc, La, L = len(conv_layers), len(attn_layers), c.layers
    group = H // KVH
    DW = D // NW  # a worker's channels of the residual
    QW, KW = H // NW * HD, KVH // NW * HD  # its query and key rows
    s_in, s_qkv, s_out, s_13, s_2 = (shapes[n] for n in ("conv.in_proj", "attn.qkv", "out_proj", "ffn.w13", "ffn.w2"))
    # The FFN's parts: w2's slices.
    PARTS, FQ = s_2[1], s_2[3]  # parts, hidden units of a part
    FQW = FQ // NW  # a worker's of a part
    RW = R // NW  # a worker's rows of a chunk, for the norms and the embedding
    assert HD == 64 and R % (2 * NW) == 0 and R % 32 == 0
    assert D % (NW * BLOCK) == 0 and FQ % (NW * BLOCK) == 0 and (KVH // NW * HD) % BLOCK == 0
    assert PARTS * FQ == F
    eps = c.eps
    HID = max(D, FQ)

    def clamped(index, count: int):
        return T.max(T.min(index, max(count, 1) - 1), 0)

    def scale_shape(w, block, group_):
        L_, S_, N_, C_ = w
        return (L_, S_, N_ // block, C_ // group_, block)

    @T.macro
    def normalize(bf, weight):
        """a = x * rsqrt(mean(x^2) + eps) * weight[k], the worker's rows, two
        at a time. Each row's factor is computed once, so that the scaling
        broadcasts it from a register; the weights are read from VTCM."""
        T.copy(weight, bf.nw)
        for r2 in T.serial(RW // 2):
            r = bf.w * RW + r2 * 2
            for i, k in T.Parallel(2, D):
                bf.sq[i, k] = bf.x[r + i, k] * bf.x[r + i, k]
            T.reduce_sum(bf.sq, bf.ss, dim=1)
            for i in T.Parallel(2):
                bf.ss[i] = T.rsqrt(bf.ss[i] * T.float32(1 / D) + T.float32(eps))
            for i, k in T.Parallel(2, D):
                bf.rows[i, k] = bf.x[r + i, k] * bf.ss[i] * bf.nw[k]
            T.copy(bf.rows, bf.a[r : r + 2, :])

    @T.macro
    def norm_rope(bf, col, weight, base, extra):
        """hsq = RoPE(RMSNorm(cc[0, :, col : col + 64]) * weight(d)) * extra, at
        positions base + i."""
        for i, d in T.Parallel(R, HD):
            bf.hsq[i, d] = bf.cc[0, i, col + d] * bf.cc[0, i, col + d]
        T.reduce_sum(bf.hsq, bf.hss, dim=1)
        for i in T.Parallel(R):
            bf.hss[i] = T.rsqrt(bf.hss[i] * T.float32(1 / HD) + T.float32(eps))
        for i, d in T.Parallel(R, HD):
            bf.head[i, d] = bf.cc[0, i, col + d] * bf.hss[i] * weight(d)
        for i, d in T.Parallel(R, HD):
            sign = T.if_then_else(d < 32, T.float32(-1), T.float32(1))
            bf.hsq[i, d] = (
                bf.head[i, d] * bf.rope_cos[base + i, d % 32] + sign * bf.head[i, (d + 32) % 64] * bf.rope_sin[base + i, d % 32]
            ) * extra

    @T.macro
    def in_proj_block(bf, b, rw, rs, k, ci, base, valid):
        """Part b % 3 (B, x, C) of channel block b // 3: after x, B * x and
        its causal conv over time; after C, the gated output. B and C go to
        slot 0, x and the conv to slot 1."""
        ch = bf.w * DW + b // 3 * BLOCK
        gemm(bf.a, D, rw, rs, k, bf.cc[b % 3 % 2, :, :], bf.ones, True)
        if b % 3 == 1:
            for i, j in T.Parallel(R, BLOCK):
                bf.cc[0, i, j] = bf.cc[0, i, j] * bf.cc[1, i, j]
            # y[t] = w0 bx[t - 2] + w1 bx[t - 1] + w2 bx[t], the state
            # before the chunk's first row.
            for i, j in T.Parallel(R, BLOCK):
                b2 = T.if_then_else(i >= 2, bf.cc[0, T.max(i - 2, 0), j], bf.conv_state[ci, T.min(i, 1), ch + j])
                b1 = T.if_then_else(i >= 1, bf.cc[0, T.max(i - 1, 0), j], bf.conv_state[ci, 1, ch + j])
                bf.cc[1, i, j] = bf.conv_w[ci, 0, ch + j] * b2 + bf.conv_w[ci, 1, ch + j] * b1 + bf.conv_w[ci, 2, ch + j] * bf.cc[0, i, j]
            # The state the next chunk, or the decode, starts from: the
            # last two valid rows' B * x.
            for j in T.Parallel(BLOCK):
                s1 = bf.cc[0, valid - 1, j]
                s0 = T.if_then_else(valid >= 2, bf.cc[0, T.max(valid - 2, 0), j], bf.conv_state[ci, 1, ch + j])
                bf.conv_state[ci, 0, ch + j] = s0
                bf.conv_state[ci, 1, ch + j] = s1
        if b % 3 == 2:
            for i, j in T.Parallel(R, BLOCK):
                bf.cc[0, i, j] = bf.cc[0, i, j] * bf.cc[1, i, j]
            T.copy(bf.cc[0, :, :], bf.hid[:, ch : ch + BLOCK])

    @T.macro
    def q_block(bf, b, rw, rs, k, ai, base):
        """Query block b: two heads, normalized, rotated and scaled by
        1/8, into their KV head's stacked queries."""
        gemm(bf.a, D, rw, rs, k, bf.cc[0, :, :], bf.ones, True)
        for hh in T.serial(BLOCK // HD):
            local = b * (BLOCK // HD) + hh  # the worker's query head
            norm_rope(bf, hh * HD, lambda d, ai=ai: bf.q_norm[ai, d], base, T.float32(0.125))
            T.copy(bf.hsq, bf.q[local // group, (local % group) * R : (local % group + 1) * R, :])

    @T.macro
    def kv_block(bf, b, rw, rs, k, ai, base, values: bool):
        """Key (or value) block b: the worker's KV heads, keys normalized
        and rotated, into the KV cache at positions base + i."""
        gemm(bf.a, D, rw, rs, k, bf.cc[0, :, :], bf.ones, True)
        for hh in T.serial(BLOCK // HD):
            kvh = bf.w * (KVH // NW) + b * (BLOCK // HD) + hh
            if values:
                T.copy(bf.cc[0, :, hh * HD : (hh + 1) * HD], bf.kv_v[ai, kvh, base : base + R, :])
            else:
                norm_rope(bf, hh * HD, lambda d, ai=ai: bf.k_norm[ai, d], base, T.float32(1))
                T.copy(bf.hsq, bf.kv_k[ai, kvh, base : base + R, :])

    @T.macro
    def attention(bf, ai, base):
        """The worker's query heads, by KV head: softmax(q K^T) V over the
        cache's positions [0, base + i] for row i, into hid. A block of keys
        and values packs once for the KV head's query heads, each of which
        keeps its own online softmax. The scores and the outputs are fp32
        rows, where the causal mask, the softmax and the rescaling are row
        maps; P packs into fp16 tiles, and HMX adds P V into the outputs."""
        for g in T.serial(KVH // NW):
            kvh = bf.w * (KVH // NW) + g
            for hq, i, d in T.Parallel(group, R, HD):
                bf.acc_o[hq, i, d] = T.float32(0)
            for hq, i in T.Parallel(group, R):
                bf.m[hq, i] = T.float32(NEG)
                bf.ls[hq, i] = T.float32(0)
            for kb in T.serial(T.ceildiv(base + R, KEYS)):
                T.copy(bf.kv_k[ai, kvh, kb * KEYS : (kb + 1) * KEYS, :], bf.kblk)
                T.copy(bf.kv_v[ai, kvh, kb * KEYS : (kb + 1) * KEYS, :], bf.vblk)
                for hq in T.serial(group):
                    T.gemm(bf.q[g, hq * R : (hq + 1) * R, :], bf.kblk, bf.s, transpose_B=True, clear_accum=True)
                    # Causal: the chunk's own positions are visible to the
                    # rows at or after them.
                    if kb * KEYS + KEYS - 1 > base:
                        for i in T.serial(R):
                            for j in T.vectorized(KEYS):
                                bf.s[i, j] = T.if_then_else(kb * KEYS + j <= base + i, bf.s[i, j], T.float32(NEG))
                    T.reduce_max(bf.s, bf.rmax, dim=1)
                    for i in T.Parallel(R):
                        bf.rmax[i] = T.max(bf.m[hq, i], bf.rmax[i])
                        bf.alpha[i] = T.exp(bf.m[hq, i] - bf.rmax[i])
                        bf.m[hq, i] = bf.rmax[i]
                    # P in fp16, as HMX takes it: exp then takes 64 lanes a
                    # register; the row sums accumulate in fp32.
                    for i, j in T.Parallel(R, KEYS):
                        bf.ph[i, j] = T.exp(T.Cast("float16", bf.s[i, j] - bf.m[hq, i]))
                    T.reduce_sum(bf.ph, bf.rsum, dim=1)
                    for i in T.Parallel(R):
                        bf.ls[hq, i] = bf.ls[hq, i] * bf.alpha[i] + bf.rsum[i]
                    T.copy(bf.ph, bf.p)
                    for i, d in T.Parallel(R, HD):
                        bf.acc_o[hq, i, d] = bf.acc_o[hq, i, d] * bf.alpha[i]
                    T.gemm(bf.p, bf.vblk, bf.acc_o[hq, :, :], clear_accum=False)
            for hq in T.serial(group):
                for i in T.Parallel(R):
                    bf.alpha[i] = T.float32(1) / bf.ls[hq, i]
                for i, d in T.Parallel(R, HD):
                    bf.acc_o[hq, i, d] = bf.acc_o[hq, i, d] * bf.alpha[i]
                col = (bf.w * (H // NW) + g * group + hq) * HD
                T.copy(bf.acc_o[hq, :, :], bf.hid[:, col : col + HD])

    @T.macro
    def glu_block(bf, b, rw, rs, k):
        """w1 (even b) or w3 of hidden block b // 2 of the worker's part;
        after w3, silu(w1 a) * (w3 a) into hid."""
        gemm(bf.a, D, rw, rs, k, bf.cc[b % 2, :, :], bf.ones, True)
        if b % 2 == 1:
            # In fp16, which hid holds: the sigmoid then takes 64 lanes a
            # register.
            for i, j in T.Parallel(R, BLOCK):
                g = T.Cast("float16", bf.cc[0, i, j])
                bf.gl[i, j] = g * T.sigmoid(g) * T.Cast("float16", bf.cc[1, i, j])
            T.copy(bf.gl, bf.hid[:, bf.w * FQW + b // 2 * BLOCK : bf.w * FQW + b // 2 * BLOCK + BLOCK])

    @T.macro
    def residual_block(bf, b, rw, rs, k, cols: int):
        """x[:, channels] += hid[:, :cols] @ block^T: the residual add in the
        GEMM itself."""
        ch = bf.w * DW + b * BLOCK
        gemm(bf.hid[:, 0:cols], cols, rw, rs, k, bf.x[:, ch : ch + BLOCK], bf.ones, False)

    @T.prim_func
    def prefill(
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
        embed: T.Tensor((c.vocab, D), "int8"),
        embed_s: T.Tensor((c.vocab,), "float32"),
        op_norm: T.Tensor((L, D), "float32"),
        ffn_norm: T.Tensor((L, D), "float32"),
        q_norm: T.Tensor((La, HD), "float32"),
        k_norm: T.Tensor((La, HD), "float32"),
        conv_w: T.Tensor((Lc, 3, D), "float32"),
        rope_cos: T.Tensor((max_pos, HD // 2), "float32"),
        rope_sin: T.Tensor((max_pos, HD // 2), "float32"),
        kv_k: T.Tensor((La, KVH, max_pos, HD), "float32"),
        kv_v: T.Tensor((La, KVH, max_pos, HD), "float32"),
        conv_state: T.Tensor((Lc, 2, D), "float32"),
        kinds: T.Tensor((L, 2), "int32"),
        tokens: T.Tensor((max_tokens,), "int32"),
        pos0: T.int32,
        n: T.int32,
    ):
        # fp16-grade math (1e-4 relative): the activations it computes are
        # fp16 for HMX anyway.
        T.annotate_pass_configs({"tl.enable_fast_math": True})
        with T.Kernel(NW, num_workers=NW) as w:
            p_in = Projection(in_w, in_s, 64, BLOCK)
            p_qkv = Projection(qkv_w, qkv_s, 64, BLOCK)
            p_out = Projection(out_w, out_s, 64, BLOCK)
            p_13 = Projection(w13_w, w13_s, 64, BLOCK)
            p_2 = Projection(w2_w, w2_s, 64, BLOCK)

            # The chunk's activations, which every worker reads.
            x = T.alloc_global((R, D), "float32", scope="global.vtcm")
            a = T.alloc_global((R, D), "float16", scope="global.vtcm")
            hid = T.alloc_global((R, HID), "float16", scope="global.vtcm")

            # A worker's: the outputs of up to two blocks, and the rows it
            # normalizes or embeds two at a time.
            ones = T.alloc_shared((R, 1), "float32")
            cc = T.alloc_shared((2, R, BLOCK), "float32")
            gl = T.alloc_shared((R, BLOCK), "float16")
            rows = T.alloc_fragment((2, D), "float32")
            sq = T.alloc_fragment((2, D), "float32")
            ss = T.alloc_fragment((2,), "float32")
            nw = T.alloc_shared((D,), "float32")
            erow = T.alloc_fragment((1, D), "int8")
            # q/k norms: one head of the block's rows at a time.
            head = T.alloc_shared((R, HD), "float32")
            hsq = T.alloc_shared((R, HD), "float32")
            hss = T.alloc_shared((R,), "float32")
            # Attention: the worker's queries, by KV head; a block of keys and
            # values; the KV head's query heads' softmax states and outputs;
            # and one query head's scores.
            q = T.alloc_shared((KVH // NW, group * R, HD), "float16")
            kblk = T.alloc_shared((KEYS, HD), "float16")
            vblk = T.alloc_shared((KEYS, HD), "float16")
            acc_o = T.alloc_shared((group, R, HD), "float32")
            m = T.alloc_shared((group, R), "float32")
            ls = T.alloc_shared((group, R), "float32")
            s = T.alloc_shared((R, KEYS), "float32")
            ph = T.alloc_shared((R, KEYS), "float16")
            p = T.alloc_shared((R, KEYS), "float16")
            rmax = T.alloc_shared((R,), "float32")
            rsum = T.alloc_shared((R,), "float32")
            alpha = T.alloc_shared((R,), "float32")

            for i in T.Parallel(R):
                ones[i, 0] = T.float32(1)
            # What the macros below work on.
            bf = SimpleNamespace(
                w=w, x=x, a=a, hid=hid, ones=ones, cc=cc, rows=rows, sq=sq, ss=ss, nw=nw, head=head, hsq=hsq, hss=hss,
                q=q, kblk=kblk, vblk=vblk, s=s, ph=ph, p=p, acc_o=acc_o, m=m, ls=ls, rmax=rmax, rsum=rsum, alpha=alpha, gl=gl,
                rope_cos=rope_cos, rope_sin=rope_sin, conv_state=conv_state, conv_w=conv_w, q_norm=q_norm, k_norm=k_norm,
                kv_k=kv_k, kv_v=kv_v,
            )  # fmt: skip

            # Every projection streams blocks of one shape through one ring.
            rings = {}
            projections = (p_in, p_qkv, p_out, p_13, p_2)
            layouts = Stream([Phase(p, 0, 1, lambda b: 0) for p in projections], DEPTH, rings=rings).layouts()
            T.annotate_layout(layouts)

            for chunk in T.serial(T.ceildiv(n, R)):
                base = pos0 + chunk * R
                valid = T.min(R, n - chunk * R)

                # The chunk's embeddings: the worker's rows of x.
                for r in T.serial(RW):
                    t = chunk * R + w * RW + r
                    token = T.if_then_else(t < n, tokens[T.min(t, n - 1)], 0)
                    T.copy(embed[token : token + 1, :], erow)
                    for k in T.Parallel(D):
                        x[w * RW + r, k] = T.Cast("float32", erow[0, k]) * embed_s[token]
                T.sync_grid()

                for layer in T.serial(L):
                    # The layer's index among the conv (ci) or the attention
                    # (ai) layers, clamped so that every weight block it reads
                    # is provably in its tensor.
                    ci = clamped(kinds[layer, 1], Lc)
                    ai = clamped(kinds[layer, 1], La)
                    # The layer's phases, worker w's blocks of each, in order:
                    # its operator's (the short conv's or attention's), then
                    # the output projection and the FFN's, each part's w1/w3
                    # blocks and then its w2 slice. The streams run one after
                    # another.
                    st_conv = Stream(
                        [Phase(p_in, ci, 3 * DW // BLOCK, lambda b: (b % 3) * 2 % 3 * D + w * DW + b // 3 * BLOCK)], DEPTH, rings=rings
                    )
                    st_attn = Stream(
                        [
                            Phase.rows(p_qkv, ai, w * QW, QW // BLOCK),
                            Phase.rows(p_qkv, ai, D + w * KW, KW // BLOCK),
                            Phase.rows(p_qkv, ai, D + KVH * HD + w * KW, KW // BLOCK),
                        ],
                        DEPTH,
                        rings=rings,
                    )
                    rest = [Phase.rows(p_out, layer, w * DW, DW // BLOCK)]
                    for part in list(range(PARTS)):
                        rest.append(
                            Phase(p_13, layer, 2 * FQW // BLOCK, lambda b, part=part: b % 2 * F + part * FQ + w * FQW + b // 2 * BLOCK)
                        )
                        rest.append(Phase(p_2, layer, DW // BLOCK, lambda b: w * DW + b * BLOCK, lambda b, part=part: part))
                    st_rest = Stream(rest, DEPTH, rings=rings)
                    normalize(bf, op_norm[layer, :])
                    T.sync_grid()
                    if kinds[layer, 0] == 1:
                        st_attn.prologue()
                        st_attn.run(0, lambda b, rw, rs, k, ai=ai, base=base: q_block(bf, b, rw, rs, k, ai, base))
                        st_attn.run(1, lambda b, rw, rs, k, ai=ai, base=base: kv_block(bf, b, rw, rs, k, ai, base, False))
                        st_attn.run(2, lambda b, rw, rs, k, ai=ai, base=base: kv_block(bf, b, rw, rs, k, ai, base, True))
                        attention(bf, ai, base)
                    else:
                        st_conv.prologue()
                        st_conv.run(0, lambda b, rw, rs, k, ci=ci, base=base, valid=valid: in_proj_block(bf, b, rw, rs, k, ci, base, valid))
                    T.sync_grid()
                    st_rest.prologue()
                    st_rest.run(0, lambda b, rw, rs, k: residual_block(bf, b, rw, rs, k, D))
                    T.sync_grid()
                    normalize(bf, ffn_norm[layer, :])
                    T.sync_grid()
                    for part in list(range(PARTS)):
                        st_rest.run(1 + 2 * part, lambda b, rw, rs, k: glu_block(bf, b, rw, rs, k))
                        T.sync_grid()
                        st_rest.run(2 + 2 * part, lambda b, rw, rs, k: residual_block(bf, b, rw, rs, k, FQ))
                        T.sync_grid()

    return prefill
