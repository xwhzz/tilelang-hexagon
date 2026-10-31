"""The fp16 prefill on Hexagon: a prompt of LFM2 or LFM3 in chunks of R rows on
HMX, in one call, from the fp16 bundle (convert.py).

The kernel shares the decode's bundle, KV cache and conv state
(``decode_fp16``, ``Decoder``) and runs on its NW worker programs, which meet
at ``T.sync_grid()`` between the phases of a layer. Each worker streams the
decode's share of every projection's output channels by DMA, a panel of 32 a
block (``Weights``), and multiplies each panel by the whole chunk on HMX: the
bundle's fp16 tiles are HMX's weight operand as they are. The workers take
turns at HMX, each running its epilogues on HVX while another multiplies.

A chunk's activations stay in VTCM the workers share (``global.vtcm``):

- the normalized input a and hid, the operator's output or a part of the
  FFN's hidden units, which HMX reads;
- each branch's output f, in fp32, into which the output projections' GEMMs
  write and the FFN's parts add, and which LFM3's sandwich norm normalizes
  over its rows before the residual add.

The residual x, in fp32, is in DDR, a worker's rows moving by DMA alone, so
that its transfers need no cache maintenance (``T.annotate_dma_only``). The
queries are in DDR too, which attention reads a block of rows at a time.

The layers are one loop at run time, which reads each layer's kind, its index
among the layers of that kind and its attention's RoPE table from a table
(``kinds``), so the code does not grow with the depth. The epilogues fuse the
rest: the short conv's gates and its causal conv over time, q/k norms and
RoPE, the KV cache's writes, the head gates and SwiGLU.

Attention runs per query head, in blocks of ROWS rows against the KV cache in
blocks of KEYS positions with an online softmax in base 2, on HMX; the cache
keeps HMX's tiles, so a block of it comes into HMX's operands by DMA. The FFN's
hidden units run in parts of FQ (of at most 64 tiles): a part's w1/w3 panels
fill hid, and its w2 slab adds the part's product into f.

The prompt's last token is left to the decode, whose first step runs it and
the LM head (``Decoder.generate``).
"""

import os
import sys
from types import SimpleNamespace

from tilelang.hexagon import language as T
from tilelang.hexagon.intrinsics.crouton import make_hmx_activation_layout

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convert import TILE, qkv_group  # noqa: E402
from decode_fp16 import NW, kv_layouts, slice_cols  # noqa: E402
from model import SLIDING, Config  # noqa: E402
from stream import Phase, Stream, Weights  # noqa: E402

DEPTH = 2  # blocks in flight per worker: a block multiplies longer than it streams
RB = 4  # rows a worker's row phases (residual adds, norms) move at once
ROWS = 64  # rows of an attention block
KEYS = 128  # KV positions per attention step
NEG = -3.0e38
NEG16 = -65504.0  # the scores' mask, in fp16
LOG2E = 1.4426950408889634  # the queries' scale folds it in: 2^(s log2 e) = e^s


def kinds_table(c: Config) -> list:
    """Per layer: whether it is attention, its index among the layers of its
    kind, its RoPE table (decode_fp16.kinds) and its window (0: none)."""
    attn = [int(c.is_attention(i)) for i in range(c.layers)]
    tables = {k: i for i, k in enumerate(c.rope)}
    rows = []
    for i, a in enumerate(attn):
        kind = c.layer_types[i]
        rows.append([a, sum(attn[:i]) if a else i - sum(attn[:i]), tables.get(kind, 0), c.window if kind == SLIDING else 0])
    return rows


def make_prefill(c: Config, R: int, max_pos: int, max_tokens: int):
    """The fp16 prefill kernel of config c, in chunks of R rows."""
    D, F, HD = c.dim, c.ff, c.head_dim
    H, KVH, taps = c.heads, c.kv_heads, c.conv_taps
    conv_layers = [i for i in range(c.layers) if not c.is_attention(i)]
    attn_layers = list(c.attention_layers)
    Lc, La, L = len(conv_layers), len(attn_layers), c.layers
    g = qkv_group(c)
    group = H // KVH
    KH = KVH // NW  # a worker's KV heads
    DW = D // NW  # its channels of the residual
    QW = KH * g.width  # its columns of attn.qkv
    RW = R // NW  # its rows of a chunk, for the norms and the embedding
    FQ = slice_cols(F)  # hidden units of a part of the FFN
    PARTS = F // FQ
    FQW = FQ // NW  # a worker's of a part
    NK = len(c.rope)
    half = HD // 2
    assert R % ROWS == 0 and RW % (2 * RB) == 0 and max_pos % KEYS == 0 and FQW % TILE == 0
    assert D <= 64 * TILE, "a slot of the ring holds a projection's whole K"
    eps = c.eps
    HID = max(D, FQ)
    SLOT = max(D, FQ)  # K of the ring's slots: in_proj, qkv, out_proj and w13's D, w2's FQ

    def clamped(index, count: int):
        return T.max(T.min(index, max(count, 1) - 1), 0)

    @T.macro
    def gemm(bf, A, K: int, ring, k, C, clear):
        """C (+)= A @ (block k of the ring), A [R, K] fp16, on HMX."""
        T.gemm(A, ring[k, 0:K, :], C, clear_accum=clear)

    @T.macro
    def normalize(bf, s, squared: bool = False):
        """rows = xr[s] * rsqrt(mean(xr[s]^2) + eps) * nw[k], RB rows; sq
        already holds xr[s]^2 when `squared`."""
        if not squared:
            for i, k in T.Parallel(RB, D):
                bf.sq[i, k] = bf.xr[s, i, k] * bf.xr[s, i, k]
        T.reduce_sum(bf.sq, bf.ss, dim=1)
        for i in T.Parallel(RB):
            bf.ss[i] = T.rsqrt(bf.ss[i] * T.float32(1 / D) + T.float32(eps))
        for i, k in T.Parallel(RB, D):
            bf.rows[i, k] = bf.xr[s, i, k] * bf.ss[i] * bf.nw[k]

    @T.macro
    def embed(bf, chunk, n, tokens, head_rows, weight):
        """The chunk's embeddings, the worker's rows of x, RB at a time: each
        token's pairs from the head's tiles. Each batch goes to x by DMA, and
        is normalized by weight[k] into a, the first layer's input."""
        T.copy(weight, bf.nw)
        for rb in T.serial(RW // RB):
            r = bf.w * RW + rb * RB
            for i in T.serial(RB):
                t = chunk * R + r + i
                token = T.max(T.min(T.if_then_else(t < n, tokens[T.min(t, n - 1)], 0), c.vocab - 1), 0)
                T.dma_copy(
                    head_rows[token // TILE * (D // 2) : (token // TILE + 1) * (D // 2), token % TILE * 2 : token % TILE * 2 + 2], bf.erow
                )
                # The pairs are in, and the batch before is stored.
                T.dma_wait(0)
                for k in T.Parallel(D):
                    bf.xr[0, i, k] = T.Cast("float32", bf.erow[k // 2, k % 2])
            T.dma_copy(bf.xr[0, :, :], bf.x[r : r + RB, :])
            normalize(bf, 0)
            T.copy(bf.rows, bf.a[r : r + RB, :])
        T.dma_wait(0)

    @T.macro
    def add_branch(bf, post, sandwich: bool, weight, normed):
        """x += f, the branch's output, normalized over each row first by
        post[k] with sandwich norms; then, if `normed`, a = the new x
        normalized by weight[k] (the next GEMMs' input), in the same pass:
        the worker's rows, RB at a time, x by DMA."""
        if sandwich:
            T.copy(post, bf.pw)
            # The sandwich norm's scales of the worker's rows, as vectors: from
            # the workers' parts of the branch's rows' sums of squares, which
            # the GEMMs' epilogues made.
            for i in T.Parallel(RW):
                total = bf.psum[0, bf.w * RW + i]
                for v in list(range(1, NW)):
                    total = total + bf.psum[v, bf.w * RW + i]
                bf.sw[i] = T.rsqrt(total * T.float32(1 / D) + T.float32(eps))
        T.copy(weight, bf.nw)
        # Two slots of rows: a batch's rows come in while the one before is
        # added, stored and normalized. The batches go in pairs, so that each
        # loop over a slot's rows addresses it at a constant offset, which the
        # software pipeliner needs: with the slot a run-time offset, LLVM did
        # not pipeline the add, and it ran 2.3 times slower.
        nb = RW // RB
        T.dma_copy(bf.x[bf.w * RW : bf.w * RW + RB, :], bf.xr[0, :, :])
        for pair in T.serial(nb // 2):
            for s in list(range(2)):
                rb = pair * 2 + s
                r = bf.w * RW + rb * RB
                # The batch's rows are in, and the slot the next takes is
                # stored.
                T.dma_wait(0)
                if rb + 1 < nb:
                    nxt = bf.w * RW + T.min(rb + 1, nb - 1) * RB
                    T.dma_copy(bf.x[nxt : nxt + RB, :], bf.xr[1 - s, :, :])
                if sandwich:
                    # The new rows, and their squares for the next norm.
                    for i, k in T.Parallel(RB, D):
                        v = bf.xr[s, i, k] + bf.f[r + i, k] * bf.sw[rb * RB + i] * bf.pw[k]
                        bf.xr[s, i, k] = v
                        bf.sq[i, k] = v * v
                else:
                    for i, k in T.Parallel(RB, D):
                        v = bf.xr[s, i, k] + bf.f[r + i, k]
                        bf.xr[s, i, k] = v
                        bf.sq[i, k] = v * v
                T.dma_copy(bf.xr[s, :, :], bf.x[r : r + RB, :])
                if normed:
                    normalize(bf, s, True)
                    T.copy(bf.rows, bf.a[r : r + RB, :])
        T.dma_wait(0)

    @T.macro
    def head_sums(bf, c0):
        """The rows' sums of squares of the head in hd, a panel at a time: the
        one at column c0 just arrived. A panel's row of fp32 is a register,
        where its fp16 would be half of one."""
        for i, j in T.Parallel(R, TILE):
            bf.sq32[i, j] = bf.hd[i, c0 + j] * bf.hd[i, c0 + j]
        T.reduce_sum(bf.sq32, bf.hs32, dim=1)
        for i in T.Parallel(R):
            bf.hss[i] = T.if_then_else(c0 == 0, bf.hs32[i], bf.hss[i] + bf.hs32[i])

    @T.macro
    def norm_rope(bf, weight, table, extra):
        """rot = RoPE(RMSNorm(hd) * weight(d)) * extra at the chunk's
        positions, whose rows of RoPE table `table` are in VTCM; the rows'
        sums of squares are in hss."""
        for i in T.Parallel(R):
            bf.hss[i] = T.rsqrt(bf.hss[i] * T.float32(1 / HD) + T.float32(eps)) * extra
        for i, d in T.Parallel(R, HD):
            d2 = (d + half) % HD
            sign = T.if_then_else(d < half, T.float32(-1), T.float32(1))
            v = (
                bf.hd[i, d] * weight(d) * bf.cos_rows[table, i, d % half]
                + sign * bf.hd[i, d2] * weight(d2) * bf.sin_rows[table, i, d % half]
            )
            bf.rot[i, d] = T.Cast("float16", v * bf.hss[i])

    @T.macro
    def in_proj_block(bf, b, ring, k, ci, valid):
        """Part b % 3 (B, x, C, as the stream visits them) of channel panel
        b // 3: after x, B * x and its causal conv over time, from the state
        before the chunk; after C, the gated output into hid."""
        ch = bf.w * DW + b // 3 * TILE
        gemm(bf, bf.a, D, ring, k, bf.cc[b % 3 % 2, :, :], True)
        if b % 3 == 1:
            for i, j in T.Parallel(R, TILE):
                bf.cc[0, i, j] = bf.cc[0, i, j] * bf.cc[1, i, j]
            # y[t] = sum_j w_j bx[t - (taps - 1) + j], bx before the chunk
            # from the state.
            for i, j in T.Parallel(R, TILE):
                y = bf.conv_w[ci, taps - 1, ch + j] * bf.cc[0, i, j]
                for tap in list(range(taps - 1)):
                    back = taps - 1 - tap
                    prev = T.if_then_else(i >= back, bf.cc[0, T.max(i - back, 0), j], bf.conv_state[ci, T.min(i + tap, taps - 2), ch + j])
                    y = y + bf.conv_w[ci, tap, ch + j] * prev
                bf.cc[1, i, j] = y
            # The state the next chunk, or the decode, starts from: the last
            # taps - 1 valid rows' B * x, the state's own rows before them.
            for s, j in T.Parallel(taps - 1, TILE):
                src = valid - (taps - 1) + s
                bf.st[s, j] = T.if_then_else(
                    src >= 0, bf.cc[0, T.max(src, 0), j], bf.conv_state[ci, T.min(src + taps - 1, taps - 2), ch + j]
                )
            for s, j in T.Parallel(taps - 1, TILE):
                bf.conv_state[ci, s, ch + j] = bf.st[s, j]
        if b % 3 == 2:
            for i, j in T.Parallel(R, TILE):
                bf.cc[0, i, j] = bf.cc[0, i, j] * bf.cc[1, i, j]
            T.copy(bf.cc[0, :, :], bf.hid[:, ch : ch + TILE])

    @T.macro
    def qkv_block(bf, b, ring, k, ai, table, base):
        """Block b of the worker's attn.qkv columns: per KV head, 4 panels a
        head of queries, of the key and of the value, then (LFM3) the panel of
        gates. A head's last panel normalizes and rotates it (queries also
        scaled by log2(e) / sqrt(HD)) into the queries' buffer or the KV
        cache at positions base + i."""
        col = b % (g.width // TILE) * TILE  # the block's column in its KV head's group
        kvh = bf.w * KH + b // (g.width // TILE)
        part = col // HD  # its head among the group's queries, key, value
        if col < g.gate:
            gemm(bf, bf.a, D, ring, k, bf.hd[:, col % HD : col % HD + TILE], True)
            if part <= group:
                head_sums(bf, col % HD)
            if col % HD == HD - TILE:
                if part < group:
                    norm_rope(bf, lambda d, ai=ai: bf.q_norm[ai, d], table, T.float32(LOG2E * HD**-0.5))
                    T.copy(bf.rot, bf.qs[kvh, part, :, :])
                elif part == group:
                    norm_rope(bf, lambda d, ai=ai: bf.k_norm[ai, d], table, T.float32(1))
                    T.copy(bf.rot, bf.kv_k[ai, kvh, base : base + R, :])
                else:
                    T.copy(bf.hd, bf.kv_v[ai, kvh, base : base + R, :])
        else:
            gemm(bf, bf.a, D, ring, k, bf.cc[0, :, :], True)
            for i, q in T.Parallel(R, group):
                bf.gates[kvh % KH, q, i] = T.sigmoid(bf.cc[0, i, q])

    @T.macro
    def fetch(bf, ai, kvh, kb, slot):
        """Block kb of the KV head's keys and values into a slot of HMX's
        operands, by DMA, the cache's tiles as they are: clamped, so that the
        copies are provably in the cache, as a block past it is never used."""
        at = T.max(T.min(kb, max_pos // KEYS - 1), 0) * KEYS
        T.dma_copy(bf.kv_k[ai, kvh, at : at + KEYS, :], bf.kblk[slot, :, :])
        T.dma_copy(bf.kv_v[ai, kvh, at : at + KEYS, :], bf.vblk[slot, :, :])

    @T.macro
    def attention(bf, ai, base, window):
        """The worker's query heads, ROWS rows at a time: softmax(q K^T) V
        over the cache's positions [base + i - window + 1 (or 0), base + i]
        for row i, gated, into hid, in base 2 (the queries carry log2 e).

        A head's scores stay in the fp16 tiles HMX stores them in, where the
        mask and the row maxima work; exp2 writes P into another buffer's
        tiles, which HMX multiplies by V and whose rows sum in fp32. The
        blocks of keys and values come by DMA, a block ahead, straight into
        HMX's operands: the cache keeps them in its tiles."""
        for h in T.serial(KH):
            kvh = bf.w * KH + h
            for hq in T.serial(group):
                for rb in T.serial(R // ROWS):
                    row0 = base + rb * ROWS
                    T.dma_copy(bf.qs[kvh, hq, rb * ROWS : (rb + 1) * ROWS, :], bf.qb)
                    for i, d in T.Parallel(ROWS, HD):
                        bf.acc_o[i, d] = T.float32(0)
                    for i in T.Parallel(ROWS):
                        bf.m[i] = T.float32(NEG)
                        bf.ls[i] = T.float32(0)
                    # The positions a row sees: its window, or all before it.
                    span = T.if_then_else(window > 0, window, max_pos + R)
                    first = T.max(row0 - span + 1, 0) // KEYS
                    last = T.ceildiv(row0 + ROWS, KEYS)
                    fetch(bf, ai, kvh, first, 0)
                    for kb in T.serial(first, last):
                        slot = (kb - first) % 2
                        if kb + 1 < last:
                            fetch(bf, ai, kvh, kb + 1, 1 - slot)
                            T.dma_wait(2)
                        else:
                            T.dma_wait()
                        T.gemm(bf.qb, bf.kblk[slot, :, :], bf.s, transpose_B=True, clear_accum=True)
                        # Causal, and within the window: key kb * KEYS + j is
                        # visible to row i when row0 + i - span < key <= row0 + i,
                        # every key of the block to every row away from the
                        # diagonal and the window's edge.
                        if kb * KEYS + KEYS - 1 > row0 or kb * KEYS <= row0 + ROWS - 1 - span:
                            for i, j in T.Parallel(ROWS, KEYS):
                                key = kb * KEYS + j
                                causal = T.if_then_else(key <= row0 + i, bf.s[i, j], T.float16(NEG16))
                                bf.s[i, j] = T.if_then_else(key > row0 + i - span, causal, T.float16(NEG16))
                        T.reduce_max(bf.s, bf.rmax, dim=1)
                        for i in T.Parallel(ROWS):
                            bf.rmax[i] = T.max(bf.m[i], bf.rmax[i])
                            bf.alpha[i] = T.exp2(bf.m[i] - bf.rmax[i])
                            bf.m[i] = bf.rmax[i]
                            # The maxima of fp16 scores are fp16.
                            bf.mh[i] = T.Cast("float16", bf.rmax[i])
                        for i, j in T.Parallel(ROWS, KEYS):
                            bf.p[i, j] = T.exp2(bf.s[i, j] - bf.mh[i])
                        T.reduce_sum(bf.p, bf.rsum, dim=1)
                        for i in T.Parallel(ROWS):
                            bf.ls[i] = bf.ls[i] * bf.alpha[i] + bf.rsum[i]
                        for i, d in T.Parallel(ROWS, HD):
                            bf.acc_o[i, d] = bf.acc_o[i, d] * bf.alpha[i]
                        T.gemm(bf.p, bf.vblk[slot, :, :], bf.acc_o, clear_accum=False)
                    for i in T.Parallel(ROWS):
                        bf.alpha[i] = bf.gate(h, rb * ROWS + i, hq) / bf.ls[i]
                    for i, d in T.Parallel(ROWS, HD):
                        bf.acc_o[i, d] = bf.acc_o[i, d] * bf.alpha[i]
                    col = (kvh * group + hq) * HD
                    T.copy(bf.acc_o, bf.hid[rb * ROWS : (rb + 1) * ROWS, col : col + HD])

    @T.macro
    def branch_block(bf, b, ring, k, A, K: int, first, sums):
        """f[:, the worker's channels of panel b] = A @ block, or += past the
        FFN's first part: in the GEMM. With `sums` (f's channels final), the
        panel's rows' sums of squares into the worker's part of the branch's,
        which LFM3's sandwich norm reads: in the GEMM phase, where HVX has
        time the weights' stream gives it."""
        ch = bf.w * DW + b * TILE
        gemm(bf, A, K, ring, k, bf.f[:, ch : ch + TILE], first)
        if sums:
            for i, j in T.Parallel(R, TILE):
                bf.sq32[i, j] = bf.f[i, ch + j] * bf.f[i, ch + j]
            T.reduce_sum(bf.sq32, bf.hs32, dim=1)
            for i in T.Parallel(R):
                bf.psum[bf.w, i] = T.if_then_else(b == 0, bf.hs32[i], bf.psum[bf.w, i] + bf.hs32[i])

    @T.macro
    def glu_block(bf, b, ring, k):
        """w1 (even b) or w3 of hidden panel b // 2 of the worker's share of
        the part; after w3, silu(w1 a) * (w3 a) into hid."""
        gemm(bf, bf.a, D, ring, k, bf.cc[b % 2, :, :], True)
        if b % 2 == 1:
            for i, j in T.Parallel(R, TILE):
                gl = T.Cast("float16", bf.cc[0, i, j])
                bf.gl[i, j] = gl * T.sigmoid(gl) * T.Cast("float16", bf.cc[1, i, j])
            col = bf.w * FQW + b // 2 * TILE
            T.copy(bf.gl, bf.hid[:, col : col + TILE])

    @T.prim_func
    def prefill(
        in_w: T.Tensor((Lc, D, 3 * D), "float16"),
        qkv_w: T.Tensor((La, D, KVH * g.width), "float16"),
        out_w: T.Tensor((L, D, D), "float16"),
        w13_w: T.Tensor((L, D, 2 * F), "float16"),
        w2_w: T.Tensor((L, F, D), "float16"),
        head_rows: T.Tensor((c.vocab // TILE * D // 2, 2 * TILE), "float16"),
        op_norm: T.Tensor((L, D), "float32"),
        ffn_norm: T.Tensor((L, D), "float32"),
        post_op_norm: T.Tensor((L, D), "float32"),
        post_ffn_norm: T.Tensor((L, D), "float32"),
        q_norm: T.Tensor((La, HD), "float32"),
        k_norm: T.Tensor((La, HD), "float32"),
        conv_w: T.Tensor((Lc, taps, D), "float32"),
        rope_cos: T.Tensor((NK, max_pos, half), "float32"),
        rope_sin: T.Tensor((NK, max_pos, half), "float32"),
        kv_k: T.Tensor((La, KVH, max_pos, HD), "float16"),
        kv_v: T.Tensor((La, KVH, max_pos, HD), "float16"),
        conv_state: T.Tensor((Lc, taps - 1, D), "float32"),
        kinds: T.Tensor((L, 4), "int32"),
        x: T.Tensor((R, D), "float32"),
        qs: T.Tensor((KVH, group, R, HD), "float16"),
        tokens: T.Tensor((max_tokens,), "int32"),
        pos0: T.int32,
        n: T.int32,
    ):
        # fp16-grade math (1e-4 relative): the activations it computes are
        # fp16 for HMX anyway.
        T.annotate_pass_configs({"tl.enable_fast_math": True})
        with T.Kernel(NW, num_workers=NW) as w:
            p_in = Weights(in_w, TILE, D, SLOT)
            p_qkv = Weights(qkv_w, TILE, D, SLOT)
            p_out = Weights(out_w, TILE, D, SLOT)
            p_13 = Weights(w13_w, TILE, D, SLOT)
            p_2 = Weights(w2_w, TILE, FQ, SLOT)

            # The chunk's activations HMX reads, which every worker reads, and
            # its rows of the RoPE tables, which every head's rotation reads.
            a = T.alloc_global((R, D), "float16", scope="global.vtcm")
            hid = T.alloc_global((R, HID), "float16", scope="global.vtcm")
            f = T.alloc_global((R, D), "float32", scope="global.vtcm")
            # The workers' parts of f's rows' sums of squares.
            psum = T.alloc_global((NW, R), "float32", scope="global.vtcm")
            cos_rows = T.alloc_global((NK, R, half), "float32", scope="global.vtcm")
            sin_rows = T.alloc_global((NK, R, half), "float32", scope="global.vtcm")

            # A worker's: the outputs of up to two panels, or of a head's;
            # the rows it normalizes or embeds two at a time.
            cc = T.alloc_shared((2, R, TILE), "float32")
            gl = T.alloc_shared((R, TILE), "float16")
            st = T.alloc_shared((taps - 1, TILE), "float32")
            xr = T.alloc_shared((2, RB, D), "float32")
            rows = T.alloc_fragment((RB, D), "float32")
            sq = T.alloc_fragment((RB, D), "float32")
            ss = T.alloc_fragment((RB,), "float32")
            sw = T.alloc_shared((RW,), "float32")
            nw = T.alloc_shared((D,), "float32")
            pw = T.alloc_shared((D,), "float32")
            erow = T.alloc_shared((D // 2, 2), "float16")
            # q/k norms and RoPE: a head of the chunk's rows, its rows' sums of
            # squares, and the head normalized and rotated.
            hd = T.alloc_shared((R, HD), "float32")
            rot = T.alloc_shared((R, HD), "float16")
            hss = T.alloc_shared((R,), "float32")
            sq32 = T.alloc_shared((R, TILE), "float32")
            hs32 = T.alloc_shared((R,), "float32")
            gates = T.alloc_shared((KH, group, R), "float32")
            # Attention: a block of a head's queries, two slots of a block of
            # keys and values, the softmax state and output, and the scores.
            qb = T.alloc_shared((ROWS, HD), "float16")
            kblk = T.alloc_shared((2, KEYS, HD), "float16")
            vblk = T.alloc_shared((2, KEYS, HD), "float16")
            acc_o = T.alloc_shared((ROWS, HD), "float32")
            m = T.alloc_shared((ROWS,), "float32")
            ls = T.alloc_shared((ROWS,), "float32")
            s = T.alloc_shared((ROWS, KEYS), "float16")
            p = T.alloc_shared((ROWS, KEYS), "float16")
            rmax = T.alloc_shared((ROWS,), "float32")
            mh = T.alloc_shared((ROWS,), "float16")
            rsum = T.alloc_shared((ROWS,), "float32")
            alpha = T.alloc_shared((ROWS,), "float32")

            # What the macros below work on.
            bf = SimpleNamespace(
                w=w, a=a, hid=hid, x=x, f=f, psum=psum, qs=qs, cc=cc, gl=gl, st=st, xr=xr, rows=rows, sq=sq, ss=ss, sw=sw, nw=nw, pw=pw, erow=erow, hd=hd, rot=rot,
                hss=hss, sq32=sq32, hs32=hs32, gates=gates, cos_rows=cos_rows, sin_rows=sin_rows, qb=qb, kblk=kblk, vblk=vblk,
                acc_o=acc_o, m=m, ls=ls, s=s, p=p, rmax=rmax, mh=mh, rsum=rsum, alpha=alpha, conv_state=conv_state, conv_w=conv_w,
                q_norm=q_norm, k_norm=k_norm, kv_k=kv_k, kv_v=kv_v,
                gate=(lambda h, i, q: gates[h, q, i]) if c.head_gate else (lambda h, i, q: T.float32(1)),
            )  # fmt: skip

            # Every projection streams blocks of one shape through one ring.
            rings = {}
            projections = (p_in, p_qkv, p_out, p_13, p_2)
            layouts = Stream([Phase(pj, 0, 1, lambda b: 0) for pj in projections], DEPTH, rings=rings).layouts()
            # The scores: HMX's output tiles. The queries, in DDR as in VTCM,
            # in its activation tiles.
            layouts[s] = make_hmx_activation_layout(s)
            layouts[qs] = make_hmx_activation_layout(qs)
            layouts[qb] = make_hmx_activation_layout(qb)
            layouts.update(kv_layouts(kv_k, kv_v))
            layouts.update(kv_layouts(kblk, vblk))
            T.annotate_layout(layouts)
            # Only DMA moves the residual: its transfers need no cache upkeep.
            T.annotate_dma_only(x)
            # The KV cache and the queries are written once and read by DMA
            # many times: written back as they are written, their loads skip
            # the writeback. The decode declares the cache so too.
            T.annotate_write_through(kv_k, kv_v, qs)

            for chunk in T.serial(T.ceildiv(n, R)):
                base = pos0 + chunk * R
                valid = T.min(R, n - chunk * R)
                # The chunk's rows of the RoPE tables, into VTCM: worker w's
                # rows. Rows past the tables, the padding's, take their last.
                for t, i, d in T.Parallel(NK, RW, half):
                    r = w * RW + i
                    cos_rows[t, r, d] = rope_cos[t, T.min(base + r, max_pos - 1), d]
                    sin_rows[t, r, d] = rope_sin[t, T.min(base + r, max_pos - 1), d]

                embed(bf, chunk, n, tokens, head_rows, op_norm[0, :])
                T.sync_grid()

                for layer in T.serial(L):
                    # The layer's index among the conv (ci) or the attention
                    # (ai) layers, clamped so that every weight block it reads
                    # is provably in its tensor.
                    ci = clamped(kinds[layer, 1], Lc)
                    ai = clamped(kinds[layer, 1], La)
                    table = clamped(kinds[layer, 2], NK)
                    window = kinds[layer, 3]
                    # The layer's phases, worker w's blocks of each, in order:
                    # its operator's (the short conv's or attention's), then the
                    # output projection and the FFN's, each part's w1/w3
                    # panels and then its w2 slab. The streams run one after
                    # another.
                    st_conv = Stream(
                        [Phase(p_in, ci, 3 * DW // TILE, lambda b: w * 3 * DW + b // 3 * 3 * TILE + (b % 3) * 2 % 3 * TILE)],
                        DEPTH,
                        rings=rings,
                    )
                    st_attn = Stream([Phase.rows(p_qkv, ai, w * QW, QW // TILE)], DEPTH, rings=rings)
                    rest = [Phase.rows(p_out, layer, w * DW, DW // TILE)]
                    for part in list(range(PARTS)):
                        hidden0 = part * FQ + w * FQW  # the worker's first hidden unit of the part
                        rest.append(Phase.rows(p_13, layer, 2 * hidden0, 2 * FQW // TILE))
                        rest.append(Phase(p_2, layer, DW // TILE, lambda b: w * DW + b * TILE, lambda b, part=part: part))
                    st_rest = Stream(rest, DEPTH, rings=rings)
                    # a holds the layer's normalized input.
                    if kinds[layer, 0] == 1:
                        st_attn.prologue()
                        st_attn.run(0, lambda b, ring, k, ai=ai, table=table, base=base: qkv_block(bf, b, ring, k, ai, table, base))
                        attention(bf, ai, base, window)
                    else:
                        st_conv.prologue()
                        st_conv.run(0, lambda b, ring, k, ci=ci, valid=valid: in_proj_block(bf, b, ring, k, ci, valid))
                    T.sync_grid()
                    # The last layer's output projection and FFN feed nothing the
                    # prefill leaves: the decode runs the prompt's last token.
                    if layer < L - 1:
                        st_rest.prologue()
                        st_rest.run(0, lambda b, ring, k: branch_block(bf, b, ring, k, hid[:, 0:D], D, True, c.sandwich_norm))
                        T.sync_grid()
                        add_branch(bf, post_op_norm[layer, :], c.sandwich_norm, ffn_norm[layer, :], True)
                        T.sync_grid()
                        for part in list(range(PARTS)):
                            st_rest.run(1 + 2 * part, lambda b, ring, k: glu_block(bf, b, ring, k))
                            T.sync_grid()
                            st_rest.run(
                                2 + 2 * part,
                                lambda b, ring, k, part=part: branch_block(
                                    bf, b, ring, k, hid[:, 0:FQ], FQ, part == 0, c.sandwich_norm and part == PARTS - 1
                                ),
                            )
                            T.sync_grid()
                        # The residual, and the next layer's normalized input.
                        add_branch(bf, post_ffn_norm[layer, :], c.sandwich_norm, op_norm[layer + 1, :], True)
                        T.sync_grid()

    return prefill


# The kernel's parameters from the bundle, in order (decode_fp16.bundle_args'
# names): without sandwich norms, the post norms are the pre norms, unread.
PREFILL_ARGS = ("conv.in_proj", "attn.qkv", "out_proj", "ffn.w13", "ffn.w2", "head")
PREFILL_NORMS = ("op_norm", "ffn_norm", "post_op_norm", "post_ffn_norm", "q_norm", "k_norm", "conv")


def prefill_args(c: Config, weights: dict) -> list:
    alias = {} if c.sandwich_norm else {"post_op_norm": "op_norm", "post_ffn_norm": "ffn_norm"}
    return [weights[n] for n in PREFILL_ARGS] + [weights[alias.get(n, n)] for n in PREFILL_NORMS]
