"""The fp16 decode on Hexagon: LFM2 or LFM3 from the fp16 bundle (convert.py),
a token of a generation a kernel call, the Decoder running the calls as a
pipeline (Session.pipeline). Past one DSP process's 4 GB of addresses, the
layers split into stages, a kernel and a process each (``plan_stages``).

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

import itertools
import os
import sys

import torch

from tilelang.hexagon import language as T
from tilelang.hexagon.intrinsics.crouton import make_hmx_weight_layout

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import convert  # noqa: E402
from convert import TILE, qkv_group  # noqa: E402
from decode import NOISE_KEYS  # noqa: E402
from model import CONV, SLIDING, Config, Model, rope_angles  # noqa: E402
from stream import Phase, Stream, Weights  # noqa: E402

NW = 4  # workers
DEPTH = 4  # blocks in flight per worker
POS_BLOCK = 64  # KV positions per attention step
# A sliding-window layer's cache is a ring of positions: position p in row
# RING_PAD + p % ring(c) of the layer's rows. The pad, a chunk of prefill
# rows either side, takes the rows a chunk's write runs past either end of
# the ring (prefill_fp16.store_kv), which no read sees.
RING_PAD = 512
SLICE_TILES = 64  # a block's K tiles at most: 128 KB of weights
# The rows of a stage's KV cache and RoPE tables when none of its layers
# attends (make_decode): placeholders, never read.
EMPTY_ROWS = 128


def kinds(c: Config) -> list:
    """The attention kinds of config c, in its RoPE tables' order."""
    return list(c.rope)


def rope_tables(c: Config, max_pos: int) -> tuple:
    """cos and sin [kinds, max_pos, head_dim / 2] of each attention kind's RoPE
    angles, as model.rope_angles computes them."""
    positions = torch.arange(max_pos)
    angles = torch.stack([rope_angles(positions, c.head_dim, *c.rope[k]) for k in kinds(c)])
    return angles.cos().float().contiguous(), angles.sin().float().contiguous()


def ring(c: Config) -> int:
    """The positions of a sliding-window layer's ring: a power of two at least
    its window, a prefill chunk (RING_PAD) and a block of the prefill's keys
    (prefill_fp16.KEYS, 128) apart, so that no chunk overwrites a key a row
    of it still sees."""
    need = c.window + RING_PAD + 128
    return 1 << (need - 1).bit_length()


def kv_plan(c: Config, max_pos: int, layers: range | None = None) -> tuple:
    """The KV cache's rows, [kv heads, rows, head dim] for keys and values each,
    of the attention layers among `layers` (all by default): every attention
    layer a run of them, a full-attention layer's max_pos positions or a
    sliding-window layer's ring with its pads. Returns each attention layer's
    first row and whether it is a ring, and the rows."""
    first, rings, rows = [], [], 0
    for layer in c.attention_layers:
        if layers is not None and layer not in layers:
            continue
        sliding = c.layer_types[layer] == SLIDING
        first.append(rows)
        rings.append(sliding)
        rows += ring(c) + 2 * RING_PAD if sliding else max_pos
    return first, rings, rows


def kv_row(c: Config, first: int, sliding: bool, pos):
    """The cache row of position pos in an attention layer whose first row is
    `first`, a ring's if `sliding`."""
    return first + (RING_PAD + pos % ring(c) if sliding else pos)


def kv_positions(c: Config, max_pos: int, k_raw: bytes, v_raw: bytes, n: int, layers: range | None = None) -> list:
    """For tests: each attention layer's (among `layers`) keys and values, [kv
    heads, positions, head dim] in fp32, of the positions of [0, n) its cache
    still holds, with those positions. A ring holds the last ring(c) -
    RING_PAD of them at least: a prefill's last chunk writes its padding rows
    too, over older positions, which the window no longer sees."""
    first, rings, rows = kv_plan(c, max_pos, layers)
    KVH, HD = c.kv_heads, c.head_dim
    # The tiles of kv_layouts: keys as the weight operand [head dim, rows] of
    # q K^T, values as that of P V.
    k = convert.from_crouton_weight(torch.frombuffer(bytearray(k_raw), dtype=torch.float16), (KVH, HD, rows))
    v = convert.from_crouton_weight(torch.frombuffer(bytearray(v_raw), dtype=torch.float16), (KVH, rows, HD)).transpose(-1, -2)
    out = []
    for f, r in zip(first, rings):
        positions = torch.arange(max(0, n - (ring(c) - RING_PAD)) if r else 0, n)
        idx = torch.tensor([kv_row(c, f, r, int(p)) for p in positions])
        out.append((k[:, idx].float(), v[:, idx].float(), positions))
    return out


# The bundle's tensors stacked over layers: over every layer, the attention
# layers or the conv layers. A stage of a model split over DSP processes maps
# its layers' slices of each.
STACKS = {
    "conv.in_proj": "conv",
    "conv": "conv",
    "attn.qkv": "attention",
    "q_norm": "attention",
    "k_norm": "attention",
    "out_proj": "layer",
    "ffn.w13": "layer",
    "ffn.w2": "layer",
    "op_norm": "layer",
    "ffn_norm": "layer",
    "post_op_norm": "layer",
    "post_ffn_norm": "layer",
}

# The bytes of a stage's buffers (stage_bytes) a DSP process maps, of its 4
# GB of addresses: LFM3-1.7B's at 48K positions, 3947 MiB, mapped in one;
# at 64K, 4155 MiB, they did not.
PROCESS_BUDGET = 3960 << 20


def stack_range(c: Config, kind: str, layers: range) -> tuple:
    """The first index and the count, in a stack of `kind`, of the slices of
    `layers`."""

    def of(i: int) -> bool:
        return kind == "layer" or (kind == "attention") == c.is_attention(i)

    return sum(of(i) for i in range(layers.start)), sum(of(i) for i in layers)


def stage_bytes(c: Config, header: dict, max_pos: int, layers: range) -> int:
    """The bytes a stage of `layers` maps that grow with the model and the
    context: its slices of the weights, the head at either end, its KV cache
    and the RoPE tables."""
    total = sum(
        stack_range(c, STACKS[name], layers)[1] * (t["nbytes"] // t["shape"][0]) for name, t in header["tensors"].items() if name in STACKS
    )
    if layers.start == 0 or layers.stop == c.layers:
        total += header["tensors"]["head"]["nbytes"]
    total += 2 * c.kv_heads * kv_plan(c, max_pos, layers)[2] * c.head_dim * 2
    if any(c.is_attention(i) for i in layers):
        total += 2 * len(c.rope) * max_pos * (c.head_dim // 2) * 4
    return total


def plan_stages(c: Config, header: dict, max_pos: int, processes="auto") -> list:
    """The layers of each stage, runs of them in order: `processes` stages,
    or ("auto") the fewest whose stages each fit a process. Of the ways to
    cut the layers, the one whose largest stage is the smallest."""
    from tilelang.hexagon.runtime.session import Session

    most = min(c.layers, Session.MAX_PROCESSES)
    if processes != "auto" and not 1 <= processes <= most:
        raise ValueError(f"LFM's {c.layers} layers run on 1 to {most} DSP processes, not {processes}")
    for p in range(1, most + 1) if processes == "auto" else [processes]:
        best = None
        for cuts in itertools.combinations(range(1, c.layers), p - 1):
            bounds = [0, *cuts, c.layers]
            stages = [range(a, b) for a, b in zip(bounds, bounds[1:])]
            largest = max(stage_bytes(c, header, max_pos, st) for st in stages)
            if best is None or largest < best[0]:
                best = (largest, stages)
        if processes != "auto" or best[0] <= PROCESS_BUDGET:
            return best[1]
    raise ValueError(f"LFM at {max_pos} positions takes more than {most} DSP processes")


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
def attend(
    qkv, q0, kv_k, kv_v, row, kvh, pos, first, out, out0, gate, group: int, HD: int, qg, kb, vb, sc, p, p16, o, m, mb, ls, lb, alpha
):
    """out[0, out0 + g * HD + d] = gate(g) softmax(q_g K^T / sqrt(HD)) V over
    positions [first, pos] of KV head kvh, for the `group` query heads at
    qkv[0, q0 + ...]: blocks of POS_BLOCK positions with an online softmax.
    row(p) is the layer's cache row of position p, whose block's rows follow
    it (a ring's blocks do not wrap)."""
    for g, d in T.Parallel(group, HD):
        qg[g, d] = T.Cast("float16", qkv[0, q0 + g * HD + d] * T.float32(HD**-0.5))
    for g in T.Parallel(group):
        m[g] = T.float32(-3.0e38)
        ls[g] = T.float32(0)
    for g, d in T.Parallel(group, HD):
        o[g, d] = T.float32(0)
    for pb in T.serial(first // POS_BLOCK, T.ceildiv(pos + 1, POS_BLOCK)):
        r = row(pb * POS_BLOCK)
        T.copy(kv_k[kvh, r : r + POS_BLOCK, :], kb)
        T.copy(kv_v[kvh, r : r + POS_BLOCK, :], vb)
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


def make_decode(c: Config, max_pos: int, max_tokens: int, layers: range | None = None):
    """The fp16 decode kernel of config c (convert.py's bundle): step t of a
    generation, position pos0 + t, a call. The Decoder runs the steps as a
    pipeline (Session.pipeline).

    A model split over DSP processes has a kernel for each run of `layers`
    (all of them by default), a stage of the pipeline: the first embeds the
    step's token, the last runs the head and samples, and the residual goes
    from each to the next through `xs` by DMA, which crosses processes. A
    stage's tensors are its layers' slices; those of a kind it has none of,
    and the head away from the ends, are placeholders."""
    layers = range(c.layers) if layers is None else layers
    has_embedding, has_head = layers.start == 0, layers.stop == c.layers
    D, F, V, HD = c.dim, c.ff, c.vocab, c.head_dim
    H, KVH, taps = c.heads, c.kv_heads, c.conv_taps
    conv_layers = [i for i in layers if not c.is_attention(i)]
    attn_layers = [i for i in layers if c.is_attention(i)]
    Lc, La, L = len(conv_layers), len(attn_layers), len(layers)
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
    kv_first, kv_ring, KV_ROWS = kv_plan(c, max_pos, layers)
    assert ring(c) % POS_BLOCK == 0
    # The extents of placeholders: of the layers of a kind the stage has none
    # of, and of the head and its outputs away from the ends.
    Lc1, La1 = max(Lc, 1), max(La, 1)
    KV_ROWS, ROPE = (KV_ROWS, max_pos) if La else (EMPTY_ROWS, EMPTY_ROWS)
    HEAD = (D, V) if has_head else (TILE, TILE)
    HEAD_ROWS = (V // TILE * D // 2, 2 * TILE) if has_embedding else (TILE, 2 * TILE)
    VL = V if has_head else TILE
    l0 = layers.start

    @T.prim_func
    def decode(
        in_w: T.Tensor((Lc1, D, 3 * D), "float16"),
        qkv_w: T.Tensor((La1, D, KVH * g.width), "float16"),
        out_w: T.Tensor((L, D, D), "float16"),
        w13_w: T.Tensor((L, D, 2 * F), "float16"),
        w2_w: T.Tensor((L, F, D), "float16"),
        head: T.Tensor(HEAD, "float16"),
        # The head's bytes again, row-major: a token's embedding is D / 2 of
        # its pairs, a register of 32 tokens' apart (convert.embedding).
        head_rows: T.Tensor(HEAD_ROWS, "float16"),
        op_norm: T.Tensor((L, D), "float32"),
        ffn_norm: T.Tensor((L, D), "float32"),
        post_op_norm: T.Tensor((L, D), "float32"),
        post_ffn_norm: T.Tensor((L, D), "float32"),
        q_norm: T.Tensor((La1, HD), "float32"),
        k_norm: T.Tensor((La1, HD), "float32"),
        conv_w: T.Tensor((Lc1, taps, D), "float32"),
        final_norm: T.Tensor((D,), "float32"),
        rope_cos: T.Tensor((NK, ROPE, HD // 2), "float32"),
        rope_sin: T.Tensor((NK, ROPE, HD // 2), "float32"),
        kv_k: T.Tensor((KVH, KV_ROWS, HD), "float16"),
        kv_v: T.Tensor((KVH, KV_ROWS, HD), "float16"),
        conv_state: T.Tensor((Lc1, taps - 1, D), "float32"),
        mix: T.Tensor((1, D), "float16"),
        branch: T.Tensor((1, D), "float32"),
        hidden: T.Tensor((1, F), "float16"),
        sums: T.Tensor((NW,), "float32"),
        cand: T.Tensor((NW, 4), "float32"),
        tokens: T.Tensor((max_tokens,), "int32"),
        logits: T.Tensor((1, VL), "float32"),
        scores: T.Tensor((max_tokens,), "float32"),
        sampling: T.Tensor((2,), "float32"),
        # The residual from the stage before, and to the next.
        xs: T.Tensor((1, D), "float32"),
        pos0: T.int32,
        n_forced: T.int32,
        t: T.int32,
    ):
        with T.Kernel(NW, num_workers=NW) as w:
            p_in = Weights(in_w, TILE, slice_cols(D))
            p_qkv = Weights(qkv_w, TILE, slice_cols(D))
            p_out = Weights(out_w, TILE, slice_cols(D))
            p_13 = Weights(w13_w, TILE, slice_cols(D))
            p_2 = Weights(w2_w, TILE, slice_cols(F))
            p_head = Weights(head, TILE, slice_cols(D)) if has_head else None

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
            # The DMA ends of the residual between stages, and of the token
            # the first stage reads.
            xv = T.alloc_shared((1, D), "float32") if not (has_embedding and has_head) else None
            tv = T.alloc_shared((TILE,), "int32") if has_embedding and not has_head else None
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
            for layer in list(layers):
                if c.is_attention(layer):
                    phases.append(Phase.rows(p_qkv, ai, w * QW, QW // TILE))
                    ai += 1
                else:
                    # B, C and x of each of the worker's channel panels.
                    phases.append(Phase.rows(p_in, ci, w * 3 * DW, 3 * DW // TILE))
                    ci += 1
                phases.append(Phase.rows(p_out, layer - l0, w * DW, DW // TILE))
                phases.append(Phase.rows(p_13, layer - l0, w * 2 * FW, 2 * FW // TILE))
                phases.append(Phase.rows(p_2, layer - l0, w * DW, DW // TILE))
            if has_head:
                phases.append(Phase.rows(p_head, 0, w * VW, VW // TILE))
            st = Stream(phases, DEPTH)
            T.annotate_layout({**st.layouts(), **kv_layouts(kv_k, kv_v), **kv_layouts(kb, vb)})
            # The prefill reads the KV cache by DMA, so each key and value
            # written here goes back to DDR at once (prefill_fp16). Each
            # process caches its own lines, so what crosses processes goes
            # by DMA: the residual, which only DMA moves, and the token the
            # last stage generates, written back to DDR as it is written,
            # which the first stage reads.
            T.annotate_dma_only(xs)
            if has_head and not has_embedding:
                T.annotate_write_through(kv_k, kv_v, tokens)
            else:
                T.annotate_write_through(kv_k, kv_v)
            st.prologue()

            pos = pos0 + t

            if has_embedding:
                # The step's token, forced or the one the step before
                # generated (by DMA, from the last stage on another process),
                # and its embedding, which every worker reads whole: its pairs
                # from the head's tiles.
                if has_head:
                    token = T.max(T.min(tokens[t], V - 1), 0)
                else:
                    at = T.max(T.min(t, max_tokens - 1), 0)
                    T.dma_copy(tokens[at : at + 1], tv[0:1])
                    T.dma_wait(0)
                    token = T.max(T.min(tv[0], V - 1), 0)
                T.dma_copy(
                    head_rows[token // TILE * (D // 2) : (token // TILE + 1) * (D // 2), token % TILE * 2 : token % TILE * 2 + 2], erow
                )
                T.dma_wait(0)
                for i, k in T.Parallel(1, D):
                    xf[i, k] = T.Cast("float32", erow[k // 2, k % 2])
            else:
                # The residual the stage before left.
                T.dma_copy(xs, xv)
                T.dma_wait(0)
                for i, k in T.Parallel(1, D):
                    xf[i, k] = xv[i, k]

            i_ph = 0
            ci = ai = 0
            for layer in list(layers):
                rms_norm(xf, lambda k, layer=layer: op_norm[layer - l0, k], h16, sq, s1, D, eps)
                if c.is_attention(layer):
                    ki = kind[c.layer_types[layer]]
                    first = T.max(pos - c.window + 1, 0) if c.layer_types[layer] == SLIDING else 0
                    # The layer's cache row of a position.
                    row = lambda p, ai=ai: kv_row(c, kv_first[ai], kv_ring[ai], p)  # noqa: E731
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
                            kv_k[kvh, row(pos), d] = T.Cast("float16", qkv[0, base + g.k + d])
                            kv_v[kvh, row(pos), d] = T.Cast("float16", qkv[0, base + g.v + d])
                        if c.head_gate:
                            gate = lambda q, base=base: T.sigmoid(qkv[0, base + g.gate + q])  # noqa: E731
                        else:
                            gate = lambda q: T.float32(1)  # noqa: E731
                        attend(
                            qkv, base, kv_k, kv_v, row, kvh, pos, first, att, h * group * HD, gate, group, HD,
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
                add_branch(xf, branch, bf, sums, s1, lambda k, layer=layer: post_op_norm[layer - l0, k], D, NW, c.sandwich_norm, eps)
                # The FFN: worker w's hidden units, then its slice of their
                # projection, K in S2 slices.
                rms_norm(xf, lambda k, layer=layer: ffn_norm[layer - l0, k], h16, sq, s1, D, eps)
                st.run(i_ph, lambda b, ring, k: glu_block(b, ring, k, h16, acc, D, ga, ffo))
                i_ph += 1
                T.copy(ffo, hidden[0:1, w * FW : (w + 1) * FW])
                T.sync_grid()
                T.copy(hidden, f16)
                st.run(i_ph, lambda b, ring, k: sliced_block(b, ring, k, f16, acc, p_2.cols, S2, own))
                i_ph += 1
                publish(own, branch, w, DW, sums, sq_own, s1, c.sandwich_norm)
                T.sync_grid()
                add_branch(xf, branch, bf, sums, s1, lambda k, layer=layer: post_ffn_norm[layer - l0, k], D, NW, c.sandwich_norm, eps)

            if has_head:
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
            else:
                # The residual, for the next stage.
                if w == 0:
                    for i, k in T.Parallel(1, D):
                        xv[i, k] = xf[i, k]
                    T.dma_copy(xv, xs)
                    T.dma_wait(0)

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
