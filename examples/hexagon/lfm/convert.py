"""Write the fp16 bundle of an LFM2 or LFM3 checkpoint: its weights, unquantized.

    python convert.py --model LiquidAI-LFM3-1.7B-dir --out lfm3.bundle

The checkpoint's bf16 weights become fp16: exactly, for all but 0.03 % of
LFM3-1.7B's, which move by at most 3e-8; none is out of fp16's range.

A projection [N, K] is stored as the HMX weight operand it is, the math
[K, N] in Crouton tiles ordered by columns, [nt, kt] (``crouton_weight``: the
layout of ``tilelang.hexagon.intrinsics.crouton.make_hmx_weight_layout``). The
K tiles of 32 output channels, a panel, are contiguous, so one DMA brings a
run of whole panels, or of the K tiles of one, into a VTCM slot of the same
layout, where HMX (the prefill) or HVX (the decode) multiplies it. The parts
that a phase multiplies together interleave by panel, so that a panel of
channels arrives with all of its parts:

    conv.in_proj  [Lc, D, 3D]       B, C and x of each panel of channels
    attn.qkv      [La, D, KVH * W]  per KV head (``qkv_group``): its query
                                    heads, its key, its value and (LFM3) a
                                    panel of its query heads' gates, zero
                                    past them
    out_proj      [L, D, D]         each layer's operator's output projection
    ffn.w13       [L, D, 2F]        w1 and w3 of each panel of hidden units
    ffn.w2        [L, F, D]
    head          [D, V]            the tied embedding, as the LM head's
                                    operand: a token's embedding is D / 2 of
                                    its 4-byte words, 128 bytes apart
                                    (``embedding``)

The rest is fp32 and row-major: the norms, stacked over the layers that have
them (post_op_norm and post_ffn_norm with LFM3's sandwich norm), and the conv
taps [Lc, taps, D].
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import NamedTuple

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bundle  # noqa: E402
from model import Config, load_hf_weights  # noqa: E402

TILE = 32  # a Crouton tile is TILE x TILE; a panel, TILE output channels


def crouton_weight(w: torch.Tensor) -> torch.Tensor:
    """Projections w [..., N, K] as HMX weight operands: the math [..., K, N]
    in Crouton tiles [..., N / 32, K / 32, 1024], element (k, n) of tile
    (n // 32, k // 32) at (k % 32 // 2) * 64 + n % 32 * 2 + k % 2."""
    *lead, N, K = w.shape
    d = len(lead)
    t = w.reshape(*lead, N // TILE, TILE, K // TILE, TILE // 2, 2)
    return t.permute(*range(d), d, d + 2, d + 3, d + 1, d + 4).reshape(*lead, N // TILE, K // TILE, TILE * TILE)


def from_crouton_weight(t: torch.Tensor, shape) -> torch.Tensor:
    """The projections [..., N, K] whose operands, of shape [..., K, N], are
    the Crouton tiles t (``crouton_weight``), in any shape."""
    *lead, K, N = shape
    d = len(lead)
    t = t.reshape(*lead, N // TILE, K // TILE, TILE // 2, TILE, 2)
    return t.permute(*range(d), d, d + 3, d + 1, d + 2, d + 4).reshape(*lead, N, K)


def panels(*parts: torch.Tensor) -> torch.Tensor:
    """The rows of `parts`, each [N, K], interleaved by panel: panel p of
    each part in turn."""
    N, K = parts[0].shape
    return torch.stack([p.reshape(N // TILE, TILE, K) for p in parts], 1).reshape(len(parts) * N, K)


def from_panels(t: torch.Tensor, parts: int) -> list:
    """The parts that ``panels`` interleaved into t."""
    N, K = t.shape[0] // parts, t.shape[1]
    return list(t.reshape(N // TILE, parts, TILE, K).transpose(0, 1).reshape(parts, N, K))


class QKVGroup(NamedTuple):
    """Where a KV head's parts start among its columns of attn.qkv, and how
    many it has: its query heads, its key, its value, its query heads' gates."""

    q: int
    k: int
    v: int
    gate: int
    width: int


def qkv_group(c: Config) -> QKVGroup:
    group = c.heads // c.kv_heads
    gate = (group + 2) * c.head_dim
    return QKVGroup(0, group * c.head_dim, (group + 1) * c.head_dim, gate, gate + (TILE if c.head_gate else 0))


def embedding(head: torch.Tensor, token: int, D: int) -> torch.Tensor:
    """The token's embedding [D] (fp16) from the head's Crouton tiles (any
    shape): D / 2 words, pairs of it, 128 bytes apart from word
    (token // 32) * 16 * D + token % 32 of the head."""
    words = head.reshape(-1).view(torch.int32)
    first = token // TILE * (TILE * D // 2) + token % TILE
    return words[first : first + TILE * D // 2 : TILE].contiguous().view(torch.float16)


def build_tensors(c: Config, w: dict) -> dict:
    """The bundle's tensors (``bundle.write``) from the checkpoint's weights
    `w` (model.py's names): the projections made as they are written."""
    D, F, V, HD = c.dim, c.ff, c.vocab, c.head_dim
    group = c.heads // c.kv_heads
    assert c.heads * HD == D, "the operators' output projections stack"
    assert group <= TILE, "a KV head's gates fit a panel"
    conv = [i for i in range(c.layers) if not c.is_attention(i)]
    attn = list(c.attention_layers)
    every = list(range(c.layers))
    tensors = {}

    def lw(i, name):
        return w[f"model.layers.{i}.{name}.weight"]

    def projection(name, layers, N, K, rows):
        """Stack, over `layers`, the rows [N, K] (rows(i)) of each layer i."""

        def make():
            out = torch.empty(len(layers), N // TILE, K // TILE, TILE * TILE, dtype=torch.float16)
            for j, i in enumerate(layers):
                out[j] = crouton_weight(rows(i).half())
            return out

        tensors[name] = (make, "float16", (len(layers), K, N), "hmx_weight")

    def qkv(i):
        q, k, v = lw(i, "self_attn.q_proj"), lw(i, "self_attn.k_proj"), lw(i, "self_attn.v_proj")
        parts = []
        for h in range(c.kv_heads):
            parts += [q[h * group * HD : (h + 1) * group * HD], k[h * HD : (h + 1) * HD], v[h * HD : (h + 1) * HD]]
            if c.head_gate:
                gate = lw(i, "self_attn.gate_proj")[h * group : (h + 1) * group]
                parts.append(torch.cat([gate, gate.new_zeros(TILE - group, D)]))
        return torch.cat(parts)

    projection("conv.in_proj", conv, 3 * D, D, lambda i: panels(*lw(i, "conv.in_proj").chunk(3)))
    projection("attn.qkv", attn, c.kv_heads * qkv_group(c).width, D, qkv)
    projection("out_proj", every, D, D, lambda i: lw(i, "self_attn.out_proj" if c.is_attention(i) else "conv.out_proj"))
    projection("ffn.w13", every, 2 * F, D, lambda i: panels(lw(i, "feed_forward.w1"), lw(i, "feed_forward.w3")))
    projection("ffn.w2", every, D, F, lambda i: lw(i, "feed_forward.w2"))
    tensors["head"] = (lambda: crouton_weight(w["model.embed_tokens.weight"].half()), "float16", (D, V), "hmx_weight")

    def stack(name, key, layers):
        t = torch.stack([w[key.format(i)].float() for i in layers]).contiguous()
        tensors[name] = (t, "float32", tuple(t.shape), "row_major")

    stack("op_norm", "model.layers.{}.operator_norm.weight", every)
    stack("ffn_norm", "model.layers.{}.ffn_norm.weight", every)
    if c.sandwich_norm:
        stack("post_op_norm", "model.layers.{}.post_operator_norm.weight", every)
        stack("post_ffn_norm", "model.layers.{}.post_ffn_norm.weight", every)
    stack("q_norm", "model.layers.{}.self_attn.q_layernorm.weight", attn)
    stack("k_norm", "model.layers.{}.self_attn.k_layernorm.weight", attn)
    taps = torch.stack([w[f"model.layers.{i}.conv.conv.weight"][:, 0, :].t().float() for i in conv]).contiguous()
    tensors["conv"] = (taps, "float32", tuple(taps.shape), "row_major")
    fn = w["model.embedding_norm.weight"].float().contiguous()
    tensors["final_norm"] = (fn, "float32", tuple(fn.shape), "row_major")
    return tensors


def read_weights(path: str) -> tuple:
    """(config, weights): a bundle's weights in fp32, under the checkpoint's
    names (model.py's), for the reference model."""
    header = bundle.read_header(path)
    c = Config.from_dict(header["config"])
    g = qkv_group(c)
    group = c.heads // c.kv_heads
    conv = [i for i in range(c.layers) if not c.is_attention(i)]
    attn = list(c.attention_layers)

    def tensor(name):
        return bundle.read_tensor(path, header, name)

    def projection(name):
        return from_crouton_weight(tensor(name).view(torch.float16), header["tensors"][name]["shape"]).float()

    w = {}
    inp, qkv, out, w13, w2 = (projection(n) for n in ("conv.in_proj", "attn.qkv", "out_proj", "ffn.w13", "ffn.w2"))
    for j, i in enumerate(conv):
        p = f"model.layers.{i}.conv."
        w[p + "in_proj.weight"] = torch.cat(from_panels(inp[j], 3))
        w[p + "conv.weight"] = tensor("conv")[j].t()[:, None, :]
    for j, i in enumerate(attn):
        p = f"model.layers.{i}.self_attn."
        heads = qkv[j].reshape(c.kv_heads, g.width, c.dim)
        w[p + "q_proj.weight"] = heads[:, g.q : g.k].reshape(-1, c.dim)
        w[p + "k_proj.weight"] = heads[:, g.k : g.v].reshape(-1, c.dim)
        w[p + "v_proj.weight"] = heads[:, g.v : g.gate].reshape(-1, c.dim)
        if c.head_gate:
            w[p + "gate_proj.weight"] = heads[:, g.gate : g.gate + group].reshape(-1, c.dim)
        w[p + "q_layernorm.weight"] = tensor("q_norm")[j]
        w[p + "k_layernorm.weight"] = tensor("k_norm")[j]
    for i in range(c.layers):
        p = f"model.layers.{i}."
        w[p + ("self_attn" if c.is_attention(i) else "conv") + ".out_proj.weight"] = out[i]
        w[p + "feed_forward.w1.weight"], w[p + "feed_forward.w3.weight"] = from_panels(w13[i], 2)
        w[p + "feed_forward.w2.weight"] = w2[i]
        w[p + "operator_norm.weight"] = tensor("op_norm")[i]
        w[p + "ffn_norm.weight"] = tensor("ffn_norm")[i]
        if c.sandwich_norm:
            w[p + "post_operator_norm.weight"] = tensor("post_op_norm")[i]
            w[p + "post_ffn_norm.weight"] = tensor("post_ffn_norm")[i]
    w["model.embed_tokens.weight"] = w["lm_head.weight"] = projection("head")
    w["model.embedding_norm.weight"] = tensor("final_norm")
    return c, w


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="the checkpoint directory (safetensors + config.json)")
    p.add_argument("--out", required=True, help="write the bundle here")
    args = p.parse_args()
    c = Config.from_hf(args.model)
    w = load_hf_weights(args.model, dtype=None)
    head = w.get("lm_head.weight")
    assert head is None or head is w["model.embed_tokens.weight"], "the LM head is the embedding"
    changed = total = 0
    worst = 0.0
    for t in (t for name, t in w.items() if name != "lm_head.weight"):
        d = (t.half().float() - t.float()).abs()
        changed += int((d > 0).sum())
        total += t.numel()
        worst = max(worst, float(d.max()))
    print(f"fp16 moves {changed} of {total} weights ({100 * changed / total:.3f} %), by at most {worst:.3g}")
    bundle.write(args.out, c.to_dict(), build_tensors(c, w), {})
    print(f"wrote {args.out}: {os.path.getsize(args.out) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
