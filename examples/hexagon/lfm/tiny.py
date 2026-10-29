"""Small random LFM2 and LFM3 models with every structure of the real ones, for tests.

``tiny()`` returns an LFM2's config, fp32 weights under the checkpoint's names
and their round-to-nearest quantization in ``quantize.calibrate``'s format:
the same 4-bit bundle, decode kernel and reference as LFM2-1.2B, at a size
that builds in seconds. Its layers are conv, attention, conv; 16 query heads
share 8 KV heads, so each of 4 workers owns two whole KV heads.

``tiny_lfm3()`` returns an LFM3's config and weights, for the fp16 bundle
(``convert.py``): conv, full attention, conv, sliding attention, with
LFM3-1.7B's model dimension and heads (16 query heads of 128 share 4 KV
heads, one a worker), 7-tap convs, sandwich norms, head gates, and a window
of 48 positions.
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import CONV, FULL, SLIDING, Config  # noqa: E402


def rtn(w: torch.Tensor, bits: int, group: int | None) -> tuple:
    """Symmetric round to nearest of w [N, K]: (codes u = q + 2^(bits-1),
    scales per group, bits, group)."""
    N, K = w.shape
    g = K if group is None else group
    maxq = 2**bits - 1
    v = w.view(N, K // g, g)
    scale = (2 * v.abs().amax(-1) / maxq).clamp_min(1e-8)
    u = (v / scale[..., None]).round().add(2 ** (bits - 1)).clamp(0, maxq)
    return u.view(N, K).to(torch.uint8), scale, bits, g


def random_weights(c: Config, seed: int = 0) -> dict:
    """Random fp32 weights for config c, under the checkpoint's names."""
    torch.manual_seed(seed)
    D, F, V, HD = c.dim, c.ff, c.vocab, c.head_dim

    def linear(rows, cols):
        return torch.randn(rows, cols) / cols**0.5

    def norm(n):
        return 1 + 0.1 * torch.randn(n)

    # A small embedding, so that the layers, not the token, decide the logits.
    w = {"model.embed_tokens.weight": torch.randn(V, D) * 0.05, "model.embedding_norm.weight": norm(D)}
    w["lm_head.weight"] = w["model.embed_tokens.weight"]
    for i in range(c.layers):
        p = f"model.layers.{i}."
        w[p + "operator_norm.weight"], w[p + "ffn_norm.weight"] = norm(D), norm(D)
        if c.sandwich_norm:
            w[p + "post_operator_norm.weight"], w[p + "post_ffn_norm.weight"] = norm(D), norm(D)
        if c.is_attention(i):
            KV = c.kv_heads * HD
            for name, rows in (("q_proj", c.heads * HD), ("k_proj", KV), ("v_proj", KV), ("out_proj", D)):
                w[p + f"self_attn.{name}.weight"] = linear(rows, D)
            if c.head_gate:
                w[p + "self_attn.gate_proj.weight"] = linear(c.heads, D)
            w[p + "self_attn.q_layernorm.weight"], w[p + "self_attn.k_layernorm.weight"] = norm(HD), norm(HD)
        else:
            w[p + "conv.in_proj.weight"], w[p + "conv.out_proj.weight"] = linear(3 * D, D), linear(D, D)
            w[p + "conv.conv.weight"] = torch.randn(D, 1, c.conv_taps) * 0.5
        w[p + "feed_forward.w1.weight"], w[p + "feed_forward.w3.weight"] = linear(F, D), linear(F, D)
        w[p + "feed_forward.w2.weight"] = linear(D, F)
    return w


def tiny(seed: int = 0) -> tuple:
    """(config, weights, quantized projections) of a small random LFM2."""
    c = Config(vocab=4096, dim=1024, ff=2048, heads=16, kv_heads=8, head_dim=64, layer_types=(CONV, FULL, CONV))
    w = random_weights(c, seed)
    q = {name: rtn(t, 4, 64) for name, t in w.items() if name.endswith("_proj.weight") or ".feed_forward." in name}
    q["lm_head.weight"] = rtn(w["lm_head.weight"], 8, None)
    return c, w, q


def tiny_lfm3(seed: int = 0) -> tuple:
    """(config, weights) of a small random LFM3."""
    c = Config(
        vocab=4096,
        dim=2048,
        ff=1024,
        heads=16,
        kv_heads=4,
        head_dim=128,
        layer_types=(CONV, FULL, CONV, SLIDING),
        conv_taps=7,
        rope={FULL: (1e6, 0.5), SLIDING: (1e4, 1.0)},
        window=48,
        sandwich_norm=True,
        head_gate=True,
    )
    return c, random_weights(c, seed)
