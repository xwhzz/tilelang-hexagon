"""A small random LFM2 with every structure of the real one, for tests.

``tiny()`` returns a config, fp32 weights under the checkpoint's names and
their round-to-nearest quantization in ``quantize.calibrate``'s format: the
same bundle, decode kernel and reference as LFM2-1.2B, at a size that builds
in seconds. Its layers are conv, attention, conv; 16 query heads share 8 KV
heads, so each of 4 workers owns two whole KV heads.
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import Config  # noqa: E402


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


def tiny(seed: int = 0) -> tuple:
    """(config, weights, quantized projections) of a small random LFM2."""
    c = Config(vocab=4096, dim=1024, ff=2048, layers=3, heads=16, kv_heads=8, head_dim=64, attention_layers=(1,))
    torch.manual_seed(seed)
    D, F, V, KV = c.dim, c.ff, c.vocab, c.kv_heads * c.head_dim

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
        if c.is_attention(i):
            for name, rows in (("q_proj", D), ("k_proj", KV), ("v_proj", KV), ("out_proj", D)):
                w[p + f"self_attn.{name}.weight"] = linear(rows, D)
            w[p + "self_attn.q_layernorm.weight"], w[p + "self_attn.k_layernorm.weight"] = norm(c.head_dim), norm(c.head_dim)
        else:
            w[p + "conv.in_proj.weight"], w[p + "conv.out_proj.weight"] = linear(3 * D, D), linear(D, D)
            w[p + "conv.conv.weight"] = torch.randn(D, 1, c.conv_taps) * 0.5
        w[p + "feed_forward.w1.weight"], w[p + "feed_forward.w3.weight"] = linear(F, D), linear(F, D)
        w[p + "feed_forward.w2.weight"] = linear(D, F)
    q = {name: rtn(t, 4, 64) for name, t in w.items() if name.endswith("_proj.weight") or ".feed_forward." in name}
    q["lm_head.weight"] = rtn(w["lm_head.weight"], 8, None)
    return c, w, q
