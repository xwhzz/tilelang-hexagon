"""LFM2 and LFM3 in plain PyTorch, with the state a device decode keeps: the reference.

Both stack gated short-conv blocks and GQA attention blocks, each followed by a
SwiGLU FFN, with pre-norm residuals; LFM3 also norms each branch's output
(sandwich norm):

    x = x + PostNorm(Op(Norm(x))),   x = x + PostNorm(FFN(Norm(x)))

- A short-conv block projects to B, C and x (in_proj), convolves B * x with a
  causal depthwise kernel over time (3 taps in LFM2, 7 in LFM3), gates the
  result with C and projects it back (out_proj). Its state is the last
  taps - 1 values of B * x.
- An attention block has per-head RMSNorm on q and k, RoPE, and a KV cache.
  LFM3 has two kinds: full attention, whose RoPE rotates the first half of
  each head's frequencies (the rest stay unrotated), and sliding attention
  over the last 256 positions, with its own RoPE base. An LFM3 head's output
  is gated by sigmoid(gate_proj(x)), one gate per head.

``Model`` runs a prompt (``prefill``) and then one token at a time
(``decode``) over explicit state, so that the device kernels can be checked
step by step. ``linear`` is the one place a projection happens: a
quantized model swaps in the device's arithmetic there.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field

import torch

CONV, FULL, SLIDING = "conv", "full_attention", "sliding_attention"


@dataclass
class Config:
    vocab: int = 65536
    dim: int = 2048
    ff: int = 8192
    heads: int = 32
    kv_heads: int = 8
    head_dim: int = 64
    layer_types: tuple = tuple(FULL if i in (2, 5, 8, 10, 12, 14) else CONV for i in range(16))
    conv_taps: int = 3
    # Per attention kind: (RoPE base, the fraction of the head's dimensions rotated).
    rope: dict = field(default_factory=lambda: {FULL: (1e6, 1.0)})
    window: int = 0  # sliding attention's positions, the current one included
    sandwich_norm: bool = False
    head_gate: bool = False
    eps: float = 1e-5
    bos: int = 1
    eos: int = 7

    @property
    def layers(self) -> int:
        return len(self.layer_types)

    @property
    def attention_layers(self) -> tuple:
        return tuple(i for i, kind in enumerate(self.layer_types) if kind != CONV)

    def is_attention(self, layer: int) -> bool:
        return self.layer_types[layer] != CONV

    def to_dict(self) -> dict:
        """The config as JSON values, for a bundle's header."""
        d = dict(vars(self))
        d["layer_types"] = list(self.layer_types)
        d["rope"] = {k: list(v) for k, v in self.rope.items()}
        return d

    @classmethod
    def from_dict(cls, d: dict) -> Config:
        d = dict(d)
        d["layer_types"] = tuple(d["layer_types"])
        d["rope"] = {k: tuple(v) for k, v in d["rope"].items()}
        return cls(**d)

    @classmethod
    def from_hf(cls, path: str) -> Config:
        c = json.load(open(os.path.join(path, "config.json")))
        if c["model_type"] == "lfm2":
            ff = c["block_ff_dim"]
            if c.get("block_auto_adjust_ff_dim"):
                ff = int(2 * ff / 3)
                ff = int(c.get("block_ffn_dim_multiplier") or 1.0) * ff
                m = c["block_multiple_of"]
                ff = m * ((ff + m - 1) // m)
            attn = set(c["full_attn_idxs"])
            return cls(
                vocab=c["vocab_size"],
                dim=c["hidden_size"],
                ff=ff,
                heads=c["num_attention_heads"],
                kv_heads=c["num_key_value_heads"],
                head_dim=c["hidden_size"] // c["num_attention_heads"],
                layer_types=tuple(FULL if i in attn else CONV for i in range(c["num_hidden_layers"])),
                conv_taps=c["conv_L_cache"],
                rope={FULL: (c["rope_theta"], 1.0)},
                eps=c["norm_eps"],
                bos=c["bos_token_id"],
                eos=c["eos_token_id"],
            )
        if c["model_type"] == "lfm3":
            rope = {k: (v["rope_theta"], v.get("partial_rotary_factor", 1.0)) for k, v in c["rope_parameters"].items()}
            return cls(
                vocab=c["vocab_size"],
                dim=c["hidden_size"],
                ff=c["intermediate_size"],
                heads=c["num_attention_heads"],
                kv_heads=c["num_key_value_heads"],
                head_dim=c["head_dim"],
                layer_types=tuple(c["layer_types"]),
                conv_taps=c["conv_L_cache"],
                rope=rope,
                window=c["sliding_window"],
                sandwich_norm=c["use_sandwich_norm"],
                head_gate=True,
                eps=c["norm_eps"],
                bos=c["bos_token_id"],
                eos=c["eos_token_id"],
            )
        raise ValueError(f"not an LFM2 or LFM3 checkpoint: {c['model_type']}")


def is_lfm2(c: Config) -> bool:
    """Whether c has LFM2's structure: 3-tap convs and full attention with
    whole-head RoPE, no sandwich norm and no head gate."""
    return c.conv_taps == 3 and set(c.layer_types) <= {CONV, FULL} and c.rope[FULL][1] == 1.0 and not (c.sandwich_norm or c.head_gate)


def load_hf_weights(path: str, dtype: torch.dtype | None = torch.float32) -> dict:
    """The checkpoint's tensors in `dtype` (None: as stored), under one set of
    names: LFM3's q_norm/k_norm/o_proj are LFM2's q_layernorm/k_layernorm/out_proj."""
    from safetensors import safe_open

    rename = {
        ".self_attn.q_norm.": ".self_attn.q_layernorm.",
        ".self_attn.k_norm.": ".self_attn.k_layernorm.",
        ".self_attn.o_proj.": ".self_attn.out_proj.",
    }
    weights = {}
    for name in sorted(os.listdir(path)):
        if name.endswith(".safetensors"):
            with safe_open(os.path.join(path, name), "pt") as f:
                for k in f.keys():
                    v = f.get_tensor(k)
                    for a, b in rename.items():
                        k = k.replace(a, b)
                    weights[k] = v if dtype is None else v.to(dtype)
    # The LM head is the embedding.
    weights.setdefault("lm_head.weight", weights["model.embed_tokens.weight"])
    return weights


@dataclass
class State:
    """What decode carries from token to token."""

    pos: int
    # [attention layer] -> [kv_heads, positions, head_dim]
    k: list = field(default_factory=list)
    v: list = field(default_factory=list)
    # [conv layer] -> [taps - 1, dim]: the last B * x, oldest first
    conv: list = field(default_factory=list)


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight


def rope_angles(positions: torch.Tensor, head_dim: int, theta: float, fraction: float) -> torch.Tensor:
    """[positions, head_dim / 2]: the first fraction * head_dim / 2 frequencies
    of the whole head's progression, zero past them."""
    n = int(fraction * head_dim) // 2
    inv = 1.0 / (theta ** (torch.arange(0, 2 * n, 2, dtype=torch.float64) / head_dim))
    inv = torch.cat([inv, torch.zeros(head_dim // 2 - n, dtype=torch.float64)])
    return positions.to(torch.float64)[:, None] * inv[None, :]


def rope(x: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """x [..., positions, head_dim], rotating its two halves."""
    d = x.shape[-1]
    cos = torch.cat([angles.cos(), angles.cos()], -1).to(x.dtype)
    sin = torch.cat([angles.sin(), angles.sin()], -1).to(x.dtype)
    x1, x2 = x[..., : d // 2], x[..., d // 2 :]
    return x * cos + torch.cat([-x2, x1], -1) * sin


class Model:
    """LFM2 or LFM3 on explicit state; ``linear(name, x)`` computes ``x @ W[name].T``."""

    def __init__(self, config: Config, weights: dict):
        self.c = config
        self.w = weights

    def linear(self, name: str, x: torch.Tensor) -> torch.Tensor:
        return x @ self.w[name].t()

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear("lm_head.weight", rms_norm(x, self.w["model.embedding_norm.weight"], self.c.eps))

    def embed(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.w["model.embed_tokens.weight"][tokens]

    def new_state(self) -> State:
        c = self.c
        s = State(pos=0)
        for layer in range(c.layers):
            if c.is_attention(layer):
                s.k.append(torch.zeros(c.kv_heads, 0, c.head_dim))
                s.v.append(torch.zeros(c.kv_heads, 0, c.head_dim))
            else:
                s.conv.append(torch.zeros(c.conv_taps - 1, c.dim))
        return s

    def _conv(self, layer: int, x: torch.Tensor, state: torch.Tensor) -> tuple:
        """x [T, dim] -> ([T, dim], new state)."""
        p = f"model.layers.{layer}.conv."
        b, cgate, xin = self.linear(p + "in_proj.weight", x).chunk(3, dim=-1)
        bx = b * xin
        seq = torch.cat([state, bx], 0)  # [taps - 1 + T, dim]
        w = self.w[p + "conv.weight"][:, 0, :]  # [dim, taps]
        taps = self.c.conv_taps
        T = x.shape[0]
        y = sum(seq[j : j + T] * w[:, j] for j in range(taps))
        return self.linear(p + "out_proj.weight", cgate * y), seq[-(taps - 1) :]

    def _attention(self, layer: int, x: torch.Tensor, k_cache, v_cache, pos: int) -> tuple:
        c = self.c
        kind = c.layer_types[layer]
        p = f"model.layers.{layer}.self_attn."
        T = x.shape[0]
        q = self.linear(p + "q_proj.weight", x).view(T, c.heads, c.head_dim).transpose(0, 1)
        k = self.linear(p + "k_proj.weight", x).view(T, c.kv_heads, c.head_dim).transpose(0, 1)
        v = self.linear(p + "v_proj.weight", x).view(T, c.kv_heads, c.head_dim).transpose(0, 1)
        q = rms_norm(q, self.w[p + "q_layernorm.weight"], c.eps)
        k = rms_norm(k, self.w[p + "k_layernorm.weight"], c.eps)
        angles = rope_angles(torch.arange(pos, pos + T), c.head_dim, *c.rope[kind])
        q, k = rope(q, angles), rope(k, angles)
        k_cache, v_cache = torch.cat([k_cache, k], 1), torch.cat([v_cache, v], 1)
        group = c.heads // c.kv_heads
        kk = k_cache.repeat_interleave(group, 0)
        vv = v_cache.repeat_interleave(group, 0)
        scores = (q @ kk.transpose(1, 2)) / math.sqrt(c.head_dim)
        L = k_cache.shape[1]
        key, query = torch.arange(L)[None, :], torch.arange(T)[:, None] + pos
        mask = key > query
        if kind == SLIDING:
            mask |= key <= query - c.window
        scores = scores.masked_fill(mask, float("-inf"))
        out = (scores.softmax(-1) @ vv).transpose(0, 1)  # [T, heads, head_dim]
        if c.head_gate:
            out = out * torch.sigmoid(self.linear(p + "gate_proj.weight", x))[:, :, None]
        return self.linear(p + "out_proj.weight", out.reshape(T, c.heads * c.head_dim)), k_cache, v_cache

    def _ffn(self, layer: int, x: torch.Tensor) -> torch.Tensor:
        p = f"model.layers.{layer}.feed_forward."
        return self.linear(p + "w2.weight", torch.nn.functional.silu(self.linear(p + "w1.weight", x)) * self.linear(p + "w3.weight", x))

    def run(self, tokens: torch.Tensor, state: State) -> torch.Tensor:
        """Run `tokens` [T] from `state`; returns the hidden states [T, dim]."""
        c = self.c
        x = self.embed(tokens)
        ai = ci = 0
        for layer in range(c.layers):
            p = f"model.layers.{layer}."
            h = rms_norm(x, self.w[p + "operator_norm.weight"], c.eps)
            if c.is_attention(layer):
                out, state.k[ai], state.v[ai] = self._attention(layer, h, state.k[ai], state.v[ai], state.pos)
                ai += 1
            else:
                out, state.conv[ci] = self._conv(layer, h, state.conv[ci])
                ci += 1
            if c.sandwich_norm:
                out = rms_norm(out, self.w[p + "post_operator_norm.weight"], c.eps)
            x = x + out
            f = self._ffn(layer, rms_norm(x, self.w[p + "ffn_norm.weight"], c.eps))
            if c.sandwich_norm:
                f = rms_norm(f, self.w[p + "post_ffn_norm.weight"], c.eps)
            x = x + f
        state.pos += tokens.shape[0]
        return x

    def prefill(self, tokens: torch.Tensor) -> tuple:
        """The logits after `tokens`, and the state they leave."""
        state = self.new_state()
        return self.logits(self.run(tokens, state)[-1]), state

    def decode(self, token: int, state: State) -> torch.Tensor:
        return self.logits(self.run(torch.tensor([token]), state)[-1])

    def generate(self, tokens: torch.Tensor, n: int) -> list:
        """Greedy: the next `n` tokens after `tokens`."""
        logits, state = self.prefill(tokens)
        out = []
        for _ in range(n):
            token = int(logits.argmax())
            out.append(token)
            if token == self.c.eos:
                break
            logits = self.decode(token, state)
        return out
