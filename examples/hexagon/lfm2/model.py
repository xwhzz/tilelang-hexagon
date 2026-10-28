"""LFM2 in plain PyTorch, with the state a device decode keeps: the reference.

LFM2 (``LiquidAI/LFM2-1.2B``; LFM2.5 has the same architecture) stacks
gated short-conv blocks and GQA attention blocks, each followed by a SwiGLU
FFN, with pre-norm residuals:

    x = x + Op(RMSNorm(x)),   x = x + FFN(RMSNorm(x))

- A short-conv block projects to B, C and x (in_proj), convolves B * x with a
  causal depthwise kernel of 3 taps over time, gates the result with C and
  projects it back (out_proj). Its state is the last two B * x.
- An attention block has per-head RMSNorm on q and k, RoPE, and a KV cache.

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


@dataclass
class Config:
    vocab: int = 65536
    dim: int = 2048
    ff: int = 8192
    layers: int = 16
    heads: int = 32
    kv_heads: int = 8
    head_dim: int = 64
    attention_layers: tuple = (2, 5, 8, 10, 12, 14)
    conv_taps: int = 3
    rope_theta: float = 1e6
    eps: float = 1e-5
    bos: int = 1
    eos: int = 7

    @classmethod
    def from_hf(cls, path: str) -> Config:
        c = json.load(open(os.path.join(path, "config.json")))
        ff = c["block_ff_dim"]
        if c.get("block_auto_adjust_ff_dim"):
            ff = int(2 * ff / 3)
            ff = int(c.get("block_ffn_dim_multiplier") or 1.0) * ff
            m = c["block_multiple_of"]
            ff = m * ((ff + m - 1) // m)
        return cls(
            vocab=c["vocab_size"],
            dim=c["hidden_size"],
            ff=ff,
            layers=c["num_hidden_layers"],
            heads=c["num_attention_heads"],
            kv_heads=c["num_key_value_heads"],
            head_dim=c["hidden_size"] // c["num_attention_heads"],
            attention_layers=tuple(c["full_attn_idxs"]),
            conv_taps=c["conv_L_cache"],
            rope_theta=c["rope_theta"],
            eps=c["norm_eps"],
            bos=c["bos_token_id"],
            eos=c["eos_token_id"],
        )

    def is_attention(self, layer: int) -> bool:
        return layer in self.attention_layers


def load_hf_weights(path: str) -> dict:
    """The checkpoint's tensors as fp32, under its own names."""
    from safetensors.torch import load_file

    weights = {}
    for name in sorted(os.listdir(path)):
        if name.endswith(".safetensors"):
            weights.update({k: v.float() for k, v in load_file(os.path.join(path, name)).items()})
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


def rope(x: torch.Tensor, positions: torch.Tensor, theta: float) -> torch.Tensor:
    """x [..., positions, head_dim], rotating its two halves."""
    d = x.shape[-1]
    inv = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float64) / d))
    angles = positions.to(torch.float64)[:, None] * inv[None, :]
    cos = torch.cat([angles.cos(), angles.cos()], -1).to(x.dtype)
    sin = torch.cat([angles.sin(), angles.sin()], -1).to(x.dtype)
    x1, x2 = x[..., : d // 2], x[..., d // 2 :]
    return x * cos + torch.cat([-x2, x1], -1) * sin


class Model:
    """LFM2 on explicit state; ``linear(name, x)`` computes ``x @ W[name].T``."""

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
        p = f"model.layers.{layer}.self_attn."
        T = x.shape[0]
        q = self.linear(p + "q_proj.weight", x).view(T, c.heads, c.head_dim).transpose(0, 1)
        k = self.linear(p + "k_proj.weight", x).view(T, c.kv_heads, c.head_dim).transpose(0, 1)
        v = self.linear(p + "v_proj.weight", x).view(T, c.kv_heads, c.head_dim).transpose(0, 1)
        q = rms_norm(q, self.w[p + "q_layernorm.weight"], c.eps)
        k = rms_norm(k, self.w[p + "k_layernorm.weight"], c.eps)
        positions = torch.arange(pos, pos + T)
        q, k = rope(q, positions, c.rope_theta), rope(k, positions, c.rope_theta)
        k_cache, v_cache = torch.cat([k_cache, k], 1), torch.cat([v_cache, v], 1)
        group = c.heads // c.kv_heads
        kk = k_cache.repeat_interleave(group, 0)
        vv = v_cache.repeat_interleave(group, 0)
        scores = (q @ kk.transpose(1, 2)) / math.sqrt(c.head_dim)
        L = k_cache.shape[1]
        mask = torch.arange(L)[None, :] > (torch.arange(T)[:, None] + pos)
        scores = scores.masked_fill(mask, float("-inf"))
        out = (scores.softmax(-1) @ vv).transpose(0, 1).reshape(T, c.heads * c.head_dim)
        return self.linear(p + "out_proj.weight", out), k_cache, v_cache

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
            x = x + out
            x = x + self._ffn(layer, rms_norm(x, self.w[p + "ffn_norm.weight"], c.eps))
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
