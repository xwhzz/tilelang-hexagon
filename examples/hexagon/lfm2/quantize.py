"""Quantize LFM2 with GPTQ and write the device bundle; measure its perplexity.

    python quantize.py --model LiquidAI-LFM2-1.2B-dir --out lfm2.bundle
    python quantize.py --model ... --eval lfm2.bundle --text wikitext2_test.txt

Projections become symmetric 4-bit codes with one fp16 scale per `group`
columns (GPTQ, layer by layer: each layer is quantized against the outputs of
the quantized layers before it). The tied embedding / LM head is 8-bit, one
scale per row: 4 bits there cost about 1.4 % perplexity, 8 bits 69 MB per
token more. WikiText-2 perplexity (40 windows of 1024 tokens; bf16 16.29):

| projections | head | ppl |
|---|---|---:|
| 4-bit, groups of 64, GPTQ | 8-bit | 16.68 |
| 4-bit, groups of 128, GPTQ | 8-bit | 16.82 |
| 4-bit, groups of 64, round to nearest | 8-bit | 17.14 |

`--eval` runs the bundle through the reference model with the device's
arithmetic: its dequantized weights, and each projection's input rounded to
int16 with one scale per row, as the decode kernels do.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bundle  # noqa: E402
from model import Config, Model, load_hf_weights  # noqa: E402

from tilelang.hexagon.intrinsics.panel import pack_panel, unpack_panel  # noqa: E402

# Columns of a streamed block: a projection wider than this is stored in
# slices of SLICE columns, each block `block` rows of one slice.
SLICE = 2048
# Rows per block: 128 KB of 4-bit codes, or of the 8-bit LM head's.
BLOCK = 128
HEAD_BLOCK = 64


# ---------------------------------------------------------------------------
# GPTQ
# ---------------------------------------------------------------------------


def _scale_zero(w: torch.Tensor, bits: int):
    """Symmetric: scale per row of w [N, g], zero 2^(bits-1)."""
    maxq = 2**bits - 1
    m = torch.maximum(-w.amin(1, keepdim=True).clamp(max=0), w.amax(1, keepdim=True).clamp(min=0))
    return (2 * m / maxq).clamp_min(1e-12), float(2 ** (bits - 1))


@torch.no_grad()
def gptq(W: torch.Tensor, H: torch.Tensor, bits: int, group: int | None, blocksize: int = 128, percdamp: float = 0.01):
    """Codes [N, K] (uint8, u = q + zero) and scales [N, K / group] of W [N, K]
    for the input Hessian H [K, K]."""
    W = W.clone()
    N, K = W.shape
    H = H.clone()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0
    H += percdamp * torch.mean(torch.diag(H)) * torch.eye(K)
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    g = K if group is None else group
    maxq = 2**bits - 1
    codes = torch.zeros(N, K, dtype=torch.uint8)
    scales = torch.zeros(N, K // g)
    scale = zero = None
    for i1 in range(0, K, blocksize):
        i2 = min(i1 + blocksize, K)
        W1 = W[:, i1:i2].clone()
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        for i in range(i2 - i1):
            col = i1 + i
            if col % g == 0:
                # The group's weights as GPTQ has updated them so far.
                span = W1[:, i : i + g] if g <= blocksize else torch.cat([W1[:, i:], W[:, i2 : col + g]], 1)
                scale, zero = _scale_zero(span, bits)
                scales[:, col // g] = scale[:, 0]
            w = W1[:, i : i + 1]
            q = (w / scale).round().add(zero).clamp(0, maxq)
            codes[:, col] = q[:, 0].to(torch.uint8)
            err = (w - (q - zero) * scale) / Hinv1[i, i]
            W1[:, i:] -= err @ Hinv1[i : i + 1, i:]
            Err1[:, i : i + 1] = err
        W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]
    return codes, scales


def calibrate(model_dir: str, text: str, nsamples: int, bits: int, group: int, head_bits: int, device: str):
    """GPTQ every projection of the checkpoint: {hf weight name: (codes, scales, bits, group)}."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir)
    hf = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32).to(device).eval()
    ids = tok(open(text).read(), add_special_tokens=False, return_tensors="pt").input_ids[0]
    step = 1023
    wins = [torch.cat([torch.tensor([tok.bos_token_id]), ids[i * step : (i + 1) * step]]) for i in range(min(nsamples, len(ids) // step))]
    inner = hf.model
    out = {}
    t0 = time.time()

    def quantize(mods, run, spec):
        Hs, n, hooks = {}, {}, []
        for name, m in mods.items():

            def hook(mod, inp, name=name):
                x = inp[0].detach().float().reshape(-1, inp[0].shape[-1])
                Hs[name] = Hs.get(name, 0) + (x.t() @ x).cpu()
                n[name] = n.get(name, 0) + x.shape[0]

            hooks.append(m.register_forward_pre_hook(hook))
        run()
        for h in hooks:
            h.remove()
        for name, m in mods.items():
            b, g = spec
            codes, scales = gptq(m.weight.data.float().cpu(), 2 * Hs[name] / n[name], b, g)
            zero = 2 ** (b - 1)
            m.weight.data.copy_(((codes.float() - zero) * scales.repeat_interleave(m.weight.shape[1] // scales.shape[1], 1)).to(device))
            out[name + ".weight"] = (codes, scales, b, g)

    with torch.no_grad():
        hs = [inner.embed_tokens(x[None].to(device)) for x in wins]
        pos = torch.arange(hs[0].shape[1], device=device)[None]
        pe = inner.rotary_emb(hs[0], position_ids=pos)
        for li, layer in enumerate(inner.layers):
            mods = {f"model.layers.{li}.{n}": m for n, m in layer.named_modules() if isinstance(m, torch.nn.Linear)}

            def run(layer=layer, hs=hs):
                for h in hs:
                    layer(h, position_embeddings=pe, attention_mask=None, position_ids=pos)

            quantize(mods, run, (bits, group))
            hs = [layer(h, position_embeddings=pe, attention_mask=None, position_ids=pos) for h in hs]
            print(f"layer {li} quantized ({time.time() - t0:.0f} s)", flush=True)

        def run_head():
            for h in hs:
                hf.lm_head(inner.embedding_norm(h))

        quantize({"lm_head": hf.lm_head}, run_head, (head_bits, None))
    return out


# ---------------------------------------------------------------------------
# The bundle
# ---------------------------------------------------------------------------


def sliced(codes: torch.Tensor, scales: torch.Tensor, block: int, group: int) -> tuple:
    """Codes [L, N, K] and scales [L, N, K / group] as the bundle stores them:
    codes [L, S, N, C], S slices of C = min(K, SLICE) columns, and fp16 scales
    [L, S, N / block, C / group, block], group-major within each block."""
    L, N, K = codes.shape
    C = min(K, SLICE)
    S = K // C
    c = codes.view(L, N, S, C).transpose(1, 2).contiguous()
    g = scales.view(L, N, S, C // group).transpose(1, 2)
    g = g.reshape(L, S, N // block, block, C // group).transpose(3, 4).contiguous().half()
    return c, g


def build_tensors(c: Config, weights: dict, q: dict):
    """The bundle's tensors from fp32 `weights` and GPTQ results `q`."""
    conv = [i for i in range(c.layers) if not c.is_attention(i)]
    attn = list(c.attention_layers)
    tensors, quant = {}, {}

    def projection(name, layers, parts):
        """Stack, over `layers`, the rows of the projections `parts(i)` of each layer i."""
        ws = [[q[f"model.layers.{i}.{p}.weight"] for p in parts(i)] for i in layers]
        codes = torch.stack([torch.cat([w[0] for w in layer]) for layer in ws])
        scales = torch.stack([torch.cat([w[1] for w in layer]) for layer in ws])
        bits, group = ws[0][0][2:]
        # Signed levels: T.gemm_blockscaled takes the codes as they are.
        levels, sc = sliced(codes.to(torch.int64) - 2 ** (bits - 1), scales, BLOCK, group)
        tensors[name] = (pack_panel(levels, bits), f"int{bits}", tuple(levels.shape), "panel")
        tensors[name + ".scale"] = (sc, "float16", tuple(sc.shape), "row_major")
        quant[name] = {"bits": bits, "group": group, "block": BLOCK}

    every = range(c.layers)
    projection("conv.in_proj", conv, lambda i: ["conv.in_proj"])
    projection("attn.qkv", attn, lambda i: ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"])
    projection("out_proj", every, lambda i: ["self_attn.out_proj" if c.is_attention(i) else "conv.out_proj"])
    projection("ffn.w13", every, lambda i: ["feed_forward.w1", "feed_forward.w3"])
    projection("ffn.w2", every, lambda i: ["feed_forward.w2"])

    # The tied LM head: signed 8-bit levels, one fp16 scale per row, in blocks
    # of HEAD_BLOCK rows; and the embedding's rows, row-major, for lookups.
    hc, hs, hb, _ = q["lm_head.weight"]
    hq = hc.to(torch.int64) - 2 ** (hb - 1)
    V, D = hq.shape
    levels, sc = sliced(hq[None], hs[None], HEAD_BLOCK, D)
    tensors["head"] = (pack_panel(levels, hb), f"int{hb}", tuple(levels.shape), "panel")
    tensors["head.scale"] = (sc, "float16", tuple(sc.shape), "row_major")
    quant["head"] = {"bits": hb, "group": D, "block": HEAD_BLOCK}
    tensors["embed"] = (hq.to(torch.int8).contiguous(), "int8", (V, D), "row_major")
    tensors["embed.scale"] = (hs[:, 0].float().contiguous(), "float32", (V,), "row_major")

    def stack(name, key, layers):
        t = torch.stack([weights[key.format(i)] for i in layers]).float().contiguous()
        tensors[name] = (t, "float32", tuple(t.shape), "row_major")

    stack("op_norm", "model.layers.{}.operator_norm.weight", range(c.layers))
    stack("ffn_norm", "model.layers.{}.ffn_norm.weight", range(c.layers))
    stack("q_norm", "model.layers.{}.self_attn.q_layernorm.weight", attn)
    stack("k_norm", "model.layers.{}.self_attn.k_layernorm.weight", attn)
    conv_w = torch.stack([weights[f"model.layers.{i}.conv.conv.weight"][:, 0, :].t() for i in conv]).contiguous()
    tensors["conv"] = (conv_w, "float32", tuple(conv_w.shape), "row_major")
    fn = weights["model.embedding_norm.weight"].float().contiguous()
    tensors["final_norm"] = (fn, "float32", tuple(fn.shape), "row_major")
    return tensors, quant


class BundleModel(Model):
    """The reference model on a bundle, with the device's arithmetic."""

    def __init__(self, path: str):
        header = bundle.read_header(path)
        c = Config(**{k: tuple(v) if isinstance(v, list) else v for k, v in header["config"].items()})
        self.header = header
        w = {}
        conv = [i for i in range(c.layers) if not c.is_attention(i)]
        attn = list(c.attention_layers)

        def dequant(name):
            t = header["tensors"][name]
            qinfo = header["quant"][name]
            L, S, N, C = t["shape"]
            codes = unpack_panel(bundle.read_tensor(path, header, name), t["shape"], qinfo["bits"], True).float()
            s = bundle.read_tensor(path, header, name + ".scale").float()  # [L, S, nb, C / group, block]
            s = s.transpose(3, 4).reshape(L, S, N, C // qinfo["group"]).repeat_interleave(qinfo["group"], -1)
            return (codes * s).transpose(1, 2).reshape(L, N, S * C)

        inp, qkv, out, w13, w2 = (dequant(n) for n in ("conv.in_proj", "attn.qkv", "out_proj", "ffn.w13", "ffn.w2"))
        q, k = c.heads * c.head_dim, c.kv_heads * c.head_dim
        for j, i in enumerate(conv):
            w[f"model.layers.{i}.conv.in_proj.weight"] = inp[j]
            w[f"model.layers.{i}.conv.conv.weight"] = bundle.read_tensor(path, header, "conv")[j].t()[:, None, :]
        for j, i in enumerate(attn):
            p = f"model.layers.{i}.self_attn."
            w[p + "q_proj.weight"], w[p + "k_proj.weight"], w[p + "v_proj.weight"] = qkv[j][:q], qkv[j][q : q + k], qkv[j][q + k :]
            w[p + "q_layernorm.weight"] = bundle.read_tensor(path, header, "q_norm")[j]
            w[p + "k_layernorm.weight"] = bundle.read_tensor(path, header, "k_norm")[j]
        for i in range(c.layers):
            p = f"model.layers.{i}."
            w[p + ("self_attn" if c.is_attention(i) else "conv") + ".out_proj.weight"] = out[i]
            w[p + "feed_forward.w1.weight"], w[p + "feed_forward.w3.weight"] = w13[i][: c.ff], w13[i][c.ff :]
            w[p + "feed_forward.w2.weight"] = w2[i]
            w[p + "operator_norm.weight"] = bundle.read_tensor(path, header, "op_norm")[i]
            w[p + "ffn_norm.weight"] = bundle.read_tensor(path, header, "ffn_norm")[i]
        w["lm_head.weight"] = dequant("head")[0]
        embed = bundle.read_tensor(path, header, "embed").float()
        w["model.embed_tokens.weight"] = embed * bundle.read_tensor(path, header, "embed.scale")[:, None]
        w["model.embedding_norm.weight"] = bundle.read_tensor(path, header, "final_norm")
        super().__init__(c, w)

    def linear(self, name: str, x: torch.Tensor) -> torch.Tensor:
        # The input rounded to int16 with one scale per row, as on the device.
        s = x.abs().amax(-1, keepdim=True).clamp_min(1e-30) / 32767
        return super().linear(name, (x / s).round() * s)


def perplexity(model: Model, ids: torch.Tensor, bos: int, nwin: int) -> float:
    step = 1023
    nll, count = 0.0, 0
    for i in range(min(nwin, len(ids) // step)):
        x = torch.cat([torch.tensor([bos]), ids[i * step : (i + 1) * step]])
        h = model.run(x, model.new_state())
        logits = model.logits(h[:-1])
        nll += torch.nn.functional.cross_entropy(logits, x[1:], reduction="sum").item()
        count += step
    return math.exp(nll / count)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="the checkpoint directory (safetensors + config.json)")
    p.add_argument("--out", help="write the bundle here")
    p.add_argument("--calib", help="calibration text (default: WikiText-2 train)")
    p.add_argument("--samples", type=int, default=64, help="calibration windows of 1024 tokens")
    p.add_argument("--bits", type=int, default=4)
    p.add_argument("--group", type=int, default=64)
    p.add_argument("--head-bits", type=int, default=8)
    p.add_argument("--eval", help="measure the perplexity of this bundle")
    p.add_argument("--text", help="evaluation text (default: WikiText-2 test)")
    p.add_argument("--windows", type=int, default=40)
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = p.parse_args()

    if args.out:
        c = Config.from_hf(args.model)
        q = calibrate(args.model, args.calib or _wikitext("train"), args.samples, args.bits, args.group, args.head_bits, args.device)
        tensors, quant = build_tensors(c, load_hf_weights(args.model), q)
        config = {k: list(v) if isinstance(v, tuple) else v for k, v in vars(c).items()}
        bundle.write(args.out, config, tensors, quant)
        print(f"wrote {args.out}: {os.path.getsize(args.out) / 2**20:.0f} MB")
    if args.eval:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.model)
        ids = tok(open(args.text or _wikitext("test")).read(), add_special_tokens=False, return_tensors="pt").input_ids[0]
        model = BundleModel(args.eval)
        print(f"{args.eval}: WikiText-2 ppl {perplexity(model, ids, model.c.bos, args.windows):.4f}")


def _wikitext(split: str) -> str:
    from datasets import load_dataset

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"wikitext2_{split}.txt")
    if not os.path.exists(path):
        rows = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)["text"]
        open(path, "w").write("\n\n".join(rows if split == "test" else rows[:20000]))
    return path


if __name__ == "__main__":
    main()
