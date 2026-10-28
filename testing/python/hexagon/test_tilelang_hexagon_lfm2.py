"""LFM2 on Hexagon (examples/hexagon/lfm2): the bundle, on a small random model
with the real one's structure (tiny.py)."""

import os
import sys
from pathlib import Path

import pytest
import torch

LFM2 = Path(__file__).resolve().parents[3] / "examples/hexagon/lfm2"
sys.path.insert(0, str(LFM2))
import bundle  # noqa: E402
from quantize import BundleModel, build_tensors  # noqa: E402
from tiny import tiny  # noqa: E402


@pytest.fixture(scope="module")
def tiny_bundle(tmp_path_factory):
    c, w, q = tiny()
    tensors, quant = build_tensors(c, w, q)
    path = str(tmp_path_factory.mktemp("lfm2") / "tiny.bundle")
    config = {k: list(v) if isinstance(v, tuple) else v for k, v in vars(c).items()}
    bundle.write(path, config, tensors, quant)
    return c, w, q, path


def test_bundle_holds_the_quantized_weights(tiny_bundle):
    c, w, q, path = tiny_bundle
    model = BundleModel(path)
    for name in (
        "model.layers.0.conv.in_proj.weight",
        "model.layers.1.self_attn.k_proj.weight",
        "model.layers.2.feed_forward.w2.weight",
        "lm_head.weight",
    ):
        codes, scales, bits, group = q[name]
        ref = (codes.float() - 2 ** (bits - 1)) * scales.half().float().repeat_interleave(group, 1)
        torch.testing.assert_close(model.w[name], ref, rtol=0, atol=0)
    torch.testing.assert_close(model.w["model.layers.0.conv.conv.weight"], w["model.layers.0.conv.conv.weight"])


@pytest.mark.skipif(not os.environ.get("LFM2_CHECKPOINT"), reason="set LFM2_CHECKPOINT to a LiquidAI/LFM2-1.2B checkout")
def test_reference_matches_transformers():
    transformers = pytest.importorskip("transformers")
    from model import Config, Model, load_hf_weights

    path = os.environ["LFM2_CHECKPOINT"]
    hf = transformers.AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).eval()
    ref = Model(Config.from_hf(path), load_hf_weights(path))
    ids = transformers.AutoTokenizer.from_pretrained(path)(
        "The capital of France is Paris. The capital of Germany is", return_tensors="pt"
    ).input_ids[0]
    with torch.no_grad():
        want = hf(ids[None]).logits[0, -1]
        got, _ = ref.prefill(ids)
        torch.testing.assert_close(got, want, rtol=0, atol=1e-4 * want.abs().max().item())
        assert ref.generate(ids, 12) == hf.generate(ids[None], max_new_tokens=12, do_sample=False)[0, len(ids) :].tolist()
