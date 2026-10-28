"""LFM2 on Hexagon (examples/hexagon/lfm2): the bundle and the decode kernel,
on a small random model with the real one's structure (tiny.py)."""

import os
import sys
from pathlib import Path

import pytest
import torch

import tilelang
from tilelang import tvm
from tilelang.backend.target import determine_target

LFM2 = Path(__file__).resolve().parents[3] / "examples/hexagon/lfm2"
sys.path.insert(0, str(LFM2))
import bundle  # noqa: E402
from decode import Decoder, make_decode  # noqa: E402
from quantize import BundleModel, build_tensors  # noqa: E402
from tiny import tiny  # noqa: E402

DEVICE = pytest.mark.skipif(os.environ.get("TL_HEXAGON_DEVICE_TEST") != "1", reason="requires an authorized Hexagon device and SDK")


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


def test_decode_is_one_cycle_of_phases(tiny_bundle):
    c, _, _, path = tiny_bundle
    header = bundle.read_header(path)
    shapes = {k: tuple(v["shape"]) for k, v in header["tensors"].items()}
    target = determine_target("hexagon", return_object=True)
    with tvm.target.Target(target), tvm.transform.PassContext(opt_level=3):
        art = tilelang.lower(make_decode(c, shapes, 256, 64), target=target, enable_host_codegen=False, enable_device_compile=False)
    source = art.kernel_source
    # Four barriers a layer and one after the head; the GEMVs are calls.
    assert source.count("tl_grid_sync(") == 4 * c.layers + 1
    assert "TL_GEMM" not in source and "tl_hvx_gemm_int_scaled<" in source


def _reference(model, tokens, n, n_forced):
    state = model.new_state()
    out, logits = list(tokens[:n_forced]), None
    for t in range(n):
        logits = model.logits(model.run(torch.tensor([out[t]]), state)[-1])
        if t + 1 >= n_forced:
            out.append(int(logits.argmax()))
    return out, logits


@DEVICE
def test_decode_matches_the_reference(tiny_bundle):
    _, _, _, path = tiny_bundle
    prompt = [1, 523, 77, 1999, 3001, 12, 8, 4000]
    n_forced, n = len(prompt), len(prompt) + 8 - 1
    with torch.no_grad():
        ref_tokens, ref_logits = _reference(BundleModel(path), prompt, n, n_forced)
    decoder = Decoder(path, max_pos=256, max_tokens=64)
    try:
        for _ in range(2):  # the second call starts over from reset state
            decoder.reset()
            tokens, logits = decoder.run(prompt, 0, n, n_forced)
            assert tokens == ref_tokens
            torch.testing.assert_close(logits, ref_logits, rtol=1e-3, atol=1e-3 * ref_logits.abs().max().item())
    finally:
        decoder.close()
