"""LFM2 on Hexagon (examples/hexagon/lfm2): the bundle, and the decode and the
prefill kernels, on a small random model with the real one's structure
(tiny.py)."""

import os
import sys
from pathlib import Path

import pytest
import torch

import tilelang
from tilelang import tvm
from tilelang.backend.target import determine_target
from tilelang.hexagon.runtime.session import lower_kernels

LFM2 = Path(__file__).resolve().parents[3] / "examples/hexagon/lfm2"
sys.path.insert(0, str(LFM2))
import bundle  # noqa: E402
from decode import Decoder, make_decode  # noqa: E402
from model import Model  # noqa: E402
from prefill import make_prefill  # noqa: E402
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
    """The tokens of n greedy steps from tokens[:n_forced], the last step's
    logits, and the log-probabilities of the forced next tokens."""
    state = model.new_state()
    out, logits, scores = list(tokens[:n_forced]), None, []
    for t in range(n):
        logits = model.logits(model.run(torch.tensor([out[t]]), state)[-1])
        if t + 1 >= n_forced:
            out.append(int(logits.argmax()))
        else:
            scores.append(torch.log_softmax(logits.double(), -1)[out[t + 1]].item())
    return out, logits, scores


@DEVICE
def test_decode_matches_the_reference(tiny_bundle):
    _, _, _, path = tiny_bundle
    prompt = [1, 523, 77, 1999, 3001, 12, 8, 4000]
    n_forced, n = len(prompt), len(prompt) + 8 - 1
    with torch.no_grad():
        ref_tokens, ref_logits, ref_scores = _reference(BundleModel(path), prompt, n, n_forced)
    decoder = Decoder(path, max_pos=256, max_tokens=64)
    try:
        for _ in range(2):  # the second call starts over from reset state
            decoder.reset()
            tokens, logits = decoder.run(prompt, 0, n, n_forced)
            assert tokens == ref_tokens
            torch.testing.assert_close(logits, ref_logits, rtol=1e-3, atol=1e-3 * ref_logits.abs().max().item())
        # The log-probabilities of the prompt's tokens, every one forced.
        scores = decoder.score(prompt)
        torch.testing.assert_close(torch.tensor(scores), torch.tensor(ref_scores), rtol=1e-3, atol=1e-3)
    finally:
        decoder.close()


@DEVICE
def test_decode_samples_at_a_temperature(tiny_bundle):
    _, _, _, path = tiny_bundle
    prompt = [1, 523, 77, 1999, 3001, 12, 8, 4000]
    decoder = Decoder(path, max_pos=256, max_tokens=64)
    try:
        greedy = decoder.generate(prompt, 12)
        # Near zero, the noise cannot move the argmax.
        assert decoder.generate(prompt, 12, temperature=1e-6, seed=5) == greedy
        a = decoder.generate(prompt, 12, temperature=1.0, seed=1)
        assert decoder.generate(prompt, 12, temperature=1.0, seed=1) == a
        assert decoder.generate(prompt, 12, temperature=1.0, seed=2) != a
        # A stream of decode calls gives the tokens of one call.
        assert [t for part in decoder.stream(prompt, 12, temperature=1.0, seed=1, chunk=5) for t in part] == a
    finally:
        decoder.close()


def test_prefill_is_one_loop_over_the_layers(tiny_bundle):
    c, _, _, path = tiny_bundle
    header = bundle.read_header(path)
    shapes = {k: tuple(v["shape"]) for k, v in header["tensors"].items()}
    ((_, source, _, _),) = lower_kernels([(make_prefill(c, shapes, 128, 256, 64), [])])
    # A loop at run time, whatever the depth; the projections are calls of
    # the HMX GEMM, and the kernel's own pass configs give it fast math.
    assert source.count(f"for (int32_t layer = 0; layer < {c.layers}; ++layer)") == 1
    assert "tl_hexagon_hmx_gemm_blockscaled<" in source
    assert "tl_hvx_exp2_fast(" in source and "tl_hvx_exp2(" not in source


@DEVICE
def test_prefill_matches_the_reference(tiny_bundle):
    c, _, _, path = tiny_bundle
    ref = BundleModel(path)
    plain = Model(ref.c, ref.w)  # the bundle's weights, fp32 activations
    torch.manual_seed(1)
    n, max_pos = 200, 512  # a chunk of 128 rows and part of another
    prompt = torch.randint(0, c.vocab, (n,))
    La, KVH, HD = len(c.attention_layers), c.kv_heads, c.head_dim
    decoder = Decoder(path, max_pos=max_pos, max_tokens=max_pos, prefill_rows=128)
    try:
        decoder.reset()
        decoder.prefill(prompt.tolist())
        state = plain.new_state()
        with torch.no_grad():
            plain.run(prompt, state)

        def rel(a, b):
            return ((a - b).abs().max() / b.abs().max()).item()

        k = torch.frombuffer(bytearray(decoder.kv_k.read()), dtype=torch.float32).view(La, KVH, max_pos, HD)
        v = torch.frombuffer(bytearray(decoder.kv_v.read()), dtype=torch.float32).view(La, KVH, max_pos, HD)
        conv = torch.frombuffer(bytearray(decoder.conv_state.read()), dtype=torch.float32).view(-1, 2, c.dim)
        for ai in range(La):
            assert rel(k[ai, :, :n], state.k[ai]) < 5e-3 and rel(v[ai, :, :n], state.v[ai]) < 5e-3
        for ci in range(conv.shape[0]):
            assert rel(conv[ci], state.conv[ci]) < 5e-3
        # The decode goes on from the prefilled context.
        with torch.no_grad():
            want = ref.generate(prompt, 8)
        assert decoder.generate(prompt.tolist(), 8) == want
    finally:
        decoder.close()
