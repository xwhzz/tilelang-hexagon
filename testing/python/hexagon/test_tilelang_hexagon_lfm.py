"""LFM2 and LFM3 on Hexagon (examples/hexagon/lfm): the reference, the
bundles, and the 4-bit decode and prefill kernels, on small random models
with the real ones' structures (tiny.py)."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

import tilelang
from tilelang import tvm
from tilelang.backend.target import determine_target
from tilelang.hexagon.runtime.config import device_serial, remote_dir
from tilelang.hexagon.runtime.session import lower_kernels

LFM = Path(__file__).resolve().parents[3] / "examples/hexagon/lfm"
sys.path.insert(0, str(LFM))
import bundle  # noqa: E402
import convert  # noqa: E402
import decode_fp16  # noqa: E402
import prefill_fp16  # noqa: E402
from decode import Decoder, make_decode  # noqa: E402
from model import Config, Model, load_hf_weights  # noqa: E402
from prefill import make_prefill  # noqa: E402
from quantize import BundleModel, build_tensors  # noqa: E402
from tiny import tiny, tiny_lfm3  # noqa: E402

DEVICE = pytest.mark.skipif(os.environ.get("TL_HEXAGON_DEVICE_TEST") != "1", reason="requires an authorized Hexagon device and SDK")


@pytest.fixture(scope="module")
def tiny_bundle(tmp_path_factory):
    c, w, q = tiny()
    tensors, quant = build_tensors(c, w, q)
    path = str(tmp_path_factory.mktemp("lfm2") / "tiny.bundle")
    config = c.to_dict()
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


@pytest.fixture(scope="module")
def tiny_lfm3_bundle(tmp_path_factory):
    c, w = tiny_lfm3()
    path = str(tmp_path_factory.mktemp("lfm3") / "tiny.bundle")
    bundle.write(path, c.to_dict(), convert.build_tensors(c, w), {})
    return c, w, path


@pytest.mark.parametrize("checkpoint", ["LFM2_CHECKPOINT", "LFM3_CHECKPOINT"])
def test_reference_matches_transformers(checkpoint):
    """Set LFM2_CHECKPOINT to a LiquidAI/LFM2-1.2B checkout, LFM3_CHECKPOINT
    to a LiquidAI/LFM3-1.7B one (its own modeling code: transformers 5)."""
    if not os.environ.get(checkpoint):
        pytest.skip(f"set {checkpoint}")
    transformers = pytest.importorskip("transformers")
    path = os.environ[checkpoint]
    hf = transformers.AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32, trust_remote_code=True).eval()
    ref = Model(Config.from_hf(path), load_hf_weights(path))
    tok = transformers.AutoTokenizer.from_pretrained(path)
    # Past LFM3's sliding window of 256 positions.
    text = "The capital of France is Paris. The capital of Germany is Berlin. " * 24 + "The capital of Italy is"
    ids = torch.tensor([ref.c.bos] + tok(text, add_special_tokens=False).input_ids)
    with torch.no_grad():
        want = hf(ids[None]).logits[0]
        got = ref.logits(ref.run(ids, ref.new_state()))
        torch.testing.assert_close(got, want, rtol=0, atol=1e-4 * want.abs().max().item())
        # Greedy: LFM3's generation config has a repetition penalty.
        greedy = hf.generate(ids[None], max_new_tokens=12, do_sample=False, repetition_penalty=1.0)
        assert ref.generate(ids, 12) == greedy[0, len(ids) :].tolist()


def test_reference_decodes_past_the_window():
    """Decode steps after a prefill give the logits of one prefill of them
    all, across the sliding window (48 positions in the tiny LFM3)."""
    c, w = tiny_lfm3()
    model = Model(c, w)
    torch.manual_seed(1)
    ids = torch.randint(0, c.vocab, (120,))
    with torch.no_grad():
        want = model.logits(model.run(ids, model.new_state()))
        logits, state = model.prefill(ids[:90])
        got = [logits] + [model.decode(int(t), state) for t in ids[90:119]]
    torch.testing.assert_close(torch.stack(got), want[89:119], rtol=0, atol=1e-5 * want.abs().max().item())


def test_crouton_weight_is_the_hmx_weight_layout():
    from tilelang.hexagon.intrinsics.crouton import make_hmx_weight_layout

    L, K, N = 2, 64, 96
    layout = make_hmx_weight_layout((L, K, N))
    shape = [int(s) for s in layout.get_output_shape()]
    w = torch.arange(L * N * K, dtype=torch.float32).reshape(L, N, K)  # projections [N, K]
    tiles = convert.crouton_weight(w).reshape(-1)
    torch.testing.assert_close(convert.from_crouton_weight(tiles, (L, K, N)), w, rtol=0, atol=0)
    for i, k, n in ((0, 0, 0), (0, 1, 0), (0, 0, 1), (0, 31, 31), (0, 32, 5), (1, 33, 40), (1, 2, 64), (1, 63, 95)):
        flat = 0
        for index, extent in zip(layout.map_forward_index([i, k, n]), shape):
            flat = flat * extent + int(index)
        assert tiles[flat] == w[i, n, k]


def test_fp16_bundle_holds_the_weights(tiny_lfm3_bundle):
    c, w, path = tiny_lfm3_bundle
    c2, w2 = convert.read_weights(path)
    assert c2 == c and w2.keys() == w.keys()
    for name, t in w.items():
        # Projections and the embedding in fp16, the rest in fp32.
        want = t.half().float() if t.dim() == 2 else t
        torch.testing.assert_close(w2[name], want, rtol=0, atol=0, msg=name)
    head = bundle.read_tensor(path, bundle.read_header(path), "head")
    for token in (0, 1, 31, 32, 1234, c.vocab - 1):
        assert torch.equal(convert.embedding(head, token, c.dim), w["model.embed_tokens.weight"][token].half())


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


@DEVICE
def test_a_decoder_reads_its_bundle_on_the_device(tiny_bundle, tmp_path):
    """With the bundle on the device, the host needs only its header."""
    _, _, _, path = tiny_bundle
    head = tmp_path / "head.bundle"
    with open(path, "rb") as f:
        head.write_bytes(f.read(bundle.read_header(path)["data_offset"]))
    adb = ["adb", "-s", device_serial()]
    on_device = f"{remote_dir('android')}/test_lfm_{os.getpid()}"
    subprocess.run(adb + ["push", path, f"{on_device}.bundle"], check=True, capture_output=True)
    subprocess.run(adb + ["push", str(head), f"{on_device}.head"], check=True, capture_output=True)
    prompt = [1, 523, 77, 1999, 3001, 12, 8, 4000]
    with torch.no_grad():
        want = BundleModel(path).generate(torch.tensor(prompt), 8)
    try:
        decoder = Decoder(str(head), f"{on_device}.bundle", max_pos=256, max_tokens=64)
        try:
            assert decoder.generate(prompt, 8) == want
            with pytest.raises(ValueError, match="ends inside"):
                bundle.load(decoder.session, str(head), f"{on_device}.head")
        finally:
            decoder.close()
    finally:
        subprocess.run(adb + ["shell", f"rm -f {on_device}.bundle {on_device}.head"], capture_output=True)


def _fp16_bundle(tmp_path, model: str) -> str:
    """The fp16 bundle of the tiny LFM2 or LFM3."""
    c, w = tiny_lfm3() if model == "lfm3" else tiny()[:2]
    path = str(tmp_path / f"{model}.bundle")
    bundle.write(path, c.to_dict(), convert.build_tensors(c, w), {})
    return path


@pytest.mark.parametrize("model", ["lfm2", "lfm3"])
def test_fp16_decode_is_one_cycle_of_phases(model, tmp_path):
    c = Config.from_dict(bundle.read_header(_fp16_bundle(tmp_path, model))["config"])
    target = determine_target("hexagon", return_object=True)
    with tvm.target.Target(target), tvm.transform.PassContext(opt_level=3):
        art = tilelang.lower(decode_fp16.make_decode(c, 256, 64), target=target, enable_host_codegen=False, enable_device_compile=False)
    source = art.kernel_source
    # Four barriers a layer and one after the head; the GEMVs are calls of
    # the weight-layout GEMV.
    assert source.count("tl_grid_sync(") == 4 * c.layers + 1
    assert "tl_hvx_gemm_weight<" in source and "TL_GEMM" not in source


@DEVICE
@pytest.mark.parametrize("model", ["lfm2", "lfm3"])
def test_fp16_decode_matches_the_reference(model, tmp_path):
    path = _fp16_bundle(tmp_path, model)
    c, w = convert.read_weights(path)
    ref = decode_fp16.DeviceModel(c, w)
    torch.manual_seed(2)
    # Past the tiny LFM3's sliding window of 48 positions.
    prompt = torch.randint(0, c.vocab, (60,)).tolist()
    n_forced, n = len(prompt), len(prompt) + 8 - 1
    with torch.no_grad():
        ref_tokens, ref_logits, ref_scores = _reference(ref, prompt, n, n_forced)
    decoder = Decoder(path, max_pos=256, max_tokens=128)
    try:
        tokens, logits = decoder.run(prompt, 0, n, n_forced)
        assert tokens == ref_tokens
        torch.testing.assert_close(logits, ref_logits, rtol=0, atol=2e-3 * ref_logits.abs().max().item())
        # The log-probabilities of the prompt's tokens, every one forced.
        scores = decoder.score(prompt)
        torch.testing.assert_close(torch.tensor(scores), torch.tensor(ref_scores), rtol=2e-3, atol=2e-3)
    finally:
        decoder.close()


@pytest.mark.parametrize("model", ["lfm2", "lfm3"])
def test_fp16_prefill_is_one_loop_over_the_layers(model, tmp_path):
    c = Config.from_dict(bundle.read_header(_fp16_bundle(tmp_path, model))["config"])
    ((_, source, _, _),) = lower_kernels([(prefill_fp16.make_prefill(c, 128, 256, 64), [])])
    # A loop at run time, whatever the depth; the projections are calls of
    # the HMX GEMM.
    assert source.count(f"for (int32_t layer = 0; layer < {c.layers}; ++layer)") == 1
    assert "tl_hexagon_hmx_gemm<" in source


@DEVICE
@pytest.mark.parametrize("model", ["lfm2", "lfm3"])
def test_fp16_prefill_matches_the_reference(model, tmp_path):
    path = _fp16_bundle(tmp_path, model)
    c, w = convert.read_weights(path)
    ref = decode_fp16.DeviceModel(c, w)
    torch.manual_seed(1)
    # A chunk of 128 rows and part of another, past the tiny LFM3's window.
    n, max_pos = 200, 512
    prompt = torch.randint(0, c.vocab, (n,))
    La, KVH, HD = len(c.attention_layers), c.kv_heads, c.head_dim
    decoder = Decoder(path, max_pos=max_pos, max_tokens=max_pos, prefill_rows=128)
    try:
        decoder.reset()
        decoder.prefill(prompt.tolist())
        state = ref.new_state()
        with torch.no_grad():
            ref.run(prompt, state)

        def rel(a, b):
            return ((a - b).abs().max() / b.abs().max()).item()

        # The cache's tiles (decode_fp16.kv_layouts): keys as the weight
        # operand [head_dim, positions] of q K^T, values as that of P V.
        k = torch.frombuffer(bytearray(decoder.kv_k.read()), dtype=torch.float16)
        k = convert.from_crouton_weight(k, (La, KVH, HD, max_pos)).float()
        v = torch.frombuffer(bytearray(decoder.kv_v.read()), dtype=torch.float16)
        v = convert.from_crouton_weight(v, (La, KVH, max_pos, HD)).transpose(-1, -2).float()
        conv = torch.frombuffer(bytearray(decoder.conv_state.read()), dtype=torch.float32).view(-1, c.conv_taps - 1, c.dim)
        errors = [rel(k[ai, :, :n], state.k[ai]) for ai in range(La)] + [rel(v[ai, :, :n], state.v[ai]) for ai in range(La)]
        errors += [rel(conv[ci], state.conv[ci]) for ci in range(conv.shape[0])]
        assert max(errors) < 1e-2, errors
        # The decode goes on from the prefilled context.
        with torch.no_grad():
            want = ref.generate(prompt, 8)
        assert decoder.generate(prompt.tolist(), 8) == want
    finally:
        decoder.close()


@DEVICE
@pytest.mark.parametrize("model", ["lfm2", "lfm3"])
def test_fp16_prefill_after_decode_steps(model, tmp_path):
    """A chat's second turn: a prefill, decode steps, then a prefill from where
    they stopped. The second prefill's attention reads by DMA the keys and
    values the decode wrote with scalar stores, which the write-through cache
    wrote back as they were written."""
    path = _fp16_bundle(tmp_path, model)
    c, w = convert.read_weights(path)
    ref = decode_fp16.DeviceModel(c, w)
    torch.manual_seed(3)
    seq = torch.randint(0, c.vocab, (150,))
    first, steps = 60, 20
    La, KVH, HD, max_pos = len(c.attention_layers), c.kv_heads, c.head_dim, 512
    decoder = Decoder(path, max_pos=max_pos, max_tokens=max_pos, prefill_rows=128)
    try:
        decoder.reset()
        decoder.prefill(seq[:first].tolist())
        decoder.run(seq[first : first + steps].tolist(), first, steps, steps)
        decoder.prefill(seq[first + steps :].tolist(), first + steps)
        state = ref.new_state()
        with torch.no_grad():
            ref.run(seq, state)

        def rel(a, b):
            return ((a - b).abs().max() / b.abs().max()).item()

        n = len(seq)
        k = torch.frombuffer(bytearray(decoder.kv_k.read()), dtype=torch.float16)
        k = convert.from_crouton_weight(k, (La, KVH, HD, max_pos)).float()
        v = torch.frombuffer(bytearray(decoder.kv_v.read()), dtype=torch.float16)
        v = convert.from_crouton_weight(v, (La, KVH, max_pos, HD)).transpose(-1, -2).float()
        conv = torch.frombuffer(bytearray(decoder.conv_state.read()), dtype=torch.float32).view(-1, c.conv_taps - 1, c.dim)
        errors = [rel(k[ai, :, :n], state.k[ai]) for ai in range(La)] + [rel(v[ai, :, :n], state.v[ai]) for ai in range(La)]
        errors += [rel(conv[ci], state.conv[ci]) for ci in range(conv.shape[0])]
        assert max(errors) < 1e-2, errors
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
