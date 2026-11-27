"""LFM2 and LFM3 on Hexagon (examples/hexagon/lfm): the reference, the
bundles, and the 4-bit decode and prefill kernels, on small random models
with the real ones' structures (tiny.py)."""

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import torch

import tilelang
from tilelang import tvm
from tilelang.backend.target import determine_target
from tilelang.hexagon import language as T
from tilelang.hexagon.intrinsics.crouton import make_hmx_activation_layout
from tilelang.hexagon.intrinsics.panel import make_hvx_panel_layout, pack_panel
from tilelang.hexagon.runtime.config import device_serial, domain_id, hlos, remote_dir
from tilelang.hexagon.runtime.device import AdbConnection
from tilelang.hexagon.runtime.linux_runtime import LinuxConnection
from tilelang.hexagon.runtime.session import lower_kernels

LFM = Path(__file__).resolve().parents[3] / "examples/hexagon/lfm"
sys.path.insert(0, str(LFM))
import bundle  # noqa: E402
import convert  # noqa: E402
import decode_fp16  # noqa: E402
import prefill_fp16  # noqa: E402
from decode import Decoder, add_gumbel, make_decode  # noqa: E402
from model import CONV, FULL, SLIDING, Config, Model, load_hf_weights, rope_angles  # noqa: E402
from prefill import make_prefill, pack_w8a8_activations, project_w8a8  # noqa: E402
from quantize import BundleModel, build_tensors  # noqa: E402
from tiny import random_weights, rtn, tiny, tiny_lfm3  # noqa: E402

DEVICE = pytest.mark.skipif(os.environ.get("TL_HEXAGON_DEVICE_TEST") != "1", reason="requires an authorized Hexagon device and SDK")
SIMULATOR = pytest.mark.skipif(os.environ.get("TL_HEXAGON_SIMULATOR_TEST") != "1", reason="requires the Hexagon SDK simulator")
FP16_SOURCE_TARGET = {"kind": "hexagon", "mcpu": "hexagonv79"}
MULTI_PROCESS = pytest.mark.skipif(
    hlos() == "android" and domain_id() != 3,
    reason="multiple DSP processes on Android require the default cDSP domain",
)


@pytest.mark.parametrize("opt_in", [False, True])
def test_interactive_chat_prefix_reuse_is_opt_in(monkeypatch, opt_in):
    from io import StringIO
    from types import SimpleNamespace

    import generate

    calls = []
    monkeypatch.setattr(generate, "Decoder", lambda *_, **__: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(generate, "Text", lambda *_: object())
    monkeypatch.setattr(generate, "reply", lambda *_, reuse=False: calls.append(reuse) or "answer")
    monkeypatch.setattr(sys, "stdin", StringIO("one\ntwo\n"))
    monkeypatch.setattr(sys, "argv", ["generate.py", "host", "device", "checkpoint", "--chat"] + (["--reuse-prefix"] if opt_in else []))
    generate.main()
    assert calls == [opt_in, opt_in]


@pytest.fixture(scope="module")
def tiny_bundle(tmp_path_factory):
    c, w, q = tiny()
    tensors, quant = build_tensors(c, w, q)
    path = str(tmp_path_factory.mktemp("lfm2") / "tiny.bundle")
    config = c.to_dict()
    bundle.write(path, config, tensors, quant)
    return c, w, q, path


@pytest.mark.parametrize(
    ("tensor", "field", "value"),
    [
        ("ffn.w13", "dtype", "int8"),
        ("ffn.w13.scale", "dtype", "float32"),
        ("ffn.w13.scale", "layout", "panel"),
        ("ffn.w13.scale", "shape", [1, 1, 1, 1, 128]),
    ],
)
def test_decoder_rejects_incompatible_w8a8_bundle(tiny_bundle, monkeypatch, tensor, field, value):
    *_, path = tiny_bundle
    header = bundle.read_header(path)
    header["tensors"][tensor][field] = value
    monkeypatch.setattr(bundle, "read_header", lambda _: header)
    with pytest.raises(ValueError, match="requires signed Q4 panels and row-major FP16 scales"):
        Decoder(path, w8a8_w13=True)


@pytest.mark.parametrize("prefix", [0, 2])
def test_decoder_scores_the_targets_after_the_forced_suffix_input(monkeypatch, prefix):
    from types import SimpleNamespace

    decoder = Decoder.__new__(Decoder)
    decoder.prefill_rows = 64 if prefix else None
    pending = []
    monkeypatch.setattr(decoder, "reset", lambda: pending.append("reset"))
    monkeypatch.setattr(decoder, "prefill", lambda tokens: pending.append(("prefill", tokens)))
    monkeypatch.setattr(decoder, "run", lambda *args: pending.append(("run", *args)))
    decoder.scores = SimpleNamespace(read=lambda size: bytes(size))
    tokens = [101, 11, 22, 33, 44]
    result = decoder.score(tokens, prefill_prefix=prefix)
    if prefix:
        assert pending == ["reset", ("prefill", [101, 11]), ("run", [22, 33, 44], 2, 2, 3)]
        assert len(result) == 2  # scores targets 33 and 44
    else:
        assert pending == ["reset", ("run", tokens, 0, 4, 5)]
        assert len(result) == 4


@pytest.mark.parametrize("scale", [0x0000, 0x8000, 0x0001, 0x8001, 0x7C00, 0xFC00, 0x7E01, 0x3C00, 0xBC00])
def test_bundle_scale_classification_and_header_only(tmp_path, scale):
    path = str(tmp_path / "scales.bundle")
    values = torch.tensor([0x3C00, 0xBC00, scale], dtype=torch.int32).to(torch.int16).view(torch.float16)
    header = bundle.write(path, {}, {"w.scale": (values, "float16", (3,), "row_major")}, {"w": {"bits": 4}})
    assert bundle.has_normal_scales(path, header) == (scale in (0x3C00, 0xBC00))
    with open(path, "r+b") as f:
        f.truncate(header["data_offset"])
    assert not bundle.has_normal_scales(path, header)


@pytest.mark.parametrize("scale", [1e-12, 2**-24, 2**-14, 0.1])
def test_quantization_records_stored_scale_classification(tmp_path, scale):
    # A small shape sufficient for packing; no device kernel is instantiated.
    c = Config(vocab=128, dim=128, ff=128, heads=2, kv_heads=1, head_dim=64, layer_types=(CONV, FULL))
    weights = random_weights(c)
    q = {name: rtn(t, 4, 64) for name, t in weights.items() if name.endswith("_proj.weight") or ".feed_forward." in name}
    q["lm_head.weight"] = rtn(weights["lm_head.weight"], 8, None)
    q["model.layers.0.conv.in_proj.weight"][1][0, 0] = scale
    tensors, quant = build_tensors(c, weights, q)
    expected = scale >= 2**-14
    assert quant["conv.in_proj"]["normal_scales"] is expected
    assert all(v["normal_scales"] for name, v in quant.items() if name != "conv.in_proj")
    path = str(tmp_path / "quantized.bundle")
    bundle.write(path, c.to_dict(), tensors, quant)
    header = bundle.read_header(path)
    assert bundle.has_normal_scales(path, header) is expected
    with open(path, "r+b") as f:
        f.truncate(header["data_offset"])
    assert bundle.has_normal_scales(path, bundle.read_header(path)) is expected


@pytest.mark.parametrize("recorded", [{}, {"a": True}, {"a": True, "b": True}, {"a": False, "b": True}, {"a": True, "b": False}])
def test_bundle_accepts_mixed_new_and_legacy_scale_metadata(tmp_path, recorded):
    path = str(tmp_path / "mixed.bundle")
    tensors = {name + ".scale": (torch.ones(2).half(), "float16", (2,), "row_major") for name in ("a", "b")}
    quant = {name: {"bits": 4, **({"normal_scales": recorded[name]} if name in recorded else {})} for name in ("a", "b")}
    bundle.write(path, {}, tensors, quant)
    header = bundle.read_header(path)
    assert bundle.has_normal_scales(path, header) is all(recorded.values())
    with open(path, "r+b") as f:
        f.truncate(header["data_offset"])
    assert bundle.has_normal_scales(path, header) is (len(recorded) == 2 and all(recorded.values()))


@pytest.mark.parametrize("recorded", ["true", 1, None, []])
def test_bundle_scale_metadata_requires_a_boolean(tmp_path, recorded):
    path = str(tmp_path / "invalid.bundle")
    header = bundle.write(path, {}, {}, {"w": {"bits": 4, "normal_scales": recorded}})
    with pytest.raises(ValueError, match="normal_scales must be a JSON boolean"):
        bundle.has_normal_scales(path, header)


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


@pytest.mark.parametrize("fractions", [(1.0, 0.5), (0.125, 0.0), (0.0, 0.0)])
def test_fp16_rope_tables_match_reference_across_chunks(monkeypatch, fractions):
    c = Config(head_dim=128, rope={SLIDING: (1e4, fractions[0]), FULL: (1e6, fractions[1])})
    positions = torch.arange(4099)
    layout = decode_fp16.RopeLayout(c)

    def chunk_angles(rows, *args, **kwargs):
        assert len(rows) < len(positions)
        return rope_angles(rows, *args, **kwargs)

    monkeypatch.setattr(decode_fp16, "rope_angles", chunk_angles)
    cos, sin = decode_fp16.rope_tables(c, len(positions))
    assert cos.shape == sin.shape == layout.shape(len(positions))
    assert cos.numel() * cos.element_size() == layout.nbytes(len(positions))
    for kind, span in layout.spans.items():
        reference = rope_angles(positions, c.head_dim, *c.rope[kind])
        if not layout.width:
            assert cos.item() == 1 and sin.item() == 0
            continue
        for actual, expected in ((cos, reference.cos()), (sin, reference.sin())):
            assert actual.dtype == torch.float32 and actual.is_contiguous()
            assert torch.equal(actual[:, span.start : span.stop].view(torch.int32), expected[:, : len(span)].float().view(torch.int32))
        assert (reference[:, len(span) :] == 0).all()


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


def test_fp16_projection_stream_matches_stacked_bundle(tmp_path):
    c = Config(vocab=32, dim=32, ff=32, heads=1, kv_heads=1, head_dim=32, layer_types=(CONV, FULL, CONV))
    w = random_weights(c)
    tensors = convert.build_tensors(c, w)
    tail = tensors["final_norm"]
    projection = tensors["ffn.w2"]
    streamed = tmp_path / "streamed.bundle"
    header = bundle.write(streamed, c.to_dict(), {"ffn.w2": projection, "final_norm": tail}, {})

    packed = convert.crouton_weight(torch.stack([w[f"model.layers.{i}.feed_forward.w2.weight"].half() for i in range(c.layers)]))
    assert torch.equal(next(projection[0]()), packed[0])
    stacked = tmp_path / "stacked.bundle"
    bundle.write(stacked, c.to_dict(), {"ffn.w2": (packed, *projection[1:]), "final_norm": tail}, {})
    assert header["tensors"]["ffn.w2"]["nbytes"] == 6144
    assert header["tensors"]["final_norm"]["offset"] == 8192
    assert streamed.read_bytes() == stacked.read_bytes()


def test_fp16_projection_checks_each_layer_shape(tmp_path):
    c = Config(vocab=32, dim=32, ff=64, heads=1, kv_heads=1, head_dim=32, layer_types=(CONV, FULL))
    w = random_weights(c)
    key = "model.layers.1.feed_forward.w2.weight"
    w[key] = w[key].T
    projection = convert.build_tensors(c, w)["ffn.w2"]
    with pytest.raises(ValueError, match="ffn.w2 layer 1 has shape"):
        bundle.write(tmp_path / "invalid.bundle", {}, {"ffn.w2": projection}, {})


@pytest.mark.parametrize("length", [2, 4])
def test_bundle_checks_streamed_byte_count(tmp_path, length):
    def chunks():
        yield torch.ones(2, dtype=torch.uint8)
        yield torch.ones(length, dtype=torch.uint8)

    with pytest.raises(ValueError, match="weights.*5"):
        bundle.write(tmp_path / "invalid.bundle", {}, {"weights": (chunks, "uint8", (5,), "row_major")}, {})


@pytest.mark.parametrize("arch", ("v73", "v75", "v79"))
@pytest.mark.parametrize("cooperative", (False, True))
def test_decode_is_one_cycle_of_phases(tiny_bundle, arch, cooperative):
    c, _, _, path = tiny_bundle
    header = bundle.read_header(path)
    shapes = {k: tuple(v["shape"]) for k, v in header["tensors"].items()}
    target = determine_target({"kind": "hexagon", "mcpu": f"hexagon{arch}"}, return_object=True)
    # Exercise the production default as well as the explicit ordinary opt-out.
    options = {} if cooperative else {"cooperative": False}
    with tvm.target.Target(target), tvm.transform.PassContext(opt_level=3):
        art = tilelang.lower(
            make_decode(c, shapes, 256, 64, **options), target=target, enable_host_codegen=False, enable_device_compile=False
        )
    source = art.kernel_source
    if cooperative:
        assert "#define TL_HEXAGON_USES_COOPERATIVE 1" in source
        assert "tl_cooperative_parallel_for<false>" in source
        assert source.count("tl_cooperative_checkpoint(") > 4 * c.layers + 1
        assert "tl_cooperative_dma_scope<decltype(tl_dma_async)>" in source
    else:
        # Four barriers a layer and one after the head; the GEMVs are calls.
        assert source.count("tl_grid_sync(") == 4 * c.layers + 1
        assert "TL_HEXAGON_USES_COOPERATIVE" not in source
    assert "tl_hvx_gemm_int_scaled<" in source


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


@pytest.mark.parametrize(
    "rows,capacity,n_prompt,n_new,chunk,fp16",
    [
        (None, 32, 1, 32, 32, False),
        (None, 8, 23, 15, 12, True),
        (None, 2, 5, 3, 1, False),
        (8, 11, 29, 18, 16, True),
        (8, 3, 10, 4, 16, False),
        (8, 3, 10, 4, 16, True),
        (None, 16, 4, 6, 16, False),
    ],
)
def test_stream_bounds_token_storage_without_repeating_positions(rows, capacity, n_prompt, n_new, chunk, fp16):
    class Calls(Decoder):
        def reset(self):
            self.consumed = []
            self.calls = []

        def prefill(self, ids, pos0):
            assert pos0 == len(self.consumed)
            assert len(ids) <= capacity
            assert self.fp16 or pos0 + -(-len(ids) // rows) * rows <= self.max_pos
            self.consumed.extend(ids)
            self.calls.append("prefill")

        def run(self, ids, pos0, n, n_forced, temperature, seed):
            assert pos0 == len(self.consumed) and n + 1 <= capacity
            assert n_forced == len(ids) and (temperature, seed) == (0.7, 21)
            out = list(ids)
            for step in range(n):
                self.consumed.append(out[step])
                if step + 1 >= n_forced:
                    out.append((out[step] + pos0 + step + seed) % 101)
            self.calls.append("decode")
            return out, None

    decoder = Calls.__new__(Calls)
    decoder.prefill_rows, decoder.max_tokens, decoder.max_pos, decoder.fp16 = rows, capacity, n_prompt + n_new - 1, fp16
    prompt, expected = list(range(n_prompt)), []
    for pos in range(n_prompt - 1, n_prompt - 1 + n_new):
        expected.append(((expected[-1] if expected else prompt[-1]) + pos + 21) % 101)
    parts = list(decoder.stream(prompt, n_new, 0.7, 21, chunk))
    assert all(1 <= len(part) <= chunk for part in parts)
    assert [token for part in parts for token in part] == expected
    assert decoder.consumed == prompt + expected[:-1]
    if not rows and n_prompt + n_new <= capacity:
        assert decoder.calls == ["decode"]
    assert decoder.generate(prompt, n_new, 0.7, 21) == expected


def test_generation_rejects_impossible_requests_before_mutating_state():
    decoder = Decoder.__new__(Decoder)
    decoder.max_pos, decoder.max_tokens = 4, 4
    assert decoder.generate([1], 0) == []
    for prompt, n, chunk in (([], 1, 1), ([1], -1, 1), ([1], 1, 0), ([1, 2], 4, 2)):
        with pytest.raises(ValueError):
            list(decoder.stream(prompt, n, chunk=chunk))
    decoder.max_tokens = 1
    with pytest.raises(ValueError, match="storage"):
        decoder.generate([1], 1)


@DEVICE
@pytest.mark.parametrize("model", ["quantized", "lfm3"])
def test_generation_with_short_token_buffer_matches_single_steps(model, tiny_bundle, fp16_bundle):
    path = tiny_bundle[-1] if model == "quantized" else fp16_bundle(model)
    decoder = Decoder(path, max_pos=256, max_tokens=8, prefill_rows=None)
    prompt = [1, 523, 77, 1999, 3001, 12, 8, 4000, 5, 93, 78]
    try:
        for temperature in (0, 0.8):
            decoder.reset()
            for pos, token in enumerate(prompt[:-1]):
                decoder.run([token, prompt[pos + 1]], pos, 1, 2, temperature, 17)
            expected, token = [], prompt[-1]
            for pos in range(len(prompt) - 1, len(prompt) - 1 + 15):
                out, _ = decoder.run([token], pos, 1, 1, temperature, 17)
                token = out[-1]
                expected.append(token)
            assert decoder.generate(prompt, 15, temperature, 17) == expected

        # Stop inside the first chunk, then append a turn longer than scratch.
        # The saved convolution state precedes prompt[-1]. The reused suffix
        # and sampled continuation must agree with a seeded full replay.
        first = next(decoder.stream(prompt, 10, 0.8, 17, chunk=5, reuse=True))
        second = prompt + [first[0], (first[1] + 1) % decoder.c.vocab] + prompt[:8]
        timings = []
        actual = [t for part in decoder.stream(second, 15, 0.8, 17, timings=timings, reuse=True) for t in part]
        assert sum(steps for _, steps, _ in timings) == len(second) - len(prompt) + 15
        assert actual == decoder.generate(second, 15, 0.8, 17)
    finally:
        decoder.close()


@DEVICE
def test_sampling_highest_bin():
    # At seed 26, position 0, the hash of vocabulary index 118408 ends
    # in the highest bin. Its fp32 midpoint rounds to 1 before clamping.
    @T.prim_func
    def sample(scores: T.Tensor((1, 32), "float32")):
        with T.Kernel(1):
            hx = T.alloc_shared((1, 32), "uint32")
            hy = T.alloc_shared((1, 32), "uint32")
            add_gumbel(scores, T.float32(1), 26, 0, 118400, hx, hy)

    kernel = tilelang.compile(sample, target="hexagon")
    try:
        scores = torch.zeros((1, 32))
        kernel(scores)
        assert scores.isfinite().all()
        expected = -torch.log(-torch.log(torch.tensor(1.0 - 2.0**-24)))
        torch.testing.assert_close(scores[0, 8], expected)
    finally:
        kernel.adapter.close()


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
    """Header-only metadata selects normal-scale prefill before device decode."""
    _, _, _, path = tiny_bundle
    head = tmp_path / "head.bundle"
    with open(path, "rb") as f:
        head.write_bytes(f.read(bundle.read_header(path)["data_offset"]))
    assert bundle.has_normal_scales(str(head), bundle.read_header(str(head)))
    connection_cls = LinuxConnection if hlos() == "linux" else AdbConnection
    connection = connection_cls(serial=device_serial(), workdir=f"{remote_dir(hlos())}/test_lfm_{os.getpid()}_{tmp_path.name}")
    prompt = [1, 523, 77, 1999, 3001, 12, 8, 4000]
    with torch.no_grad():
        want = BundleModel(path).generate(torch.tensor(prompt), 8)
    try:
        on_device = connection.push(path, "weights.bundle")
        on_device_head = connection.push(str(head), "header.bundle")
        decoder = Decoder(str(head), on_device, max_pos=256, max_tokens=256, prefill_rows=128)
        try:
            assert decoder.generate(prompt, 8) == want
            with pytest.raises(ValueError, match="ends inside"):
                bundle.load(decoder.session, str(head), on_device_head)
        finally:
            decoder.close()
    finally:
        connection.cleanup()


@pytest.fixture(scope="module")
def fp16_bundle(tmp_path_factory):
    """The fp16 bundle of the tiny LFM2 or LFM3 (`fp16_bundle(model)`), written
    once for the module: each is a few hundred MB."""
    paths = {}

    def path(model: str) -> str:
        if model not in paths:
            c, w = tiny_lfm3() if model == "lfm3" else tiny()[:2]
            paths[model] = str(tmp_path_factory.mktemp(f"fp16_{model}") / f"{model}.bundle")
            bundle.write(paths[model], c.to_dict(), convert.build_tensors(c, w), {})
        return paths[model]

    return path


@pytest.mark.parametrize("model", ["lfm2", "lfm3"])
@pytest.mark.parametrize("arch", ["v73", "v75", "v79"])
@pytest.mark.parametrize("cooperative", [False, True])
def test_fp16_decode_is_one_step_of_phases(model, arch, cooperative, fp16_bundle):
    c = Config.from_dict(bundle.read_header(fp16_bundle(model))["config"])
    target = determine_target({"kind": "hexagon", "mcpu": f"hexagon{arch}"}, return_object=True)
    options = {} if cooperative else {"cooperative": False}
    with tvm.target.Target(target), tvm.transform.PassContext(opt_level=3):
        art = tilelang.lower(
            decode_fp16.make_decode(c, 256, 64, **options), target=target, enable_host_codegen=False, enable_device_compile=False
        )
    source = art.kernel_source
    if cooperative:
        assert "tl_cooperative_parallel_for<false>" in source
        assert source.count("tl_cooperative_checkpoint(") > 4 * c.layers + 1
        assert "tl_cooperative_dma_scope<decltype(tl_dma_async)>" in source
        assert "tl_grid_sync(" not in source
    else:
        assert source.count("tl_grid_sync(") == 4 * c.layers + 1
        assert "TL_HEXAGON_USES_COOPERATIVE" not in source
    assert "tl_hvx_gemm_weight<" in source


def test_fp16_embeds_row_major_where_it_takes_no_process_more(fp16_bundle, monkeypatch):
    """The copy maps in the first of several stages instead of the head, and
    in one stage beside the head only if both fit a process."""
    header = bundle.read_header(fp16_bundle("lfm3"))
    c = Config.from_dict(header["config"])
    one, two = [range(4)], [range(2), range(2, 4)]
    head = header["tensors"]["head"]["nbytes"]
    alone = decode_fp16.stage_bytes(c, header, 256, range(4))
    assert decode_fp16.stage_bytes(c, header, 256, range(4), True) == alone + decode_fp16.embedding_nbytes(c)
    assert decode_fp16.stage_bytes(c, header, 256, range(2)) == decode_fp16.stage_bytes(c, header, 256, range(2), True)
    assert decode_fp16.embedding_nbytes(c) == head
    for budget, rows in ((alone + head, True), (alone + head - 1, False)):
        monkeypatch.setattr(decode_fp16, "PROCESS_BUDGET", budget)
        assert decode_fp16.row_major_embedding(c, header, 256, one) is rows
        assert decode_fp16.row_major_embedding(c, header, 256, two)


@pytest.mark.parametrize("cooperative", [False, True])
def test_fp16_stages_split_the_layers(cooperative, fp16_bundle):
    """A model over several DSP processes: each stage a run of the layers, the
    largest as small as can be; the embedding in the first, the head in the
    last, and the residual handed on between."""
    path = fp16_bundle("lfm3")
    header = bundle.read_header(path)
    c = Config.from_dict(header["config"])
    assert decode_fp16.plan_stages(c, header, 256) == [range(4)]
    assert decode_fp16.plan_stages(c, header, 256, 2) == [range(2), range(2, 4)]
    assert decode_fp16.plan_stages(c, header, 256, 4) == [range(i, i + 1) for i in range(4)]
    target = determine_target(FP16_SOURCE_TARGET, return_object=True)
    options = {} if cooperative else {"cooperative": False}
    for layers in (range(2), range(1, 3), range(2, 4), range(1), range(1, 2), range(3, 4)):
        with tvm.target.Target(target), tvm.transform.PassContext(opt_level=3):
            art = tilelang.lower(
                decode_fp16.make_decode(c, 256, 64, layers, **options),
                target=target,
                enable_host_codegen=False,
                enable_device_compile=False,
            )
        barrier = "tl_cooperative_checkpoint(" if cooperative else "tl_grid_sync("
        minimum = 4 * len(layers) + (layers.stop == c.layers)
        assert art.kernel_source.count(barrier) >= minimum
        # Conv-only, full-only and sliding-only stages must all retain uniform
        # checkpoints, including stages without the embedding or the LM head.
        ((_, stage_source, _, _),) = lower_kernels(
            [(prefill_fp16.make_prefill(c, 128, 256, 64, layers, **options), [])], target=FP16_SOURCE_TARGET
        )
        assert ("TL_HEXAGON_USES_COOPERATIVE" in stage_source) is cooperative
    ((_, source, _, _),) = lower_kernels(
        [(prefill_fp16.make_prefill(c, 128, 256, 64, range(1, 3), **options), [])], target=FP16_SOURCE_TARGET
    )
    assert source.count("for (int32_t layer = 0; layer < 2; ++layer)") == 1


# Runs over DSP processes: the tiny LFM3's layers two a stage, and the tiny
# LFM2's one a stage, of which the first and the last have no attention and
# the middle one neither the embedding nor the head.
STAGES = [
    ("lfm2", 1),
    ("lfm3", 1),
    pytest.param("lfm3", 2, marks=MULTI_PROCESS),
    pytest.param("lfm2", 3, marks=MULTI_PROCESS),
]


@DEVICE
@pytest.mark.parametrize(("model", "processes"), STAGES)
def test_fp16_decode_matches_the_reference(model, processes, fp16_bundle):
    path = fp16_bundle(model)
    c, w = convert.read_weights(path)
    ref = decode_fp16.DeviceModel(c, w)
    torch.manual_seed(2)
    # Past the tiny LFM3's sliding window of 48 positions.
    prompt = torch.randint(0, c.vocab, (60,)).tolist()
    n_forced, n = len(prompt), len(prompt) + 8 - 1
    with torch.no_grad():
        ref_tokens, ref_logits, ref_scores = _reference(ref, prompt, n, n_forced)
    decoder = Decoder(path, max_pos=256, max_tokens=128, processes=processes)
    assert len(decoder.stages) == processes
    try:
        tokens, logits = decoder.run(prompt, 0, n, n_forced)
        assert tokens == ref_tokens
        torch.testing.assert_close(logits, ref_logits, rtol=0, atol=2e-3 * ref_logits.abs().max().item())
        # The log-probabilities of the prompt's tokens, every one forced.
        scores = decoder.score(prompt)
        torch.testing.assert_close(torch.tensor(scores), torch.tensor(ref_scores), rtol=2e-3, atol=2e-3)
    finally:
        decoder.close()


@DEVICE
@pytest.mark.parametrize(
    ("processes", "blocks"), [(1, None), (1, 32), pytest.param(2, None, marks=MULTI_PROCESS), pytest.param(2, 32, marks=MULTI_PROCESS)]
)
def test_fp16_embedding_is_the_head_row_major(processes, blocks, fp16_bundle, monkeypatch):
    """The Decoder writes the embedding row-major from the head's tiles
    (decode_fp16.make_embedding): from the head the first process maps, or
    from the bundle a part at a time; in one call, or (`blocks`) several."""
    if blocks is not None:
        monkeypatch.setattr(decode_fp16, "embedding_blocks", lambda c: blocks)
    path = fp16_bundle("lfm3")
    c, w = convert.read_weights(path)
    decoder = Decoder(path, max_pos=256, max_tokens=128, prefill_rows=None, processes=processes)
    try:
        assert len(decoder.stages) == processes
        rows = torch.frombuffer(bytearray(decoder.embedding.read()), dtype=torch.float16).view(c.vocab, c.dim)
        assert torch.equal(rows, w["model.embed_tokens.weight"].half())
    finally:
        decoder.close()


def _assert_prefill_norm_path(source, direct=False):
    if not direct:
        assert re.search(r"\bvoid\* rows =", source)
        assert "tl_crouton_copy" in source
        return
    # Both paths scale FP32 values, round once to FP16 and store a whole
    # physical register into a, with no intermediate row-major buffer.
    assert not re.search(r"\bvoid\* rows =", source)
    stores = [line for line in source.splitlines() if re.search(r"\*\(half(?:32|64)\*\)\(a_w \+", line)]
    assert stores and all("nw" in line and "ss" in line for line in stores), stores
    assert not re.search(r"a_w\[[^\]]*\] =", source)


@pytest.mark.parametrize("model", ["lfm2", "lfm3"])
@pytest.mark.parametrize("arch", ["v73", "v75", "v79"])
@pytest.mark.parametrize("cooperative", [False, True])
def test_fp16_prefill_is_one_loop_over_the_layers(model, arch, cooperative, fp16_bundle):
    c = Config.from_dict(bundle.read_header(fp16_bundle(model))["config"])
    target = {"kind": "hexagon", "mcpu": f"hexagon{arch}"}
    options = {} if cooperative else {"cooperative": False}
    ((_, source, _, _),) = lower_kernels([(prefill_fp16.make_prefill(c, 128, 256, 64, **options), [])], target=target)
    # A loop at run time, whatever the depth; the projections are calls of
    # the HMX GEMM.
    assert source.count(f"for (int32_t layer = 0; layer < {c.layers}; ++layer)") == 1
    assert "tl_hexagon_hmx_gemm<" in source
    assert ("#define TL_HEXAGON_USES_COOPERATIVE 1" in source) is cooperative
    if cooperative:
        assert "tl_cooperative_parallel_for<true>" in source
        assert re.search(r"tl_cooperative_checkpoint\(\w+, tl_dma_async\)", source)
        assert "tl_grid_sync(" not in source
    _assert_prefill_norm_path(source)


@pytest.mark.parametrize("model", ["lfm2", "lfm3"])
@pytest.mark.parametrize("phase", ["decode", "prefill"])
@pytest.mark.parametrize("row_major", [True, False])
def test_fp16_kernels_lower_v73_dma_descriptors(model, phase, row_major, fp16_bundle):
    c = Config.from_dict(bundle.read_header(fp16_bundle(model))["config"])
    options = {"row_major_embedding": row_major}
    kernel = decode_fp16.make_decode(c, 256, 64, **options) if phase == "decode" else prefill_fp16.make_prefill(c, 128, 256, 64, **options)
    ((_, source, _, _),) = lower_kernels([(kernel, [])], target={"kind": "hexagon", "mcpu": "hexagonv73"})
    calls = [line for line in source.splitlines() if "tl_hexagon_dma_async_submit(" in line]
    geometry = [
        tuple(map(int, match.groups()))
        for line in calls
        if (match := re.search(r"\(uint32_t\)(\d+), \(uint32_t\)(\d+), \(uint32_t\)(\d+), \(uint32_t\)(\d+), \d+\)\)", line))
    ]
    assert len(geometry) == len(calls) and calls
    # The embedding is a row of the row-major copy (make_embedding), D fp16,
    # or D / 2 rows of a token's pair from the head's tiles.
    if row_major:
        assert (2 * c.dim, 2 * c.dim, 2 * c.dim, 1) in geometry
    else:
        assert (4, 128, 4, c.dim // 2) in geometry
    for dst_stride, src_stride, width, rows in geometry:
        if rows > 1:
            assert 0 < width <= min(dst_stride, src_stride) <= max(dst_stride, src_stride) <= 65535
            assert rows <= 65535
        else:
            assert rows == 1 and dst_stride == src_stride == width
    assert "tl_hexagon_dma_async_wait(" in source


def _kv_errors(decoder, state, n, rel) -> list:
    """The relative errors of each attention layer's keys and values in the
    device's caches against the reference's, at the positions of [0, n) the
    caches hold: all of them, or a sliding window's ring's."""
    errors = []
    for ai, (k, v, positions) in enumerate(decoder.kv_positions(n)):
        errors += [rel(k, state.k[ai][:, positions]), rel(v, state.v[ai][:, positions])]
    return errors


@DEVICE
@pytest.mark.parametrize(("model", "processes"), STAGES)
@pytest.mark.parametrize("row_major", [True, False])
def test_fp16_prefill_matches_the_reference(model, processes, row_major, fp16_bundle, monkeypatch):
    """The prefill, then the decode, against the reference: the first stage
    embeds from the row-major copy, or from the head's tiles where the copy
    would take a process more (decode_fp16.row_major_embedding)."""
    monkeypatch.setattr(decode_fp16, "row_major_embedding", lambda *args: row_major)
    path = fp16_bundle(model)
    c, w = convert.read_weights(path)
    ref = decode_fp16.DeviceModel(c, w)
    torch.manual_seed(1)
    # The tiny LFM2 ends in a convolution. A final chunk of one token keeps
    # one token from its previous state. The LFM3 runs past its sliding window.
    n, max_pos = (129 if model == "lfm2" else 200), 512
    prompt = torch.randint(0, c.vocab, (n,))
    decoder = Decoder(path, max_pos=max_pos, max_tokens=max_pos, prefill_rows=128, processes=processes)
    assert decoder.row_major_embedding is row_major
    try:
        decoder.reset()
        decoder.prefill(prompt.tolist())
        state = ref.new_state()
        with torch.no_grad():
            ref.run(prompt, state)

        def rel(a, b):
            return ((a - b).abs().max() / b.abs().max()).item()

        errors = _kv_errors(decoder, state, n, rel)
        conv = decoder.conv_states()
        errors += [rel(conv[ci], state.conv[ci]) for ci in range(conv.shape[0])]
        assert max(errors) < 1e-2, errors
        # The decode goes on from the prefilled context.
        with torch.no_grad():
            want = ref.generate(prompt, 8)
        assert decoder.generate(prompt.tolist(), 8) == want
    finally:
        decoder.close()


@pytest.mark.parametrize("rows", [None, 128])
def test_chat_reuses_only_consumed_state(rows):
    class TrackingDecoder(Decoder):
        def __init__(self):
            self.prefill_rows, self.fp16, self.max_pos, self.max_tokens = rows, False, 4096, 5
            self.calls = []
            self.reset()

        def reset(self):
            self._chat_tokens = None
            self.state = []

        def _copy_chat_conv(self, *, restore):
            if restore:
                self.state = self.saved.copy()
            else:
                self.saved = self.state.copy()

        def prefill(self, tokens, pos0=0):
            assert pos0 == len(self.state) and len(tokens) <= self.max_tokens
            self._chat_tokens = None
            self.calls.append((pos0, len(tokens)))
            self.state.extend(tokens)

        def run(self, tokens, pos0, n, n_forced, temperature=0.0, seed=0):
            assert pos0 == len(self.state) and n + 1 <= self.max_tokens
            self._chat_tokens = None
            self.calls.append((pos0, n))
            result = list(tokens)
            for i in range(n):
                self.state.append(result[i])
                if i + 1 >= n_forced:
                    result.append((sum(self.state) + len(self.state)) % 101)
            return result, None

    d = TrackingDecoder()
    prompt = [2, 5, 7, 11]
    stream = d.stream(prompt, 16, chunk=4, reuse=True)
    predicted = next(stream)
    assert d.state == prompt + predicted[:-1]
    stream.close()  # The caller can stop in the middle of the first result.

    def turn(tokens, start):
        d.calls.clear()
        (actual,) = d.stream(tokens, 1, reuse=True)
        assert d.calls[0][0] == start
        assert d.state == tokens
        reference = TrackingDecoder()
        (expected,) = reference.stream(tokens, 1, reuse=True)
        assert actual == expected

    # A chunk's final prediction has not advanced KV or the convolution.
    more = prompt + predicted + [1, 3]
    turn(more, len(prompt) + 3)
    # The template changed the just-consumed final prompt token. The input
    # before it is exactly the previous prefill checkpoint.
    changed = more[:-1] + [6, 7]
    turn(changed, len(more) - 1)
    # Early EOS discards multiple predictions that the DSP already consumed.
    chunk = next(d.stream(changed + [9, 10], 16, chunk=4, reuse=True))
    trimmed = changed + [9, 10, chunk[0], (chunk[1] + 1) % 101, 23]
    turn(trimmed, len(changed) + 1)
    turn([99, 87, 33], 0)  # Re-rendering changed even the saved prefix.
    d.reset()
    turn([99, 87, 33, 19], 0)


def test_chat_checkpoint_cannot_restore_overwritten_sliding_keys():
    from types import SimpleNamespace

    d = Decoder.__new__(Decoder)
    d.fp16, d.c = True, SimpleNamespace(window=48)
    limit = decode_fp16.ring(d.c) - d.c.window
    saved = 2200
    called = []
    d._chat_saved_pos = saved
    d._copy_chat_conv = lambda *, restore: called.append("restore")
    d.reset = lambda: called.append("reset")
    prompt = [2] * saved + [8, 3]
    for length, result in ((limit, saved), (limit + 1, 0)):
        d._chat_tokens = [2] * saved + [5] * length
        assert d._chat_position(prompt) == result
    assert called == ["restore", "reset"]


@DEVICE
@pytest.mark.parametrize("model,processes", [("quantized", 1), ("lfm2", 1), pytest.param("lfm3", 2, marks=MULTI_PROCESS)])
def test_chat_continuation_matches_full_replay(model, processes, tiny_bundle, fp16_bundle):
    path = tiny_bundle[-1] if model == "quantized" else fp16_bundle(model)
    decoder = Decoder(path, max_pos=512, max_tokens=256, prefill_rows=128, processes=processes)
    first = list(range(24, 84))
    try:
        predictions = next(decoder.stream(first, 16, chunk=8, reuse=True))
        # Treat the second returned ID as EOS. The remaining IDs in the
        # result still advanced the resident state despite the caller's break.
        second = first + predictions[:1] + [(predictions[1] + 1) % decoder.c.vocab, 53, 77]
        timings = []
        (actual,) = decoder.stream(second, 1, reuse=True, timings=timings)
        logits = torch.frombuffer(bytearray(decoder.logits.read()), dtype=torch.float32)
        assert timings[0][:2] == ("prefill", len(second) - len(first))
        # At a chunk boundary the whole consumed prefix matches. Only the
        # prediction and the next user input still need to run.
        third = second + actual + [55, 79]
        timings.clear()
        (third_token,) = decoder.stream(third, 1, reuse=True, timings=timings)
        third_logits = torch.frombuffer(bytearray(decoder.logits.read()), dtype=torch.float32)
        assert timings[0][:2] == ("prefill", 2)
        for prompt, token, actual_logits in ((second, actual, logits), (third, third_token, third_logits)):
            (expected,) = decoder.stream(prompt, 1)
            reference = torch.frombuffer(bytearray(decoder.logits.read()), dtype=torch.float32)
            assert token == expected
            torch.testing.assert_close(actual_logits, reference, rtol=0, atol=0.01 * reference.abs().max().item())
    finally:
        decoder.close()


@pytest.mark.parametrize(
    ("pos0", "n", "capacity", "calls"),
    [
        (128, 65, 65, [(1, 128, 65, 1)]),
        (255, 1, 1, [(0, 255, 1, 1)]),
        (120, 136, 136, [(1, 120, 128, 1), (0, 248, 8, 8)]),
    ],
)
def test_fp16_prefill_only_decodes_chunks_that_cross_the_cache_end(pos0, n, capacity, calls):
    from types import SimpleNamespace

    decoder = Decoder.__new__(Decoder)
    decoder.fp16, decoder.max_pos, decoder.max_tokens, decoder.prefill_rows = True, 256, capacity, 128
    decoder.c = SimpleNamespace(vocab=4096)
    writes, actual = [], []
    decoder.tokens = SimpleNamespace(write=lambda ids: writes.extend(ids.tolist()))
    decoder.stages = [SimpleNamespace(prefill=[], decode=[])]
    decoder.seconds = 99.0

    def pipeline(first, args, steps):
        actual.append((first, *args[0], steps))
        decoder.seconds = 1.0

    decoder._pipeline = pipeline
    ids = list(range(n))
    decoder.prefill(ids, pos0)
    assert writes == ids and actual == calls
    assert decoder.last_run_seconds() == len(calls)
    with pytest.raises(AssertionError):
        decoder.prefill([1], decoder.max_pos)
    assert writes == ids and actual == calls


@DEVICE
def test_fp16_incremental_prefill_at_context_end(fp16_bundle):
    path = fp16_bundle("lfm3")
    c, w = convert.read_weights(path)
    ref = decode_fp16.DeviceModel(c, w)
    seq = torch.randint(c.vocab, (256,), generator=torch.Generator().manual_seed(3))
    # The suffix ends inside the cache but its last physical 128-row chunk
    # would extend beyond it. The final decode consumes the last legal slot.
    with torch.no_grad():
        want, state = ref.prefill(seq)
    decoder = Decoder(path, max_pos=256, max_tokens=256, prefill_rows=128, processes=1)
    try:
        decoder.prefill(seq[:120].tolist())
        decoder.prefill(seq[120:-1].tolist(), 120)
        _, logits = decoder.run(seq[-1:].tolist(), 255, 1, 1)

        def relative(a, b):
            return ((a - b).abs().max() / b.abs().max()).item()

        errors = _kv_errors(decoder, state, len(seq), relative)
        states = decoder.conv_states()
        errors.extend(relative(got, expected) for got, expected in zip(states, state.conv))
        errors.append(relative(logits, want))
        assert all(error < 1e-2 for error in errors), errors
    finally:
        decoder.close()


@DEVICE
@pytest.mark.parametrize(("model", "processes"), STAGES[:3])
def test_fp16_prefill_after_decode_steps(model, processes, fp16_bundle):
    """A chat's second turn: a prefill, decode steps, then a prefill from where
    they stopped. The second prefill's attention reads by DMA the keys and
    values the decode wrote with scalar stores, which the write-through cache
    wrote back as they were written."""
    path = fp16_bundle(model)
    c, w = convert.read_weights(path)
    ref = decode_fp16.DeviceModel(c, w)
    torch.manual_seed(3)
    seq = torch.randint(0, c.vocab, (150,))
    first, steps = 60, 20
    max_pos = 512
    decoder = Decoder(path, max_pos=max_pos, max_tokens=max_pos, prefill_rows=128, processes=processes)
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
        errors = _kv_errors(decoder, state, n, rel)
        conv = decoder.conv_states()
        errors += [rel(conv[ci], state.conv[ci]) for ci in range(conv.shape[0])]
        assert max(errors) < 1e-2, errors
    finally:
        decoder.close()


@DEVICE
@MULTI_PROCESS
def test_fp16_stages_compute_what_one_process_does(fp16_bundle):
    """The tiny LFM3 over two DSP processes against one, after a prefill and
    greedy decode steps, and through forced decode steps: the same tokens,
    and caches and logits within about an fp16 step of each other, where a
    residual or a token that crossed wrong would move them by its whole size.
    Distinct stage kernels may round differently. In #220, adding a barrier
    exposed spilled qfloat temporaries in a Crouton reduction; changing
    worker order inside the same binary did not reproduce that difference."""
    path = fp16_bundle("lfm3")
    c = Config.from_dict(bundle.read_header(path)["config"])
    torch.manual_seed(7)
    prompt = torch.randint(0, c.vocab, (200,)).tolist()
    steps = 8
    runs = []
    for processes in (1, 2):
        decoder = Decoder(path, max_pos=512, max_tokens=512, prefill_rows=128, processes=processes)
        try:
            decoder.prefill(prompt[:-1])
            tokens, _ = decoder.run(prompt[-1:], len(prompt) - 1, steps, 1)
            caches = [t for k, v, _ in decoder.kv_positions(len(prompt) + steps - 1) for t in (k, v)]
            caches.append(decoder.conv_states())
            decoder.reset()
            _, logits = decoder.run(prompt[:41], 0, 40, 41)
            runs.append((tokens, caches, logits))
        finally:
            decoder.close()
    (tokens, caches, logits), (tokens2, caches2, logits2) = runs
    assert tokens == tokens2

    def rel(a, b):
        return ((a - b).abs().max() / b.abs().max()).item()

    errors = [rel(a, b) for a, b in zip(caches2, caches)] + [rel(logits2, logits)]
    assert len(caches) == 2 * len(c.attention_layers) + 1 and max(errors) < 2e-3, errors


@DEVICE
def test_fp16_prefill_and_decode_run_past_the_ring(fp16_bundle):
    """The sliding-window layers' caches are rings of 1024 positions
    (decode_fp16.ring): a prompt past one, and decode steps after it, read
    the window from the ring."""
    path = fp16_bundle("lfm3")
    c, w = convert.read_weights(path)
    ref = decode_fp16.DeviceModel(c, w)
    assert decode_fp16.ring(c) == 1024
    torch.manual_seed(5)
    n, max_pos = 1300, 2048
    prompt = torch.randint(0, c.vocab, (n,))
    decoder = Decoder(path, max_pos=max_pos, max_tokens=max_pos, prefill_rows=128)
    try:
        decoder.reset()
        decoder.prefill(prompt.tolist())
        state = ref.new_state()
        with torch.no_grad():
            ref.run(prompt, state)

        def rel(a, b):
            return ((a - b).abs().max() / b.abs().max()).item()

        errors = _kv_errors(decoder, state, n, rel)
        assert max(errors) < 1e-2, errors
        with torch.no_grad():
            want = ref.generate(prompt, 8)
        assert decoder.generate(prompt.tolist(), 8) == want
    finally:
        decoder.close()


@DEVICE
def test_fp16_prefill_from_decode_steps_wraps_the_ring(fp16_bundle):
    """A prefill that starts where decode steps stopped, at a position no
    chunk boundary: its first chunk's rows run past the ring's end, and its
    write goes again from the ring's start (prefill_fp16.store_kv)."""
    path = fp16_bundle("lfm3")
    c, w = convert.read_weights(path)
    ref = decode_fp16.DeviceModel(c, w)
    torch.manual_seed(6)
    seq = torch.randint(0, c.vocab, (1310,))
    first, steps, max_pos = 1000, 10, 2048
    assert (first + steps) % decode_fp16.ring(c) + 128 > decode_fp16.ring(c)
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

        errors = _kv_errors(decoder, state, len(seq), rel)
        assert max(errors) < 1e-2, errors
    finally:
        decoder.close()


@pytest.mark.parametrize("arch", ("v73", "v75", "v79"))
@pytest.mark.parametrize("cooperative", (False, True))
def test_prefill_is_one_loop_over_the_layers(tiny_bundle, arch, cooperative):
    c, _, _, path = tiny_bundle
    header = bundle.read_header(path)
    shapes = {k: tuple(v["shape"]) for k, v in header["tensors"].items()}
    target = determine_target({"kind": "hexagon", "mcpu": f"hexagon{arch}"}, return_object=True)
    options = {} if cooperative else {"cooperative": False}
    ((_, source, _, _),) = lower_kernels([(make_prefill(c, shapes, 128, 256, 64, target=target, **options), [])], target=target)
    # A loop at run time, whatever the depth; the projections are calls of
    # the HMX GEMM, and the kernel's own pass configs give it fast math.
    assert source.count(f"for (int32_t layer = 0; layer < {c.layers}; ++layer)") == 1
    assert "tl_hexagon_hmx_gemm_blockscaled<" in source
    assert "tl_hvx_exp2_fast(" in source and "tl_hvx_exp2(" not in source
    assert ("#define TL_HEXAGON_USES_COOPERATIVE 1" in source) is cooperative
    _assert_prefill_norm_path(source)


@pytest.mark.parametrize("arch", ("v73", "v79"))
def test_w8a8_w13_prefill_lowering(arch):
    c = Config(vocab=4096, dim=2048, ff=8192, heads=32, kv_heads=8, head_dim=64, layer_types=(CONV, FULL, CONV))
    shapes = {
        "conv.in_proj": (2, 1, 6144, 2048),
        "attn.qkv": (1, 1, 3072, 2048),
        "out_proj": (3, 1, 2048, 2048),
        "ffn.w13": (3, 1, 16384, 2048),
        "ffn.w2": (3, 4, 2048, 2048),
    }
    target = determine_target({"kind": "hexagon", "mcpu": f"hexagon{arch}"}, return_object=True)
    ordinary = make_prefill(c, shapes, 128, 256, 64, target=target)
    integer = make_prefill(c, shapes, 128, 256, 64, target=target, w8a8_w13=True)
    (_, baseline, _, _), (_, source, _, _) = lower_kernels([(ordinary, []), (integer, [])], target=target)
    assert "#include <tl_templates/hexagon/w8a8.h>" not in baseline and "tl_hexagon_hmx_gemm_u8i8" not in baseline
    assert "#include <tl_templates/hexagon/w8a8.h>" in source
    assert source.count("tl_hexagon_quantize_crouton_u8(") == 1  # one call inside the layer loop
    assert "tl_hexagon_requantize_panel_s4_s8(" in source and "tl_hexagon_hmx_gemm_u8i8(" in source
    assert "tl_hexagon_hmx_gemm_blockscaled<" in source  # the other projections
    assert source.count("tl_hexagon_dma_async_submit(") == baseline.count("tl_hexagon_dma_async_submit(")


def _w8a8_w13_projection(R, K, normal_scales):
    @T.prim_func
    def projection(
        A: T.Tensor((R, K), "float16"),
        W: T.Tensor((2, 128, K), "int4"),
        S: T.Tensor((2, K // 64, 128), "float16"),
        Out: T.Tensor((1, R, 128), "float32"),
        PackedA: T.Tensor(((R + 63) // 64, K // 32, 64, 32), "uint8"),
    ):
        with T.Kernel(4, num_workers=4) as worker:
            a = T.alloc_global((R, K), "float16", scope="global.vtcm")
            aq = T.alloc_global(((R + 63) // 64, K // 32, 64, 32), "uint8", scope="global.vtcm")
            sa = T.alloc_global((R,), "float32", scope="global.vtcm")
            w = T.alloc_shared((2, 128, K), "int4")
            s = T.alloc_shared((2, K // 64, 128), "float16")
            tmp = T.alloc_shared((64 * 32,), "float16")
            wq = T.alloc_shared((K // 32, 8, 32, 4), "int8")
            sw = T.alloc_shared((32,), "float32")
            correction = T.alloc_shared((32,), "int32")
            dots = T.alloc_shared((64, 32), "int32")
            result = T.alloc_shared((1, R, 128), "float32")
            T.annotate_layout({a: make_hmx_activation_layout(a), W: make_hvx_panel_layout(W), w: make_hvx_panel_layout(w)})

            first = worker * (R // 4)
            T.copy(A[first : first + R // 4, :], a[first : first + R // 4, :])
            pack_w8a8_activations(a, aq, sa, R, K, first, R // 4)
            T.sync_grid()
            if worker == 0:
                # Use a nonzero ring slot and all four Q4 panels in the block.
                T.dma_copy(W[1, :, :], w[1, :, :])
                T.dma_copy(S[1, :, :], s[1, :, :])
                T.dma_wait()
                project_w8a8(aq, sa, w, s, 1, tmp, wq, sw, correction, dots, result, 0, R, K, normal_scales)
                T.copy(result, Out)
                T.copy(aq, PackedA)

    return projection


def _symmetric_q8_from_fp16(x):
    values = x.to(torch.float64)
    amax = values.abs().amax(-1, keepdim=True)
    safe = torch.where(amax == 0, 1.0, amax)
    return (values * 127 / safe).round().to(torch.int32), (amax / 127).to(torch.float32)


def _w8a8_w13_inputs(R, K, normal_scales):
    rng = torch.Generator().manual_seed(30)
    a = torch.randn((R, K), generator=rng).half()
    a[0].zero_()
    a[1, 0] = 254
    a[1, 1:255] = torch.arange(-127, 127, dtype=torch.float16)  # half-integer ties in both directions
    a[2] *= 2**-20
    w = torch.randint(-8, 8, (2, 128, K), generator=rng, dtype=torch.int8)
    w[1, 0].zero_()
    w[1, 1, :64] = 0
    w[1, 1, 0] = 1
    scales = (0.005 + torch.rand((2, K // 64, 128), generator=rng) * 0.025).half()
    scales[1, :, 1] = 1
    scales[1, 0, 1] = 254
    scales[1, 1::2, 4].neg_()
    if not normal_scales:
        scales[1, :, 2] = 0
        scales[1, ::2, 3] = 2**-24
        scales[1, :, 5:7] = 2**-24
        w[1, 5].clamp_(-1, 1)
        w[1, 6].clamp_(-2, 2)
    return a, w, scales


@SIMULATOR
@pytest.mark.parametrize("arch", ["v73", "v79"])
def test_w8a8_w13_packing_on_simulator(arch, tmp_path):
    sdk = Path(os.environ["HEXAGON_TOOLS_ROOT"]) / "Tools/bin"
    repo = LFM.parents[2]
    fixture = Path(__file__).with_name("hvx_w8a8_lfm_sim.cc")
    executable = tmp_path / "pack"
    compile_command = [
        sdk / "hexagon-clang++", f"-m{arch}", "-mhvx", "-mhvx-length=128B", "-std=c++17", "-O2",
        "-mllvm", "--disable-handle-qfp", "-mllvm", "-hexagon-insert=false", "-I", repo / "src", fixture, "-o", executable,
    ]  # fmt: skip
    compiled = subprocess.run(compile_command, capture_output=True, text=True, timeout=180)
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    K = 2048
    for R, normal_scales in [(32, False), (128, True)]:
        a, w, scales = _w8a8_w13_inputs(R, K, normal_scales)
        crouton = a.reshape(R // 32, 16, 2, K // 32, 32).permute(0, 3, 1, 4, 2).contiguous()
        src, dst = tmp_path / "input.bin", tmp_path / "output.bin"
        src.write_bytes(crouton.numpy().tobytes() + pack_panel(w[1], 4).numpy().tobytes() + scales[1].numpy().tobytes())
        simulated = subprocess.run(
            [sdk / "hexagon-sim", f"-m{arch}", "--simulated_returnval", executable, "--", src, dst, str(R), str(int(normal_scales))],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert simulated.returncode == 0, simulated.stdout + simulated.stderr
        assert "LFM_W8A8_PACK_OK" in simulated.stdout

        a_bytes = max(64, R) * K
        w_bytes = 128 * K
        data = bytearray(dst.read_bytes())
        packed_a = torch.frombuffer(data, dtype=torch.uint8, count=a_bytes).reshape(-1, K // 32, 64, 32)
        packed_w = torch.frombuffer(data, dtype=torch.int8, count=w_bytes, offset=a_bytes).reshape(4, K // 4, 32, 4)
        sa = torch.frombuffer(data, dtype=torch.float32, count=R, offset=a_bytes + w_bytes)
        sw = torch.frombuffer(data, dtype=torch.float32, count=128, offset=a_bytes + w_bytes + R * 4)
        correction = torch.frombuffer(data, dtype=torch.int32, count=128, offset=a_bytes + w_bytes + (R + 128) * 4)
        logical_a = packed_a.permute(0, 2, 1, 3).reshape(-1, K)
        logical_w = packed_w.permute(0, 2, 1, 3).reshape(128, K)

        aq, sa_ref = _symmetric_q8_from_fp16(a)
        w16 = (w[1].float() * scales[1].T.repeat_interleave(64, -1).float()).half()
        wq, sw_ref = _symmetric_q8_from_fp16(w16)
        assert torch.equal(logical_a[:R], (aq + 128).to(torch.uint8))
        assert (logical_a[R:] == 128).all()
        assert torch.equal(logical_w, wq.to(torch.int8))
        assert torch.equal(correction, 128 * wq.sum(-1).to(torch.int32))
        torch.testing.assert_close(sa, sa_ref[:, 0], rtol=2e-7, atol=0)
        torch.testing.assert_close(sw, sw_ref[:, 0], rtol=2e-7, atol=0)


@DEVICE
@pytest.mark.parametrize(("R", "normal_scales"), [(32, False), (128, True)])
def test_w8a8_w13_projection_matches_reference(R, normal_scales):
    K = 2048
    a, w, scales = _w8a8_w13_inputs(R, K, normal_scales)

    aw, sa = _symmetric_q8_from_fp16(a)
    w16 = (w[1].float() * scales[1].T.repeat_interleave(64, -1).float()).half()
    ww, sw = _symmetric_q8_from_fp16(w16)
    # Signed logical dot uses no centering identity, making an independent
    # check of the native unsigned activation and zero-point correction.
    want = (aw.to(torch.int64) @ ww.to(torch.int64).T).float() * sa * sw.T
    target = determine_target("hexagon", return_object=True)
    kernel = tilelang.compile(_w8a8_w13_projection(R, K, normal_scales), out_idx=[3, 4], target=target)
    try:
        actual, packed = kernel(a, pack_panel(w, 4), scales)
        rows = packed.permute(0, 2, 1, 3).reshape(-1, K)
        assert torch.equal(rows[:R], (aw + 128).to(torch.uint8))
        assert (rows[R:] == 128).all()
        torch.testing.assert_close(actual[0], want, rtol=1e-6, atol=1e-7)
    finally:
        kernel.adapter.close()


@DEVICE
def test_w8a8_w13_tiny_model_state_and_logits(tiny_bundle, record_property):
    c, _, _, path = tiny_bundle
    decode_ref = BundleModel(path)

    class PrefillReference(Model):
        def __init__(self, integer_w13):
            super().__init__(c, decode_ref.w)
            self.integer_w13 = integer_w13

        def linear(self, name, x):
            a, w = x.half(), self.w[name].half()
            if self.integer_w13 and name.endswith((".feed_forward.w1.weight", ".feed_forward.w3.weight")):
                aq, sa = _symmetric_q8_from_fp16(a)
                wq, sw = _symmetric_q8_from_fp16(w)
                # For the tiny model's K=1024, every integer partial sum fits exactly in FP32.
                return (aq.float() @ wq.float().T) * sa * sw.T
            return a.float() @ w.float().T

        def _ffn(self, layer, x):
            name = f"model.layers.{layer}.feed_forward."
            gate = self.linear(name + "w1.weight", x).half()
            up = self.linear(name + "w3.weight", x).half()
            return self.linear(name + "w2.weight", gate * gate.sigmoid() * up)

    prefix_len, max_pos = 66, 256  # two R=32 chunks followed by a short chunk
    tokens = torch.randint(c.vocab, (prefix_len + 1,), generator=torch.Generator().manual_seed(30))

    def flatten_state(conv, keys, values):
        return torch.cat([x.reshape(-1) for x in (conv, keys, values)])

    @torch.no_grad()
    def reference(integer):
        model = PrefillReference(integer)
        state = model.new_state()
        model.run(tokens[:-1], state)
        rows = flatten_state(torch.stack(state.conv), torch.stack(state.k), torch.stack(state.v))
        hidden = decode_ref.run(tokens[-1:], state)
        return rows, decode_ref.logits(hidden)[0]

    def device(integer):
        decoder = Decoder(path, max_pos=max_pos, max_tokens=max_pos, prefill_rows=32, w8a8_w13=integer)
        try:
            decoder.prefill(tokens[:-1].tolist())
            conv = torch.frombuffer(bytearray(decoder.conv_state.read()), dtype=torch.float32)
            kv_shape = (len(c.attention_layers), c.kv_heads, max_pos, c.head_dim)
            keys, values = (
                torch.frombuffer(bytearray(buf.read()), dtype=torch.float32).view(kv_shape)[:, :, :prefix_len]
                for buf in (decoder.kv_k, decoder.kv_v)
            )
            rows = flatten_state(conv, keys, values)
            _, logits = decoder.run(tokens[-1:].tolist(), prefix_len, 1, 1)
            return rows, logits
        finally:
            decoder.close()

    reference_fp16, reference_int = reference(False), reference(True)
    device_fp16, device_int = device(False), device(True)
    for name, host_a, host_b, dsp_a, dsp_b in zip(
        ("prefill_state", "continuation_logits"), reference_fp16, reference_int, device_fp16, device_int, strict=True
    ):
        # Compare the incremental effect of changing w13. The FP16 control
        # accounts for common discrepancies from DSP attention and epilogues.
        # Leaving w13 on the ordinary route produces a relative delta error of 1.
        expected = host_b - host_a
        actual = dsp_b - dsp_a
        delta_error = ((actual - expected).norm() / expected.norm()).item()
        baseline_error = ((dsp_a - host_a).norm() / host_a.norm()).item()
        integer_error = ((dsp_b - host_b).norm() / host_b.norm()).item()
        record_property(f"{name}_w8a8_delta_relative_l2", delta_error)
        record_property(f"{name}_w4a16_relative_l2", baseline_error)
        record_property(f"{name}_w8a8_relative_l2", integer_error)
        assert delta_error < 0.5, (name, delta_error, baseline_error, integer_error)


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


@pytest.mark.parametrize("fp16", [False, True])
@pytest.mark.parametrize("arch,dim", [(None, 2048), (73, 2048), (75, 2048), (79, 2048), (81, 2048), (79, 1024)])
def test_prefill_normalization_uses_explicit_target_and_qualified_width(fp16, arch, dim, monkeypatch):
    # Factory defaults never derive optimization eligibility from the process
    # environment. Only the explicit compilation target enables the large map.
    monkeypatch.setenv("HEXAGON_DSP_ARCH", "v79")
    c = Config(vocab=4096, dim=dim, ff=2048, heads=dim // 64, kv_heads=8, head_dim=64, layer_types=(CONV, FULL, CONV))
    target = None if arch is None else determine_target({"kind": "hexagon", "mcpu": f"hexagonv{arch}"}, return_object=True)
    if fp16:
        kernel = prefill_fp16.make_prefill(c, 128, 256, 64, target=target)
    else:
        shapes = {
            "conv.in_proj": (2, 1, 3 * dim, dim),
            "attn.qkv": (1, 1, dim + 1024, dim),
            "out_proj": (3, 1, dim, dim),
            "ffn.w13": (3, 1, 4096, dim),
            "ffn.w2": (3, 1, dim, 2048),
        }
        kernel = make_prefill(c, shapes, 128, 256, 64, target=target)
    compile_target = target if target is not None else FP16_SOURCE_TARGET
    ((_, source, _, _),) = lower_kernels([(kernel, [])], target=compile_target)
    _assert_prefill_norm_path(source, direct=arch == 79 and dim >= 2048)


@pytest.mark.parametrize("fp16", [False, True])
@pytest.mark.parametrize("arch", [73, 79])
def test_decoder_passes_one_resolved_target_to_factory_and_session(fp16, arch, monkeypatch):
    import decode
    import tilelang.backend.target as target_module
    import tilelang.hexagon.runtime.session as session_module

    target = determine_target({"kind": "hexagon", "mcpu": f"hexagonv{arch}"}, return_object=True)
    observed = []

    def resolve(value, return_object):
        assert value == "hexagon" and return_object
        observed.append("resolve")
        return target

    def factory(*args, **kwargs):
        assert kwargs["target"] is target
        observed.append("factory")
        return "prefill"

    class StopBeforeTransport(Exception):
        pass

    def session(kernels, **kwargs):
        assert kwargs["target"] is target
        assert any(kernel == "prefill" for kernel, _ in kernels)
        observed.append("session")
        raise StopBeforeTransport

    monkeypatch.setattr(target_module, "determine_target", resolve)
    monkeypatch.setattr(session_module, "Session", session)
    decoder = Decoder.__new__(Decoder)
    decoder.c = Config(vocab=4096, dim=2048, ff=2048, heads=32, kv_heads=8, head_dim=64, layer_types=(CONV, FULL, CONV))
    decoder.header = {"tensors": {}}
    decoder.max_pos, decoder.max_tokens, decoder.prefill_rows = 256, 64, 128
    decoder.w8a8_w13 = False
    if fp16:
        monkeypatch.setattr(decode_fp16, "plan_stages", lambda *args: [range(3)])
        monkeypatch.setattr(decode_fp16, "row_major_embedding", lambda *args: True)
        monkeypatch.setattr(decode_fp16, "make_decode", lambda *args, **kwargs: "decode")
        monkeypatch.setattr(prefill_fp16, "make_prefill", factory)
        with pytest.raises(StopBeforeTransport):
            decoder._open_fp16("unused", None, 1)
    else:
        monkeypatch.setattr(decode, "make_decode", lambda *args: "decode")
        monkeypatch.setattr(decode, "make_prefill", factory)
        monkeypatch.setattr(bundle, "has_normal_scales", lambda *args: False)
        with pytest.raises(StopBeforeTransport):
            decoder._open_4bit("unused", None)
    assert observed == ["resolve", "factory", "session"]


@pytest.mark.parametrize("fp16", [False, True])
@pytest.mark.parametrize("capability", ["width", "aligned_only"])
def test_prefill_factory_keeps_staging_for_unqualified_vector_capability(fp16, capability):
    c = Config(vocab=4096, dim=2048, ff=2048, heads=32, kv_heads=8, head_dim=64, layer_types=(CONV, FULL, CONV))
    attrs = {"kind": "hexagon", "mcpu": "hexagonv79"}
    if capability == "width":
        attrs.update(min_vector_bits=512, max_vector_bits=512)
    else:
        attrs["supports_unaligned_vector_access"] = False
    target = tvm.target.Target(attrs)
    if fp16:
        original = prefill_fp16.make_prefill(c, 128, 256, 64)
        candidate = prefill_fp16.make_prefill(c, 128, 256, 64, target=target)
    else:
        shapes = {
            "conv.in_proj": (2, 1, 6144, 2048),
            "attn.qkv": (1, 1, 3072, 2048),
            "out_proj": (3, 1, 2048, 2048),
            "ffn.w13": (3, 1, 4096, 2048),
            "ffn.w2": (3, 1, 2048, 2048),
        }
        original = make_prefill(c, shapes, 128, 256, 64)
        candidate = make_prefill(c, shapes, 128, 256, 64, target=target)
    tvm.ir.assert_structural_equal(original, candidate)


def test_compact_rope_coefficients_stay_inside_each_row():
    # A short final kind wraps speculative vector lanes within the row. The
    # identity still applies to each pair beyond that kind's active span.
    c = Config(head_dim=128, rope={SLIDING: (1e4, 0.125), FULL: (1e6, 0.34375)})
    layout = decode_fp16.RopeLayout(c)
    values = {(0, i): tvm.tirx.FloatImm("float32", i + 2) for i in range(layout.width)}
    analyzer = tvm.arith.Analyzer()
    for span in layout.spans.values():
        for lane in range(c.head_dim // 2):
            for identity in (0.0, 1.0):
                coeff = decode_fp16.rope_coefficient(values, 0, lane, c.head_dim // 2, span, layout.width, identity)
                expected = span.start + lane + 2 if lane < len(span) else identity
                assert analyzer.simplify(coeff).value == expected
