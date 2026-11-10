"""Opt-in full-attention blocks: normal lowering and host entry contracts."""

import sys
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import pytest

from tilelang.hexagon.runtime import session
from tvm import arith, tirx

LFM = Path(__file__).resolve().parents[3] / "examples/hexagon/lfm"
sys.path.insert(0, str(LFM))
import bench  # noqa: E402
import decode  # noqa: E402
import decode_fp16  # noqa: E402
import prefill_fp16  # noqa: E402
from model import CONV, FULL, SLIDING, Config  # noqa: E402


def config():
    # The existing tiny LFM3 shapes, without allocating model weights.
    return Config(
        vocab=4096,
        dim=2048,
        ff=1024,
        heads=16,
        kv_heads=4,
        head_dim=128,
        layer_types=(CONV, FULL, CONV, SLIDING),
        conv_taps=7,
        rope={FULL: (1e6, 0.5), SLIDING: (1e4, 1.0)},
        window=256,
        sandwich_norm=True,
        head_gate=True,
    )


@lru_cache(None)
def source(keys=None, start=0, stop=4, max_pos=512, arch="v79"):
    kwargs = {} if keys is None else {"full_keys": keys}
    kernel = prefill_fp16.make_prefill(config(), 128, max_pos, 64, range(start, stop), **kwargs)
    ((_, text, _, _),) = session.lower_kernels([(kernel, [])], target={"kind": "hexagon", "mcpu": "hexagon" + arch})
    return text


@pytest.mark.parametrize("value", [0, 64, 1024, "256"])
def test_full_keys_rejects_invalid_factory_and_decoder(value, monkeypatch):
    def no_header(*args):
        pytest.fail("invalid option reached bundle or Session setup")

    monkeypatch.setattr(decode.bundle, "read_header", no_header)
    with pytest.raises(ValueError, match="full_keys"):
        prefill_fp16.make_prefill(config(), 128, 512, 64, full_keys=value)
    with pytest.raises(ValueError, match="prefill_full_keys"):
        decode.Decoder("unused", prefill_full_keys=value)


@pytest.mark.parametrize("keys", [256, 512])
def test_full_keys_rejects_quantized_bundle_before_open(keys, monkeypatch):
    monkeypatch.setattr(decode.bundle, "read_header", lambda _: {"config": config().to_dict(), "quant": {"weight": {"bits": 4}}})
    monkeypatch.setattr(decode.Decoder, "_open_4bit", lambda *args: pytest.fail("quantized Session opened"))
    with pytest.raises(ValueError, match="requires an fp16 bundle"):
        decode.Decoder("unused", prefill_full_keys=keys)


@pytest.mark.parametrize("keys", [128, 256, 512])
def test_full_keys_forwarded_to_each_fp16_stage(keys, monkeypatch):
    class BeforeDevice(Exception):
        pass

    import tilelang.backend.target as target_module

    resolved = target_module.determine_target({"kind": "hexagon", "mcpu": "hexagonv79"}, return_object=True)
    resolutions = []

    def resolve(value, return_object):
        assert value == "hexagon" and return_object
        resolutions.append(value)
        return resolved

    monkeypatch.setattr(target_module, "determine_target", resolve)
    stages = [range(2), range(2, 4)]
    calls = []
    monkeypatch.setattr(decode.bundle, "read_header", lambda _: {"config": config().to_dict(), "quant": {}})
    monkeypatch.setattr(decode_fp16, "plan_stages", lambda *args: stages)
    monkeypatch.setattr(decode_fp16, "make_decode", lambda *args: "decode")

    def prefill(c, rows, max_pos, max_tokens, layers, *, target, full_keys):
        assert target is resolved
        calls.append((rows, max_pos, max_tokens, layers, full_keys))
        return "prefill"

    def stop(kernels, *, target, processes):
        assert target is resolved
        assert processes == 2
        assert kernels == [("decode", []), ("decode", []), ("prefill", []), ("prefill", [])]
        raise BeforeDevice

    monkeypatch.setattr(prefill_fp16, "make_prefill", prefill)
    monkeypatch.setattr(session, "Session", stop)
    with pytest.raises(BeforeDevice):
        decode.Decoder("unused", max_pos=512, max_tokens=64, prefill_rows=128, processes=2, prefill_full_keys=keys)
    assert calls == [(128, 512, 64, r, keys) for r in stages]
    assert resolutions == ["hexagon"]


@pytest.mark.parametrize("keys", [None, 256, 512])
def test_bench_full_keys_option_and_default(keys, monkeypatch):
    calls = []
    closed = []

    def decoder(*args, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(c=SimpleNamespace(vocab=2048), close=lambda: closed.append(True))

    argv = ["bench.py", "unused", "unused", "--pp", "0", "--tg", "0"]
    if keys is not None:
        argv += ["--prefill-full-keys", str(keys)]
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(bench, "Decoder", decoder)
    bench.main()
    assert calls[0]["prefill_full_keys"] == (128 if keys is None else keys)
    assert calls[0]["prefill_rows"] == "auto" and closed == [True]


def test_full_keys_default_and_incompatible_max_pos_preserve_source():
    assert source() == source(128)
    assert source(512, max_pos=256) == source(128, max_pos=256)


@pytest.mark.parametrize("stage", [(0, 1), (3, 4)])
@pytest.mark.parametrize("keys", [256, 512])
def test_full_keys_does_not_change_nonfull_stages(stage, keys):
    assert source(keys, *stage) == source(128, *stage)


@pytest.mark.parametrize("arch", ["v73", "v75", "v79"])
@pytest.mark.parametrize("keys", [256, 512])
def test_full_keys_normal_lowering_retains_sliding_and_four_workers(arch, keys):
    text = source(keys, arch=arch)
    assert f"tl_hexagon_hmx_gemm<false, true, true, true>(64, {keys}, 128," in text
    assert "tl_hexagon_hmx_gemm<false, true, true, true>(64, 128, 128," in text
    assert "tl_parallel_for<tl_hmx_worker_scope, true>(4, 4," in text
    # full_keys is an upper bound. Every selected width retains compact
    # score/probability matrices and the existing vector reduction helpers.
    for width in [128, 256] if keys == 256 else [128, 256, 512]:
        assert f"tl_hexagon_hmx_gemm<false, true, true, true>(64, {width}, 128," in text
        for op in ("MAX", "SUM"):
            assert any(f"tl_crouton_reduce<TL_REDUCE_{op}" in line and f", 2, {width // 32}, 1," in line for line in text.splitlines())
    # The table's full layer still starts after zero rows and the sliding
    # layer after max_pos / 128 metadata units, irrespective of compute block.
    assert prefill_fp16.kinds_table(config(), 512)[1][4] == 0
    assert prefill_fp16.kinds_table(config(), 512)[3][4] == 4


@pytest.mark.parametrize("rows", [64, 128, 256])
@pytest.mark.parametrize("keys", [256, 512])
@pytest.mark.parametrize("stage", ["mixed", "full", "sliding"])
def test_full_keys_selection_uses_the_physical_chunk(rows, keys, stage):
    layers = {"mixed": range(4), "full": range(1, 2), "sliding": range(3, 4)}[stage]
    func = prefill_fp16.make_prefill(config(), rows, 4608, rows, layers, full_keys=keys)
    bindings = {}

    def collect(node):
        if isinstance(node, tirx.Bind):
            bindings[str(node.var.name)] = node

    tirx.stmt_functor.post_order_visit(func.body, collect)
    if stage == "sliding":
        assert "selected_keys" not in bindings
        return
    expression = bindings["selected_keys"].value
    for name in ("full_block", "base"):
        if name in bindings:
            binding = bindings[name]
            expression = tirx.stmt_functor.substitute(expression, {binding.var: binding.value})
    variables = []
    tirx.stmt_functor.post_order_visit(expression, lambda node: variables.append(node) if isinstance(node, tirx.Var) else None)
    # n/valid never enter the actual IR selector. In particular a padded
    # n=20 call still selects for all rows, rather than for 20 queries.
    assert {str(var.name) for var in variables} <= {"pos0", "chunk", "window"}
    for pos0, chunk in [(0, 0), (0, 1), (1, 0), (64, 0), (65, 0), (128, 0), (129, 0), (4096, 0)]:
        for window in [0, config().window] if stage == "mixed" else [0]:
            values = {"pos0": pos0, "chunk": chunk, "window": window}
            result = arith.Analyzer().simplify(
                tirx.stmt_functor.substitute(expression, {var: tirx.IntImm(var.dtype, values[str(var.name)]) for var in variables})
            )
            end = pos0 + chunk * rows + rows
            expected = 128 if window > 0 or end <= 128 else (256 if keys == 512 and end <= 256 else keys)
            assert isinstance(result, tirx.IntImm) and int(result) == expected
