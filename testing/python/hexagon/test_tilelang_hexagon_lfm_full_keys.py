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
def test_full_keys_has_one_state_loop_per_attention_kind(rows, keys, stage):
    layers = {"mixed": range(4), "full": range(1, 2), "sliding": range(3, 4)}[stage]
    func = prefill_fp16.make_prefill(config(), rows, 4608, rows, layers, full_keys=keys)
    bindings, loops = [], []

    def collect(node):
        if isinstance(node, tirx.Bind):
            bindings.append(str(node.var.name))
        elif isinstance(node, tirx.For) and str(node.loop_var.name) == "rb":
            row0 = []
            tirx.stmt_functor.post_order_visit(
                node.body, lambda child: row0.append(child) if isinstance(child, tirx.Bind) and str(child.var.name) == "row0" else None
            )
            if row0:
                loops.append(node)

    tirx.stmt_functor.post_order_visit(func.body, collect)
    # Narrow tails share the full path's state. Do not duplicate whole
    # head/query loops for early physical chunks or depend on valid n.
    assert not {"selected_keys", "full_block"} & set(bindings)
    assert len(loops) == (2 if stage == "mixed" else 1)
    wide = []
    for loop in loops:
        remaining = []
        tirx.stmt_functor.post_order_visit(
            loop.body,
            lambda node, remaining=remaining: (
                remaining.append(node) if isinstance(node, tirx.Bind) and str(node.var.name) == "remaining" else None
            ),
        )
        if remaining:
            wide.append(int(remaining[0].value.a.b))
    assert wide == ([] if stage == "sliding" else [keys])


def _trace_attention_row(loop, *, base, row, kv0):
    """Interpret scalar control only; check actual DMA regions before GEMMs.

    Parallel arithmetic is not evaluated here. Its numerical contract is
    exercised by the public model tests; this checks the double-buffered
    schedule and addresses of the generated TIR instead.
    """
    values = {}
    seeds = {"base": base, "rb": row, "window": 0, "kv0": kv0, "kvh": 0, "hq": 0}

    def seed(node):
        if isinstance(node, tirx.Var) and str(node.name) in seeds:
            values[node] = tirx.IntImm(node.dtype, seeds[str(node.name)])

    tirx.stmt_functor.post_order_visit(loop.body, seed)
    pending, loaded, events = [], {}, []
    accumulator = []

    def integer(expr):
        value = arith.Analyzer().simplify(tirx.stmt_functor.substitute(expr, values))
        assert isinstance(value, tirx.IntImm), value
        return int(value)

    def visit(stmt):
        if isinstance(stmt, tirx.SeqStmt):
            for child in stmt.seq:
                visit(child)
        elif isinstance(stmt, tirx.Bind):
            values[stmt.var] = tirx.IntImm(stmt.var.dtype, integer(stmt.value))
        elif isinstance(stmt, tirx.IfThenElse):
            branch = stmt.then_case if integer(stmt.condition) else stmt.else_case
            if branch is not None:
                visit(branch)
        elif isinstance(stmt, tirx.For):
            if stmt.kind == tirx.ForKind.PARALLEL:
                return
            start, count = integer(stmt.min), integer(stmt.extent)
            for i in range(start, start + count):
                values[stmt.loop_var] = tirx.IntImm(stmt.loop_var.dtype, i)
                visit(stmt.body)
        elif isinstance(stmt, tirx.Evaluate) and isinstance(stmt.value, tirx.Call):
            call = stmt.value
            if call.op.name == "tl.tileop.dma_copy":
                src, dst = call.args
                load = src.args[0]
                if load.buffer.name in ("kv_k", "kv_v"):
                    kind = str(load.buffer.name)
                    origin = integer(load.indices[1])
                    width = integer(src.args[-2])
                    slot = integer(dst.args[0].indices[0])
                    assert integer(dst.args[-2]) == width
                    pending.append((kind, slot, origin, width))
                else:
                    assert load.buffer.name == "qs"
                    pending.append(None)
                assert len(pending) <= 5  # Query plus current/next K and V.
            elif call.op.name == "tl.hexagon_dma_wait":
                keep = integer(call.args[0])
                while len(pending) > keep:
                    item = pending.pop(0)
                    if item is not None:
                        kind, slot, origin, width = item
                        loaded[kind, slot] = (origin, width)
            elif call.op.name == "tl.tileop.gemm":
                operand = call.args[1]
                slot = integer(operand.args[0].indices[0])
                width = integer(operand.args[-2])
                key0 = next(integer(v) for v in reversed(values) if str(v.name) == "key0")
                kind = "kv_k" if integer(call.args[4]) else "kv_v"
                assert loaded[kind, slot] == (kv0 * 128 + key0, width)
                assert not any(item is not None and item[:2] == (kind, slot) for item in pending)
                if kind == "kv_k":
                    assert integer(call.args[6]) == width
                    events.append((key0, width, slot))
                else:
                    assert integer(call.args[7]) == width and not integer(call.args[9])
                    accumulator.append(call.args[2].args[0].buffer.data)

    visit(loop.body)
    assert not pending
    assert len(accumulator) == len(events)
    assert all(value.same_as(accumulator[0]) for value in accumulator)
    return events


@pytest.mark.parametrize("rows", [64, 128, 256])
@pytest.mark.parametrize("keys", [256, 512])
def test_causal_tail_dma_and_compute_use_the_same_key_interval(rows, keys):
    func = prefill_fp16.make_prefill(config(), rows, 4608, rows, full_keys=keys)
    loops = []

    def collect(node):
        if isinstance(node, tirx.For) and str(node.loop_var.name) == "rb":
            remaining = []
            tirx.stmt_functor.post_order_visit(
                node.body, lambda x: remaining.append(x) if isinstance(x, tirx.Bind) and str(x.var.name) == "remaining" else None
            )
            if remaining:
                loops.append((node, int(remaining[0].value.a.b)))

    tirx.stmt_functor.post_order_visit(func.body, collect)
    assert {width for _, width in loops} == {keys}
    for loop, width in loops:
        bulk_loops = []

        def find_bulk(node, bulk_loops=bulk_loops):
            if isinstance(node, tirx.For):
                variables = []
                tirx.stmt_functor.post_order_visit(
                    node.extent, lambda value, variables=variables: variables.append(value) if isinstance(value, tirx.Var) else None
                )
                if any(str(v.name) == "bulk_last" for v in variables):
                    bulk_loops.append(node)

        tirx.stmt_functor.post_order_visit(loop.body, find_bulk)
        assert len(bulk_loops) == 1
        gemms = []
        tirx.stmt_functor.post_order_visit(
            bulk_loops[0].body,
            lambda value, gemms=gemms: gemms.append(value) if isinstance(value, tirx.Call) and value.op.name == "tl.tileop.gemm" else None,
        )
        # The fixed-width bulk loop must not contain small-width arithmetic
        # dispatch. Only its final one-ahead DMA may fetch the narrow tail.
        assert len(gemms) == 2
        assert int(gemms[0].args[6]) == int(gemms[1].args[7]) == width
        variables = []
        tirx.stmt_functor.post_order_visit(
            loop.body, lambda value, variables=variables: variables.append(value) if isinstance(value, tirx.Var) else None
        )
        assert not {"n", "valid"} & {str(x.name) for x in variables}
        # Boundaries, unaligned query origins and a long prefix. The wider
        # block's absolute origin must not change when its final width does.
        for base in (0, 1, 64, 65, 128, 129, 192, 256, 257, 320, 384, 448, 1010, 4096):
            for row in range(rows // 64):
                end = base + row * 64 + 64
                blocks = (end + width - 1) // width
                remainder = end - (blocks - 1) * width
                tail = 128 if remainder <= 128 else (256 if remainder <= 256 else width)
                expected = [(i * width, width, i % 2) for i in range(blocks - 1)]
                expected.append(((blocks - 1) * width, tail, (blocks - 1) % 2))
                assert _trace_attention_row(loop, base=base, row=row, kv0=8) == expected
