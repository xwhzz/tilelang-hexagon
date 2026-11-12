"""The Flash example selects measured defaults without changing compiler policy."""

import importlib.util
from pathlib import Path
import sys

import pytest

from tilelang import tvm
from tilelang.hexagon.target import hexagon_arch, normalize_hexagon_target
from test_tilelang_hexagon_pipeline import _source

X = tvm.tirx
ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location("flash_schedule_example", ROOT / "examples/hexagon/example_flash_attention.py")
flash = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(flash)


@pytest.mark.parametrize("arch", ["v73", "v75", "v79"])
@pytest.mark.parametrize("seq", [64, 256, 1024, 4096])
def test_auto_uses_only_measured_target_and_shape(arch, seq):
    target = normalize_hexagon_target({"kind": "hexagon", "mcpu": "hexagon" + arch})
    assert flash.select_schedule("auto", target, 64, seq, 64, 64) == ("pipeline" if arch in {"v73", "v79"} and seq >= 256 else "serial")


@pytest.mark.parametrize("shape", [(32, 256, 64, 64), (64, 256, 128, 64), (64, 256, 64, 128)])
def test_auto_keeps_unmeasured_shapes_serial(shape):
    target = normalize_hexagon_target({"kind": "hexagon", "mcpu": "hexagonv79"})
    assert flash.select_schedule("auto", target, *shape) == "serial"


@pytest.mark.parametrize("arch", ["v73", "v75", "v79"])
@pytest.mark.parametrize("schedule", ["serial", "pipeline"])
def test_explicit_schedule_overrides_auto(arch, schedule):
    target = normalize_hexagon_target({"kind": "hexagon", "mcpu": "hexagon" + arch})
    assert flash.select_schedule(schedule, target, 32, 64, 64, 128) == schedule


@pytest.mark.parametrize(
    "explicit,environment,expected", [("v73", "v79", "pipeline"), ("v79", "v73", "pipeline"), ("v75", "v79", "serial")]
)
def test_explicit_target_architecture_wins_over_environment(monkeypatch, explicit, environment, expected):
    monkeypatch.setenv("HEXAGON_DSP_ARCH", environment)
    target = normalize_hexagon_target({"kind": "hexagon", "mcpu": "hexagon" + explicit})
    assert flash.select_schedule("auto", target, 64, 256, 64, 64) == expected


def test_unknown_schedule_is_rejected():
    with pytest.raises(ValueError, match="Unknown Flash schedule"):
        flash.select_schedule("unknown", normalize_hexagon_target("hexagon"), 64, 256, 64, 64)


@pytest.mark.parametrize("arch", ["v73", "v75", "v79"])
def test_default_factory_preserves_plain_serial_codegen(arch):
    target = {"kind": "hexagon", "mcpu": "hexagon" + arch}
    default = flash.make_flash(64, 256, 64, 64)

    def remove_pipeline_request(node):
        if isinstance(node, X.For) and "num_stages" in node.annotations:
            assert int(node.annotations["num_stages"]) == 0
            return X.For(node.loop_var, node.min, node.extent, node.kind, node.body, node.thread_binding, {}, step=node.step)
        return None

    reference = default.with_body(X.stmt_functor.ir_transform(default.body, None, remove_pipeline_request))
    source = _source(default, target)
    assert source == _source(reference, target) == _source(flash.make_flash(64, 256, 64, 64, pipelined=False), target)
    assert "tl_hexagon_dma_" not in source
    pipelined = _source(flash.make_flash(64, 256, 64, 64, pipelined=True), target)
    assert "tl_hexagon_dma_compact_load_context<2, 2>" in pipelined
    assert pipelined.count("tl_hexagon_dma_compact_submit(") == 2


@pytest.mark.parametrize(
    "arch,requested,expected",
    [("v73", "auto", "pipeline"), ("v75", "pipeline", "pipeline"), ("v79", "auto", "pipeline"), ("v79", "serial", "serial")],
)
def test_cli_normalizes_once_and_passes_the_same_target(monkeypatch, capsys, arch, requested, expected):
    monkeypatch.setenv("HEXAGON_DSP_ARCH", arch)
    monkeypatch.setattr(sys, "argv", ["example_flash_attention.py", "--schedule", requested])
    normalized = []

    def normalize(value):
        target = normalize_hexagon_target(value)
        normalized.append(target)
        return target

    class StopBeforeDevice(Exception):
        pass

    def compile_probe(func, **kwargs):
        assert len(normalized) == 1 and kwargs["target"] is normalized[0]
        assert hexagon_arch(kwargs["target"]) == arch
        reference = flash.make_flash(64, 256, 64, 64, pipelined=expected == "pipeline")
        tvm.ir.assert_structural_equal(func, reference, map_free_vars=True)
        raise StopBeforeDevice

    monkeypatch.setattr(flash, "normalize_hexagon_target", normalize)
    monkeypatch.setattr(flash.tilelang, "compile", compile_probe)
    with pytest.raises(StopBeforeDevice):
        flash.main()
    assert f"Flash schedule: {expected} (requested {requested}, {arch})" in capsys.readouterr().out
