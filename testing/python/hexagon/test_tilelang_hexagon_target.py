"""Hexagon target normalization and the capabilities passes read from it."""

import json

import pytest

import tilelang
from tilelang import tvm
from tilelang.backend.target import determine_target
from tilelang.hexagon import language as T
from tilelang.hexagon.target import SUPPORTED_ARCHS, hexagon_arch, normalize_hexagon_target
from tvm.target import Target

V79_VTCM = 8 << 20


@pytest.fixture(autouse=True)
def _no_arch_env(monkeypatch):
    monkeypatch.delenv("HEXAGON_DSP_ARCH", raising=False)


def test_every_hexagon_target_has_the_hvx_vector_width():
    # Kind defaults, so even targets that skip normalization carry them. HVX
    # instructions work only on whole 1024-bit registers.
    assert int(Target("hexagon").attrs["max_vector_bits"]) == 1024
    assert int(Target("hexagon").attrs["min_vector_bits"]) == 1024


@pytest.mark.parametrize("spec", ["hexagon", {"kind": "hexagon"}, Target("hexagon"), json.dumps({"kind": "hexagon"})])
def test_plain_hexagon_normalizes_to_v79(spec):
    target = determine_target(spec, return_object=True)
    assert target.attrs["mcpu"] == "hexagonv79"
    assert hexagon_arch(target) == "v79"
    assert int(target.attrs["max_vector_bits"]) == 1024
    assert int(target.attrs["vtcm-capacity"]) == V79_VTCM
    assert int(target.attrs["num-cores"]) == 6


@pytest.mark.parametrize("spec", [{"kind": "hexagon", "mcpu": "hexagonv73"}, Target({"kind": "hexagon", "mcpu": "hexagonv73"})])
def test_explicit_arch_wins_over_the_environment(spec, monkeypatch):
    monkeypatch.setenv("HEXAGON_DSP_ARCH", "v79")
    target = determine_target(spec, return_object=True)
    assert target.attrs["mcpu"] == "hexagonv73"
    # Not measured on v73 hardware: left to the runtime.
    assert "vtcm-capacity" not in target.attrs
    assert "num-cores" not in target.attrs


@pytest.mark.parametrize("value", ["v75", "hexagonv75", "75"])
def test_environment_sets_the_default_arch(value, monkeypatch):
    monkeypatch.setenv("HEXAGON_DSP_ARCH", value)
    assert determine_target("hexagon", return_object=True).attrs["mcpu"] == "hexagonv75"


def test_explicit_capabilities_are_kept():
    target = determine_target({"kind": "hexagon", "mcpu": "hexagonv79", "vtcm-capacity": 4 << 20}, return_object=True)
    assert int(target.attrs["vtcm-capacity"]) == 4 << 20


@pytest.mark.parametrize("mcpu", ["hexagonv60", "hexagonv99", "cortex-a76"])
def test_unsupported_arch_is_rejected(mcpu):
    with pytest.raises(ValueError, match="not supported"):
        normalize_hexagon_target({"kind": "hexagon", "mcpu": mcpu})


def test_supported_archs_normalize():
    for arch in SUPPORTED_ARCHS:
        assert hexagon_arch(normalize_hexagon_target({"kind": "hexagon", "mcpu": f"hexagon{arch}"})) == arch


def test_other_targets_are_left_alone():
    for spec in ["c", "llvm", {"kind": "c"}, Target("c"), json.dumps({"kind": "c"})]:
        assert normalize_hexagon_target(spec) is None


def _lower(func, spec):
    target = determine_target(spec, return_object=True)
    with tvm.target.Target(target):
        return tilelang.lower(func, target=target, enable_host_codegen=False, enable_device_compile=False)


def _dma_2d_kernel():
    @T.prim_func
    def kernel(A: T.Tensor((64, 128), "float16"), B: T.Tensor((32, 64), "float16")):
        with T.Kernel(1):
            A_s = T.alloc_shared((32, 64), "float16")
            T.dma_copy(A[0:32, 0:64], A_s)  # strided rows: a 2D descriptor
            T.dma_wait()
            T.copy(A_s, B)

    return kernel


def test_2d_dma_needs_v75():
    source = str(_lower(_dma_2d_kernel(), {"kind": "hexagon", "mcpu": "hexagonv75"}).kernel_source)
    # A 32-row descriptor: the v75+ 2D form.
    assert "(uint32_t)32, 0)) return tl_status;" in source
    with pytest.raises(Exception, match="2D T.dma_copy needs Hexagon v75"):
        _lower(_dma_2d_kernel(), {"kind": "hexagon", "mcpu": "hexagonv73"})


def test_vtcm_capacity_is_checked_at_compile_time():
    @T.prim_func
    def kernel(A: T.Tensor((1024, 1024), "float16"), B: T.Tensor((1024, 1024), "float16")):
        with T.Kernel(1):
            A_s = T.alloc_shared((1024, 1024), "float16")  # 2 MiB
            T.copy(A, A_s)
            T.copy(A_s, B)

    _lower(kernel, "hexagon")
    with pytest.raises(Exception, match="exceeds the target's VTCM capacity"):
        _lower(kernel, {"kind": "hexagon", "mcpu": "hexagonv79", "vtcm-capacity": 1 << 20})
