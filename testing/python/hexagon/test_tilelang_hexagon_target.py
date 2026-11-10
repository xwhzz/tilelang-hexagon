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


@pytest.mark.parametrize("spec", [{"kind": "hexagon", "mcpu": "hexagonv73"}, Target({"kind": "hexagon", "mcpu": "hexagonv73"})])
def test_explicit_arch_wins_over_the_environment(spec, monkeypatch):
    monkeypatch.setenv("HEXAGON_DSP_ARCH", "v79")
    target = determine_target(spec, return_object=True)
    assert target.attrs["mcpu"] == "hexagonv73"
    # Not measured on v73 hardware: left to the runtime.
    assert "vtcm-capacity" not in target.attrs


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


@pytest.mark.parametrize("arch", ["v73", "v75", "v79"])
def test_2d_dma_uses_one_descriptor_for_small_strides(arch):
    source = str(_lower(_dma_2d_kernel(), {"kind": "hexagon", "mcpu": f"hexagon{arch}"}).kernel_source)
    submits = [line for line in source.splitlines() if "tl_hexagon_dma_async_submit(" in line]
    assert len(submits) == 1
    assert "tl_hexagon_dma_async_context<1, 1>" in source
    assert source.count("tl_hexagon_dma_async_commit(tl_dma_async)") == 1
    assert source.count("tl_hexagon_dma_async_wait(tl_dma_async, 0)") == 1
    assert "(uint32_t)128, (uint32_t)256, (uint32_t)128, (uint32_t)32, 3)" in submits[0]


@pytest.mark.parametrize("stride", [65535, 65536])
@pytest.mark.parametrize("arch", ["v73", "v75", "v79"])
def test_2d_dma_stride_field_boundary(arch, stride):
    @T.prim_func
    def kernel(A: T.Tensor((2, stride), "uint8")):
        with T.Kernel(1):
            scratch = T.alloc_shared((2, 32), "uint8")
            T.dma_copy(A[:, :32], scratch)
            T.dma_wait()

    source = str(_lower(kernel, {"kind": "hexagon", "mcpu": f"hexagon{arch}"}).kernel_source)
    submits = [line for line in source.splitlines() if "tl_hexagon_dma_async_submit(" in line]
    split = arch == "v73" and stride == 65536
    assert len(submits) == (2 if split else 1)
    assert source.count("tl_hexagon_dma_async_commit(tl_dma_async)") == 1
    if split:
        for row, call in enumerate(submits):
            assert f"(&(A[{row * stride}]))" in call
            assert "(uint32_t)32, (uint32_t)32, (uint32_t)32, (uint32_t)1, 3)" in call
        assert "tl_hexagon_dma_async_context<2, 1>" in source
    else:
        assert f"(uint32_t)32, (uint32_t){stride}, (uint32_t)32, (uint32_t)2, 3)" in submits[0]


def test_vtcm_capacity_is_checked_at_compile_time():
    @T.prim_func
    def kernel(A: T.Tensor((1024, 1024), "float16"), B: T.Tensor((1024, 1024), "float16")):
        with T.Kernel(1):
            A_s = T.alloc_shared((1024, 1024), "float16")  # 2 MiB
            T.copy(A, A_s)
            T.copy(A_s, B)

    _lower(kernel, "hexagon")
    with pytest.raises(Exception, match="take 2097152 bytes of VTCM, more than the target's 1048576"):
        _lower(kernel, {"kind": "hexagon", "mcpu": "hexagonv79", "vtcm-capacity": 1 << 20})


def _worker_arena_kernel(programs, requested, elements=512 * 1024, sync=False):
    @T.prim_func
    def kernel(A: T.Tensor((programs, elements), "float16"), B: T.Tensor((programs, elements), "float16")):
        with T.Kernel(programs, num_workers=requested) as bx:
            A_s = T.alloc_shared((elements,), "float16")
            T.copy(A[bx, :], A_s)
            if sync:
                T.sync_grid()
            T.copy(A_s, B[bx, :])

    return kernel


@pytest.mark.parametrize("arch", ["v73", "v79"])
@pytest.mark.parametrize("requested", [2, 6, 8])
@pytest.mark.parametrize("sync", [False, True])
def test_arena_budget_uses_program_count_not_worker_request(arch, requested, sync):
    source = _lower(
        _worker_arena_kernel(2, requested, sync=sync),
        {"kind": "hexagon", "mcpu": f"hexagon{arch}", "vtcm-capacity": 4 << 20},
    ).kernel_source
    assert f"(2, {requested}, tl_vtcm_worker_arenas(1048576u)," in source
    assert ("tl_parallel_for<tl_worker_scope, true>" in source) == sync


@pytest.mark.parametrize("arch", ["v73", "v79"])
@pytest.mark.parametrize("requested", [6, 8])
def test_independent_programs_can_share_fewer_arenas(arch, requested):
    source = _lower(
        _worker_arena_kernel(8, requested),
        {"kind": "hexagon", "mcpu": f"hexagon{arch}", "vtcm-capacity": 4 << 20},
    ).kernel_source
    assert f"tl_parallel_for(8, {requested}, tl_vtcm_worker_arenas(1048576u)," in source


@pytest.mark.parametrize("arch", ["v73", "v79"])
def test_coresident_program_arenas_must_fit_together(arch):
    with pytest.raises(Exception, match="T.sync_grid.*needs 6 simultaneous VTCM arenas.*only 4 fit"):
        _lower(
            _worker_arena_kernel(6, 8, sync=True),
            {"kind": "hexagon", "mcpu": f"hexagon{arch}", "vtcm-capacity": 4 << 20},
        )


def test_even_one_worker_arena_must_fit():
    with pytest.raises(Exception, match="A worker's VTCM arena takes 1048576 bytes.*smaller tiles"):
        _lower(
            _worker_arena_kernel(8, 6),
            {"kind": "hexagon", "mcpu": "hexagonv79", "vtcm-capacity": 512 << 10},
        )


def test_unknown_arena_budget_remains_a_runtime_check():
    source = _lower(_worker_arena_kernel(6, 8, sync=True), {"kind": "hexagon", "mcpu": "hexagonv73"}).kernel_source
    assert "tl_parallel_for<tl_worker_scope, true>(6, 8, tl_vtcm_worker_arenas(1048576u)," in source


def test_dynamic_independent_grid_needs_only_one_arena():
    n = T.dynamic("n")

    @T.prim_func
    def kernel(A: T.Tensor((n, 64), "float32"), B: T.Tensor((n, 64), "float32")):
        with T.Kernel(n, num_workers=8) as p:
            s = T.alloc_shared((64,), "float32")
            T.copy(A[p, :], s)
            T.copy(s, B[p, :])

    source = _lower(kernel, {"kind": "hexagon", "mcpu": "hexagonv79", "vtcm-capacity": 2048}).kernel_source
    assert "tl_parallel_for(n, 8, tl_vtcm_worker_arenas(256u)," in source


@pytest.mark.parametrize("sync,capacity,accepted", [(False, 6144, True), (False, 6143, False), (True, 8192, True), (True, 8191, False)])
def test_worker_budget_keeps_global_and_private_arena_alignment(sync, capacity, accepted):
    @T.prim_func
    def kernel(A: T.Tensor((2, 64), "float32"), B: T.Tensor((2, 64), "float32")):
        with T.Kernel(2, num_workers=8) as w:
            g0 = T.alloc_global((33,), "float32", scope="global.vtcm")
            g1 = T.alloc_global((65,), "float32", scope="global.vtcm")
            s = T.alloc_shared((64,), "float32")
            T.copy(A[w, :], s)
            g0[w] = s[0]
            g1[w] = s[1]
            if sync:
                T.sync_grid()
            T.copy(s, B[w, :])
            B[w, 0] = g0[w] + g1[w]

    target = {"kind": "hexagon", "mcpu": "hexagonv79", "vtcm-capacity": capacity}
    if not accepted:
        with pytest.raises(Exception, match="leave after its 4096 bytes of global.vtcm buffers"):
            _lower(kernel, target)
    else:
        source = _lower(kernel, target).kernel_source
        assert "tl_vtcm_arena() + 2048" in source
        assert "tl_vtcm_worker_arenas(256u, 4096u)" in source


@pytest.mark.parametrize("arch", ["v73", "v79"])
def test_2d_dma_wide_rows_use_1d_on_v73(arch):
    @T.prim_func
    def kernel(A: T.Tensor((2, 32769), "float16")):
        with T.Kernel(1):
            scratch = T.alloc_shared((2, 32768), "float16")
            T.dma_copy(A[:, :32768], scratch)
            T.dma_wait()

    source = str(_lower(kernel, {"kind": "hexagon", "mcpu": f"hexagon{arch}"}).kernel_source)
    submits = [line for line in source.splitlines() if "tl_hexagon_dma_async_submit(" in line]
    assert len(submits) == (2 if arch == "v73" else 1)
    assert source.count("tl_hexagon_dma_async_commit(tl_dma_async)") == 1
    if arch == "v73":
        for call in submits:
            assert "(uint32_t)65536, (uint32_t)65536, (uint32_t)65536, (uint32_t)1, 3)" in call
    else:
        assert "(uint32_t)65536, (uint32_t)65538, (uint32_t)65536, (uint32_t)2, 3)" in submits[0]
