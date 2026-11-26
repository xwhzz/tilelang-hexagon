"""Device sessions: several kernels sharing resident buffers in one FastRPC session."""

from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import shutil
import struct
import subprocess
import time
from types import SimpleNamespace

import pytest
import torch

from tilelang.hexagon import language as T
from tilelang.hexagon.runtime import fastrpc, session
from tilelang.hexagon.runtime.config import domain_id, hlos

DEVICE = pytest.mark.skipif(os.environ.get("TL_HEXAGON_DEVICE_TEST") != "1", reason="requires an authorized Hexagon device and SDK")
MULTI_PROCESS = pytest.mark.skipif(
    hlos() == "android" and domain_id() != 3,
    reason="multiple DSP processes on Android require the default cDSP domain",
)

N = 4096


@T.prim_func
def fill(value: T.Tensor((1,), "float32"), out: T.Tensor((N,), "float32")):
    with T.Kernel(1):
        for i in T.Parallel(N):
            out[i] = value[0]


@T.prim_func
def add(a: T.Tensor((N,), "float32"), b: T.Tensor((N,), "float32"), out: T.Tensor((N,), "float32")):
    with T.Kernel(1):
        for i in T.Parallel(N):
            out[i] = a[i] + b[i]


@T.prim_func
def deep(a: T.Tensor((N,), "float32"), out: T.Tensor((N,), "float32")):
    """out = 2 a[i] + 2 a[i], through 32 KB of stack: more than the FastRPC
    thread, which runs the kernel, had by default."""
    with T.Kernel(1):
        local = T.alloc_local((2 * N,), "float32")
        for i in T.serial(2 * N):
            local[i] = a[i % N] * T.float32(2)
        for i in T.serial(N):
            out[i] = local[i] + local[i + N]


@T.prim_func
def dma_step(a: T.Tensor((N,), "float32"), out: T.Tensor((N,), "float32")):
    """out = a + 1 through VTCM, its DDR buffers moved by DMA alone: data that
    crosses DSP processes, whose caches are their own."""
    with T.Kernel(1):
        T.annotate_dma_only(a, out)
        s = T.alloc_shared((N,), "float32")
        T.dma_copy(a, s)
        T.dma_wait()
        for i in T.Parallel(N):
            s[i] = s[i] + T.float32(1)
        T.dma_copy(s, out)
        T.dma_wait()


@T.prim_func
def hmx_gemm(A: T.Tensor((64, 64), "float16"), B: T.Tensor((64, 32), "float16"), C: T.Tensor((64, 32), "float32")):
    with T.Kernel(1):
        a = T.alloc_shared((64, 64), "float16")
        b = T.alloc_shared((64, 32), "float16")
        c = T.alloc_shared((64, 32), "float32")
        T.copy(A, a)
        T.copy(B, b)
        T.gemm(a, b, c, clear_accum=True)
        T.copy(c, C)


@T.prim_func
def stamp(step: T.int32, out: T.Tensor((128,), "int32")):
    """out[step] = step + 1: a pipeline stage's step index."""
    with T.Kernel(1):
        out[step] = step + 1


def _exp(fast: bool):
    @T.prim_func
    def kernel(a: T.Tensor((N,), "float32"), out: T.Tensor((N,), "float32")):
        if fast:
            T.annotate_pass_configs({"tl.enable_fast_math": True})
        with T.Kernel(1):
            for i in T.Parallel(N):
                out[i] = T.exp(a[i])

    return kernel


def _strided_copy(src_strides, dst_strides):
    @T.prim_func
    def copy(x: T.StridedTensor((3, 2), src_strides, "float32"), y: T.StridedTensor((3, 2), dst_strides, "float32")):
        with T.Kernel(1):
            for i, j in T.grid(3, 2):
                y[i, j] = x[i, j]

    return copy


def test_dense_transfer_layout_after_parameter_serialization():
    from tilelang.engine.param import KernelParam, dump_kernel_params, load_kernel_params
    from tilelang.hexagon.runtime.tensor import pack_buffer, unpack_buffer

    ((_, _, params, _),) = session.lower_kernels([(_strided_copy((1, 3), (2, 1)), [1])])
    transposed, contiguous, legacy = load_kernel_params(dump_kernel_params([*params, KernelParam("float32", [3, 2])]))
    plans = fastrpc._plan([transposed, contiguous, legacy], [])
    backing = torch.arange(12, dtype=torch.float32).reshape(2, 6)[:, 1::2]
    logical = backing.T
    for data in (logical, logical.contiguous(), logical.numpy()):
        raw = pack_buffer(data, transposed, plans[0], "copy")
        assert raw == backing.contiguous().numpy().tobytes()
        restored = unpack_buffer(raw, transposed)
        assert restored.stride() == (1, 3)
        torch.testing.assert_close(restored, logical)
    assert plans[1].storage_axes == plans[2].storage_axes == ()
    for strides, elements in [([1, 1], 4), ([4, 1], 10)]:
        param = KernelParam("float32", [3, 2], strides)
        (plan,) = fastrpc._plan([param], [])
        assert plan.nbytes == elements * 4  # resident buffer bounds include any holes
        assert pack_buffer(bytes(plan.nbytes), param, plan, "copy") == bytes(plan.nbytes)
        with pytest.raises(ValueError, match=f"kernel takes {plan.nbytes}"):
            pack_buffer(bytes(plan.nbytes - 1), param, plan, "copy")
        with pytest.raises(NotImplementedError, match="non-dense"):
            pack_buffer(logical, param, plan, "copy")

    empty = KernelParam("float32", [2, 0, 3], [3, 3, 1])
    assert fastrpc._plan([empty], [0])[0].nbytes == 0
    assert unpack_buffer(b"", empty).shape == (2, 0, 3)


@T.prim_func
def rank_zero(x: T.Tensor((6,), "float32"), bias: T.Tensor((), "float32"), scale: T.float32, out: T.Tensor((), "float32")):
    with T.Kernel(1):
        out[()] = (x[0] + x[5]) * scale + bias[()]


def test_rank_zero_buffer_abi_survives_parameter_serialization(monkeypatch):
    from tilelang import tvm
    from tilelang.autotuner.tuner import _can_reuse_inputs
    from tilelang.engine.param import dump_kernel_params, load_kernel_params
    from tilelang.profiler import Profiler, TensorSupplyType

    ((_, _, params, _),) = session.lower_kernels([(rank_zero, [3])])
    old_records = [{"dtype": str(p.dtype), "shape": p.shape} for p in params]
    old_text = tvm.ir.save_json(tvm.runtime.convert(old_records))
    assert load_kernel_params(old_text)[1].is_scalar()  # Historical shape-only convention.
    for restored in (params, load_kernel_params(dump_kernel_params(params)), load_kernel_params(old_text, func=rank_zero)):
        plans = fastrpc._plan(restored, [3])
        assert [(p.is_scalar, p.nbytes, p.is_output) for p in plans] == [
            (False, 24, False),
            (False, 4, False),
            (True, 0, False),
            (False, 4, True),
        ]
        idl = fastrpc.gen_idl("rank_zero", plans)
        assert "in sequence<uint8> p1" in idl and "in int32 s2" in idl and "rout sequence<uint8> p3" in idl

    _, bias, scalar, out = restored
    assert not _can_reuse_inputs([bias], [scalar], [0.5])
    assert not _can_reuse_inputs([scalar], [bias], [torch.tensor(0.5)])
    monkeypatch.setattr("tilelang.utils.device.get_current_device", lambda: torch.device("cpu"))
    for supply in TensorSupplyType:
        profiler = Profiler([bias, out], [1], supply)
        inputs = profiler._get_inputs(with_output=True, dynamic_symbolic_constraints={"n": 6})
        assert all(x.shape == () and x.dtype == torch.float32 for x in inputs)
    with pytest.raises(ValueError, match="by-value scalar"):
        profiler.supply(scalar)


@DEVICE
def test_rank_zero_buffers_reach_jit_and_resident_calls():
    import tilelang

    x = torch.arange(6, dtype=torch.float32)
    bias = torch.tensor(0.375)
    scale = -2.5
    expected = (x[0] + x[5]) * scale + bias
    kernel = tilelang.compile(rank_zero, target="hexagon", out_idx=3)
    try:
        torch.testing.assert_close(kernel(x, bias, scale), expected)
        supplied = torch.zeros(())
        assert kernel(x, bias, scale, supplied) is None
        torch.testing.assert_close(supplied, expected)
    finally:
        kernel.adapter.close()

    with session.Session([(rank_zero, [3])]) as s:
        (kernel,) = s.kernels
        src, offset, dst = s.buffer(24), s.buffer(4), s.buffer(4)
        src.write(x)
        offset.write(bias)
        kernel(src, offset, scale, dst)
        assert dst.read() == expected.numpy().tobytes()


@DEVICE
@pytest.mark.parametrize(("src_strides", "dst_strides"), [((1, 3), (2, 1)), ((2, 1), (1, 3))])
def test_jit_copies_dense_strided_tensors(src_strides, dst_strides):
    import tilelang
    from tilelang.hexagon.runtime.adapter import HexagonKernelAdapter

    x = torch.arange(6, dtype=torch.float32).reshape(2, 3).T
    try:
        kernel = tilelang.compile(_strided_copy(src_strides, dst_strides), target="hexagon", out_idx=1)
        y = kernel(x)
        assert y.stride() == dst_strides
        torch.testing.assert_close(y, x)
    finally:
        HexagonKernelAdapter.close_all()


def test_writable_arguments_reject_overlap_and_invalid_destinations_before_dispatch():
    from tilelang.engine.param import KernelParam

    kernel = session.SessionKernel(session.Session.__new__(session.Session), 0, "update", [KernelParam("float32", [4])] * 2, [])
    kernel.session.processes, kernel._writable = 1, (1,)
    x = torch.arange(8, dtype=torch.float32)
    for a, dest, message in [
        (x[:4], x[2:6], "overlap"),
        (x[2:6].numpy(), x[:4], "overlap"),
        (x[:4], x[4:5].expand(4), "non-overlapping"),
        (x[:4], torch.zeros(4, dtype=torch.int32), "shape or dtype"),
        (x[:4], torch.zeros((2, 2)), "shape or dtype"),
        (x[:4], bytearray(16), "PyTorch tensor"),
    ]:
        with pytest.raises((ValueError, TypeError), match=message):
            kernel(a, dest)


@DEVICE
def test_jit_writes_state_and_keeps_allocated_output_order():
    import tilelang

    @T.prim_func
    def update(x: T.Tensor((4, 3), "float32"), state: T.StridedTensor((4, 3), (1, 4), "float32"), out: T.Tensor((4, 3), "float32")):
        with T.Kernel(1):
            for i, j in T.grid(4, 3):
                state[i, j] = state[i, j] + x[i, j]
                out[i, j] = 2 * state[i, j]

    kernel = tilelang.compile(update, out_idx=[2, 1], target="hexagon")
    try:
        x = torch.arange(12, dtype=torch.float32).reshape(4, 3)
        backing = torch.full((4, 6), 3.0)
        state = backing[:, ::2]
        x_version = x._version
        out = kernel(x, state, None)
        torch.testing.assert_close(state, x + 3)
        torch.testing.assert_close(out, 2 * state)
        out.fill_(-1)
        assert kernel(x, state, out) is None
        torch.testing.assert_close(state, 2 * x + 3)
        torch.testing.assert_close(out, 2 * state)
        assert (backing[:, 1::2] == 3).all() and x._version == x_version
    finally:
        kernel.adapter.close()


@DEVICE
def test_jit_and_source_only_calls_preserve_output_and_resource_contracts():
    import tilelang
    from tilelang.hexagon.runtime.adapter import HexagonKernelAdapter

    x = torch.arange(N, dtype=torch.float32)
    y, supplied = torch.ones_like(x), torch.zeros_like(x)
    versions = x._version, y._version
    kernel = tilelang.compile(add.with_attr("global_symbol", "add_into_supplied"), target="hexagon", out_idx=2)
    adapter = kernel.adapter
    params, name, source = adapter.params, adapter.kernel_name, adapter.kernel_source
    plans = fastrpc._plan(params, [])
    free_dispatch = "return tl_hexagon_run_kernel<false, false>"
    try:
        assert free_dispatch in session.gen_dsp("jit", [(name, source, plans)])
        assert kernel(x, y, supplied) is None
        torch.testing.assert_close(supplied, x + y)
        assert (x._version, y._version) == versions
    finally:
        adapter.close()

    raw = HexagonKernelAdapter(params, [2], adapter.target, str(source))
    try:
        assert free_dispatch not in session.gen_dsp("raw", [(name, raw.kernel_source, plans)])
        torch.testing.assert_close(raw(x, y), x + y)
    finally:
        raw.close()


@DEVICE
def test_eager_and_lazy_jit_write_supplied_state():
    import tilelang
    from tilelang.hexagon.runtime.adapter import HexagonKernelAdapter

    @tilelang.jit(target="hexagon")
    def eager(x, state):
        x: T.Tensor[[32], T.float32]
        state: T.Tensor[[32], T.float32]
        with T.Kernel(1):
            for i in T.Parallel(32):
                state[i] = x[i] + state[i]

    @T.prim_func
    def lazy(x: T.Tensor((32,), "float32"), state: T.Tensor((32,), "float32")):
        with T.Kernel(1):
            for i in T.Parallel(32):
                state[i] = x[i] + state[i]

    x, state = torch.arange(32, dtype=torch.float32), torch.full((32,), 3.0)
    try:
        assert eager(x, state) is None
        torch.testing.assert_close(state, x + 3)
        HexagonKernelAdapter.close_all()
        kernel = tilelang.compile(lazy, target="hexagon")
        assert kernel(x, state) is None
        torch.testing.assert_close(state, 2 * x + 3)
    finally:
        HexagonKernelAdapter.close_all()


@DEVICE
def test_jit_writes_dma_only_destination():
    import tilelang

    kernel = tilelang.compile(dma_step, target="hexagon")
    try:
        x, out = torch.arange(N, dtype=torch.float32), torch.full((N,), -1.0)
        assert kernel(x, out) is None
        torch.testing.assert_close(out, x + 1)
    finally:
        kernel.adapter.close()


@DEVICE
def test_jit_writes_through_derived_external_pointer():
    import tilelang

    @T.prim_func
    def store(x: T.Tensor((8,), "int32"), y: T.Tensor((8,), "int32"), choose: T.int32, index: T.Tensor((1,), "int32")):
        with T.Kernel(1):
            T.import_source("static inline void store_i32(void *p) { *(int32_t*)p = 31; }")
            address = T.handle_add_byte_offset(T.if_then_else(choose > 0, x.data, y.data), index[0] * 4)
            T.evaluate(T.call_extern("void", "store_i32", address))

    kernel = tilelang.compile(store, target="hexagon")
    try:
        x, y = torch.zeros(8, dtype=torch.int32), torch.zeros(8, dtype=torch.int32)
        index = torch.tensor([2], dtype=torch.int32)
        version = index._version
        kernel(x, y, 1, index)
        index[0] = 5
        kernel(x, y, 0, index)
        assert x.tolist() == [0, 0, 31, 0, 0, 0, 0, 0]
        assert y.tolist() == [0, 0, 0, 0, 0, 31, 0, 0]
        assert index._version == version + 1
    finally:
        kernel.adapter.close()


@DEVICE
def test_by_value_float32_reaches_jit_and_resident_calls():
    import tilelang

    @T.prim_func
    def scale(x: T.Tensor((32,), "float32"), y: T.Tensor((32,), "float32"), alpha: T.float32):
        with T.Kernel(1):
            for i in T.Parallel(32):
                y[i] = x[i] * alpha

    x = torch.arange(1, 33, dtype=torch.float32)
    coefficients = (0.375, -0.5, 0.0, -0.0)
    jit = tilelang.compile(scale, target="hexagon", out_idx=1)
    try:
        for alpha in coefficients:
            assert torch.equal(jit(x, alpha).view(torch.int32), (x * alpha).view(torch.int32))
    finally:
        jit.adapter.close()

    with session.Session([(scale, [1])]) as s:
        (kernel,) = s.kernels
        src, dst = s.buffer(128), s.buffer(128)
        src.write(x)
        for alpha in coefficients:
            expected = (x * alpha).numpy().tobytes()
            args = (src, dst, alpha)
            poison = bytes([0xA5]) * 128
            dst.write(poison)
            kernel(*args)
            assert dst.read() == expected
            dst.write(poison)
            s.pipeline([(kernel, 0, args)], 2)
            assert dst.read() == expected
            dst.write(poison)
            s.run_sequence([session.Run(kernel, args)])
            assert dst.read() == expected
        with pytest.raises(ValueError, match="integer scalar"):
            s.pipeline([(kernel, 0, (src, dst, s.STEP))], 1)


def test_imported_helper_preceding_the_kernel_entry(monkeypatch):
    import tilelang
    from tilelang import tvm
    from tilelang.hexagon.runtime.adapter import HexagonKernelAdapter
    from tilelang.hexagon.runtime.env import HexagonSDK

    @T.prim_func
    def add_one(x: T.Tensor((1,), "int32"), y: T.Tensor((1,), "int32")):
        with T.Kernel(1):
            T.import_source("int increment(int x) { return x + 1; }\n")
            y[0] = T.call_extern("int32", "increment", x[0])

    (lowered,) = session.lower_kernels([(add_one, [1])])
    name, source, _, _ = lowered
    assert name == "add_one_kernel"
    assert source.index("int increment(") < source.index(f"int32_t {name}(")

    captured = []
    monkeypatch.setattr(HexagonSDK, "validate", lambda self: self)
    fake = SimpleNamespace(kernels=[lambda *args: None], close=lambda: None)
    monkeypatch.setattr(session.Session, "of_lowered", classmethod(lambda cls, kernels, **kwargs: captured.extend(kernels) or fake))
    tilelang.compile(add_one, target="hexagon", out_idx=[1])
    assert captured[0][:2] == lowered[:2]
    assert "tl_hexagon_run_kernel<false, false>" not in session.gen_dsp("jit", [(captured[0][0], captured[0][1], [])])
    # The exported C symbol and the IRModule's lookup key can differ.
    func = add_one.with_attr("global_symbol", name).with_attr("tl.readonly_param_indices", [0])
    mod = tvm.IRModule({"internal": func})
    adapter = HexagonKernelAdapter(lowered[2], [1], None, source, device_mod=mod)
    assert adapter.kernel_name == name and adapter._run._writable == (1,)


def test_session_kernels_lower_under_their_own_pass_configs():
    (_, fast, _, _), (_, accurate, _, _) = session.lower_kernels([(_exp(True), [1]), (_exp(False), [1])])
    assert "tl_hvx_exp_fast(" in fast
    assert "tl_hvx_exp(" in accurate and "tl_hvx_exp_fast(" not in accurate


def test_session_kernels_lower_once(tmp_path, monkeypatch):
    import tilelang
    from tilelang.env import env

    monkeypatch.setenv("TILELANG_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(env, "TILELANG_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("TILELANG_DISABLE_CACHE", raising=False)
    calls = []
    lower = tilelang.lower
    monkeypatch.setattr(tilelang, "lower", lambda *a, **k: calls.append(1) or lower(*a, **k))
    enabled = env.is_cache_enabled()
    env.enable_cache()
    try:
        (first,) = session.lower_kernels([(_exp(False), [1])])
        (again,) = session.lower_kernels([(_exp(False), [1])])
        assert len(calls) == 1 and again[:2] == first[:2] and [p.dtype for p in again[2]] == [p.dtype for p in first[2]]
        # Other pass configs make another kernel.
        session.lower_kernels([(_exp(True), [1])])
        assert len(calls) == 2
    finally:
        if not enabled:
            env.disable_cache()


def test_session_skel_dispatches_by_kernel():
    plans = fastrpc._plan([], [])
    source = session.gen_dsp(
        "tl_session_x",
        [
            ("fill_kernel", "int32_t fill_kernel(float* a) { return 0; }", plans),
            ("add_kernel", "int32_t add_kernel(float* a) { return 0; }", plans),
        ],
    )
    # Each kernel's entry takes its index, so equal names cannot collide.
    assert "#define fill_kernel tl_session_kernel_0" in source
    assert "#define add_kernel tl_session_kernel_1" in source
    assert "case 1: return tl_session_call_1(a);" in source
    # Resident buffers resolve by fd, once.
    assert "HAP_mmap_get(fd, &vaddr, &paddr)" in source


@pytest.mark.parametrize("arch", ["v73", "v79"])
def test_raw_session_skel_compiles_without_kernel_template_includes(tmp_path, arch):
    from tilelang.hexagon.runtime.env import HexagonSDK, HexagonEnvError

    sdk = HexagonSDK()
    try:
        compiler = Path(sdk.tools_root) / "bin" / "hexagon-clang++"
    except HexagonEnvError:
        pytest.skip("a Hexagon SDK is required for the offline skel compile")
    qurt = Path(sdk.sdk_root) / "rtos" / "qurt" / f"compute{arch}" / "include"
    if not compiler.is_file() or not qurt.is_dir():
        pytest.skip(f"the Hexagon SDK must include the {arch} toolchain and QuRT headers")
    # of_lowered also accepts raw kernels with no codegen headers or USES markers.
    # The wrapper itself owns the common.h status ABI dependency.
    source = session.gen_dsp("tl_raw", [("raw", "int32_t raw() { return 0; }", [])])
    (tmp_path / "raw.cc").write_text(source)
    (tmp_path / "tl_raw.h").write_text("#include <remote.h>\n")
    includes = [
        Path(__file__).resolve().parents[3] / "src",
        tmp_path,
        Path(sdk.sdk_root) / "incs",
        Path(sdk.sdk_root) / "incs" / "stddef",
        qurt,
        qurt / "qurt",
    ]
    result = subprocess.run(
        [
            str(compiler),
            f"-m{arch}",
            "-mhvx",
            "-mhvx-length=128b",
            "-O2",
            "-fPIC",
            "-std=c++17",
            *[flag for path in includes for flag in ("-I", str(path))],
            "-c",
            str(tmp_path / "raw.cc"),
            "-o",
            str(tmp_path / "raw.o"),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_ordinary_run_uses_default_resource_policy():
    plans = fastrpc._plan([], [])
    source = session.gen_dsp("tl_session_x", [("fill_kernel", "int32_t fill_kernel(float* a) { return 0; }", plans)])
    assert "return tl_hexagon_run_kernel([&]() { return tl_session_kernel_0(" in source


def test_mixed_session_selects_resource_policy_per_kernel():
    kernels = [
        ("ordinary", "int ordinary() { return 0; }", []),
        ("cooperative", "#define TL_HEXAGON_USES_COOPERATIVE 1\nint cooperative() { return 0; }", []),
        ("ordinary_again", "int ordinary_again() { return 0; }", []),
    ]
    source = session.gen_dsp("mixed", kernels)
    for i in (0, 2):
        assert f"return tl_hexagon_run_kernel([&]() {{ return tl_session_kernel_{i}(); }});" in source
    assert "return tl_hexagon_run_kernel<true>([&]() { return tl_session_kernel_1(); });" in source
    assert fastrpc.run_kernel_entry("// #define TL_HEXAGON_USES_COOPERATIVE 1") == "tl_hexagon_run_kernel"
    oneshot = fastrpc.gen_dsp("single", "cooperative", kernels[1][1], [])
    assert "return tl_hexagon_run_kernel<true>([&]() { return cooperative(); });" in oneshot


def test_session_skips_take_and_give_only_for_proven_resource_free_kernels(tmp_path, monkeypatch):
    import json
    import tilelang
    from tilelang.env import env

    monkeypatch.setattr(env, "TILELANG_CACHE_DIR", str(tmp_path))

    @T.prim_func
    def imported(a: T.Tensor((1,), "float32")):
        with T.Kernel(1):
            T.import_source("#define PRIVATE_MACRO 3")
            a[0] = 1

    @T.prim_func
    def external(a: T.Tensor((1,), "float32")):
        with T.Kernel(1):
            a[0] = T.call_extern("float32", "external", a[0])

    def sources(functions):
        return [(name, source, []) for name, source, _, _ in session.lower_kernels([(f, []) for f in functions])]

    kernels = sources([add, dma_step, hmx_gemm])
    wrappers = session.gen_dsp("mixed", kernels)
    assert "tl_hexagon_run_kernel<false, false>([&]() { return tl_session_kernel_0(" in wrappers
    for i in (1, 2):
        assert f"tl_hexagon_run_kernel([&]() {{ return tl_session_kernel_{i}(" in wrappers
    name, source, plans = kernels[0]
    for opaque in [
        *sources([imported, external]),
        ("raw", "int raw() { return 0; }", []),
        (name, source + "\n// edited after lowering", plans),
        (name, json.loads(json.dumps(source)), plans),
    ]:
        # Source is combined into one TU. An import can affect the other kernels.
        assert "tl_hexagon_run_kernel<false, false>" not in session.gen_dsp("mixed", [*kernels, opaque])

    from tilelang.hexagon.runtime.adapter import HexagonKernelAdapter
    from tilelang.hexagon.runtime.env import HexagonSDK

    seen = []
    fake = SimpleNamespace(kernels=[lambda *args: None], close=lambda: None)
    monkeypatch.setattr(HexagonSDK, "validate", lambda self: self)
    monkeypatch.setattr(session.Session, "of_lowered", classmethod(lambda cls, lowered, **kw: seen.extend(lowered) or fake))
    tilelang.compile(add, target="hexagon", out_idx=2)
    HexagonKernelAdapter(kernels[0][2], [], None, str(source)).close()
    generated, supplied = [(n, src, []) for n, src, _, _ in seen]
    assert "tl_hexagon_run_kernel<false, false>" in session.gen_dsp("jit", [generated])
    assert "tl_hexagon_run_kernel<false, false>" not in session.gen_dsp("raw", [supplied])

    # Only a cache entry written for these exact source bytes restores the proof.
    monkeypatch.setattr(tilelang, "lower", lambda *args, **kwargs: pytest.fail("expected a lowering cache hit"))
    assert session.gen_dsp("mixed", sources([add, dma_step, hmx_gemm])) == wrappers
    for path in (tmp_path / "hexagon_lowered").glob("*.json"):
        record = json.loads(path.read_text())
        if record["source"] == source:
            del record["source_digest"]  # Historical shape, prior to the proof.
            path.write_text(json.dumps(record))
            break
    else:
        pytest.fail("missing cached source")
    assert "tl_hexagon_run_kernel<false, false>" not in session.gen_dsp("mixed", sources([add, dma_step, hmx_gemm]))


def test_normal_close_cannot_hide_unsent_shutdown():
    def fail_send(data):
        raise OSError("connection lost")

    closed = []
    instance = session.Session.__new__(session.Session)
    instance._closed, instance._holds_device = False, False
    instance._sock = SimpleNamespace(sendall=fail_send, close=lambda: closed.append("socket"))
    instance._conn = SimpleNamespace(cleanup=lambda: closed.append("remote"))
    with pytest.raises(RuntimeError, match="shutdown request failed: connection lost"):
        instance.close()
    assert closed == ["socket", "remote"]
    assert instance._closed


def test_context_manager_keeps_body_error_if_close_also_fails():
    instance = session.Session.__new__(session.Session)

    def fail_close():
        raise RuntimeError("close failure")

    instance.close = fail_close
    with (
        pytest.warns(RuntimeWarning, match="close failed after an earlier error"),
        pytest.raises(ValueError, match="body failure"),
        instance,
    ):
        raise ValueError("body failure")


def test_a_pipeline_stage_runs_its_steps_in_turns():
    plans = fastrpc._plan([], [])
    source = session.gen_dsp("tl_session_x", [("fill_kernel", "int32_t fill_kernel(float* a) { return 0; }", plans)])
    stage = source[source.index("AEEResult tl_session_x_pipeline(") :]
    assert (
        stage.index("tl_turn_wait(word, mine)")
        < stage.index("tl_session_dispatch_run(kernel, a)")
        < stage.index("tl_turn_write(word, mine + 1)")
    )
    # The processes' caches are their own: the turn word's line is dropped
    # before a read and cleaned after a write.
    assert 'asm volatile("dcinva(%0)"' in source and 'asm volatile("dccleaninva(%0)"' in source


def test_pipeline_stages_take_resident_buffers_and_scalars_on_their_processes():
    s = session.Session.__new__(session.Session)
    s.processes = 2
    s._closed, s._holds_device = False, False
    s._conn = SimpleNamespace(cleanup=lambda: None)
    s._allocations, s._buffer_generations = {}, {}
    ((name, _, params, out_idx),) = session.lower_kernels([(dma_step, [1])])
    k = session.SessionKernel(s, 0, name, params, out_idx)
    a0 = session._BufferAllocation(0, N * 4, 1, processes=(0,))
    a1 = session._BufferAllocation(1, N * 4, 1, processes=(1,))
    s._allocations = {0: a0, 1: a1}
    on0 = session.DeviceBuffer(s, 0, N * 4, processes=(0,), _allocation=a0)
    on1 = session.DeviceBuffer(s, 1, N * 4, processes=(1,), _allocation=a1)
    with pytest.raises(ValueError, match="distinct processes"):
        s.pipeline([(k, 0, [on0, on0]), (k, 0, [on0, on0])], 1)
    with pytest.raises(ValueError, match=r"runs on process 1, its buffer is mapped into \[0\]"):
        s.pipeline([(k, 1, [on0, on1])], 1)
    with pytest.raises(ValueError, match="Session.STEP stands for a scalar"):
        s.pipeline([(k, 0, [session.Session.STEP, on0])], 1)
    with pytest.raises(TypeError, match="resident buffers and scalars"):
        s.pipeline([(k, 0, [torch.zeros(N), on0])], 1)


def test_session_agent_maps_a_buffer_into_the_processes_it_names():
    agent = session.gen_agent("tl_session_x")
    assert "rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_FLAG_UNCACHED" in agent
    # Processes past the first are further sessions of the cDSP: reserved on
    # Android, opened by URI on Linux, whose BSP has no reservation request,
    # with PD controls on the domain and mappings on the session's extended
    # domain.
    assert "remote_session_control(FASTRPC_RESERVE_NEW_SESSION" in agent
    linux = agent[agent.index("#ifndef __ANDROID__") : agent.index("#else", agent.index("#ifndef __ANDROID__"))]
    assert '"%s%s&_session=%d", tl_session_x_URI, d->uri, p' in linux
    assert "mapped = domain_id | p << 16;" in linux
    assert "g_domains[p] = mapped;" in agent
    assert "fastrpc_mmap(g_domains[q], fd, p, 0, size, FASTRPC_MAP_FD)" in agent
    # A host access syncs the lines every process caches.
    assert "if (ok) ok = sync_buffer(id, offset, size, 1) == AEE_SUCCESS;" in agent


def test_a_session_runs_on_one_to_four_processes():
    for processes in (0, 5):
        with pytest.raises(ValueError, match="1 to 4 DSP processes"):
            session.Session.of_lowered([], processes=processes)


def test_buffers_and_runs_name_processes_of_their_session():
    s = session.Session.__new__(session.Session)
    s.processes = 2
    s._closed, s._holds_device = False, False
    s._conn = SimpleNamespace(cleanup=lambda: None)
    s._allocations, s._buffer_generations = {}, {}
    with pytest.raises(ValueError, match="processes 0 to 1"):
        s.buffer(64, processes=[2])
    with pytest.raises(ValueError, match="processes 0 to 1"):
        s.buffer(64, processes=[])
    kernel = session.SessionKernel(s, 0, "fill", [], [])
    with pytest.raises(ValueError, match="fill runs on one of processes 0 to 1, not 2"):
        kernel(process=2)


def test_session_dispatch_lifecycle_on_host(tmp_path):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("a host C++ compiler is required for the dispatch lifecycle test")
    # Exercise the generated implementation with real host synchronization.
    # This checks resource ownership/error cleanup, not DSP stack capacity.
    dispatch = fastrpc.RUN_KERNEL + session._dispatch_source("  case 0: return tl_hexagon_run_kernel([&]() { return test_kernel(a); });\n")
    harness = Path(__file__).with_name("session_dispatch_host.cc").read_text()
    source, executable = tmp_path / "dispatch.cc", tmp_path / "dispatch"
    source.write_text(harness.replace("// GENERATED_DISPATCH", dispatch))
    subprocess.run([compiler, "-std=c++17", "-O2", "-pthread", str(source), "-o", str(executable)], check=True, timeout=30)
    for mode, failures in [
        ("direct", ["none", "query", "resources", "take"]),
        ("worker", ["none", "query", "allocate", "create", "resources", "take", "join", "exited"]),
    ]:
        for failure in failures:
            subprocess.run([str(executable), mode, failure], check=True, timeout=10)


def test_session_pipeline_error_propagation_on_host(tmp_path):
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("a host C compiler is required for the pipeline error test")
    # Compile the actual generated handler and launch function. Only the
    # FastRPC/I/O boundary is replaced; pthread synchronization remains real.
    agent = session.gen_agent("tl_session_x")
    launch = agent[agent.index("struct launch_gate {") : agent.index("static int sync_buffer(")]
    start = agent.index(f"  if (op == {session._OP_PIPELINE}) {{")
    handler = agent[start : agent.index(f"  if (op == {session._OP_ALLOC}) {{", start)]
    harness = Path(__file__).with_name("session_pipeline_host.c").read_text()
    source, executable = tmp_path / "pipeline.c", tmp_path / "pipeline"
    source.write_text(harness.replace("// GENERATED_PIPELINE_LAUNCH", launch).replace("// GENERATED_PIPELINE_HANDLER", handler))
    subprocess.run(
        [
            compiler,
            "-std=c11",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-pthread",
            f"-DTEST_PIPELINE_OP={session._OP_PIPELINE}",
            f"-DTEST_RESIDENT={session._RESIDENT}",
            f"-DTEST_STEP={session._STEP}",
            f"-DTEST_TURN_GAVE_UP={session._TURN_GAVE_UP}",
            f"-DTEST_KERNEL_STATUS_BASE={fastrpc.KERNEL_STATUS_BASE}",
            str(source),
            "-o",
            str(executable),
        ],
        check=True,
        timeout=30,
    )
    subprocess.run([str(executable)], check=True, timeout=10)


class _LocalAgent:
    """A session transport whose agent is a local shell command."""

    def __init__(self, agent):
        self.agent = agent

    def push(self, local):
        return local

    def start_host_binary(self, exe, args):
        self.args = args
        return subprocess.Popen(["sh", "-c", self.agent], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    def forward_port(self, port):
        return 1234

    def remove_forward(self, port):
        pass

    def cleanup(self):
        pass


def _start(monkeypatch, agent, domain=3):
    sock = SimpleNamespace(setsockopt=lambda *args: None, settimeout=lambda timeout: None)
    monkeypatch.setattr(session, "socket", SimpleNamespace(create_connection=lambda *args, **kwargs: sock, IPPROTO_TCP=0, TCP_NODELAY=0))
    s = session.Session.__new__(session.Session)
    s._conn, s._closed, s._holds_device = _LocalAgent(agent), False, False
    s._device_dir, s._domain = "/tmp/session", domain
    s.processes = 1
    s._artifacts = SimpleNamespace(dsp_skel="skel.so", agent_exe="agent")
    try:
        s._start(5.0)
    except BaseException:
        s.close()
        raise
    return s


def test_a_session_reports_what_its_agent_printed_when_it_exits(monkeypatch):
    with pytest.raises(RuntimeError, match=r"exited \(3\) before ready:\ncannot load the skel"):
        _start(monkeypatch, "echo 'cannot load the skel'; exit 3")


@pytest.mark.parametrize("domain", [3, 4])
def test_a_session_starts_its_agent_on_its_domain(monkeypatch, domain):
    s = _start(monkeypatch, "echo 'AGENT_READY port=1'; sleep 30", domain)
    assert s.domain == domain
    assert s._conn.args[2] == str(domain)
    s._proc.kill()
    s._broken = True
    s.close()


@pytest.mark.parametrize("newline", ["\\n", "\\r\\n"])
def test_a_session_sees_its_agent_ready_after_other_output(monkeypatch, newline):
    # Both lines reach the pipe at once.
    s = _start(monkeypatch, f"printf 'loading{newline}AGENT_READY port=1{newline}'; sleep 30")
    s._proc.kill()
    s._broken = True  # This test deliberately kills its local fake agent.
    s.close()


@pytest.fixture(scope="module")
def sess():
    s = session.Session([(fill, [1]), (add, [2])])
    yield s
    s.close()


@DEVICE
def test_kernels_share_a_resident_buffer(sess):
    k_fill, k_add = sess.kernels
    buf = sess.buffer(N * 4)
    # fill writes into the resident buffer; add reads it in another call.
    assert k_fill(torch.tensor([2.5]), buf) is None
    x = torch.randn(N)
    torch.testing.assert_close(k_add(x, buf), x + 2.5)
    # The host sees what the kernel wrote.
    torch.testing.assert_close(torch.frombuffer(bytearray(buf.read()), dtype=torch.float32), torch.full((N,), 2.5))
    buf.free()


@DEVICE
def test_host_writes_and_device_files_reach_kernels(sess):
    _, k_add = sess.kernels
    a, b = sess.buffer(N * 4), sess.buffer(2 * N * 4)
    x, y = torch.randn(N), torch.randn(N)
    a.write(x)
    b.write(y, offset=N * 4)
    torch.testing.assert_close(k_add(a, b.at(N * 4)), x + y)
    # Rewriting a buffer the DSP has read invalidates its cached lines.
    a.write(2 * x)
    torch.testing.assert_close(k_add(a, b.at(N * 4)), 2 * x + y)
    a.free()
    b.free()


@DEVICE
def test_a_buffer_loads_a_range_of_a_device_file(sess, tmp_path):
    _, k_add = sess.kernels
    data = torch.randn(4 * N)
    (tmp_path / "data.bin").write_bytes(data.numpy().tobytes())
    path = f"{sess._device_dir}/data.bin"
    sess._conn.push(str(tmp_path / "data.bin"), "data.bin")
    buf = sess.buffer(3 * N * 4)
    buf.write(torch.full((3 * N,), 7.0))
    x = torch.randn(N)
    # The DSP caches the lines it reads; a load drops them.
    torch.testing.assert_close(k_add(x, buf), x + 7)
    # The file's second N floats into the buffer's: those bytes alone.
    assert buf.at(N * 4, N * 4).load(path, N * 4) == N * 4
    # Past the file's end, a load reads what is left.
    assert buf.load(path, 3 * N * 4) == N * 4
    sevens = torch.full((N,), 7.0)
    got = torch.frombuffer(bytearray(buf.read()), dtype=torch.float32)
    torch.testing.assert_close(got, torch.cat([data[3 * N :], data[N : 2 * N], sevens]))
    torch.testing.assert_close(k_add(x, buf), x + data[3 * N :])
    with pytest.raises(RuntimeError, match="could not read"):
        buf.load(f"{sess._device_dir}/missing.bin")
    buf.free()


def test_a_buffer_is_under_2_gib():
    s = session.Session.__new__(session.Session)
    for nbytes in (0, 1 << 31):
        with pytest.raises(ValueError, match="1 byte to 2 GiB - 1"):
            s.buffer(nbytes)


@DEVICE
def test_a_host_write_replaces_what_a_kernel_wrote(sess):
    k_fill, k_add = sess.kernels
    buf = sess.buffer(N * 4)
    # fill leaves the DSP's lines of buf dirty; the host's zeros must win.
    k_fill(torch.tensor([3.0]), buf)
    buf.write(torch.zeros(N))
    torch.testing.assert_close(torch.frombuffer(bytearray(buf.read()), dtype=torch.float32), torch.zeros(N))
    x = torch.randn(N)
    torch.testing.assert_close(k_add(x, buf), x)
    buf.free()


@DEVICE
def test_a_kernel_has_a_deep_stack():
    with session.Session([(deep, [1])]) as s:
        (k_deep,) = s.kernels
        x = torch.randn(N)
        torch.testing.assert_close(k_deep(x), 4 * x)


@DEVICE
def test_a_kernel_has_the_requested_execution_stack():
    from tilelang.engine.param import KernelParam

    source = """#include <stdint.h>
#include <qurt.h>
int32_t stack_info(int32_t* out) {
  qurt_thread_attr_t attr;
  if (qurt_thread_attr_get(qurt_thread_get_id(), &attr) != QURT_EOK) return 1;
  out[0] = attr.stack_size;
  out[1] = (intptr_t)attr.stack_addr;
  out[2] = (intptr_t)&attr;
  return 0;
}
"""
    with session.Session.of_lowered([("stack_info", source, [KernelParam("int32", [3])], [0])]) as s:
        size, base, local = s.kernels[0]().tolist()
        # Check the execution thread, not the host's accepted stack request.
        assert size >= fastrpc.PD_THREAD_STACK
        assert base & 0xFFFFFFFF <= local & 0xFFFFFFFF < (base & 0xFFFFFFFF) + size


@DEVICE
def test_a_large_resident_argument_adds_no_cost(sess):
    _, k_add = sess.kernels
    x = torch.randn(N)
    small = sess.buffer(N * 4)
    big = sess.buffer(512 << 20)

    def per_call(buf):
        k_add(x, buf)
        start = time.perf_counter()
        for _ in range(20):
            k_add(x, buf)
        return (time.perf_counter() - start) / 20

    # The kernel reads the first N floats of either buffer; 512 MB more
    # resident bytes add no per-call maintenance.
    assert per_call(big) < 1.5 * per_call(small) + 0.002
    big.free()
    small.free()


def _f32(buf):
    return torch.frombuffer(bytearray(buf.read()), dtype=torch.float32)


@DEVICE
@MULTI_PROCESS
def test_processes_have_their_own_addresses():
    # The same 2.5 GiB mapped into two processes: 5 GiB of mappings, more than
    # one process's 4 GB of addresses hold.
    with session.Session([(fill, [1])], processes=2) as s:
        bufs = [s.buffer(512 << 20) for _ in range(5)]
        for buf in bufs:
            buf.free()


@DEVICE
@MULTI_PROCESS
def test_the_host_sees_what_another_process_wrote():
    with session.Session([(fill, [1]), (add, [2])], processes=2) as s:
        k_fill, k_add = s.kernels
        buf = s.buffer(N * 4)
        # Process 1's kernel leaves its lines dirty; the read syncs process 1 too.
        k_fill(torch.tensor([2.5]), buf, process=1)
        torch.testing.assert_close(_f32(buf), torch.full((N,), 2.5))
        # The host's write reaches process 1's kernel, not lines it cached.
        buf.write(torch.ones(N))
        x = torch.randn(N)
        torch.testing.assert_close(k_add(x, buf, process=1), x + 1)
        buf.free()


@DEVICE
@MULTI_PROCESS
def test_processes_take_turns_at_vtcm_and_pass_data_by_dma():
    with session.Session([(dma_step, [1])], processes=2) as s:
        (k,) = s.kernels
        a, b = s.buffer(N * 4), s.buffer(N * 4)
        x = torch.randn(N)
        a.write(x)
        for _ in range(5):
            k(a, b, process=0)
            k(b, a, process=1)
        torch.testing.assert_close(_f32(a), x + 10)
        a.free()
        b.free()


@DEVICE
@MULTI_PROCESS
def test_processes_take_turns_at_hmx():
    with session.Session([(hmx_gemm, [2])], processes=2) as s:
        (k,) = s.kernels
        for i in range(6):
            a = (torch.randn(64, 64) * 0.25).half()
            b = (torch.randn(64, 32) * 0.25).half()
            torch.testing.assert_close(k(a, b, process=i % 2), a.float() @ b.float(), atol=2e-2, rtol=2e-2)


@DEVICE
@pytest.mark.parametrize("processes", [1, pytest.param(2, marks=MULTI_PROCESS), pytest.param(4, marks=MULTI_PROCESS)])
def test_a_pipeline_hands_data_on_by_dma(processes):
    with session.Session([(dma_step, [1])], processes=processes) as s:
        (k,) = s.kernels
        bufs = [s.buffer(N * 4) for _ in range(processes)]
        x = torch.randn(N)
        bufs[0].write(x)
        # Stage p adds 1 to what stage p - 1 left, and the last hands on to the first.
        stages = [(k, p, [bufs[p], bufs[(p + 1) % processes]]) for p in range(processes)]
        s.pipeline(stages, 100 // processes)
        torch.testing.assert_close(_f32(bufs[0]), x + 100)
        for buf in bufs:
            buf.free()


@DEVICE
@MULTI_PROCESS
def test_steps_reach_the_stages():
    with session.Session([(stamp, [])], processes=2) as s:
        (k,) = s.kernels
        out = [s.buffer(128 * 4, processes=[p]) for p in (0, 1)]
        # Steps 0-63, then 64-127 in a second call.
        for first in (0, 64):
            step = session.Session.STEP + first
            s.pipeline([(k, 0, [step, out[0]]), (k, 1, [step, out[1]])], 64)
        for buf in out:
            torch.testing.assert_close(torch.frombuffer(bytearray(buf.read()), dtype=torch.int32), torch.arange(1, 129, dtype=torch.int32))
            buf.free()


@DEVICE
@MULTI_PROCESS
def test_a_failing_stage_ends_the_pipeline():
    with session.Session([(dma_step, [1])], processes=2) as s:
        (k,) = s.kernels
        a, b = s.buffer(N * 4), s.buffer(N * 4)
        # The skel has no kernel 9: stage 0 fails at its first step, and stage 1
        # gives up waiting for its turn.
        missing = session.SessionKernel(s, 9, "missing", [], [])
        with pytest.raises(RuntimeError, match="a pipeline failed"):
            s.pipeline([(missing, 0, []), (k, 1, [a, b])], 4)
        # The session carries on.
        x = torch.randn(N)
        a.write(x)
        s.pipeline([(k, 0, [a, b]), (k, 1, [b, a])], 2)
        torch.testing.assert_close(_f32(a), x + 4)
        a.free()
        b.free()


@pytest.mark.parametrize("domain", ["0", "4"])
@pytest.mark.parametrize("processes", [2, 3, 4])
def test_unsupported_multi_process_platforms_fail_before_build(monkeypatch, domain, processes):
    monkeypatch.setenv("TILELANG_HEXAGON_HLOS", "android")
    monkeypatch.setenv("TILELANG_HEXAGON_DOMAIN", domain)
    monkeypatch.setattr(session, "write_project", lambda *a: pytest.fail("must reject before building"))
    with pytest.raises(NotImplementedError, match="multiple DSP processes on Android require the default cDSP"):
        session.Session.of_lowered([], processes=processes)


class _Built(Exception):
    pass


def _stop_at_build(*args):
    raise _Built


@pytest.mark.parametrize(("environment", "domain", "rejected"), [("3", 4, True), ("4", 3, False), ("4", None, True)])
def test_a_session_domain_overrides_the_environment(monkeypatch, environment, domain, rejected):
    # Android's multi-process guard sees the session's domain, not the variable.
    monkeypatch.setenv("TILELANG_HEXAGON_HLOS", "android")
    monkeypatch.setenv("TILELANG_HEXAGON_DOMAIN", environment)
    monkeypatch.setattr(session, "write_project", _stop_at_build)
    with pytest.raises(NotImplementedError if rejected else _Built):
        session.Session.of_lowered([], processes=2, domain=domain)


@pytest.mark.parametrize("domain", [-1, 1 << 31, True, "4", 4.0])
def test_an_invalid_session_domain_fails_before_build(monkeypatch, domain):
    monkeypatch.setattr(session, "write_project", lambda *a: pytest.fail("must reject before building"))
    with pytest.raises(ValueError, match="domain must be a nonnegative int32"):
        session.Session.of_lowered([], domain=domain)


# The FastRPC domains of the device's cDSPs, "3,4" on an SA8650.
DOMAINS = [int(d) for d in os.environ.get("TL_HEXAGON_DOMAINS", str(domain_id())).split(",")]


@DEVICE
def test_sessions_on_each_cdsp_run_at_the_same_time():
    # A session on each cDSP (two on the one cDSP of a device that has one),
    # called from two threads; then a kernel on the first adds what both made.
    domains = (DOMAINS * 2)[:2]
    sessions = [session.Session([(fill, [1]), (add, [2])], domain=d) for d in domains]
    try:
        assert [s.domain for s in sessions] == domains
        values = [torch.tensor([1.5]), torch.tensor([-4.0])]
        with ThreadPoolExecutor(2) as pool:
            made = list(pool.map(lambda i: [sessions[i].kernels[0](values[i]) for _ in range(20)][-1], range(2)))
        for value, out in zip(values, made):
            torch.testing.assert_close(out, value.expand(N))
        torch.testing.assert_close(sessions[0].kernels[1](*made), torch.full((N,), -2.5))
    finally:
        for s in sessions:
            s.close()


def test_vtcm_cached_context_releases_partial_grants_on_host(tmp_path):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("a host C++ compiler is required")
    # The harness tracks actual cached-context grants and failure cleanup.
    (tmp_path / "HAP_compute_res.h").write_text("")
    source = Path(__file__).with_name("vtcm_lifetime_host.cc")
    root = Path(__file__).resolve().parents[3]
    binary = tmp_path / "vtcm"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-I",
            str(tmp_path),
            "-I",
            str(root / "src"),
            str(source),
            "-o",
            str(binary),
        ],
        check=True,
        timeout=30,
    )
    subprocess.run([str(binary)], check=True, timeout=10)


class _WireSocket:
    def __init__(self, reply=b"", failure=None, fail_send=False):
        self.reply, self.failure, self.fail_send = reply, failure, fail_send
        self.sent, self.timeouts = [], []
        self.timeout, self.closed = session.CALL_TIMEOUT, False

    def sendall(self, data):
        self.sent.append(data)
        if self.fail_send:
            raise self.failure

    def recv(self, count):
        if self.reply:
            data, self.reply = self.reply[:count], self.reply[count:]
            return data
        if self.failure is not None:
            raise self.failure
        return b""

    def close(self):
        self.closed = True

    def gettimeout(self):
        return self.timeout

    def settimeout(self, timeout):
        self.timeout = timeout
        self.timeouts.append(timeout)


def _wire_session(sock):
    instance = session.Session.__new__(session.Session)
    instance._closed, instance._holds_device, instance.processes = False, False, 1
    instance._conn = SimpleNamespace(cleanup=lambda: None)
    instance._sock = sock
    allocation = session._BufferAllocation(0, 128, 1)
    instance._allocations = {0: allocation}
    instance._buffer_generations = {0: 1}
    instance.kernels = [session.SessionKernel(instance, 0, "empty", [], [])]
    return instance, session.DeviceBuffer(instance, 0, 128, _allocation=allocation)


@pytest.mark.parametrize("reply_kind", ["success", "kernel error", "truncated"])
def test_inline_writeback_waits_for_the_complete_success_reply(reply_kind):
    from tilelang.engine.param import KernelParam

    src, dest = torch.arange(4, dtype=torch.float32), torch.full((4,), 3.0)
    block = struct.pack("<4f", 3, 4, 5, 6).ljust(128, b"\0")
    reply = {"success": b"\0" + block * 2, "kernel error": b"\1", "truncated": b"\0" + block}[reply_kind]
    sock = _WireSocket(reply)
    instance, _ = _wire_session(sock)
    kernel = session.SessionKernel(instance, 0, "update", [KernelParam("float32", [4])] * 3, [2])
    kernel._writable = (1, 2)
    if reply_kind == "success":
        returned = kernel(src, dest)
        torch.testing.assert_close(returned, src + 3)
        torch.testing.assert_close(dest, src + 3)
        (wire,) = sock.sent
        assert struct.unpack_from("<B4i9iI", wire) == (
            session._OP_RUN,
            0,
            3,
            256,
            0,
            session._INPUT,
            128,
            0,
            session._INOUT,
            0,
            16,
            session._OUTPUT,
            128,
            0,
            256,
        )
        assert wire[-256:-240] == struct.pack("<4f", 3, 3, 3, 3)
    else:
        with pytest.raises(RuntimeError):
            kernel(src, dest)
        assert (dest == 3).all()
        assert instance._closed == (reply_kind == "truncated")
    instance.close()


def test_resident_writes_keep_large_contiguous_storage_and_preserve_byte_order():
    import numpy as np

    sock = _WireSocket(reply=b"\0" * 8)
    instance, buffer = _wire_session(sock)
    data = torch.arange(32769, dtype=torch.int32)
    size = data.numel() * data.element_size()
    buffer.nbytes = buffer._allocation.nbytes = size + 128
    buffer.write(data, 128)
    head, payload = sock.sent
    assert struct.unpack("<BiQQ", head) == (session._OP_WRITE, 0, 128, size)
    assert np.shares_memory(np.frombuffer(payload, dtype=np.int32), data.numpy())
    assert bytes(payload) == data.numpy().tobytes()

    array = np.arange(12, dtype=np.uint16).reshape(3, 4)
    for value, expected in [
        (memoryview(array[:, ::2]), array[:, ::2].tobytes()),
        (torch.from_numpy(array.T), array.T.tobytes()),
        (memoryview(array), array.tobytes()),
        (torch.ones(4, dtype=torch.bfloat16), torch.ones(4, dtype=torch.bfloat16).view(torch.int16).numpy().tobytes()),
        (b"", b""),
    ]:
        sock.sent.clear()
        buffer.write(value)
        assert sock.sent == [struct.pack("<BiQQ", session._OP_WRITE, 0, 0, len(expected)) + expected]
    with pytest.raises(ValueError, match="overrun"):
        buffer.write(data, 129)
    instance._closed = True


@pytest.mark.parametrize("failure", [TimeoutError, KeyboardInterrupt])
@pytest.mark.parametrize(
    "operation,phase",
    [
        (op, phase)
        for op in ("run", "pipeline", "write", "read", "load", "timing", "allocate", "free", "sequence")
        for phase in ("send", "reply")
        if (op, phase) != ("free", "reply")
    ],
)
def test_interrupted_wire_operation_invalidates_session_and_buffers(operation, phase, failure):
    # For multibyte replies, consume a prefix before failing. No future
    # allocation may mistake the missing reply's tail for a buffer ID.
    prefix = b"\0" if phase == "reply" and operation in ("read", "load", "timing", "allocate", "sequence") else b""
    sock = _WireSocket(prefix, failure("interrupted exchange"), phase == "send")
    instance, buffer = _wire_session(sock)
    kernel = instance.kernels[0]
    invoke = {
        "run": lambda: kernel(),
        "pipeline": lambda: instance.pipeline([(kernel, 0, [])], 1),
        "write": lambda: buffer.write(b"test"),
        "read": lambda: buffer.read(4),
        "load": lambda: buffer.load("/model"),
        "timing": instance.last_run_seconds,
        "allocate": lambda: instance.buffer(128),
        "free": buffer.free,
        "sequence": lambda: instance.run_sequence([session.Run(kernel, ())]),
    }[operation]
    with pytest.raises(failure, match="interrupted exchange"):
        invoke()
    assert instance._closed and sock.closed
    assert len(sock.sent) == 1  # Do not append SHUTDOWN to an incomplete frame.
    with pytest.raises(ValueError, match="no longer live"):
        buffer.read(4)
    with pytest.raises(ValueError, match="closed"):
        instance.buffer(128)
    assert len(sock.sent) == 1


@pytest.mark.parametrize("failed", [False, True])
def test_sequence_timeout_is_scoped_and_complete_errors_keep_session(failed):
    # A complete failure has the failed Run's timing too; it leaves framing
    # intact and permits a subsequent request on the same Session.
    reply = struct.pack("<IIIIIq", 0 if failed else 1, 0 if failed else 0xFFFFFFFF, 2 if failed else 0, 0x1001 if failed else 0, 1, 7)
    reply += struct.pack("<IIq", 0, 0, 5)
    sock = _WireSocket(reply)
    instance, _ = _wire_session(sock)
    try:
        if failed:
            with pytest.raises(session.SequenceError):
                instance.run_sequence([session.Run(instance.kernels[0], ())], timeout=1800)
        else:
            assert instance.run_sequence([session.Run(instance.kernels[0], ())], timeout=1800).completed_operations == 1
        assert not instance._closed
        assert sock.timeouts == [1800, session.CALL_TIMEOUT]
        sock.reply = struct.pack("<q", 250000000)
        assert instance.last_run_seconds() == 0.25
    finally:
        instance.close()


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf"), None, "invalid"])
def test_invalid_sequence_timeout_sends_nothing(timeout):
    sock = _WireSocket()
    instance, _ = _wire_session(sock)
    try:
        with pytest.raises(ValueError, match="finite and positive"):
            instance.run_sequence([session.Run(instance.kernels[0], ())], timeout=timeout)
        assert not sock.sent and not instance._closed
    finally:
        instance.close()


def test_sequence_timeout_cleanup_does_not_mask_original_exception(monkeypatch):
    sock = _WireSocket(failure=KeyboardInterrupt("original interruption"))
    instance, _ = _wire_session(sock)
    calls = []

    def timeout(value):
        calls.append(value)
        if len(calls) > 1:
            raise RuntimeError("must not restore a closed socket")

    monkeypatch.setattr(sock, "settimeout", timeout)
    with pytest.raises(KeyboardInterrupt, match="original interruption"):
        instance.run_sequence([session.Run(instance.kernels[0], ())], timeout=1800)
    assert instance._closed and calls == [1800]


def test_sequence_timeout_restore_failure_preserves_complete_error(monkeypatch):
    reply = struct.pack("<IIIIIq", 0, 0, 2, 0x1001, 1, 7) + struct.pack("<IIq", 0, 0, 5)
    sock = _WireSocket(reply)
    instance, _ = _wire_session(sock)

    def timeout(value):
        if value == session.CALL_TIMEOUT:
            raise RuntimeError("timeout restore failed")

    monkeypatch.setattr(sock, "settimeout", timeout)
    with pytest.raises(session.SequenceError) as error:
        instance.run_sequence([session.Run(instance.kernels[0], ())], timeout=1800)
    assert error.value.raw_status == 0x1001
    assert instance._closed and sock.closed


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan"), 1e308])
def test_invalid_startup_timeout_fails_before_build(monkeypatch, timeout):
    monkeypatch.setattr(session, "write_project", lambda *a: pytest.fail("must reject before build"))
    with pytest.raises(ValueError, match="ready_timeout"):
        session.Session.of_lowered([], ready_timeout=timeout)
