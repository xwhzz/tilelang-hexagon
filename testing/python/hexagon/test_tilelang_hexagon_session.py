"""Device sessions: several kernels sharing resident buffers in one FastRPC session."""

import os
import time

import pytest
import torch

from tilelang.hexagon import language as T
from tilelang.hexagon.runtime import fastrpc, session

DEVICE = pytest.mark.skipif(os.environ.get("TL_HEXAGON_DEVICE_TEST") != "1", reason="requires an authorized Hexagon device and SDK")

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


def _exp(fast: bool):
    @T.prim_func
    def kernel(a: T.Tensor((N,), "float32"), out: T.Tensor((N,), "float32")):
        if fast:
            T.annotate_pass_configs({"tl.enable_fast_math": True})
        with T.Kernel(1):
            for i in T.Parallel(N):
                out[i] = T.exp(a[i])

    return kernel


def test_session_kernels_lower_under_their_own_pass_configs():
    (_, fast, _, _), (_, accurate, _, _) = session.lower_kernels([(_exp(True), [1]), (_exp(False), [1])])
    assert "tl_hvx_exp_fast(" in fast
    assert "tl_hvx_exp(" in accurate and "tl_hvx_exp_fast(" not in accurate


def test_session_kernels_lower_once(tmp_path, monkeypatch):
    import tilelang
    from tilelang.env import env

    monkeypatch.setenv("TILELANG_CACHE_DIR", str(tmp_path))
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


def test_session_agent_maps_buffers_into_the_dsp_once():
    agent = session.gen_agent("tl_session_x")
    assert "rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_FLAG_UNCACHED" in agent
    assert "fastrpc_mmap(CDSP_DOMAIN_ID, fd, p, 0, size, FASTRPC_MAP_FD)" in agent


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
