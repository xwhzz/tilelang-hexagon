"""Device sessions: several kernels sharing resident buffers in one FastRPC session."""

import os
from pathlib import Path
import shutil
import subprocess
import time
from types import SimpleNamespace

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


def test_session_agent_maps_buffers_into_the_dsp_once():
    agent = session.gen_agent("tl_session_x")
    assert "rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_FLAG_UNCACHED" in agent
    assert "fastrpc_mmap(g_domain, fd, p, 0, size, FASTRPC_MAP_FD)" in agent


@pytest.mark.parametrize("lifetime", ["session", "call"])
def test_session_dispatch_lifecycle_on_host(tmp_path, monkeypatch, lifetime):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("a host C++ compiler is required for the dispatch lifecycle test")
    # Exercise the generated implementation with real host synchronization.
    # This checks resource ownership/error cleanup, not DSP stack capacity.
    monkeypatch.setenv("TILELANG_HEXAGON_RESOURCE_LIFETIME", lifetime)
    monkeypatch.setattr(fastrpc, "SKEL_RESOURCES_OPEN", "  return resources_open();\n")
    monkeypatch.setattr(fastrpc, "SKEL_RESOURCES_CLOSE", "  resources_close();\n")
    dispatch = session._dispatch_source("  case 0: return test_kernel(a);\n")
    harness = Path(__file__).with_name("session_dispatch_host.cc").read_text()
    source, executable = tmp_path / "dispatch.cc", tmp_path / "dispatch"
    source.write_text(harness.replace("// GENERATED_DISPATCH", dispatch))
    subprocess.run([compiler, "-std=c++17", "-O2", "-pthread", str(source), "-o", str(executable)], check=True, timeout=30)
    for mode, failures in [
        ("direct", ["none", "query", "resources"]),
        ("worker", ["none", "query", "allocate", "create", "resources", "join"]),
    ]:
        for failure in failures:
            subprocess.run([str(executable), mode, failure], check=True, timeout=10)


@pytest.mark.parametrize("value", ["", "CALL", "per_call", "session "])
def test_invalid_resource_lifetime_fails_before_lowering_or_build(monkeypatch, value):
    monkeypatch.setenv("TILELANG_HEXAGON_RESOURCE_LIFETIME", value)
    monkeypatch.setattr(session, "lower_kernels", lambda *args: pytest.fail("must reject before lowering"))
    monkeypatch.setattr(session, "write_project", lambda *args: pytest.fail("must reject before building"))
    for create in (session.Session, session.Session.of_lowered):
        with pytest.raises(ValueError, match="TILELANG_HEXAGON_RESOURCE_LIFETIME"):
            create([])


class _LocalAgent:
    """A session transport whose agent is a local shell command."""

    def __init__(self, agent):
        self.agent = agent

    def push(self, local):
        return local

    def start_host_binary(self, exe, args):
        return subprocess.Popen(["sh", "-c", self.agent], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    def forward_port(self, port):
        return 1234

    def remove_forward(self, port):
        pass

    def cleanup(self):
        pass


def _start(monkeypatch, agent):
    sock = SimpleNamespace(setsockopt=lambda *args: None, settimeout=lambda timeout: None)
    monkeypatch.setattr(session, "socket", SimpleNamespace(create_connection=lambda *args, **kwargs: sock, IPPROTO_TCP=0, TCP_NODELAY=0))
    s = session.Session.__new__(session.Session)
    s._conn, s._closed, s._holds_device = _LocalAgent(agent), False, False
    s._device_dir, s._domain = "/tmp/session", 3
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


def test_a_session_sees_its_agent_ready_after_other_output(monkeypatch):
    # Both lines reach the pipe at once.
    s = _start(monkeypatch, "printf 'loading\\nAGENT_READY port=1\\n'; sleep 30")
    s._proc.kill()
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
