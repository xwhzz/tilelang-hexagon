"""Device sessions: several kernels sharing resident buffers in one FastRPC session."""

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


def test_every_run_takes_the_processes_resources_and_gives_them_back():
    plans = fastrpc._plan([], [])
    source = session.gen_dsp("tl_session_x", [("fill_kernel", "int32_t fill_kernel(float* a) { return 0; }", plans)])
    assert "return tl_hexagon_run_kernel([&]() { return tl_session_kernel_0(" in source
    run = source[source.index("static AEEResult tl_hexagon_run_kernel(") :]
    assert run.index("tl_hexagon_take()") < run.index("fn()") < run.index("tl_hexagon_give()")


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
    # Processes past the first are reserved sessions of the cDSP.
    assert "remote_session_control(FASTRPC_RESERVE_NEW_SESSION" in agent
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


@pytest.mark.parametrize("newline", ["\\n", "\\r\\n"])
def test_a_session_sees_its_agent_ready_after_other_output(monkeypatch, newline):
    # Both lines reach the pipe at once.
    s = _start(monkeypatch, f"printf 'loading{newline}AGENT_READY port=1{newline}'; sleep 30")
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


def _f32(buf):
    return torch.frombuffer(bytearray(buf.read()), dtype=torch.float32)


@DEVICE
def test_processes_have_their_own_addresses():
    # The same 2.5 GiB mapped into two processes: 5 GiB of mappings, more than
    # one process's 4 GB of addresses hold.
    with session.Session([(fill, [1])], processes=2) as s:
        bufs = [s.buffer(512 << 20) for _ in range(5)]
        for buf in bufs:
            buf.free()


@DEVICE
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
def test_processes_take_turns_at_hmx():
    with session.Session([(hmx_gemm, [2])], processes=2) as s:
        (k,) = s.kernels
        for i in range(6):
            a = (torch.randn(64, 64) * 0.25).half()
            b = (torch.randn(64, 32) * 0.25).half()
            torch.testing.assert_close(k(a, b, process=i % 2), a.float() @ b.float(), atol=2e-2, rtol=2e-2)


@DEVICE
@pytest.mark.parametrize("processes", [1, 2, 4])
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


@pytest.mark.parametrize("hlos,domain", [("linux", "3"), ("android", "4")])
def test_unsupported_multi_process_platforms_fail_before_build(monkeypatch, hlos, domain):
    monkeypatch.setenv("TILELANG_HEXAGON_HLOS", hlos)
    monkeypatch.setenv("TILELANG_HEXAGON_DOMAIN", domain)
    monkeypatch.setattr(session, "write_project", lambda *a: pytest.fail("must reject before building"))
    with pytest.raises(NotImplementedError, match="multiple DSP processes require Android"):
        session.Session.of_lowered([], processes=2)


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
