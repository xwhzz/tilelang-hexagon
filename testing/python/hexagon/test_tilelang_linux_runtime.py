"""Host-only regression checks for the optional Linux BSP transport/build path."""

import importlib
import os
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest

from tilelang.hexagon.runtime import adapter, build, device, elf_crc, env, linux_build, linux_runtime


@pytest.fixture(autouse=True)
def clean_runtime_environment(monkeypatch):
    # These tests also run inside the SA8650 image, whose defaults enable Linux.
    for name in list(os.environ):
        if name.startswith("TILELANG_HEXAGON_") or name == "HEXAGON_DSP_ARCH":
            monkeypatch.delenv(name)
    # They check the build commands themselves, over fake projects.
    monkeypatch.setattr(importlib.import_module("tilelang.env").env, "is_cache_enabled", lambda: False)


class FakeSDK:
    dsp_arch = "v73"
    environ = {}
    build_cmake = "/sdk/build_cmake"

    def validate(self):
        return self


@pytest.mark.parametrize(
    "configured, explicit, expected",
    [
        (None, None, "v79"),
        ("v73", None, "v73"),
        ("v73", "v79", "v79"),
    ],
)
def test_adapter_architecture_respects_environment_and_explicit_argument(monkeypatch, tmp_path, configured, explicit, expected):
    monkeypatch.delenv("HEXAGON_DSP_ARCH", raising=False)
    if configured:
        monkeypatch.setenv("HEXAGON_DSP_ARCH", configured)
    monkeypatch.setattr(env.HexagonSDK, "validate", lambda sdk: sdk)
    monkeypatch.setattr(adapter.HexagonKernelAdapter, "_open", lambda self: None)
    instance = adapter.HexagonKernelAdapter([], [], None, "int test() {}", dsp_arch=explicit)
    assert instance.dsp_arch == expected
    assert instance.sdk.dsp_arch == expected


def test_adapter_runs_a_session_on_both_android_and_linux(monkeypatch):
    from tilelang.hexagon.runtime import session

    monkeypatch.setattr(env.HexagonSDK, "validate", lambda sdk: sdk)
    opened = []
    fake = SimpleNamespace(kernels=[lambda *args: "session"], close=lambda: None)
    monkeypatch.setattr(session.Session, "of_lowered", classmethod(lambda cls, lowered, **kwargs: opened.append(lowered) or fake))

    kernel = adapter.HexagonKernelAdapter([], [], None, "int32_t k() {}", dsp_arch="v79")
    assert kernel.func() == "session"
    assert opened == [[("k", "int32_t k() {}", [], [])]]
    monkeypatch.setenv("TILELANG_HEXAGON_HLOS", "linux")
    assert adapter.HexagonKernelAdapter([], [], None, "int32_t k() {}", dsp_arch="v73").func() == "session"


def test_a_linux_session_does_not_require_adb(monkeypatch):
    from tilelang.hexagon.runtime import session

    monkeypatch.setenv("TILELANG_HEXAGON_HLOS", "linux")
    monkeypatch.setenv("TILELANG_HEXAGON_SSH_HOST", "root@board")
    monkeypatch.setenv("TILELANG_HEXAGON_DOMAIN", "4")
    monkeypatch.setattr(device.AdbConnection, "list_devices", lambda: pytest.fail("SSH must not discover ADB devices"))
    monkeypatch.setattr(build, "build_cmake_fastrpc", lambda *a, **kw: build.FastRPCArtifacts("skel", agent_exe="agent"))
    monkeypatch.setattr(session.Session, "_start", lambda *a: None)
    with session.Session.of_lowered([], sdk=FakeSDK()) as instance:
        assert instance._conn.ssh_host == "root@board"
        assert instance._conn.serial is None
        assert instance._domain == 4


@pytest.mark.parametrize("legacy", ["0", "1"])
def test_linux_build_does_not_build_android_agent(monkeypatch, legacy):
    monkeypatch.setenv("TILELANG_HEXAGON_HLOS", "linux")
    monkeypatch.setenv("TILELANG_HEXAGON_ELF_CRC", "0")
    monkeypatch.setenv("TILELANG_HEXAGON_LEGACY_COMPUTE_RES", legacy)
    calls = []
    monkeypatch.setattr(build, "_run", lambda argv, **kwargs: calls.append(argv))
    monkeypatch.setattr(build, "_find_one", lambda patterns: "/project/skel.so")
    monkeypatch.setattr(linux_build, "build_linux_host", lambda *args: "/project/linux/driver")
    monkeypatch.setattr(linux_build, "validate_dsp_artifact", lambda *args: None)
    artifacts = build.build_cmake_fastrpc("/project", "example", sdk=FakeSDK())
    assert calls == [["/sdk/build_cmake", "hexagon", "DSP_ARCH=v73", "TL_HEXAGON_LEGACY_COMPUTE_RES=" + ("ON" if legacy == "1" else "OFF")]]
    assert artifacts.host_exe == "/project/linux/driver"
    assert artifacts.agent_exe == ""


def test_android_build_remains_default(monkeypatch):
    monkeypatch.delenv("TILELANG_HEXAGON_HLOS", raising=False)
    monkeypatch.setenv("TILELANG_HEXAGON_LEGACY_COMPUTE_RES", "1")
    calls = []
    monkeypatch.setattr(build, "_run", lambda argv, **kwargs: calls.append(argv))
    products = iter(["/project/skel.so", "/project/agent"])
    monkeypatch.setattr(build, "_find_one", lambda patterns: next(products))
    artifacts = build.build_cmake_fastrpc("/project", "example", sdk=FakeSDK())
    assert [argv[1] for argv in calls] == ["hexagon", "android"]
    assert "TL_HEXAGON_LEGACY_COMPUTE_RES=OFF" in calls[0]
    assert artifacts.host_exe == ""
    assert artifacts.agent_exe == "/project/agent"


@pytest.mark.parametrize("known_hosts", ["/tmp/home with spaces/known_hosts", ""])
def test_ssh_checks_host_key_and_preserves_exit_status(monkeypatch, known_hosts):
    monkeypatch.setenv("TILELANG_HEXAGON_SSH_HOST", "root@board")
    monkeypatch.setenv("TILELANG_HEXAGON_SSH_KNOWN_HOSTS", known_hosts)
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=7, stdout="out", stderr="err")

    monkeypatch.setattr(linux_runtime.subprocess, "run", run)
    assert linux_runtime.LinuxConnection(serial="board").shell("exit 7") == (7, "outerr")
    assert "StrictHostKeyChecking=yes" in calls[0]
    assert ("UserKnownHostsFile=" + known_hosts in calls[0]) == bool(known_hosts)
    assert "UserKnownHostsFile=" not in calls[0]
    assert calls[0][-2:] == ["root@board", "exit 7"]


def test_linux_adb_reports_remote_failure_even_when_transport_succeeds(monkeypatch):
    monkeypatch.delenv("TILELANG_HEXAGON_SSH_HOST", raising=False)
    monkeypatch.setenv("TILELANG_HEXAGON_HLOS", "linux")
    conn = linux_runtime.LinuxConnection(serial="board")
    marker = f"__TILELANG_REMOTE_EXIT_{os.getpid()}__="
    monkeypatch.setattr(conn, "_run", lambda *args, **kwargs: (0, f"failure\r\n{marker}17\r\n"))
    assert conn.shell("exit 17") == (17, "failure")


def test_android_adb_ignores_linux_ssh_configuration(monkeypatch):
    monkeypatch.setenv("TILELANG_HEXAGON_SSH_HOST", "root@board")
    conn = device.AdbConnection(serial="android")
    calls = []
    monkeypatch.setattr(conn, "_run", lambda argv, **kwargs: calls.append(argv) or (0, "android"))
    assert conn.shell("echo android") == (0, "android")
    assert calls == [["shell", "echo android"]]


def test_linux_connection_uses_configured_device_and_workdir(monkeypatch):
    monkeypatch.setenv("TILELANG_HEXAGON_DEVICE", "board:5555")
    monkeypatch.setenv("TILELANG_HEXAGON_REMOTE_DIR", "/tmp/pr_test")
    conn = linux_runtime.LinuxConnection.from_environment()
    assert conn.serial == "board:5555"
    assert Path(conn.workdir).parent == Path("/tmp/pr_test")
    assert conn.workdir != linux_runtime.LinuxConnection.from_environment().workdir


def test_linux_default_parent_is_tmp(monkeypatch):
    monkeypatch.setenv("TILELANG_HEXAGON_DEVICE", "board")
    assert Path(linux_runtime.LinuxConnection.from_environment().workdir).parent == Path("/tmp/tilelang_hexagon")


def test_linux_transport_does_not_change_with_environment(monkeypatch):
    monkeypatch.setenv("TILELANG_HEXAGON_DEVICE", "first:5555")
    monkeypatch.setenv("TILELANG_HEXAGON_SSH_HOST", "root@first")
    monkeypatch.setenv("TILELANG_HEXAGON_SSH_KNOWN_HOSTS", "/first/known_hosts")
    monkeypatch.setenv("ADB_SERVER_SOCKET", "tcp:localhost:5037")
    conn = linux_runtime.LinuxConnection.from_environment()
    monkeypatch.setenv("TILELANG_HEXAGON_DEVICE", "second:5555")
    monkeypatch.setenv("TILELANG_HEXAGON_SSH_HOST", "root@second")
    monkeypatch.setenv("TILELANG_HEXAGON_SSH_KNOWN_HOSTS", "/second/known_hosts")
    monkeypatch.setenv("ADB_SERVER_SOCKET", "tcp:localhost:5038")
    calls = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs["env"]))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(linux_runtime.subprocess, "run", run)
    conn.shell("true")
    conn._run(["push", "input", "output"])
    assert calls[0][0][-2] == "root@first"
    assert "UserKnownHostsFile=/first/known_hosts" in calls[0][0]
    assert "-s" not in calls[1][0]  # SSH selection ignores the ADB serial entirely.
    assert all(env["ADB_SERVER_SOCKET"] == "tcp:localhost:5037" for _, env in calls)


def test_two_same_named_one_shot_kernels_keep_separate_deployments(monkeypatch, tmp_path):
    monkeypatch.setenv("TILELANG_HEXAGON_HLOS", "linux")
    monkeypatch.setenv("TILELANG_HEXAGON_DEVICE", "board")
    monkeypatch.setenv("TILELANG_HEXAGON_REMOTE_DIR", "/tmp/shared-parent")
    remote_files = {}

    def adb(self, argv, **kwargs):
        if argv[0] == "push":
            remote_files[argv[2]] = Path(argv[1]).read_bytes()
            return 0, "copied"
        assert argv[0] == "shell"
        return 0, f"__TILELANG_REMOTE_EXIT_{os.getpid()}__=0\n"

    def compile_project(project, iface, **kwargs):
        project = Path(project)
        source = (project / (iface + "_dsp.cc")).read_bytes()
        skel, host = project / ("lib" + iface + "_skel.so"), project / (iface + "_test")
        skel.write_bytes(source)
        host.write_bytes(source)
        return build.FastRPCArtifacts(str(skel), host_exe=str(host))

    monkeypatch.setattr(linux_runtime.LinuxConnection, "_run", adb)
    monkeypatch.setattr(build, "build_cmake_fastrpc", compile_project)
    one_shot = [
        linux_runtime.OneShotKernel("same_name", f"int32_t same_name() {{ return {value}; }}", [], [], sdk=FakeSDK(), dsp_arch="v73")
        for value in (11, 22)
    ]
    assert one_shot[0]._exe != one_shot[1]._exe
    for kernel, value in zip(one_shot, (11, 22)):
        assert f"return {value}".encode() in remote_files[kernel._exe]
        kernel._conn.push(str(Path(kernel._workdir) / "tl_same_name_test"), "p0.bin")
    assert len([path for path in remote_files if path.endswith("/p0.bin")]) == 2


def test_android_honors_explicit_device(monkeypatch):
    from tilelang.hexagon.runtime import config

    monkeypatch.setenv("TILELANG_HEXAGON_DEVICE", "phone")
    monkeypatch.setattr(device.AdbConnection, "list_devices", lambda: ["board", "phone"])
    assert config.device_serial() == "phone"


def test_linux_workdir_failure_is_reported_even_with_ssh(monkeypatch):
    monkeypatch.setenv("TILELANG_HEXAGON_SSH_HOST", "root@board")
    conn = linux_runtime.LinuxConnection(serial="board")
    monkeypatch.setattr(conn, "shell", lambda *args, **kwargs: (1, "denied"))
    with pytest.raises(device.HexagonConnectionError, match="denied"):
        conn._ensure_workdir()
    assert not conn._ensured


def test_linux_execution_sets_cdsp_path_without_diagnostic_options(monkeypatch):
    monkeypatch.setenv("TILELANG_HEXAGON_RUN_LOG", "/missing/diagnostic.log")
    monkeypatch.setenv("TILELANG_HEXAGON_RUN_TIMEOUT", "invalid")
    conn = linux_runtime.LinuxConnection(serial="board")
    calls = []
    monkeypatch.setattr(conn, "shell", lambda cmd, **kwargs: calls.append(cmd) or (0, "done"))
    assert conn.run_host_binary("/tmp/test/driver", ["input with spaces"]) == (0, "done")
    assert "CDSP_LIBRARY_PATH=" in calls[-1]
    assert "/vendor/dsp/cdsp;" in calls[-1]
    assert "'input with spaces'" in calls[-1]
    assert "timeout " not in calls[-1]


def hexagon_elf():
    size = 136
    data = bytearray(size)
    ident = b"\x7fELF\x01\x01\x01" + bytes(9)
    struct.pack_into("<16sHHIIIIIHHHHHH", data, 0, ident, 3, 164, 1, 0, 52, 96, 0x73, 52, 32, 1, 40, 1, 0)
    struct.pack_into("<8I", data, 52, 1, 0, 0x10000, 0x10000, size, size, 5, 4096)
    return data


def test_load_crc_is_idempotent_and_detects_modified_load_bytes(tmp_path):
    path = tmp_path / "skel.so"
    path.write_bytes(hexagon_elf())
    first = elf_crc.add_load_crc(path)
    complete = path.read_bytes()
    assert elf_crc.add_load_crc(path) == first
    assert path.read_bytes() == complete
    corrupt = bytearray(complete)
    corrupt[84] ^= 1
    with pytest.raises(ValueError, match="ELF CRC mismatch"):
        elf_crc.verify(corrupt)


def test_dsp_architecture_must_match_requested_target(tmp_path):
    path = tmp_path / "skel.so"
    path.write_bytes(hexagon_elf())
    linux_build.validate_dsp_artifact(path, "v73")
    with pytest.raises(RuntimeError, match="architecture mismatch"):
        linux_build.validate_dsp_artifact(path, "v79")


def test_linux_build_returns_a_resident_agent(monkeypatch, tmp_path):
    monkeypatch.setenv("TILELANG_HEXAGON_HLOS", "linux")
    (tmp_path / "example_agent.c").write_text("int main(void) { return 0; }\n")
    calls = []
    monkeypatch.setattr(build, "_run", lambda argv, **kwargs: calls.append(argv))
    monkeypatch.setattr(build, "_find_one", lambda patterns: "/project/skel.so")
    monkeypatch.setattr(linux_build, "build_linux_host", lambda *args: "/project/linux/example_agent")
    monkeypatch.setattr(linux_build, "validate_dsp_artifact", lambda *args: None)
    artifacts = build.build_cmake_fastrpc(str(tmp_path), "example", sdk=FakeSDK())
    assert artifacts.agent_exe == "/project/linux/example_agent"
    assert artifacts.host_exe == ""
    assert [command[1] for command in calls] == ["hexagon"]
    assert "TL_HEXAGON_LEGACY_COMPUTE_RES=OFF" in calls[0]


def test_ssh_streams_binary_files_without_adb_or_sftp(monkeypatch, tmp_path):
    import subprocess

    monkeypatch.setenv("TILELANG_HEXAGON_HLOS", "linux")
    monkeypatch.setenv("TILELANG_HEXAGON_SSH_HOST", "root@board")
    monkeypatch.setattr(device.AdbConnection, "list_devices", lambda: pytest.fail("unexpected ADB discovery"))
    conn = linux_runtime.LinuxConnection.from_environment()
    conn.workdir = str(tmp_path / "remote's directory")
    # Run the exact remote shell commands locally, including their file streams.
    monkeypatch.setattr(conn, "_shell_argv", lambda command: ["sh", "-c", command])
    source, result = tmp_path / "source", tmp_path / "result"
    source.write_bytes(bytes(range(256)) * 1024)
    remote = conn.push(str(source), "input's bytes.bin")
    assert conn.pull(remote, str(result)) == str(result)
    assert result.read_bytes() == source.read_bytes()
    with pytest.raises(device.HexagonConnectionError, match="SSH download failed"):
        conn.pull(str(tmp_path / "missing"), str(result))
    assert result.read_bytes() == source.read_bytes()
    assert subprocess.run(["test", "-d", conn.workdir]).returncode == 0


def test_ssh_transfer_timeout_is_captured_and_can_be_overridden(monkeypatch):
    assert linux_runtime.LinuxConnection(ssh_host="board").transfer_timeout == 600
    monkeypatch.setenv("TILELANG_HEXAGON_SSH_TRANSFER_TIMEOUT", "900")
    conn = linux_runtime.LinuxConnection(ssh_host="board")
    assert conn.transfer_timeout == 900
    monkeypatch.setenv("TILELANG_HEXAGON_SSH_TRANSFER_TIMEOUT", "1800")
    assert conn.transfer_timeout == 900
    assert linux_runtime.LinuxConnection(ssh_host="board", transfer_timeout=0.5).transfer_timeout == 0.5


@pytest.mark.parametrize("timeout", ["0", "-1", "nan", "inf", "bad"])
def test_invalid_ssh_transfer_timeout_is_rejected(monkeypatch, timeout):
    monkeypatch.setenv("TILELANG_HEXAGON_SSH_TRANSFER_TIMEOUT", timeout)
    with pytest.raises(ValueError, match="finite and positive"):
        linux_runtime.LinuxConnection(ssh_host="board")
    with pytest.raises(ValueError, match="finite and positive"):
        linux_runtime.LinuxConnection(ssh_host="board", transfer_timeout=timeout)


def _blocked_ssh(pid_file):
    """A real subprocess that records its PID, emits partial data and stalls."""
    import sys

    return [
        sys.executable,
        "-c",
        "import os, sys, time; from pathlib import Path; "
        f"Path({str(pid_file)!r}).write_text(str(os.getpid())); "
        "sys.stdout.buffer.write(b'partial'); sys.stdout.buffer.flush(); time.sleep(60)",
    ]


@pytest.mark.parametrize("stage", ["mkdir", "upload"])
def test_blocked_ssh_deployment_times_out_and_releases_session_lock(monkeypatch, tmp_path, stage):
    import fcntl
    import subprocess
    from tilelang.hexagon.runtime import session

    monkeypatch.setenv("TILELANG_HEXAGON_HLOS", "linux")
    lock_path = tmp_path / "device.lock"
    monkeypatch.setenv("TILELANG_HEXAGON_DEVICE_LOCK", str(lock_path))
    lock = device.DeviceLock()
    monkeypatch.setattr(session, "DEVICE_LOCK", lock)
    source = tmp_path / "skel.so"
    source.write_bytes(b"skel")
    pid_file = tmp_path / "blocked.pid"
    conn = linux_runtime.LinuxConnection(workdir=str(tmp_path / "private"), ssh_host="local-test", transfer_timeout=1)

    def shell_argv(command):
        blocked = command.startswith("mkdir -p") if stage == "mkdir" else command.startswith("cat >")
        return _blocked_ssh(pid_file) if blocked else ["sh", "-c", command]

    monkeypatch.setattr(conn, "_shell_argv", shell_argv)
    monkeypatch.setattr(linux_runtime.LinuxConnection, "from_environment", classmethod(lambda *args: conn))
    monkeypatch.setattr(build, "build_cmake_fastrpc", lambda *a, **kw: build.FastRPCArtifacts(str(source), agent_exe=str(source)))
    with pytest.raises(subprocess.TimeoutExpired) as timed_out:
        session.Session.of_lowered([], sdk=FakeSDK())
    assert timed_out.value.timeout == 1
    with pytest.raises(ProcessLookupError):
        os.kill(int(pid_file.read_text()), 0)
    assert not Path(conn.workdir).exists()
    assert not conn._workdir_attempted
    assert lock._depth == 0 and lock._fd is None
    with lock_path.open("rb") as fd:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_blocked_ssh_download_preserves_destination_and_removes_staging(monkeypatch, tmp_path):
    import subprocess

    local = tmp_path / "downloads"
    local.mkdir()
    destination = local / "result"
    destination.write_bytes(b"completed previous result")
    pid_file = tmp_path / "blocked.pid"
    conn = linux_runtime.LinuxConnection(ssh_host="local-test", transfer_timeout=1)
    monkeypatch.setattr(conn, "_shell_argv", lambda command: _blocked_ssh(pid_file))
    with pytest.raises(subprocess.TimeoutExpired) as timed_out:
        conn.pull("/remote/model", str(destination))
    assert timed_out.value.timeout == 1
    with pytest.raises(ProcessLookupError):
        os.kill(int(pid_file.read_text()), 0)
    assert destination.read_bytes() == b"completed previous result"
    assert list(local.iterdir()) == [destination]


@pytest.mark.parametrize("fails", [False, True])
def test_ssh_tunnel_reports_ready_or_error_and_is_reaped(monkeypatch, fails):
    import subprocess

    popen = subprocess.Popen
    calls = []
    processes = []

    def start(argv, **kwargs):
        calls.append(argv)
        command = "echo 'forwarding denied'; exit 7" if fails else "printf 'TILELANG_TUNNEL_READY\\n'; cat"
        proc = popen(["sh", "-c", command], **kwargs)
        processes.append(proc)
        return proc

    monkeypatch.setattr(linux_runtime.subprocess, "Popen", start)
    conn = linux_runtime.LinuxConnection(ssh_host="root@board")
    if fails:
        with pytest.raises(device.HexagonConnectionError, match="forwarding denied"):
            conn.forward_port(4321)
    else:
        port = conn.forward_port(4321)
        assert f"127.0.0.1:{port}:127.0.0.1:4321" in calls[0]
        assert "ExitOnForwardFailure=yes" in calls[0]
        assert "StrictHostKeyChecking=yes" in calls[0]
        assert processes[0].poll() is None
        conn.remove_forward(port)
    assert not conn._tunnels
    assert processes[0].poll() is not None
    assert processes[0].stdin.closed and processes[0].stdout.closed


def test_session_transport_cleans_up_agent_on_shutdown(monkeypatch, tmp_path):
    import sys
    from tilelang.hexagon.runtime import session

    monkeypatch.setenv("TILELANG_HEXAGON_HLOS", "linux")
    monkeypatch.setenv("TILELANG_HEXAGON_DOMAIN", "4")
    local = tmp_path / "build"
    local.mkdir()
    skel, agent = local / "skel.so", local / "agent"
    skel.write_bytes(b"skel")
    agent.write_text(
        f"#!{sys.executable}\n"
        "import socket, sys\n"
        "assert sys.argv[1:] == ['0', '4'], sys.argv\n"
        "s=socket.socket(); s.bind(('127.0.0.1', 0)); s.listen(1)\n"
        "print('AGENT_READY port=%d' % s.getsockname()[1], flush=True)\n"
        "conn, _=s.accept()\n"
        "assert conn.recv(1) == bytes([2])\n"
        "conn.close(); s.close()\n"
    )
    conn = linux_runtime.LinuxConnection(workdir=str(tmp_path / "device's session"))
    monkeypatch.setattr(conn, "_shell_argv", lambda command: ["sh", "-c", command])
    # SSH runs the local shell stand-in and exercises stream uploads.
    conn.ssh_host = "root@board"
    forwards = []
    monkeypatch.setattr(conn, "forward_port", lambda port: forwards.append(port) or port)
    monkeypatch.setattr(conn, "remove_forward", lambda port: forwards.remove(port))
    monkeypatch.setattr(linux_runtime.LinuxConnection, "from_environment", classmethod(lambda cls, serial=None: conn))
    monkeypatch.setattr(build, "build_cmake_fastrpc", lambda *a, **kw: build.FastRPCArtifacts(str(skel), agent_exe=str(agent)))
    with session.Session.of_lowered([], sdk=FakeSDK()) as instance:
        assert len(forwards) == 1
        assert Path(conn.workdir, "skel.so").read_bytes() == b"skel"
        assert instance._proc.poll() is None
    assert instance._proc.returncode == 0
    assert not Path(conn.workdir).exists()
    assert not forwards


def test_linux_sdk_validation_does_not_require_android_ndk(monkeypatch, tmp_path):
    for name in (
        "setup_sdk_env.source",
        "tools/HEXAGON_Tools/19.0.07/Tools/bin/hexagon-clang",
        "ipc/fastrpc/qaic/bin/qaic",
        "bin/cmake",
        "bin/ninja",
    ):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    sdk = env.HexagonSDK(sdk_root=str(tmp_path), cmake_root=str(tmp_path), ndk_root=str(tmp_path / "missing-ndk"))
    monkeypatch.setenv("TILELANG_HEXAGON_HLOS", "linux")
    assert sdk.validate() is sdk
    monkeypatch.setenv("TILELANG_HEXAGON_HLOS", "android")
    with pytest.raises(env.HexagonEnvError, match="Android NDK root"):
        sdk.validate()


@pytest.mark.parametrize("stage", ["push", "start", "forward", "connect"])
def test_partial_session_start_releases_lock_and_deployment(monkeypatch, stage):
    import subprocess
    from tilelang.hexagon.runtime import session

    monkeypatch.setenv("TILELANG_HEXAGON_HLOS", "linux")
    lock = device.DeviceLock()
    monkeypatch.setattr(session, "DEVICE_LOCK", lock)
    acquired = []
    monkeypatch.setattr(lock, "acquire", lambda: acquired.append(True) or True)
    monkeypatch.setattr(lock, "release", lambda: acquired.pop())
    cleaned, forwards, processes = [], [], []

    def fail(where):
        if stage == where:
            raise RuntimeError(where)

    def start(*args):
        fail("start")
        proc = subprocess.Popen(["sh", "-c", "echo 'AGENT_READY port=1234'; read unused"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        processes.append(proc)
        return proc

    def forward(port):
        fail("forward")
        forwards.append(port)
        return port

    conn = SimpleNamespace(
        workdir="/private/session",
        push=lambda path: fail("push"),
        start_host_binary=start,
        forward_port=forward,
        remove_forward=forwards.remove,
        cleanup=lambda: cleaned.append(True),
    )
    monkeypatch.setattr(linux_runtime.LinuxConnection, "from_environment", classmethod(lambda *a: conn))
    monkeypatch.setattr(build, "build_cmake_fastrpc", lambda *a, **kw: build.FastRPCArtifacts("skel", agent_exe="agent"))
    monkeypatch.setattr(session.socket, "create_connection", lambda *a, **kw: fail("connect"))
    with pytest.raises(RuntimeError, match=stage):
        session.Session.of_lowered([], sdk=FakeSDK())
    assert cleaned == [True]
    assert not acquired and not forwards
    assert all(proc.poll() is not None for proc in processes)


def test_agent_watchdog_reaps_on_transport_eof(monkeypatch, tmp_path):
    import sys

    conn = linux_runtime.LinuxConnection(workdir=str(tmp_path / "private"))
    monkeypatch.setattr(conn, "_shell_argv", lambda command: ["sh", "-c", command])
    conn.ssh_host = "local-test"
    source = tmp_path / "agent"
    source.write_text(f"#!{sys.executable}\nimport time\nprint('running', flush=True)\ntime.sleep(60)\n")
    remote = conn.push(str(source))
    proc = conn.start_host_binary(remote, [])
    try:
        assert proc.stdout.readline() == b"running\n"
        proc.stdin.close()  # Simulates the SSH stream closing without protocol shutdown.
        proc.wait(timeout=3)
        assert not Path(conn.workdir).exists()
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        proc.stdout.close()
        conn.cleanup()


def test_linux_domain_search_paths_are_configurable_and_captured(monkeypatch):
    import shlex

    monkeypatch.setenv("TILELANG_HEXAGON_DSP_LIBRARY_PATH", "/bsp/cdsp1;/bsp/shared with spaces")
    conn = linux_runtime.LinuxConnection(workdir="/private/session")
    monkeypatch.setenv("TILELANG_HEXAGON_DSP_LIBRARY_PATH", "/another/bsp")
    command = shlex.split(conn._host_command("/private/session/agent", ["0", "4"]))
    expected = "/private/session;/bsp/cdsp1;/bsp/shared with spaces"
    assert "ADSP_LIBRARY_PATH=" + expected in command
    assert "CDSP_LIBRARY_PATH=" + expected in command
    assert "CDSP1_LIBRARY_PATH=" + expected in command


def test_cleanup_error_never_retains_the_device_lock(monkeypatch):
    from tilelang.hexagon.runtime import session

    released = []
    monkeypatch.setattr(session, "DEVICE_LOCK", SimpleNamespace(release=lambda: released.append(True)))
    instance = session.Session.__new__(session.Session)
    instance._closed, instance._holds_device = False, True

    def disconnected():
        raise RuntimeError("network disconnected")

    instance._conn = SimpleNamespace(cleanup=disconnected)
    with pytest.warns(RuntimeWarning, match="network disconnected"):
        instance.close()
    assert released == [True]
    instance.close()
    assert released == [True]
