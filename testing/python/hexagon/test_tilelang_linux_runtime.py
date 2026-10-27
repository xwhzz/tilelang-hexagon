"""Host-only regression checks for the optional Linux BSP transport/build path."""

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
    monkeypatch.setattr(adapter.HexagonKernelAdapter, "_deploy", lambda self: None)
    monkeypatch.setattr(adapter.HexagonKernelAdapter, "_convert_torch_func", lambda self: lambda: None)
    instance = adapter.HexagonKernelAdapter([], [], None, "int test() {}", dsp_arch=explicit, workdir=str(tmp_path))
    assert instance.dsp_arch == expected
    assert instance.sdk.dsp_arch == expected
    assert instance._use_agent is True


def test_adapter_selects_linux_connection_only_for_linux(monkeypatch):
    monkeypatch.delenv("TILELANG_HEXAGON_HLOS", raising=False)
    monkeypatch.setenv("TILELANG_HEXAGON_SSH_HOST", "root@board")
    monkeypatch.setattr(device.AdbConnection, "list_devices", lambda: ["android"])
    instance = object.__new__(adapter.HexagonKernelAdapter)
    instance._conn = None
    assert type(instance._connection()) is device.AdbConnection
    monkeypatch.setenv("TILELANG_HEXAGON_HLOS", "linux")
    monkeypatch.setenv("TILELANG_HEXAGON_DEVICE", "board")
    instance._conn = None
    assert type(instance._connection()) is linux_runtime.LinuxConnection


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
    products = iter(["/project/skel.so", "/project/host", "/project/agent"])
    monkeypatch.setattr(build, "_find_one", lambda patterns: next(products))
    artifacts = build.build_cmake_fastrpc("/project", "example", sdk=FakeSDK())
    assert [argv[1] for argv in calls] == ["hexagon", "android"]
    assert "TL_HEXAGON_LEGACY_COMPUTE_RES=OFF" in calls[0]
    assert artifacts.host_exe == "/project/host"
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
    assert calls[1][0][1:3] == ["-s", "first:5555"]
    assert all(env["ADB_SERVER_SOCKET"] == "tcp:localhost:5037" for _, env in calls)


def test_two_same_named_linux_kernels_keep_separate_deployments(monkeypatch, tmp_path):
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
        return build.FastRPCArtifacts(str(skel), str(host), "")

    monkeypatch.setattr(linux_runtime.LinuxConnection, "_run", adb)
    monkeypatch.setattr(adapter, "build_cmake_fastrpc", compile_project)
    kernels = [
        adapter.HexagonKernelAdapter(
            [],
            [],
            None,
            f"int32_t same_name() {{ return {value}; }}",
            sdk=FakeSDK(),
            workdir=str(tmp_path / str(value)),
        )
        for value in (11, 22)
    ]
    assert kernels[0]._exe_remote != kernels[1]._exe_remote
    for kernel, value in zip(kernels, (11, 22)):
        assert f"return {value}".encode() in remote_files[kernel._exe_remote]
        kernel._connection().push(str(Path(kernel._workdir) / "tl_same_name_test"), "p0.bin")
    assert len([path for path in remote_files if path.endswith("/p0.bin")]) == 2


def test_android_honors_explicit_device(monkeypatch):
    monkeypatch.setenv("TILELANG_HEXAGON_DEVICE", "phone")
    monkeypatch.setattr(device.AdbConnection, "list_devices", lambda: ["board", "phone"])
    instance = object.__new__(adapter.HexagonKernelAdapter)
    instance._conn = None
    assert instance._connection().serial == "phone"


def test_linux_workdir_failure_is_reported_even_with_ssh(monkeypatch):
    monkeypatch.setenv("TILELANG_HEXAGON_SSH_HOST", "root@board")
    conn = linux_runtime.LinuxConnection(serial="board")
    marker = f"__TILELANG_REMOTE_EXIT_{os.getpid()}__="
    monkeypatch.setattr(conn, "_run", lambda *args, **kwargs: (0, f"denied\n{marker}1\n"))
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
