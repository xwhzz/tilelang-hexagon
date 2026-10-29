"""A cached Hexagon kernel owns a live device deployment, not just code."""

import importlib
import os
from types import SimpleNamespace

import pytest

from tilelang.hexagon.runtime import device
from tilelang.hexagon.runtime.kernel_cache import HexagonKernelCache


@pytest.fixture
def cache(monkeypatch):
    for name in list(os.environ):
        if name.startswith("TILELANG_HEXAGON_") or name == "HEXAGON_DSP_ARCH":
            monkeypatch.delenv(name)
    monkeypatch.setenv("TILELANG_HEXAGON_HLOS", "linux")
    monkeypatch.setenv("HEXAGON_DSP_ARCH", "v73")
    monkeypatch.setenv("TILELANG_HEXAGON_DEVICE", "board")
    instance = HexagonKernelCache()
    monkeypatch.setattr(instance, "_memory_cache", {})
    monkeypatch.setattr(instance, "_get_base_key", lambda: {"version": "test"})
    return instance


def key(cache):
    func = SimpleNamespace(script=lambda **kwargs: "same PrimFunc")
    return cache._generate_key(func, [2], "hexagon", args=(), target="hexagon")


@pytest.mark.parametrize(
    "name,value",
    [
        ("TILELANG_HEXAGON_HLOS", "android"),
        ("HEXAGON_DSP_ARCH", "v79"),
        ("TILELANG_HEXAGON_DEVICE", "phone"),
        ("TILELANG_HEXAGON_SSH_HOST", "root@other"),
        ("TILELANG_HEXAGON_SSH_KNOWN_HOSTS", "/other/known_hosts"),
        ("TILELANG_HEXAGON_REMOTE_DIR", "/other/remote"),
        ("TILELANG_HEXAGON_LEGACY_COMPUTE_RES", "1"),
        ("TILELANG_HEXAGON_ELF_CRC", "1"),
        ("TILELANG_HEXAGON_LINUX_RPC_LIB_DIR", "/other/bsp"),
        ("TILELANG_HEXAGON_LINUX_CC", "/other/gcc"),
        ("HEXAGON_SDK_ROOT", "/other/sdk"),
        ("ANDROID_NDK_ROOT", "/other/ndk"),
        ("CMAKE_ROOT_PATH", "/other/cmake"),
        ("ADB_SERVER_SOCKET", "tcp:localhost:9000"),
    ],
)
def test_configuration_changes_invalidate_key(cache, monkeypatch, name, value):
    original = key(cache)
    assert key(cache) == original
    monkeypatch.setenv(name, value)
    assert key(cache) != original


def test_automatic_device_selection_is_part_of_key(cache, monkeypatch):
    monkeypatch.delenv("TILELANG_HEXAGON_DEVICE")
    monkeypatch.setattr(device.AdbConnection, "list_devices", lambda: ["first"])
    original = key(cache)
    monkeypatch.setattr(device.AdbConnection, "list_devices", lambda: ["second"])
    assert key(cache) != original


def test_cached_does_not_return_linux_adapter_after_switch_to_android(cache, monkeypatch):
    module = importlib.import_module("tilelang.cache.kernel_cache")
    monkeypatch.setattr(module.env, "is_cache_enabled", lambda: True)
    monkeypatch.setattr(module, "JITKernel", lambda *a, **kw: SimpleNamespace(adapter=None))
    monkeypatch.setattr(module, "get_prim_func_name", lambda *a: "test")
    func = SimpleNamespace(script=lambda **kwargs: "same PrimFunc")

    def compile_kernel():
        return cache.cached(func, [2], target="hexagon", execution_backend="hexagon", verbose=False)

    linux = compile_kernel()
    assert compile_kernel() is linux
    monkeypatch.setenv("TILELANG_HEXAGON_HLOS", "android")
    monkeypatch.setenv("HEXAGON_DSP_ARCH", "v79")
    monkeypatch.setenv("TILELANG_HEXAGON_DEVICE", "phone")
    android = compile_kernel()
    assert android is not linux
    assert compile_kernel() is android
    monkeypatch.setenv("TILELANG_HEXAGON_HLOS", "linux")
    monkeypatch.setenv("HEXAGON_DSP_ARCH", "v73")
    monkeypatch.setenv("TILELANG_HEXAGON_DEVICE", "board")
    assert compile_kernel() is linux


# ---------------------------------------------------------------------------
# Build products persist across processes, keyed by their inputs.
# ---------------------------------------------------------------------------


@pytest.fixture
def builds(monkeypatch, tmp_path):
    """build_cmake_fastrpc over a fake build that counts its runs."""
    from tilelang.hexagon.runtime import build

    tl_env = importlib.import_module("tilelang.env")

    templates = tmp_path / "templates"
    (templates / "tl_templates" / "hexagon").mkdir(parents=True)
    (templates / "tl_templates" / "hexagon" / "common.h").write_text("// v1\n")
    monkeypatch.setattr(tl_env, "TILELANG_TEMPLATE_PATH", str(templates))
    monkeypatch.setattr(tl_env.env, "TILELANG_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(tl_env.env, "is_cache_enabled", lambda: True)
    runs = []

    def fake_build(project_dir, name, sdk, dsp_arch, hlos, verbose):
        runs.append(project_dir)
        paths = {}
        for field, fn in (("dsp_skel", f"lib{name}_skel.so"), ("host_exe", f"{name}_test"), ("agent_exe", f"{name}_agent")):
            paths[field] = os.path.join(project_dir, fn)
            with open(paths[field], "w") as f:
                f.write(f"{field} of {project_dir}")
        return build.FastRPCArtifacts(**paths)

    monkeypatch.setattr(build, "_build", fake_build)
    sdk = SimpleNamespace(sdk_root="/sdk", ndk_root="/ndk", cmake_root="/cmake", dsp_arch="v79", validate=lambda: sdk)
    counter = iter(range(1000))

    def run(source="kernel v1"):
        project = tmp_path / f"project{next(counter)}"
        project.mkdir()
        (project / "tl_k_dsp.cc").write_text(source)
        return build.build_cmake_fastrpc(str(project), "tl_k", sdk=sdk)

    return SimpleNamespace(run=run, runs=runs, templates=templates, tl_env=tl_env)


def test_an_unchanged_project_reuses_its_build(builds):
    first = builds.run()
    second = builds.run()
    assert len(builds.runs) == 1
    assert second == first
    assert "hexagon_builds" in first.dsp_skel
    with open(second.agent_exe) as f:
        assert f.read() == f"agent_exe of {builds.runs[0]}"


def test_sources_and_headers_are_part_of_the_key(builds):
    builds.run()
    builds.run("kernel v2")
    assert len(builds.runs) == 2
    (builds.templates / "tl_templates" / "hexagon" / "common.h").write_text("// v2\n")
    builds.run()
    assert len(builds.runs) == 3


def test_the_toolchain_is_part_of_the_key(builds, monkeypatch):
    builds.run()
    monkeypatch.setenv("TILELANG_HEXAGON_LINUX_CC", "/other/gcc")
    builds.run()
    assert len(builds.runs) == 2
    # The device lock does not change what a build produces.
    monkeypatch.setenv("TILELANG_HEXAGON_DEVICE_LOCK", "/tmp/lock")
    builds.run()
    assert len(builds.runs) == 2


def test_a_disabled_cache_always_builds(builds, monkeypatch):
    monkeypatch.setattr(builds.tl_env.env, "is_cache_enabled", lambda: False)
    builds.run()
    builds.run()
    assert len(builds.runs) == 2


# ---------------------------------------------------------------------------
# The device lock: one session at a time across processes.
# ---------------------------------------------------------------------------


def _locked_elsewhere(path):
    """Whether another process finds the lock file taken."""
    import subprocess
    import sys

    probe = "import fcntl, os, sys; fd = os.open(sys.argv[1], os.O_RDWR)\ntry:\n    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)\nexcept BlockingIOError:\n    sys.exit(1)\n"
    return subprocess.run([sys.executable, "-c", probe, path]).returncode == 1


def test_the_device_lock_is_off_without_its_variable(monkeypatch):
    monkeypatch.delenv("TILELANG_HEXAGON_DEVICE_LOCK", raising=False)
    assert device.DeviceLock().acquire() is False


def test_device_lock_holds_nest_within_a_process(monkeypatch, tmp_path):
    path = str(tmp_path / "device.lock")
    monkeypatch.setenv("TILELANG_HEXAGON_DEVICE_LOCK", path)
    lock = device.DeviceLock()
    assert lock.acquire() and lock.acquire()
    assert _locked_elsewhere(path)
    lock.release()
    assert _locked_elsewhere(path)
    lock.release()
    assert not _locked_elsewhere(path)


def _fake_kernel(monkeypatch):
    """A compiled kernel whose build and device are fakes: its session starts
    and runs without a phone."""
    from tilelang.hexagon.runtime import adapter, build, session

    def compile_project(project, iface, **_):
        skel, agent = os.path.join(project, f"lib{iface}_skel.so"), os.path.join(project, f"{iface}_agent")
        for path in (skel, agent):
            with open(path, "w") as f:
                f.write("built")
        return build.FastRPCArtifacts(skel, agent_exe=agent)

    starts = []
    monkeypatch.setenv("TILELANG_HEXAGON_DEVICE", "phone")
    monkeypatch.setattr(build, "build_cmake_fastrpc", compile_project)
    monkeypatch.setattr(session.Session, "_start", lambda self, timeout: starts.append(len(starts)))
    monkeypatch.setattr(session.Session, "_run", lambda self, kernel, given: None)
    sdk = SimpleNamespace(sdk_root="/sdk", ndk_root="/ndk", cmake_root="/cmake", dsp_arch="v79", validate=lambda: sdk)
    return adapter.HexagonKernelAdapter([], [], None, "int32_t k() { return 0; }", sdk=sdk), starts


def test_closed_kernels_reopen_and_hold_the_lock_only_while_open(monkeypatch, tmp_path):
    from tilelang.hexagon.runtime import adapter

    lock = str(tmp_path / "device.lock")
    monkeypatch.setenv("TILELANG_HEXAGON_DEVICE_LOCK", lock)
    kernel, starts = _fake_kernel(monkeypatch)
    assert len(starts) == 1
    assert _locked_elsewhere(lock)
    adapter.HexagonKernelAdapter.close_all()
    assert not _locked_elsewhere(lock)
    # A call after close() starts another session, which takes the lock again.
    kernel.func()
    assert len(starts) == 2
    assert _locked_elsewhere(lock)
    kernel.close()
    assert not _locked_elsewhere(lock)


def test_kernels_remove_the_work_directories_they_make(monkeypatch, tmp_path):
    import gc
    import tempfile

    monkeypatch.setenv("TILELANG_HEXAGON_DEVICE_LOCK", str(tmp_path / "device.lock"))
    temp = tmp_path / "temp"
    temp.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(temp))
    kernel, _ = _fake_kernel(monkeypatch)
    assert [d.name.startswith("tl_hexagon_session_") for d in temp.iterdir()] == [True]
    kernel.close()
    del kernel
    gc.collect()
    assert list(temp.iterdir()) == []
