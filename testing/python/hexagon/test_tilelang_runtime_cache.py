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
        ("TILELANG_HEXAGON_DOMAIN", "4"),
        ("TILELANG_HEXAGON_DSP_LIBRARY_PATH", "/custom/dsp"),
        ("TILELANG_HEXAGON_SSH_HOST", "root@other"),
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
    headers = []
    monkeypatch.setattr(build, "_external_headers", lambda *_: list(headers))
    sdk = SimpleNamespace(sdk_root="/sdk", ndk_root="/ndk", cmake_root="/cmake", dsp_arch="v79", validate=lambda: sdk)
    counter = iter(range(1000))

    def run(source="kernel v1", files=None):
        project = tmp_path / f"project{next(counter)}"
        project.mkdir()
        (project / "tl_k_dsp.cc").write_text(source)
        for name, contents in (files or {}).items():
            path = project / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(contents)
        return build.build_cmake_fastrpc(str(project), "tl_k", sdk=sdk)

    return SimpleNamespace(run=run, runs=runs, templates=templates, tl_env=tl_env, headers=headers)


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


def test_external_and_relocated_project_headers_invalidate_only_their_build(builds, tmp_path):
    leaf = tmp_path / "user headers" / "weights.h"
    leaf.parent.mkdir()
    leaf.write_text("11")
    builds.headers.extend(["include/bias.h", str(leaf)])
    project_headers = {"include/bias.h": "weights.h"}
    first = builds.run(files=project_headers)
    assert builds.run(files=project_headers) == first
    leaf.write_text("77")
    second = builds.run(files=project_headers)
    assert second != first and builds.run(files=project_headers) == second
    assert builds.run(files={"include/bias.h": "other.h"}) != second
    assert len(builds.runs) == 3


def test_dsp_compiler_dependencies_include_transitive_and_project_relative_headers(tmp_path, monkeypatch):
    from tilelang.hexagon.runtime import build

    project = tmp_path / "project"
    tree = project / "hexagon_ReleaseG_toolv19_v79"
    tree.mkdir(parents=True)
    (tree / ".ninja_deps").touch()
    app = tmp_path / "app headers"
    sdk = SimpleNamespace(ninja="ninja", sdk_root=str(tmp_path / "sdk"))
    artifacts = build.FastRPCArtifacts(str(tree / "ship" / "libtl_k_skel.so"))
    inputs = [
        project / "tl_k_dsp.cc",
        tree / "tl_k.h",
        tmp_path / "sdk" / "stdio.h",
        project / "parts" / "local.h",
        app / "a.h",
        app / "b.h",
    ]
    output = "object: #deps 6\n" + "".join(f"    {path}\n" for path in inputs)
    monkeypatch.setattr(build.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout=output, returncode=0))
    assert build._external_headers(str(project), "tl_k", sdk, artifacts) == sorted([str(app / "a.h"), str(app / "b.h"), "parts/local.h"])
    assert build._external_headers(str(project), "other", sdk, artifacts) is None


def test_build_does_not_publish_under_a_retargeted_directory_symlink(tmp_path):
    from tilelang.hexagon.runtime import build

    for dirname, value in (("first", "11"), ("second", "77")):
        (tmp_path / dirname).mkdir()
        (tmp_path / dirname / "value.h").write_text(value)
    link = tmp_path / "active"
    link.symlink_to("first")
    (tmp_path / "includes").symlink_to("active")
    headers = ["includes/value.h"]
    first = build._header_variant("entry", tmp_path, headers)
    # Use filesystem ctimes so the check also exercises coarse timestamp resolution.
    link.unlink()
    link.symlink_to("second")
    assert build._header_variant("entry", tmp_path, headers, link.lstat().st_ctime_ns) is None
    second = build._header_variant("entry", tmp_path, headers)
    assert second != first


def test_the_toolchain_is_part_of_the_key(builds, monkeypatch):
    builds.run()
    monkeypatch.setenv("TILELANG_HEXAGON_LINUX_CC", "/other/gcc")
    builds.run()
    assert len(builds.runs) == 2
    # Neither the device lock nor runtime FastRPC domain changes a build.
    monkeypatch.setenv("TILELANG_HEXAGON_DOMAIN", "4")
    monkeypatch.setenv("TILELANG_HEXAGON_DEVICE_LOCK", "/tmp/lock")
    monkeypatch.setenv("TILELANG_HEXAGON_SSH_TRANSFER_TIMEOUT", "1800")
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
    monkeypatch.setattr(session.Session, "_run", lambda self, kernel, given, process: None)
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


def test_ssh_cache_identity_ignores_adb_and_tracks_host_keys(cache, monkeypatch):
    monkeypatch.setenv("TILELANG_HEXAGON_SSH_HOST", "root@board")
    monkeypatch.setattr(device.AdbConnection, "list_devices", lambda: pytest.fail("SSH must not discover ADB"))
    original = key(cache)
    monkeypatch.setenv("TILELANG_HEXAGON_DEVICE", "unrelated-phone")
    monkeypatch.setenv("ADB_SERVER_SOCKET", "tcp:localhost:9999")
    assert key(cache) == original
    monkeypatch.setenv("TILELANG_HEXAGON_SSH_KNOWN_HOSTS", "/different/known_hosts")
    assert key(cache) != original
    monkeypatch.setenv("TILELANG_HEXAGON_SSH_HOST", "")
    empty = key(cache)
    monkeypatch.delenv("TILELANG_HEXAGON_SSH_HOST")
    assert key(cache) == empty
    assert empty != original


def test_ssh_transfer_timeout_changes_only_live_cache_identity(cache, monkeypatch):
    monkeypatch.setenv("TILELANG_HEXAGON_SSH_HOST", "root@board")
    original = key(cache)
    monkeypatch.setenv("TILELANG_HEXAGON_SSH_TRANSFER_TIMEOUT", "600.0")
    assert key(cache) == original
    monkeypatch.setenv("TILELANG_HEXAGON_SSH_TRANSFER_TIMEOUT", "1800")
    assert key(cache) != original


@pytest.mark.parametrize("mode", ["lazy", "eager"])
def test_decorated_jit_uses_current_deployment(mode, monkeypatch):
    import torch
    import tilelang
    from tilelang.hexagon import language as T
    from tilelang.hexagon.runtime.config import hlos, remote_dir
    from tilelang.hexagon.runtime.env import HexagonSDK
    from tilelang.hexagon.runtime.session import Session

    opened = []

    def open_session(cls, *args, **kwargs):
        if not opened:
            # Another caller changes deployment after key capture but before
            # this Session reads its connection settings.
            monkeypatch.setenv("TILELANG_HEXAGON_REMOTE_DIR", "second")
        parent = remote_dir(hlos())
        opened.append(parent)
        return SimpleNamespace(kernels=[lambda x: parent], close=lambda: None)

    monkeypatch.setenv("TILELANG_HEXAGON_DEVICE", "phone")
    monkeypatch.setattr(HexagonSDK, "validate", lambda self: self)
    monkeypatch.setattr(Session, "of_lowered", classmethod(open_session))

    @tilelang.jit(target="hexagon", out_idx=1)
    def lazy(n):
        @T.prim_func
        def copy(x: T.Tensor((n,), "float32"), y: T.Tensor((n,), "float32")):
            with T.Kernel(1):
                for i in T.Parallel(n):
                    y[i] = x[i]

        return copy

    @tilelang.jit(target="hexagon")
    def eager(x):
        x: T.Tensor[[32], T.float32]
        y = T.empty((32,), T.float32)
        with T.Kernel(1):
            for i in T.Parallel(32):
                y[i] = x[i]
        return y

    x = torch.empty(32)
    monkeypatch.setenv("TILELANG_HEXAGON_REMOTE_DIR", "first")
    for parent in ("first", "second", "second", "first", "first"):
        if parent == "first" and len(opened) == 2:
            monkeypatch.setenv("TILELANG_HEXAGON_REMOTE_DIR", parent)
        result = lazy(32)(x) if mode == "lazy" else eager(x)
        assert result == parent
    assert opened == ["first", "second"]
    if mode == "lazy":
        first = lazy(32)
        first.adapter.close()
        monkeypatch.setenv("TILELANG_HEXAGON_REMOTE_DIR", "second")
        assert first(x) == "first"  # reopening the handle retains its identity
        assert lazy(32)(x) == "second"
        first.adapter.close()


@pytest.mark.parametrize("fast", [False, True])
def test_deployment_lookup_tracks_byte_and_string_environment(cache, monkeypatch, fast):
    from tilelang.hexagon.runtime import config

    if not os.supports_bytes_environ:
        pytest.skip("byte environment is unavailable")
    if not fast:
        monkeypatch.setattr(config, "_posix_environ", None)
    setting = "TILELANG_HEXAGON_REMOTE_DIR"
    monkeypatch.setenv(setting, "/first")
    first = config.deployment_cache_key()
    monkeypatch.setitem(os.environb, os.fsencode(setting), b"/second")
    assert config.deployment_cache_key() != first
    monkeypatch.setenv(setting, "/first")
    assert config.deployment_cache_key() == first


def test_decorated_isa_is_bound_from_frontend_to_session(cache, monkeypatch):
    import tilelang
    from tvm.target import Target
    from tilelang.hexagon import language as T
    from tilelang.hexagon.runtime.session import Session
    from tilelang.hexagon.target import hexagon_arch

    def make_factory(target):
        @tilelang.jit(target=target, out_idx=1)
        def factory(n):
            arch = int(hexagon_arch(Target.current())[1:])

            @T.prim_func
            def add(x: T.Tensor((n,), "int32"), y: T.Tensor((n,), "int32")):
                with T.Kernel(1):
                    for i in T.Parallel(n):
                        y[i] = x[i] + arch

            return add

        return factory

    monkeypatch.setattr(Session, "of_lowered", classmethod(lambda *a, **kw: SimpleNamespace(kernels=[lambda *_: None], close=lambda: None)))
    monkeypatch.setenv("HEXAGON_DSP_ARCH", "v79")
    original = HexagonKernelCache.cached

    def change_after_target_resolution(self, *args, **kwargs):
        monkeypatch.setenv("HEXAGON_DSP_ARCH", "v73")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(HexagonKernelCache, "cached", change_after_target_resolution)
    factory = make_factory("hexagon")
    first = factory(32)
    monkeypatch.setattr(HexagonKernelCache, "cached", original)
    second = factory(32)
    assert (first.adapter.dsp_arch, second.adapter.dsp_arch) == ("v79", "v73")
    assert second.get_kernel_source() == make_factory("hexagon")(32).get_kernel_source()
    assert first.get_kernel_source() != second.get_kernel_source()
    # An explicit target keeps its ISA when the default changes.
    explicit = make_factory({"kind": "hexagon", "mcpu": "hexagonv79"})(32)
    assert explicit.get_kernel_source() == first.get_kernel_source()
    monkeypatch.setenv("HEXAGON_DSP_ARCH", "v79")
    assert factory(32) is first


def test_implicit_target_can_enter_and_leave_live_deployments(cache, monkeypatch):
    import tilelang
    from tilelang.hexagon import language as T
    from tilelang.hexagon.runtime.session import Session

    monkeypatch.setattr(Session, "of_lowered", classmethod(lambda *a, **kw: SimpleNamespace(kernels=[lambda *_: None], close=lambda: None)))

    @tilelang.jit(out_idx=1)
    def factory(n):
        @T.prim_func
        def copy(x: T.Tensor((n,), "int32"), y: T.Tensor((n,), "int32")):
            with T.Kernel(1):
                for i in T.Parallel(n):
                    y[i] = x[i]

        return copy

    monkeypatch.setenv("TILELANG_DEFAULT_TARGET", "c")
    cpu = factory(32)
    assert cpu.target.kind.name == "c"
    monkeypatch.setenv("TILELANG_DEFAULT_TARGET", "hexagon")
    first = factory(32)
    assert first.target.kind.name == "hexagon"
    monkeypatch.setenv("TILELANG_HEXAGON_REMOTE_DIR", "/two")
    assert factory(32) is not first
    monkeypatch.setenv("TILELANG_DEFAULT_TARGET", "c")
    assert factory(32) is cpu
    assert factory(64).target.kind.name == "c"
