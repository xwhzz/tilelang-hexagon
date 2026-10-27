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
