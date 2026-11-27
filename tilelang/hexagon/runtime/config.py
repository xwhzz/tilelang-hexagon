"""Device selection, the run's settings and cache identity for live Hexagon
deployments: the one place the runtime reads its environment's settings."""

from __future__ import annotations

import json
import math
import os
import shutil
import sys
from functools import lru_cache, partial
from types import MappingProxyType

from .device import AdbConnection, HexagonConnectionError
from .env import HexagonSDK, current_environment, use_environment


def hlos() -> str:
    """The OS the device runs (TILELANG_HEXAGON_HLOS): "android", the default,
    or "linux", a BSP. Either HLOS supports resident sessions."""
    name = current_environment().get("TILELANG_HEXAGON_HLOS", "android")
    if name not in ("android", "linux"):
        raise ValueError(f"TILELANG_HEXAGON_HLOS is android or linux, not {name!r}")
    return name


def remote_dir(os_name: str) -> str:
    """The directory under which runs on a device of `os_name` (see hlos) put
    their files (TILELANG_HEXAGON_REMOTE_DIR)."""
    default = "/tmp/tilelang_hexagon" if os_name == "linux" else "/data/local/tmp/tilelang_hexagon"
    return current_environment().get("TILELANG_HEXAGON_REMOTE_DIR", default)


def ssh_host() -> str | None:
    """An explicit SSH destination selects SSH exclusively, never ADB fallback."""
    host = current_environment().get("TILELANG_HEXAGON_SSH_HOST") or None
    if host is not None and (host.startswith("-") or any(c.isspace() for c in host)):
        raise ValueError("TILELANG_HEXAGON_SSH_HOST must be a nonempty SSH host or user@host")
    return host


def ssh_transfer_timeout(value: float | None = None) -> float:
    """Seconds per SSH file operation; raise the 600s default for large files."""
    message = "TILELANG_HEXAGON_SSH_TRANSFER_TIMEOUT / transfer_timeout must be finite and positive"
    try:
        timeout = float(current_environment().get("TILELANG_HEXAGON_SSH_TRANSFER_TIMEOUT", "600") if value is None else value)
    except (ValueError, TypeError):
        raise ValueError(message) from None
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError(message)
    return timeout


def device_serial() -> str | None:
    if hlos() == "linux" and ssh_host() is not None:
        return None
    serial = current_environment().get("TILELANG_HEXAGON_DEVICE")
    if serial:
        return serial
    serials = AdbConnection.list_devices()
    if not serials:
        raise HexagonConnectionError("no authorized adb device found (check `adb devices`)")
    return serials[0]


def check_domain(domain, name: str = "domain") -> int:
    """A FastRPC domain: a nonnegative int32 (3 is cDSP0, 4 cDSP1 where a SoC has it)."""
    if isinstance(domain, bool) or not isinstance(domain, int) or not 0 <= domain < 1 << 31:
        raise ValueError(f"{name} must be a nonnegative int32, not {domain!r}")
    return domain


def domain_id() -> int:
    """FastRPC domain, independent of the DSP ISA and transport (default cDSP 3)."""
    return check_domain(int(current_environment().get("TILELANG_HEXAGON_DOMAIN", "3")), "TILELANG_HEXAGON_DOMAIN")


def cache_identity() -> dict:
    """Resolve before cache lookup, including the automatically selected device.

    A cache entry owns a deployment/session, not just generated machine code.
    Configuration changes must therefore create a new adapter. Toolchain paths
    identify installations; replacing tools in place requires clearing the cache.
    """
    sdk = HexagonSDK()
    identity = {
        "hlos": hlos(),
        "domain": domain_id(),
        "dsp_arch": sdk.dsp_arch,
        "sdk_root": sdk.sdk_root,
        "ndk_root": sdk.ndk_root,
        "cmake_root": sdk.cmake_root,
        "remote_dir": remote_dir(hlos()),
        "dsp_library_path": current_environment().get("TILELANG_HEXAGON_DSP_LIBRARY_PATH"),
    }
    host = ssh_host() if identity["hlos"] == "linux" else None
    if host is not None:
        identity["transport"] = {
            "kind": "ssh",
            "host": host,
            "ssh": shutil.which("ssh", path=current_environment().get("PATH", os.defpath)),
            "known_hosts": current_environment().get("TILELANG_HEXAGON_SSH_KNOWN_HOSTS"),
            "transfer_timeout": ssh_transfer_timeout(),
        }
    else:
        identity["transport"] = {
            "kind": "adb",
            "device": device_serial(),
            "adb": shutil.which("adb", path=current_environment().get("PATH", os.defpath)),
            "adb_server": {
                name: current_environment().get(name)
                for name in (
                    "ADB_SERVER_SOCKET",
                    "ANDROID_ADB_SERVER_ADDRESS",
                    "ANDROID_ADB_SERVER_PORT",
                )
            },
        }
    if identity["hlos"] == "linux":
        identity["linux"] = {
            name: current_environment().get(name)
            for name in (
                "TILELANG_HEXAGON_REMOTE_DIR",
                "TILELANG_HEXAGON_LINUX_RPC_LIB_DIR",
                "TILELANG_HEXAGON_LINUX_CC",
                "TILELANG_HEXAGON_ELF_CRC",
                "TILELANG_HEXAGON_LEGACY_COMPUTE_RES",
                "PATH",
            )
        }
    return identity


# Cache the normalization of environment settings. Path searches, toolchain
# discovery and JSON construction are independent of the kernel's arguments.
_SETTINGS = (
    "TILELANG_HEXAGON_DEVICE",
    "TILELANG_HEXAGON_HLOS",
    "TILELANG_HEXAGON_SSH_HOST",
    "TILELANG_HEXAGON_DOMAIN",
    "HEXAGON_DSP_ARCH",
    "HEXAGON_SDK_ROOT",
    "ANDROID_NDK_ROOT",
    "CMAKE_ROOT_PATH",
    "TILELANG_HEXAGON_REMOTE_DIR",
    "TILELANG_HEXAGON_DSP_LIBRARY_PATH",
    "PATH",
    "TILELANG_HEXAGON_LINUX_RPC_LIB_DIR",
    "TILELANG_HEXAGON_LINUX_CC",
    "TILELANG_HEXAGON_ELF_CRC",
    "TILELANG_HEXAGON_LEGACY_COMPUTE_RES",
    "TILELANG_HEXAGON_SSH_KNOWN_HOSTS",
    "TILELANG_HEXAGON_SSH_TRANSFER_TIMEOUT",
    "ADB_SERVER_SOCKET",
    "ANDROID_ADB_SERVER_ADDRESS",
    "ANDROID_ADB_SERVER_PORT",
)
_BYTE_SETTINGS = tuple(map(os.fsencode, _SETTINGS))
_posix_environ = getattr(os.environ, "_data", None) if sys.implementation.name == "cpython" and os.supports_bytes_environ else None


@lru_cache(maxsize=32)
def _normalized_deployment(values, selected):
    env = {k: os.fsdecode(v) for k, v in zip(_SETTINGS, values) if v is not None}
    if selected is not None:
        env["TILELANG_HEXAGON_DEVICE"] = selected
    with use_environment(env):
        key = json.dumps(cache_identity(), sort_keys=True)
    return key, partial(_capture_scope, env)


def current_deployment():
    env = current_environment()
    # CPython's POSIX environment stores bytes. Reading the same mapping
    # avoids twenty key encodings/decodings on each decorated call. The
    # public mapping is sufficient for snapshots and other implementations.
    if env is os.environ and _posix_environ is not None and getattr(env, "_data", None) is _posix_environ:
        values = tuple(map(_posix_environ.get, _BYTE_SETTINGS))
    else:
        values = tuple(map(env.get, _SETTINGS))
    device, os_name, host = values[:3]
    selected = None
    if not device and not (os_name in (b"linux", "linux") and host):
        # Device enumeration must use the ADB server and PATH from the same
        # selection as the kernel. Its answer may change on every call.
        settings = {k: os.fsdecode(v) for k, v in zip(_SETTINGS, values) if v is not None}
        scope = _capture_scope(settings)
        with scope():
            selected = device_serial()
    return _normalized_deployment(values, selected)


def _capture_scope(settings):
    # Supply the full build/subprocess environment, overriding only the
    # settings that selected this particular deployment, including absence.
    current = current_environment()
    env = {name: value for name in current if (value := current.get(name)) is not None}
    for name in _SETTINGS:
        if name in settings:
            env[name] = settings[name]
        else:
            env.pop(name, None)
    return partial(use_environment, MappingProxyType(env))


def deployment_cache_key() -> str:
    return current_deployment()[0]


def deployment_scope():
    current = current_environment()
    if current is not os.environ:
        return partial(use_environment, current)
    _, capture = current_deployment()
    return capture()
