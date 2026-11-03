"""Device selection, the run's settings and cache identity for live Hexagon
deployments: the one place the runtime reads its environment's settings."""

from __future__ import annotations

import math
import os
import shutil

from .device import AdbConnection, HexagonConnectionError
from .env import HexagonSDK


def hlos() -> str:
    """The OS the device runs (TILELANG_HEXAGON_HLOS): "android", the default,
    or "linux", a BSP. Either HLOS supports resident sessions."""
    name = os.environ.get("TILELANG_HEXAGON_HLOS", "android")
    if name not in ("android", "linux"):
        raise ValueError(f"TILELANG_HEXAGON_HLOS is android or linux, not {name!r}")
    return name


def remote_dir(os_name: str) -> str:
    """The directory under which runs on a device of `os_name` (see hlos) put
    their files (TILELANG_HEXAGON_REMOTE_DIR)."""
    default = "/tmp/tilelang_hexagon" if os_name == "linux" else "/data/local/tmp/tilelang_hexagon"
    return os.environ.get("TILELANG_HEXAGON_REMOTE_DIR", default)


def ssh_host() -> str | None:
    """An explicit SSH destination selects SSH exclusively, never ADB fallback."""
    host = os.environ.get("TILELANG_HEXAGON_SSH_HOST")
    if host is not None and (not host or host.startswith("-") or any(c.isspace() for c in host)):
        raise ValueError("TILELANG_HEXAGON_SSH_HOST must be a nonempty SSH host or user@host")
    return host


def ssh_transfer_timeout(value: float | None = None) -> float:
    """Seconds per SSH file operation; raise the 600s default for large files."""
    message = "TILELANG_HEXAGON_SSH_TRANSFER_TIMEOUT / transfer_timeout must be finite and positive"
    try:
        timeout = float(os.environ.get("TILELANG_HEXAGON_SSH_TRANSFER_TIMEOUT", "600") if value is None else value)
    except (ValueError, TypeError):
        raise ValueError(message) from None
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError(message)
    return timeout


def device_serial() -> str | None:
    if hlos() == "linux" and ssh_host() is not None:
        return None
    serial = os.environ.get("TILELANG_HEXAGON_DEVICE")
    if serial:
        return serial
    serials = AdbConnection.list_devices()
    if not serials:
        raise HexagonConnectionError("no authorized adb device found (check `adb devices`)")
    return serials[0]


def domain_id() -> int:
    """FastRPC domain, independent of the DSP ISA and transport (default cDSP 3)."""
    domain = int(os.environ.get("TILELANG_HEXAGON_DOMAIN", "3"))
    if not 0 <= domain < 1 << 31:
        raise ValueError("TILELANG_HEXAGON_DOMAIN must be a nonnegative int32")
    return domain


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
        "dsp_library_path": os.environ.get("TILELANG_HEXAGON_DSP_LIBRARY_PATH"),
    }
    host = ssh_host() if identity["hlos"] == "linux" else None
    if host is not None:
        identity["transport"] = {
            "kind": "ssh",
            "host": host,
            "ssh": shutil.which("ssh"),
            "known_hosts": os.environ.get("TILELANG_HEXAGON_SSH_KNOWN_HOSTS"),
            "transfer_timeout": ssh_transfer_timeout(),
        }
    else:
        identity["transport"] = {
            "kind": "adb",
            "device": device_serial(),
            "adb": shutil.which("adb"),
            "adb_server": {
                name: os.environ.get(name)
                for name in (
                    "ADB_SERVER_SOCKET",
                    "ANDROID_ADB_SERVER_ADDRESS",
                    "ANDROID_ADB_SERVER_PORT",
                )
            },
        }
    if identity["hlos"] == "linux":
        identity["linux"] = {
            name: os.environ.get(name)
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
