"""Device selection and cache identity for live Hexagon deployments."""

from __future__ import annotations

import os
import shutil

from .device import AdbConnection, HexagonConnectionError
from .env import HexagonSDK


def device_serial() -> str:
    serial = os.environ.get("TILELANG_HEXAGON_DEVICE")
    if serial:
        return serial
    serials = AdbConnection.list_devices()
    if not serials:
        raise HexagonConnectionError("no authorized adb device found (check `adb devices`)")
    return serials[0]


def cache_identity() -> dict:
    """Resolve before cache lookup, including the automatically selected device.

    A cache entry owns a deployment/session, not just generated machine code.
    Configuration changes must therefore create a new adapter. Toolchain paths
    identify installations; replacing tools in place requires clearing the cache.
    """
    sdk = HexagonSDK()
    hlos = os.environ.get("TILELANG_HEXAGON_HLOS", "android")
    identity = {
        "hlos": hlos,
        "dsp_arch": sdk.dsp_arch,
        "sdk_root": sdk.sdk_root,
        "ndk_root": sdk.ndk_root,
        "cmake_root": sdk.cmake_root,
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
    if hlos == "linux":
        identity["linux"] = {
            name: os.environ.get(name)
            for name in (
                "TILELANG_HEXAGON_REMOTE_DIR",
                "TILELANG_HEXAGON_SSH_HOST",
                "TILELANG_HEXAGON_SSH_KNOWN_HOSTS",
                "TILELANG_HEXAGON_LINUX_RPC_LIB_DIR",
                "TILELANG_HEXAGON_LINUX_CC",
                "TILELANG_HEXAGON_ELF_CRC",
                "TILELANG_HEXAGON_LEGACY_COMPUTE_RES",
                "PATH",
            )
        }
    return identity
