"""Hexagon SDK / toolchain discovery and environment setup.

This module is intentionally dependency-free (stdlib only) so it can be used by
the build/deploy harness without importing the (heavy, native) ``tilelang``
package.  It encapsulates the knowledge that would otherwise live in an ad-hoc
``setup_sdk_env.source`` shell incantation:

  * where the Hexagon SDK, its hexagon-clang toolchain, and ``qaic`` live;
  * where a usable ``cmake`` + ``ninja`` live (this dev host has no system cmake,
    so a portable one under ``~/.local/cmake`` is supported);
  * where the Android NDK lives (for the aarch64 HLOS / FastRPC stub side);
  * a fully-populated ``environ`` dict to hand to ``subprocess`` for builds.

The SDK ships ``setup_sdk_env.source`` which exports a large set of variables.
Rather than re-implement it, :meth:`HexagonSDK.environ` *sources it in a subshell
and snapshots the result*, then overlays our additions.  That keeps us correct
against SDK version churn.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

# Default locations on this host.  All overridable via environment variables or
# explicit constructor args so nothing is hard-wired for production use.
_DEFAULT_SDK_ROOT = "/home/xwh/Downloads/Hexagon_SDK_Linux/Hexagon_SDK/6.6.0.0"
_DEFAULT_NDK_ROOT = "/home/xwh/Downloads/android-ndk-r25c-linux/android-ndk-r25c"
_DEFAULT_CMAKE_ROOT = str(Path.home() / ".local" / "cmake")  # has bin/cmake + bin/ninja


class HexagonEnvError(RuntimeError):
    """Raised when a required SDK/toolchain component cannot be located."""


@dataclass
class HexagonSDK:
    """Locates the Hexagon SDK + companion toolchains and builds a build env.

    Parameters are resolved in priority order: explicit arg > environment
    variable > on-disk default.  Construction is cheap; validation is lazy
    (call :meth:`validate` to fail fast).
    """

    sdk_root: str = field(default_factory=lambda: os.environ.get("HEXAGON_SDK_ROOT", _DEFAULT_SDK_ROOT))
    ndk_root: str = field(default_factory=lambda: os.environ.get("ANDROID_NDK_ROOT", _DEFAULT_NDK_ROOT))
    cmake_root: str = field(default_factory=lambda: os.environ.get("CMAKE_ROOT_PATH", _DEFAULT_CMAKE_ROOT))
    dsp_arch: str = field(default_factory=lambda: os.environ.get("HEXAGON_DSP_ARCH", "v73"))

    # ---- core paths -------------------------------------------------------
    @property
    def setup_script(self) -> str:
        return os.path.join(self.sdk_root, "setup_sdk_env.source")

    @cached_property
    def tools_root(self) -> str:
        """``.../tools/HEXAGON_Tools/<ver>/Tools`` (holds bin/hexagon-clang)."""
        base = os.path.join(self.sdk_root, "tools", "HEXAGON_Tools")
        if not os.path.isdir(base):
            raise HexagonEnvError(f"HEXAGON_Tools not found under {base}")
        # pick the highest version directory present (numeric, not lexicographic:
        # "8.10.07" must outrank "8.7.06").
        versions = [d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))]
        if not versions:
            raise HexagonEnvError(f"no toolchain version under {base}")

        def _ver_key(v: str):
            try:
                return (0, tuple(int(x) for x in v.split(".")))
            except ValueError:
                return (1, v)  # non-numeric names sort after, lexicographically

        versions.sort(key=_ver_key)
        return os.path.join(base, versions[-1], "Tools")

    @property
    def hexagon_clang(self) -> str:
        return os.path.join(self.tools_root, "bin", "hexagon-clang")

    @cached_property
    def qaic(self) -> str:
        """The FastRPC IDL compiler.  The cmake ``build_idl`` macro expects it at
        ``ipc/fastrpc/qaic/bin/qaic``; the prebuilt ships at ``.../Ubuntu/qaic``."""
        cand = [
            os.path.join(self.sdk_root, "ipc", "fastrpc", "qaic", "bin", "qaic"),
            os.path.join(self.sdk_root, "ipc", "fastrpc", "qaic", "Ubuntu", "qaic"),
        ]
        for c in cand:
            if os.path.exists(c):
                return c
        raise HexagonEnvError(f"qaic not found (looked in {cand})")

    @property
    def cmake(self) -> str:
        return os.path.join(self.cmake_root, "bin", "cmake")

    @property
    def ninja(self) -> str:
        return os.path.join(self.cmake_root, "bin", "ninja")

    # ---- validation -------------------------------------------------------
    def validate(self) -> "HexagonSDK":
        """Raise :class:`HexagonEnvError` if anything required is missing."""
        checks = {
            "Hexagon SDK root": self.sdk_root,
            "setup_sdk_env.source": self.setup_script,
            "hexagon-clang": self.hexagon_clang,
            "qaic": self.qaic,
            "cmake": self.cmake,
            "ninja": self.ninja,
            "Android NDK root": self.ndk_root,
        }
        missing = [f"{name}: {path}" for name, path in checks.items() if not os.path.exists(path)]
        if missing:
            raise HexagonEnvError("Hexagon toolchain incomplete:\n  " + "\n  ".join(missing))
        return self

    # ---- build environment ------------------------------------------------
    def _ensure_qaic_bin_symlink(self) -> None:
        """The cmake ``build_idl`` macro invokes ``<qaic>/bin/qaic``; ensure it
        exists (symlink to the prebuilt) so SDK-cmake builds work."""
        bin_qaic = os.path.join(self.sdk_root, "ipc", "fastrpc", "qaic", "bin", "qaic")
        if not os.path.exists(bin_qaic):
            os.makedirs(os.path.dirname(bin_qaic), exist_ok=True)
            try:
                os.symlink(os.path.join("..", "Ubuntu", "qaic"), bin_qaic)
            except OSError:
                pass

    @cached_property
    def environ(self) -> dict[str, str]:
        """A full environment dict for build subprocesses.

        Sources ``setup_sdk_env.source`` in a subshell (after clearing the guard
        var it checks) and snapshots the result, then overlays cmake/ninja on
        PATH, ``CMAKE_ROOT_PATH``, and the Android NDK location.
        """
        self._ensure_qaic_bin_symlink()
        # Snapshot the SDK's own environment by sourcing its script.
        script = self.setup_script
        # ``unset HEXAGON_SDK_ROOT`` defeats the "already setup" early-return guard.
        cmd = f'unset HEXAGON_SDK_ROOT DEFAULT_HEXAGON_TOOLS_ROOT; source "{script}" >/dev/null 2>&1; env -0'
        try:
            raw = subprocess.run(
                ["bash", "-c", cmd], capture_output=True, text=True, timeout=120,
                stdin=subprocess.DEVNULL,
            ).stdout
            env = dict(
                kv.split("=", 1) for kv in raw.split("\0") if "=" in kv
            )
        except Exception:
            env = dict(os.environ)
        if not env.get("HEXAGON_SDK_ROOT"):
            env["HEXAGON_SDK_ROOT"] = self.sdk_root

        # Overlay our additions.
        env["CMAKE_ROOT_PATH"] = self.cmake_root
        env["ANDROID_ROOT_DIR"] = self.ndk_root
        env["ANDROID_NDK_ROOT"] = self.ndk_root
        cmake_bin = os.path.join(self.cmake_root, "bin")
        local_bin = str(Path.home() / ".local" / "bin")
        env["PATH"] = os.pathsep.join([cmake_bin, local_bin, env.get("PATH", os.environ.get("PATH", ""))])

        # Strip toolchain flags injected by a conda/host build env: they target
        # x86 (e.g. ``-march=nocona``, ``-isystem $CONDA_PREFIX/include``) and
        # break the Hexagon / Android cross-compilers, which set their own flags
        # via the SDK/NDK toolchain files.
        for var in (
            "CFLAGS", "CXXFLAGS", "CPPFLAGS", "LDFLAGS", "FFLAGS", "FCFLAGS",
            "DEBUG_CFLAGS", "DEBUG_CXXFLAGS", "DEBUG_CPPFLAGS", "DEBUG_FFLAGS",
            "DEBUG_FCFLAGS", "CPATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH",
            "OBJC_INCLUDE_PATH", "LIBRARY_PATH", "CMAKE_ARGS", "CMAKE_PREFIX_PATH",
            "CONDA_BUILD_SYSROOT", "CC", "CXX", "CPP", "LD", "AR", "AS", "NM",
            "RANLIB", "STRIP", "READELF", "OBJCOPY", "OBJDUMP", "ADDR2LINE",
        ):
            env.pop(var, None)
        return env

    @property
    def build_cmake(self) -> str:
        """Path to the SDK's ``build_cmake`` wrapper binary."""
        p = os.path.join(self.sdk_root, "build", "cmake", "Ubuntu", "build_cmake")
        if not os.path.exists(p):
            found = shutil.which("build_cmake", path=self.environ.get("PATH"))
            if found:
                return found
            raise HexagonEnvError(f"build_cmake not found at {p}")
        return p
