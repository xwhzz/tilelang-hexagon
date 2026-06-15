"""Build orchestration for Hexagon FastRPC projects.

For M0 this drives the Hexagon SDK's ``build_cmake`` over a project that already
has a ``CMakeLists.txt`` (e.g. ``mini-htp``), producing the two FastRPC
artifacts:

  * the **DSP skel** ``lib<name>_skel.so`` (Hexagon ELF, runs on the cDSP), and
  * the **HLOS host** binary (aarch64 Android, opens the FastRPC session).

For generated kernels (M1+) a separate code path will invoke ``qaic`` +
``hexagon-clang`` directly instead of cmake; this module is the orchestration
seam where that will plug in.

Dependency-free (stdlib only).
"""

from __future__ import annotations

import glob
import os
import subprocess
from dataclasses import dataclass

from .env import HexagonSDK


class HexagonBuildError(RuntimeError):
    pass


@dataclass
class FastRPCArtifacts:
    """Resolved paths to the products of a FastRPC build."""

    dsp_skel: str  # lib<name>_skel.so for the cDSP
    host_exe: str  # aarch64 Android one-shot host driver
    agent_exe: str = ""  # aarch64 persistent socket agent


def _run(argv: list[str], cwd: str, env: dict[str, str], verbose: bool) -> None:
    if verbose:
        print(f"[hexagon-build] $ {' '.join(argv)}  (cwd={cwd})")
    proc = subprocess.run(
        argv, cwd=cwd, env=env, capture_output=True, text=True, stdin=subprocess.DEVNULL,
    )
    if verbose or proc.returncode != 0:
        print(proc.stdout[-4000:])
        print(proc.stderr[-4000:])
    if proc.returncode != 0:
        raise HexagonBuildError(f"command failed ({proc.returncode}): {' '.join(argv)}")


def _find_one(patterns: list[str]) -> str | None:
    for pat in patterns:
        hits = glob.glob(pat, recursive=True)
        if hits:
            # prefer the shallowest match (top-level copy_binaries output)
            return min(hits, key=lambda p: (p.count(os.sep), len(p)))
    return None


def build_cmake_fastrpc(
    project_dir: str,
    name: str,
    sdk: HexagonSDK | None = None,
    dsp_arch: str | None = None,
    build_host: bool = True,
    verbose: bool = False,
) -> FastRPCArtifacts:
    """Build a CMake-based FastRPC project for both the DSP and (optionally) the
    aarch64 host, returning the resolved artifact paths.

    Parameters
    ----------
    project_dir : directory containing ``CMakeLists.txt`` and the ``.idl``.
    name        : the interface/library base name (skel is ``lib<name>_skel.so``).
    sdk         : a :class:`HexagonSDK` (constructed+validated if omitted).
    dsp_arch    : Hexagon arch (e.g. ``v73``); defaults to ``sdk.dsp_arch``.
    build_host  : also build the HLOS host binary (needs the Android NDK).
    """
    sdk = (sdk or HexagonSDK()).validate()
    dsp_arch = dsp_arch or sdk.dsp_arch
    env = sdk.environ
    build_cmake = sdk.build_cmake

    _run([build_cmake, "hexagon", f"DSP_ARCH={dsp_arch}"], cwd=project_dir, env=env, verbose=verbose)
    if build_host:
        _run([build_cmake, "android"], cwd=project_dir, env=env, verbose=verbose)

    skel = _find_one([
        os.path.join(project_dir, f"hexagon_*_{dsp_arch}", "**", f"lib{name}_skel.so"),
        os.path.join(project_dir, f"hexagon_*_{dsp_arch}", f"lib{name}_skel.so"),
    ])
    host = _find_one([
        os.path.join(project_dir, "android_*_aarch64", "**", f"{name}_test"),
        os.path.join(project_dir, "android_*_aarch64", f"{name}_test"),
    ]) if build_host else ""
    agent = _find_one([
        os.path.join(project_dir, "android_*_aarch64", "**", f"{name}_agent"),
        os.path.join(project_dir, "android_*_aarch64", f"{name}_agent"),
    ]) if build_host else ""

    if not skel:
        raise HexagonBuildError(f"DSP skel lib{name}_skel.so not found after build in {project_dir}")
    if build_host and not host:
        raise HexagonBuildError(f"host binary {name}_test not found after build in {project_dir}")

    return FastRPCArtifacts(dsp_skel=skel, host_exe=host or "", agent_exe=agent or "")
