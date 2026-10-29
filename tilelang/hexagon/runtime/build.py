"""Build orchestration for Hexagon FastRPC projects.

This drives the Hexagon SDK's ``build_cmake`` over a generated project
(``fastrpc.write_project``), producing the FastRPC artifacts:

  * the **DSP skel** ``lib<name>_skel.so`` (Hexagon ELF, runs on the cDSP), and
  * the **HLOS host** binaries (aarch64 Android, open the FastRPC session).

Build products are cached by content under the TileLang cache directory: the
project files, the Hexagon template headers they include and the toolchain
identify a build, so an unchanged kernel skips both CMake builds.

Dependency-free (stdlib only).
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass

from .env import HexagonSDK


class HexagonBuildError(RuntimeError):
    pass


@dataclass
class FastRPCArtifacts:
    """Resolved paths to the products of a FastRPC build."""

    dsp_skel: str  # lib<name>_skel.so for the cDSP
    host_exe: str = ""  # a Linux BSP's one-shot host driver
    agent_exe: str = ""  # a session's Android agent


def _run(argv: list[str], cwd: str, env: dict[str, str], verbose: bool) -> None:
    if verbose:
        print(f"[hexagon-build] $ {' '.join(argv)}  (cwd={cwd})")
    proc = subprocess.run(
        argv,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
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


# What picks or reaches the device, which a build does not depend on.
_TRANSPORT = {
    "TILELANG_HEXAGON_DEVICE",
    "TILELANG_HEXAGON_DEVICE_LOCK",
    "TILELANG_HEXAGON_REMOTE_DIR",
    "TILELANG_HEXAGON_SSH_HOST",
    "TILELANG_HEXAGON_SSH_KNOWN_HOSTS",
}


def _cache_entry(project_dir: str, name: str, sdk: HexagonSDK, dsp_arch: str, hlos: str) -> str | None:
    """The build cache directory of this project, or None when caching is off."""
    from tilelang.env import TILELANG_TEMPLATE_PATH, env

    if not env.is_cache_enabled() or not TILELANG_TEMPLATE_PATH:
        return None
    digest = hashlib.sha256()

    def add(label: str, path: str) -> None:
        digest.update(label.encode() + b"\0")
        with open(path, "rb") as f:
            digest.update(f.read())
        digest.update(b"\0")

    for fn in sorted(os.listdir(project_dir)):
        path = os.path.join(project_dir, fn)
        if os.path.isfile(path):
            add(fn, path)
    templates = os.path.join(TILELANG_TEMPLATE_PATH, "tl_templates", "hexagon")
    for root, dirs, files in os.walk(templates):
        dirs.sort()
        for fn in sorted(files):
            path = os.path.join(root, fn)
            add(os.path.relpath(path, templates), path)
    toolchain = {
        "name": name,
        "dsp_arch": dsp_arch,
        "hlos": hlos,
        "sdk_root": sdk.sdk_root,
        "ndk_root": sdk.ndk_root,
        "cmake_root": sdk.cmake_root,
        # Build switches and the Linux HLOS toolchain (linux_build.py), not
        # the device a build runs on. Like the paths above, they identify
        # tools; replacing a tool in place needs a cleared cache.
        "env": {k: v for k, v in os.environ.items() if k.startswith("TILELANG_HEXAGON_") and k not in _TRANSPORT},
    }
    digest.update(json.dumps(toolchain, sort_keys=True).encode())
    return os.path.join(env.TILELANG_CACHE_DIR, "hexagon_builds", digest.hexdigest())


def _load_cached(entry: str) -> FastRPCArtifacts | None:
    try:
        with open(os.path.join(entry, "artifacts.json")) as f:
            names = json.load(f)
    except (OSError, ValueError):
        return None
    paths = {key: os.path.join(entry, fn) if fn else "" for key, fn in names.items()}
    if not all(os.path.isfile(p) for p in paths.values() if p):
        return None
    return FastRPCArtifacts(**paths)


def _store_cached(entry: str, artifacts: FastRPCArtifacts) -> FastRPCArtifacts:
    """Copies the build products into the cache; another process may race us."""
    parent = os.path.dirname(entry)
    os.makedirs(parent, exist_ok=True)
    staging = tempfile.mkdtemp(prefix=".staging-", dir=parent)
    try:
        names = {}
        for key, path in asdict(artifacts).items():
            names[key] = os.path.basename(path) if path else ""
            if path:
                shutil.copy2(path, os.path.join(staging, names[key]))
        with open(os.path.join(staging, "artifacts.json"), "w") as f:
            json.dump(names, f)
        os.rename(staging, entry)
    except OSError:
        # The entry exists (a concurrent build stored it first) or the cache
        # is not writable; the fresh products serve this build either way.
        shutil.rmtree(staging, ignore_errors=True)
        return artifacts
    return _load_cached(entry) or artifacts


def build_cmake_fastrpc(
    project_dir: str,
    name: str,
    sdk: HexagonSDK | None = None,
    dsp_arch: str | None = None,
    verbose: bool = False,
) -> FastRPCArtifacts:
    """Build a FastRPC project: its DSP skel, and its Android agent (a
    session's) or, on a Linux BSP, its one-shot host driver. Returns the
    products' paths.

    A build whose inputs match a cached one returns the cached products.

    Parameters
    ----------
    project_dir : directory containing ``CMakeLists.txt`` and the ``.idl``.
    name        : the interface/library base name (skel is ``lib<name>_skel.so``).
    sdk         : a :class:`HexagonSDK` (constructed+validated if omitted).
    dsp_arch    : Hexagon arch (e.g. ``v73``); defaults to ``sdk.dsp_arch``.
    """
    sdk = (sdk or HexagonSDK()).validate()
    dsp_arch = dsp_arch or sdk.dsp_arch
    hlos = os.environ.get("TILELANG_HEXAGON_HLOS", "android")
    if hlos not in {"android", "linux"}:
        raise HexagonBuildError(f"Unsupported HLOS: {hlos}")
    entry = _cache_entry(project_dir, name, sdk, dsp_arch, hlos)
    if entry:
        cached = _load_cached(entry)
        if cached:
            if verbose:
                print(f"[hexagon-build] cached: {entry}")
            return cached
    artifacts = _build(project_dir, name, sdk, dsp_arch, hlos, verbose)
    return _store_cached(entry, artifacts) if entry else artifacts


def _build(project_dir: str, name: str, sdk: HexagonSDK, dsp_arch: str, hlos: str, verbose: bool) -> FastRPCArtifacts:
    if hlos == "linux":
        from .linux_runtime import build_linux_fastrpc

        return build_linux_fastrpc(project_dir, name, sdk, dsp_arch, verbose)
    env = sdk.environ
    build_cmake = sdk.build_cmake

    # Never inherit a Linux BSP compatibility setting from a reused CMake tree.
    _run([build_cmake, "hexagon", f"DSP_ARCH={dsp_arch}", "TL_HEXAGON_LEGACY_COMPUTE_RES=OFF"], cwd=project_dir, env=env, verbose=verbose)
    _run([build_cmake, "android"], cwd=project_dir, env=env, verbose=verbose)

    skel = _find_one(
        [
            os.path.join(project_dir, f"hexagon_*_{dsp_arch}", "**", f"lib{name}_skel.so"),
            os.path.join(project_dir, f"hexagon_*_{dsp_arch}", f"lib{name}_skel.so"),
        ]
    )
    agent = _find_one(
        [
            os.path.join(project_dir, "android_*_aarch64", "**", f"{name}_agent"),
            os.path.join(project_dir, "android_*_aarch64", f"{name}_agent"),
        ]
    )
    if not skel:
        raise HexagonBuildError(f"DSP skel lib{name}_skel.so not found after build in {project_dir}")
    if not agent:
        raise HexagonBuildError(f"agent {name}_agent not found after build in {project_dir}")
    return FastRPCArtifacts(dsp_skel=skel, agent_exe=agent)
