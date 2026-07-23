"""End-to-end harness smoke test: build + deploy + run the ``mini-htp`` HMX
self-test on a real device, entirely from Python.

This validates the layer-5 harness (:mod:`.env`, :mod:`.device`, :mod:`.build`)
against a known-good reference kernel before any tilelang codegen exists.  It is
the Python equivalent of ``mini-htp/go.sh all`` and returns the measured
``max_err`` (≈0 means the HMX matmul ran correctly on the cDSP).

Run directly::

    python -m hexagon.mini_htp           # (with tilelang/ on sys.path)
    python -m tilelang.hexagon.mini_htp  # (once tilelang is installed)
"""

from __future__ import annotations

import os
import re
import sys

from .build import build_cmake_fastrpc
from .device import AdbConnection, HexagonConnection
from .env import HexagonSDK

# repo_root/mini-htp  (this file is repo_root/tilelang/hexagon/mini_htp.py)
_DEFAULT_PROJECT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "mini-htp")
)
_MAXERR_RE = re.compile(r"max_err\s*=\s*([0-9.]+)")


def selftest(
    M: int = 32,
    N: int = 32,
    K: int = 32,
    *,
    project_dir: str = _DEFAULT_PROJECT,
    name: str = "hmx_matmul",
    dsp_arch: str = "v79",
    connection: HexagonConnection | None = None,
    verbose: bool = False,
) -> float:
    """Build mini-htp, deploy it, run the ``M×N×K`` HMX self-test, return max_err.

    Raises if the toolchain/device is unavailable or the RPC call fails.
    """
    sdk = HexagonSDK(dsp_arch=dsp_arch).validate()
    if connection is None:
        serials = AdbConnection.list_devices()
        if not serials:
            raise RuntimeError("no authorized adb device found (check `adb devices`)")
        connection = AdbConnection(serial=serials[0])

    if verbose:
        print(f"[mini-htp] building {name} (DSP_ARCH={dsp_arch}) in {project_dir}")
    artifacts = build_cmake_fastrpc(project_dir, name, sdk=sdk, dsp_arch=dsp_arch, verbose=verbose)

    if verbose:
        print(f"[mini-htp] skel={artifacts.dsp_skel}\n[mini-htp] host={artifacts.host_exe}")
    connection.push(artifacts.dsp_skel)
    exe_remote = connection.push(artifacts.host_exe)

    rc, out = connection.run_host_binary(exe_remote, [str(M), str(N), str(K)], timeout=120)
    if verbose:
        print(f"[mini-htp] device output:\n{out.strip()}")
    if rc != 0:
        raise RuntimeError(f"device run failed (rc={rc}):\n{out}")
    m = _MAXERR_RE.search(out)
    if not m:
        raise RuntimeError(f"could not parse max_err from device output:\n{out}")
    return float(m.group(1))


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    dims = [int(a) for a in argv] if argv else [32, 32, 32]
    M, N, K = (dims + [dims[-1]] * 3)[:3]
    err = selftest(M, N, K, verbose=True)
    print(f"\nmini-htp HMX {M}x{N}x{K}: max_err = {err:.4f}  "
          f"({'OK' if err < 0.1 else 'WRONG'})")
    return 0 if err < 0.1 else 1


if __name__ == "__main__":
    raise SystemExit(main())
