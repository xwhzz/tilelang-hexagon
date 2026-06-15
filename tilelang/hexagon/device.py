"""On-device transport for Hexagon kernels.

The Hexagon NPU is a *remote, out-of-process* device: it lives on a separate
chip (the cDSP) running a separate OS (QuRT), reached over FastRPC from an
Android-ARM host process, which we in turn reach over ``adb`` from this x86 dev
box.  Everything that crosses that boundary is funnelled through the
:class:`HexagonConnection` interface so the rest of the backend never bakes in
"adb" assumptions — a future low-latency transport (e.g. a persistent on-device
agent, or TVM's Hexagon RPC server) can be dropped in by implementing the same
interface.

Dependency-free (stdlib only).
"""

from __future__ import annotations

import abc
import os
import shlex
import subprocess
from dataclasses import dataclass


class HexagonConnectionError(RuntimeError):
    pass


class HexagonConnection(abc.ABC):
    """Abstract transport to a Hexagon-capable device.

    Implementations move files and execute an HLOS (Android-ARM) host binary
    that performs the actual FastRPC call into the cDSP.
    """

    workdir: str

    @abc.abstractmethod
    def push(self, local_path: str, remote_name: str | None = None) -> str:
        """Copy a local file to the device workdir; return the remote path."""

    @abc.abstractmethod
    def pull(self, remote_path: str, local_path: str) -> str:
        """Copy a device file back to the host; return the local path."""

    @abc.abstractmethod
    def shell(self, command: str, timeout: float | None = None) -> tuple[int, str]:
        """Run a shell command on the device; return (returncode, combined output)."""

    @abc.abstractmethod
    def run_host_binary(
        self,
        exe_remote_path: str,
        args: list[str],
        adsp_library_path: list[str] | None = None,
        timeout: float | None = None,
    ) -> tuple[int, str]:
        """Execute an HLOS host binary on the device with the FastRPC library
        search path configured so the cDSP can find the skel ``.so``."""


# Standard locations the FastRPC runtime searches for skel libraries, in
# addition to our workdir (which must come first).
_DEFAULT_ADSP_DIRS = (
    "/vendor/lib/rfsa/adsp",
    "/system/lib/rfsa/adsp",
    "/vendor/dsp",
    "/dsp",
)


@dataclass
class AdbConnection(HexagonConnection):
    """FastRPC transport over ``adb`` to a USB-attached Android device.

    This is the lean, mini-htp-style transport: push the skel + host driver,
    then ``adb shell`` the driver with ``ADSP_LIBRARY_PATH`` pointing at the
    workdir so the cDSP loads our skel.  One process per invocation — simple and
    correct; a persistent agent can replace it later for low latency.
    """

    serial: str | None = None
    workdir: str = "/data/local/tmp/tilelang_hexagon"
    adb: str = "adb"

    def __post_init__(self) -> None:
        self._base = [self.adb] + (["-s", self.serial] if self.serial else [])
        self._ensured = False

    # ---- helpers ----------------------------------------------------------
    def _run(self, argv: list[str], timeout: float | None = None) -> tuple[int, str]:
        proc = subprocess.run(
            self._base + argv, capture_output=True, text=True, timeout=timeout,
            stdin=subprocess.DEVNULL,
        )
        return proc.returncode, (proc.stdout or "") + (proc.stderr or "")

    def _ensure_workdir(self) -> None:
        if not self._ensured:
            self.shell(f"mkdir -p {shlex.quote(self.workdir)}")
            self._ensured = True

    @classmethod
    def list_devices(cls, adb: str = "adb") -> list[str]:
        """Return serials of authorized, attached devices."""
        proc = subprocess.run([adb, "devices"], capture_output=True, text=True, stdin=subprocess.DEVNULL)
        out = []
        for line in proc.stdout.splitlines()[1:]:
            parts = line.split()
            if len(parts) >= 2 and parts[1] == "device":
                out.append(parts[0])
        return out

    # ---- interface --------------------------------------------------------
    def push(self, local_path: str, remote_name: str | None = None) -> str:
        self._ensure_workdir()
        remote = os.path.join(self.workdir, remote_name or os.path.basename(local_path))
        rc, out = self._run(["push", local_path, remote])
        if rc != 0:
            raise HexagonConnectionError(f"adb push failed: {out}")
        return remote

    def pull(self, remote_path: str, local_path: str) -> str:
        rc, out = self._run(["pull", remote_path, local_path])
        if rc != 0:
            raise HexagonConnectionError(f"adb pull failed: {out}")
        return local_path

    def shell(self, command: str, timeout: float | None = None) -> tuple[int, str]:
        return self._run(["shell", command], timeout=timeout)

    def run_host_binary(
        self,
        exe_remote_path: str,
        args: list[str],
        adsp_library_path: list[str] | None = None,
        timeout: float | None = None,
    ) -> tuple[int, str]:
        self.shell(f"chmod 755 {shlex.quote(exe_remote_path)}")
        exe_dir = os.path.dirname(exe_remote_path)
        adsp_dirs = [self.workdir, exe_dir, *_DEFAULT_ADSP_DIRS]
        # de-dup while preserving order
        seen: set[str] = set()
        adsp = os.pathsep.join(d for d in (adsp_library_path or adsp_dirs) if not (d in seen or seen.add(d)))
        argstr = " ".join(shlex.quote(a) for a in args)
        cmd = (
            f"cd {shlex.quote(exe_dir)} && "
            f"LD_LIBRARY_PATH={shlex.quote(exe_dir)} "
            f"ADSP_LIBRARY_PATH={shlex.quote(adsp)} "
            f"{shlex.quote(exe_remote_path)} {argstr}"
        )
        return self.shell(cmd, timeout=timeout)
