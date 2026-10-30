"""File transfer, process launch and loopback forwarding to Hexagon HLOS hosts.

The FastRPC agent runs on the ARM host next to the cDSP. Connections hide
ADB and SSH from the resident session's buffer and kernel protocol.
"""

from __future__ import annotations

import abc
import fcntl
import os
import shlex
import shutil
import subprocess
import threading
from dataclasses import dataclass


class HexagonConnectionError(RuntimeError):
    pass


class DeviceLock:
    """Runs one device session at a time across processes, when asked to.

    With ``TILELANG_HEXAGON_DEVICE_LOCK`` naming a file, an adapter holds an
    exclusive lock on it from deployment to ``close()``, so parallel test
    workers (``pytest -n``) build kernels concurrently but use the device one
    after another. Holds nest within a process.
    """

    def __init__(self) -> None:
        self._mutex = threading.Lock()
        self._depth = 0
        self._fd: int | None = None

    def acquire(self) -> bool:
        """Takes a hold; returns whether one was taken (the variable is set)."""
        path = os.environ.get("TILELANG_HEXAGON_DEVICE_LOCK")
        if not path:
            return False
        with self._mutex:
            if self._depth == 0:
                fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX)
                except BaseException:
                    os.close(fd)
                    raise
                self._fd = fd
            self._depth += 1
        return True

    def release(self) -> None:
        with self._mutex:
            self._depth -= 1
            if self._depth == 0:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
                os.close(self._fd)
                self._fd = None


DEVICE_LOCK = DeviceLock()


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

    @abc.abstractmethod
    def start_host_binary(self, exe_remote_path: str, args: list[str]) -> subprocess.Popen:
        """Start a resident agent; retain stdout and an open stdin for its lifetime."""

    @abc.abstractmethod
    def forward_port(self, remote_port: int) -> int:
        """Forward a device loopback TCP port and return its local port."""

    @abc.abstractmethod
    def remove_forward(self, local_port: int) -> None:
        """Release a forwarding created by this connection."""

    @abc.abstractmethod
    def cleanup(self) -> None:
        """Remove this session's private deployment directory, if creation began."""


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

    Files and commands use ADB, and ``adb forward`` reaches a persistent
    agent's loopback socket. The same connection can run a one-shot driver.
    """

    serial: str | None = None
    workdir: str = "/data/local/tmp/tilelang_hexagon"
    adb: str = "adb"

    def __post_init__(self) -> None:
        self._base = [shutil.which(self.adb) or self.adb] + (["-s", self.serial] if self.serial else [])
        self._transport_env = dict(os.environ)
        self._ensured = False
        self._workdir_attempted = False

    # ---- helpers ----------------------------------------------------------
    def _run(self, argv: list[str], timeout: float | None = None) -> tuple[int, str]:
        proc = subprocess.run(
            self._base + argv,
            env=self._transport_env,
            capture_output=True,
            text=True,
            timeout=timeout,
            stdin=subprocess.DEVNULL,
        )
        return proc.returncode, (proc.stdout or "") + (proc.stderr or "")

    def _ensure_workdir(self, timeout: float | None = None) -> None:
        if not self._ensured:
            self._workdir_attempted = True
            rc, out = self.shell(f"mkdir -p {shlex.quote(self.workdir)}", timeout=timeout)
            if rc:
                raise HexagonConnectionError(f"Creating device workdir failed: {out}")
            self._ensured = True

    @classmethod
    def list_devices(cls, adb: str = "adb") -> list[str]:
        """Return serials of authorized, attached devices; none without adb."""
        try:
            proc = subprocess.run([adb, "devices"], capture_output=True, text=True, stdin=subprocess.DEVNULL)
        except FileNotFoundError:
            return []
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

    def _shell_argv(self, command: str) -> list[str]:
        return self._base + ["shell", command]

    def start_shell(self, command: str):
        """Start a command whose output stays available until it exits."""
        return subprocess.Popen(
            self._shell_argv(command),
            env=self._transport_env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

    def forward_port(self, remote_port: int) -> int:
        """Forward a device loopback port; close it with ``remove_forward``."""
        rc, out = self._run(["forward", "tcp:0", f"tcp:{remote_port}"])
        if rc:
            raise HexagonConnectionError(f"adb forward failed: {out}")
        return int(out.split()[-1])

    def remove_forward(self, local_port: int) -> None:
        self._run(["forward", "--remove", f"tcp:{local_port}"])

    def _host_command(self, exe_remote_path, args, adsp_library_path=None):
        exe_dir = os.path.dirname(exe_remote_path)
        adsp_dirs = [self.workdir, exe_dir, *_DEFAULT_ADSP_DIRS]
        configured = self._transport_env.get("TILELANG_HEXAGON_DSP_LIBRARY_PATH")
        if configured is not None:
            adsp_dirs = [self.workdir, exe_dir, *filter(None, configured.split(";"))]
        adsp = os.pathsep.join(dict.fromkeys(adsp_library_path or adsp_dirs))
        argstr = " ".join(shlex.quote(str(a)) for a in args)
        return (
            f"cd {shlex.quote(exe_dir)} && chmod 755 {shlex.quote(exe_remote_path)} && "
            f"exec env LD_LIBRARY_PATH={shlex.quote(exe_dir)} "
            f"ADSP_LIBRARY_PATH={shlex.quote(adsp)} "
            f"{shlex.quote(exe_remote_path)} {argstr}"
        )

    def run_host_binary(self, exe_remote_path, args, adsp_library_path=None, timeout=None):
        return self.shell(self._host_command(exe_remote_path, args, adsp_library_path), timeout=timeout)

    def start_host_binary(self, exe_remote_path, args):
        """Start a session agent, removing its private directory on exit."""
        # Keep the agent a child of the transport shell so a failed startup or
        # interrupted SSH connection can terminate it as well as the shell.
        # Preserve stdin on fd 3: POSIX shells otherwise give a background
        # command /dev/null before its redirections are applied.
        command = self._host_command(exe_remote_path, args)
        cleanup = f'kill "$_tl_agent" "$_tl_watch" 2>/dev/null; rm -rf -- {shlex.quote(self.workdir)}'
        return self.start_shell(
            f"exec 3<&0; ( {command} ) 3<&- & _tl_agent=$!; "
            '( while IFS= read -r _tl_line; do :; done; kill "$_tl_agent" 2>/dev/null ) <&3 3<&- & _tl_watch=$!; exec 3<&-; '
            f"trap {shlex.quote(cleanup)} EXIT; "
            "trap 'exit 1' HUP INT TERM; "
            'wait "$_tl_agent"; _tl_status=$?; exit "$_tl_status"'
        )

    def cleanup(self) -> None:
        if self._workdir_attempted:
            rc, out = self.shell(f"rm -rf -- {shlex.quote(self.workdir)}", timeout=10)
            if rc:
                raise HexagonConnectionError(f"Removing private device workdir failed: {out}")
            self._ensured = self._workdir_attempted = False
