"""Linux BSP builds and transport: SSH alone, or ADB when SSH is not set.

Linux and Android use the same resident session protocol. The one-shot
runner remains available for small bring-up programs.
"""

from __future__ import annotations

import os
import shlex
import shutil
import socket
import select
import time
import subprocess
import tempfile
import uuid
import weakref
from dataclasses import dataclass, field

import numpy as np
import torch

from .device import DEVICE_LOCK, AdbConnection, HexagonConnectionError, _DEFAULT_ADSP_DIRS
from .config import device_serial, remote_dir, ssh_host, ssh_transfer_timeout


def build_linux_fastrpc(project_dir, name, sdk, dsp_arch, verbose):
    """Build a DSP skel and the project's Linux agent or one-shot driver."""
    from . import build, linux_build

    env = sdk.environ
    legacy = "ON" if os.environ.get("TILELANG_HEXAGON_LEGACY_COMPUTE_RES") == "1" else "OFF"
    build._run(
        [sdk.build_cmake, "hexagon", f"DSP_ARCH={dsp_arch}", f"TL_HEXAGON_LEGACY_COMPUTE_RES={legacy}"],
        cwd=project_dir,
        env=env,
        verbose=verbose,
    )
    skel = build._find_one(
        [
            os.path.join(project_dir, f"hexagon_*_{dsp_arch}", "**", f"lib{name}_skel.so"),
            os.path.join(project_dir, f"hexagon_*_{dsp_arch}", f"lib{name}_skel.so"),
        ]
    )
    if not skel:
        raise build.HexagonBuildError(f"DSP skel lib{name}_skel.so not found after build in {project_dir}")
    linux_build.validate_dsp_artifact(skel, dsp_arch)
    if os.environ.get("TILELANG_HEXAGON_ELF_CRC") == "1":
        from .elf_crc import add_load_crc

        add_load_crc(skel)
    host = linux_build.build_linux_host(project_dir, name, sdk, env, verbose)
    if not host:
        raise build.HexagonBuildError(f"host binary {name}_test not found after build in {project_dir}")
    if os.path.isfile(os.path.join(project_dir, name + "_agent.c")):
        return build.FastRPCArtifacts(dsp_skel=skel, agent_exe=host)
    return build.FastRPCArtifacts(dsp_skel=skel, host_exe=host)


@dataclass
class LinuxConnection(AdbConnection):
    """SSH files, commands and loopback tunnels, or ADB when SSH is unset.

    ``transfer_timeout`` bounds each SSH upload, download and deployment mkdir
    in seconds. It defaults to TILELANG_HEXAGON_SSH_TRANSFER_TIMEOUT (600s),
    captured on construction; larger model files may need a higher setting.
    Persistent agents and their kernel calls have separate lifetime limits.
    """

    workdir: str = field(default_factory=lambda: "/tmp/tilelang_hexagon/" + uuid.uuid4().hex)
    ssh_host: str | None = field(default_factory=ssh_host)
    known_hosts: str | None = field(default_factory=lambda: os.environ.get("TILELANG_HEXAGON_SSH_KNOWN_HOSTS"))
    transfer_timeout: float | None = None

    def __post_init__(self):
        super().__post_init__()
        self.transfer_timeout = ssh_transfer_timeout(self.transfer_timeout)
        self._tunnels = {}

    @classmethod
    def from_environment(cls, serial=None):
        return cls(
            serial=None if ssh_host() is not None else serial or device_serial(),
            workdir=os.path.join(remote_dir("linux"), uuid.uuid4().hex),
        )

    def _adb_shell(self, command, timeout=None):
        # Older Linux adbd reports transport success even when the command fails.
        marker = f"__TILELANG_REMOTE_EXIT_{os.getpid()}__="
        wrapped = f"( {command} ); _tl_status=$?; printf '\\n{marker}%s\\n' \"$_tl_status\""
        transport_rc, output = self._run(["shell", wrapped], timeout=timeout)
        output = output.replace("\r", "")
        if marker in output:
            body, status = output.rsplit(marker, 1)
            try:
                return int(status.strip()), body.rstrip()
            except ValueError:
                pass
        return transport_rc or 1, output + "\nMissing remote command exit status"

    def _ssh_argv(self):
        host_key_options = ["-o", "UserKnownHostsFile=" + self.known_hosts] if self.known_hosts else []
        return [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=10",
            "-o",
            "StrictHostKeyChecking=yes",
            *host_key_options,
        ]

    def _shell_argv(self, command):
        if self.ssh_host:
            return [*self._ssh_argv(), self.ssh_host, command]
        return super()._shell_argv(command)

    def shell(self, command, timeout=None):
        if not self.ssh_host:
            return self._adb_shell(command, timeout=timeout)
        proc = subprocess.run(
            self._shell_argv(command),
            env=self._transport_env,
            capture_output=True,
            text=True,
            timeout=timeout,
            stdin=subprocess.DEVNULL,
        )
        return proc.returncode, (proc.stdout or "") + (proc.stderr or "")

    def push(self, local_path, remote_name=None):
        if not self.ssh_host:
            return super().push(local_path, remote_name)
        self._ensure_workdir(timeout=self.transfer_timeout)
        remote = os.path.join(self.workdir, remote_name or os.path.basename(local_path))
        # A shell stream works with BSPs that have no SFTP server or scp.
        with open(local_path, "rb") as src:
            proc = subprocess.run(
                self._shell_argv(f"cat > {shlex.quote(remote)}"),
                env=self._transport_env,
                stdin=src,
                capture_output=True,
                timeout=self.transfer_timeout,
            )
        if proc.returncode:
            raise HexagonConnectionError(f"SSH upload failed: {proc.stderr.decode(errors='replace')}")
        return remote

    def pull(self, remote_path, local_path):
        if not self.ssh_host:
            return super().pull(remote_path, local_path)
        # A failed transfer must not replace an existing local result.
        fd, staging = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(local_path)))
        try:
            with os.fdopen(fd, "wb") as dst:
                proc = subprocess.run(
                    self._shell_argv(f"cat {shlex.quote(remote_path)}"),
                    env=self._transport_env,
                    stdin=subprocess.DEVNULL,
                    stdout=dst,
                    stderr=subprocess.PIPE,
                    timeout=self.transfer_timeout,
                )
            if proc.returncode:
                raise HexagonConnectionError(f"SSH download failed: {proc.stderr.decode(errors='replace')}")
            os.replace(staging, local_path)
        finally:
            if os.path.exists(staging):
                os.unlink(staging)
        return local_path

    def forward_port(self, remote_port):
        if not self.ssh_host:
            return super().forward_port(remote_port)
        # OpenSSH cannot choose a local forwarding port for us. Reserve one
        # briefly, and let ExitOnForwardFailure reject a conflicting listener.
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            local_port = listener.getsockname()[1]
        proc = subprocess.Popen(
            [
                *self._ssh_argv(),
                "-o",
                "ExitOnForwardFailure=yes",
                "-T",
                "-L",
                f"127.0.0.1:{local_port}:127.0.0.1:{remote_port}",
                self.ssh_host,
                "printf 'TILELANG_TUNNEL_READY\\n'; cat",
            ],
            env=self._transport_env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        self._tunnels[local_port] = proc
        try:
            # The remote command starts only after SSH has set up forwarding.
            # Its open stdin keeps it alive; unlike a TCP probe, this cannot
            # accidentally accept an unrelated listener that won the port race.
            deadline = time.monotonic() + 15
            printed = b""
            while b"TILELANG_TUNNEL_READY\n" not in printed:
                if proc.poll() is not None:
                    printed += proc.stdout.read()
                    raise HexagonConnectionError(f"SSH tunnel exited ({proc.returncode}): {printed.decode(errors='replace')}")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise HexagonConnectionError(f"SSH tunnel did not become ready: {printed.decode(errors='replace')}")
                ready, _, _ = select.select([proc.stdout], [], [], min(0.2, remaining))
                if ready:
                    printed += os.read(proc.stdout.fileno(), 4096)
            return local_port
        except BaseException:
            self.remove_forward(local_port)
            raise

    def remove_forward(self, local_port):
        if not self.ssh_host:
            return super().remove_forward(local_port)
        proc = self._tunnels.pop(local_port, None)
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            proc.stdin.close()
            proc.stdout.close()

    def _host_command(self, exe_remote_path, args, adsp_library_path=None):
        exe_dir = os.path.dirname(exe_remote_path)
        adsp_dirs = [
            self.workdir,
            exe_dir,
            "/vendor/dsp/cdsp",
            "/vendor/dsp/cdsp1",
            "/usr/share/fastrpc",
            "/usr/lib/rfsa/adsp",
            *_DEFAULT_ADSP_DIRS,
        ]
        configured = self._transport_env.get("TILELANG_HEXAGON_DSP_LIBRARY_PATH")
        if configured is not None:
            adsp_dirs = [self.workdir, exe_dir, *filter(None, configured.split(";"))]
        adsp = ";".join(dict.fromkeys(adsp_library_path or adsp_dirs))
        argstr = " ".join(shlex.quote(str(a)) for a in args)
        return (
            f"cd {shlex.quote(exe_dir)} && chmod 755 {shlex.quote(exe_remote_path)} && "
            f"exec env LD_LIBRARY_PATH={shlex.quote(exe_dir)} "
            f"ADSP_LIBRARY_PATH={shlex.quote(adsp)} CDSP_LIBRARY_PATH={shlex.quote(adsp)} "
            f"CDSP1_LIBRARY_PATH={shlex.quote(adsp)} "
            f"{shlex.quote(exe_remote_path)} {argstr}"
        )


class OneShotKernel:
    """A compiled kernel on a Linux BSP. Each call pushes its tensors as files,
    runs the kernel's one-shot host driver (fastrpc.py), which opens a FastRPC
    session for the call, and pulls the outputs back, in ``result_idx`` order.
    Deployed until ``close()``, it holds the device lock (device.py)."""

    def __init__(self, name: str, source: str, params, result_idx, *, sdk, dsp_arch: str, verbose: bool = False):
        from . import fastrpc
        from .build import build_cmake_fastrpc

        self.params, self.result_idx = list(params), list(result_idx)
        # The kernel's FastRPC project, build tree and argument files.
        self._workdir = tempfile.mkdtemp(prefix="tl_hexagon_")
        weakref.finalize(self, shutil.rmtree, self._workdir, True)
        project, self._iface = fastrpc.write_project(self._workdir, name, source, self.params, self.result_idx)
        artifacts = build_cmake_fastrpc(project, self._iface, sdk=sdk, dsp_arch=dsp_arch, verbose=verbose)
        self._conn = LinuxConnection.from_environment()
        self._holds_device = DEVICE_LOCK.acquire()
        try:
            self._conn.push(artifacts.dsp_skel)
            self._exe = self._conn.push(artifacts.host_exe)
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        if getattr(self, "_holds_device", False):
            self._holds_device = False
            DEVICE_LOCK.release()

    def __call__(self, *ins):
        params, outputs = self.params, set(self.result_idx)
        # Arguments in parameter order: a buffer is a pushed file, a scalar
        # (a number, or a 0-d tensor) its value.
        values, it = [None] * len(params), iter(ins)
        for i, p in enumerate(params):
            if i in outputs:
                values[i] = torch.empty(tuple(int(s) for s in p.shape), dtype=p.torch_dtype(), device="cpu")
            else:
                t = next(it)
                values[i] = t if p.is_scalar() else t.detach().to("cpu").contiguous()
        argv, pulls = [], []
        for i, p in enumerate(params):
            if p.is_scalar():
                argv.append(str(int(values[i])))
                continue
            fname = f"{self._iface}_p{i}.bin"
            local = os.path.join(self._workdir, fname)
            if i in outputs:
                pulls.append((local, fname, values[i]))
            else:
                with open(local, "wb") as f:
                    f.write(values[i].numpy().tobytes())
                self._conn.push(local, fname)
            argv.append(fname)
        rc, out = self._conn.run_host_binary(self._exe, argv, timeout=120)
        if rc != 0:
            raise RuntimeError(f"Hexagon device run failed (rc={rc}):\n{out}")
        for local, fname, tensor in pulls:
            self._conn.pull(os.path.join(self._conn.workdir, fname), local)
            with open(local, "rb") as f:
                arr = np.frombuffer(f.read(), dtype=np.dtype(str(tensor.dtype).replace("torch.", ""))).reshape(tensor.shape)
            tensor.copy_(torch.from_numpy(arr.copy()))
        results = [values[i] for i in self.result_idx]
        if not results:
            return None
        return results[0] if len(results) == 1 else results
