"""Kernels on a Linux BSP (TILELANG_HEXAGON_HLOS=linux): their build, and
their transport, ADB for files and ADB or SSH for commands.

There is no agent on a Linux BSP: a compiled kernel runs one call at a time
(:class:`OneShotKernel`). Android kernels run in sessions (session.py).
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import tempfile
import uuid
import weakref
from dataclasses import dataclass, field

import numpy as np
import torch

from .device import DEVICE_LOCK, AdbConnection, HexagonConnectionError, _DEFAULT_ADSP_DIRS
from .config import device_serial


def build_linux_fastrpc(project_dir, name, sdk, dsp_arch, verbose):
    """Build a DSP skel and its Linux one-shot driver."""
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
    return build.FastRPCArtifacts(dsp_skel=skel, host_exe=host)


@dataclass
class LinuxConnection(AdbConnection):
    """Transfer files with ADB; execute Linux commands with ADB or optional SSH."""

    workdir: str = field(default_factory=lambda: "/tmp/tilelang_hexagon/" + uuid.uuid4().hex)
    ssh_host: str | None = field(default_factory=lambda: os.environ.get("TILELANG_HEXAGON_SSH_HOST"))
    known_hosts: str | None = field(default_factory=lambda: os.environ.get("TILELANG_HEXAGON_SSH_KNOWN_HOSTS"))

    def __post_init__(self):
        super().__post_init__()
        # Keep existing deployments attached to their original server/SSH setup
        # when a caller changes the environment to compile for another device.
        self._transport_env = dict(os.environ)
        self._base[0] = shutil.which(self.adb) or self.adb

    @classmethod
    def from_environment(cls):
        parent = os.environ.get("TILELANG_HEXAGON_REMOTE_DIR", "/tmp/tilelang_hexagon")
        return cls(serial=device_serial(), workdir=os.path.join(parent, uuid.uuid4().hex))

    def _run(self, argv, timeout=None):
        proc = subprocess.run(
            self._base + argv,
            env=self._transport_env,
            capture_output=True,
            text=True,
            timeout=timeout,
            stdin=subprocess.DEVNULL,
        )
        return proc.returncode, (proc.stdout or "") + (proc.stderr or "")

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

    def _ensure_workdir(self):
        if not self._ensured:
            # The ADB uploader must own the directory, even when execution uses SSH.
            rc, out = self._adb_shell(f"mkdir -p {shlex.quote(self.workdir)}")
            if rc:
                raise HexagonConnectionError(f"Creating device workdir failed: {out}")
            self._ensured = True

    def shell(self, command, timeout=None):
        if not self.ssh_host:
            return self._adb_shell(command, timeout=timeout)
        host_key_options = ["-o", "UserKnownHostsFile=" + self.known_hosts] if self.known_hosts else []
        proc = subprocess.run(
            [
                "ssh",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=10",
                "-o",
                "StrictHostKeyChecking=yes",
                *host_key_options,
                self.ssh_host,
                command,
            ],
            env=self._transport_env,
            capture_output=True,
            text=True,
            timeout=timeout,
            stdin=subprocess.DEVNULL,
        )
        return proc.returncode, (proc.stdout or "") + (proc.stderr or "")

    def run_host_binary(self, exe_remote_path, args, adsp_library_path=None, timeout=None):
        rc, out = self.shell(f"chmod 755 {shlex.quote(exe_remote_path)}")
        if rc:
            return rc, out
        exe_dir = os.path.dirname(exe_remote_path)
        adsp_dirs = [self.workdir, exe_dir, "/vendor/dsp/cdsp", "/usr/share/fastrpc", "/usr/lib/rfsa/adsp", *_DEFAULT_ADSP_DIRS]
        adsp = ";".join(dict.fromkeys(adsp_library_path or adsp_dirs))
        argstr = " ".join(shlex.quote(a) for a in args)
        command = (
            f"cd {shlex.quote(exe_dir)} && "
            f"LD_LIBRARY_PATH={shlex.quote(exe_dir)} "
            f"ADSP_LIBRARY_PATH={shlex.quote(adsp)} "
            f"CDSP_LIBRARY_PATH={shlex.quote(adsp)} "
            f"{shlex.quote(exe_remote_path)} {argstr}"
        )
        return self.shell(command, timeout=timeout)


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
