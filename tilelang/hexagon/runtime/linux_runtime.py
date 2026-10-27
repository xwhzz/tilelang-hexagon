"""Optional Linux BSP build and transport support.

Android keeps the original build and ADB connection. These helpers are selected
only when TILELANG_HEXAGON_HLOS=linux.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import uuid
from dataclasses import dataclass, field

from .device import AdbConnection, HexagonConnectionError, _DEFAULT_ADSP_DIRS
from .config import device_serial


def build_linux_fastrpc(project_dir, name, sdk, dsp_arch, build_host, verbose):
    """Build a DSP skel and an optional Linux driver, without an Android agent."""
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
    host = linux_build.build_linux_host(project_dir, name, sdk, env, verbose) if build_host else ""
    if build_host and not host:
        raise build.HexagonBuildError(f"host binary {name}_test not found after build in {project_dir}")
    return build.FastRPCArtifacts(dsp_skel=skel, host_exe=host or "", agent_exe="")


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
