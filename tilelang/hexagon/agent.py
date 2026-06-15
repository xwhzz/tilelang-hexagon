"""Persistent on-device agent client (low-latency transport).

Holds a single long-running FastRPC session on the phone (one cDSP PD, one skel
load, HMX/VTCM acquired once) and serves repeated invokes over a TCP socket
forwarded by ``adb``.  This replaces the per-call ``adb push/shell/pull`` of the
one-shot path, so repeated calls (autotuning, benchmarking, inference loops) pay
only socket + compute, not process spawn + session open.

Wire protocol per invoke: client sends ``op`` (1=run, 0=close conn, 2=shutdown);
for ``run`` it then streams the input buffers (and any int32 scalars) in
param order, and reads back the output buffers.  Buffer sizes are baked into the
generated agent, so only raw bytes cross the wire.
"""

from __future__ import annotations

import os
import select
import socket
import subprocess
import time

import numpy as np

from . import _fastrpc
from .device import _DEFAULT_ADSP_DIRS


class HexagonAgentSession:
    """Lifecycle + invoke channel for a generated FastRPC agent."""

    def __init__(
        self,
        skel_path: str,
        agent_path: str,
        params,
        result_idx,
        *,
        serial: str | None = None,
        workdir: str = "/data/local/tmp/tilelang_hex_agent",
        device_port: int = 9777,
        host_port: int | None = None,
        adb: str = "adb",
        ready_timeout: float = 20.0,
    ) -> None:
        self.params = list(params)
        self.result_idx = list(result_idx if isinstance(result_idx, (list, tuple)) else [result_idx])
        self.plans = _fastrpc._plan(self.params, self.result_idx)
        self._base = [adb] + (["-s", serial] if serial else [])
        self.workdir = workdir
        self.device_port = device_port
        self.host_port = host_port or device_port

        self._sh(f"mkdir -p {workdir}")
        self._push(skel_path)
        exe = self._push(agent_path)
        self._sh(f"chmod 755 {exe}")
        self._exe_name = os.path.basename(exe)

        subprocess.run(self._base + ["forward", f"tcp:{self.host_port}", f"tcp:{self.device_port}"],
                       check=True, capture_output=True)

        adsp = os.pathsep.join([workdir, *_DEFAULT_ADSP_DIRS])
        cmd = (f"cd {workdir} && LD_LIBRARY_PATH={workdir} ADSP_LIBRARY_PATH={adsp} "
               f"./{self._exe_name} {self.device_port}")
        self._proc = subprocess.Popen(self._base + ["shell", cmd], stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT, text=True)
        self._wait_ready(ready_timeout)
        self._sock = self._connect()

    # ---- lifecycle --------------------------------------------------------
    def _sh(self, cmd: str):
        subprocess.run(self._base + ["shell", cmd], capture_output=True)

    def _push(self, local: str) -> str:
        remote = self.workdir + "/" + os.path.basename(local)
        subprocess.run(self._base + ["push", local, remote], check=True, capture_output=True)
        return remote

    def _wait_ready(self, timeout: float):
        # select() so a silent/hung agent can't block past the deadline: a bare
        # blocking readline() would wait forever for a line that never comes,
        # defeating *timeout* entirely.
        fd = self._proc.stdout
        t0 = time.time()
        while True:
            remaining = timeout - (time.time() - t0)
            if remaining <= 0:
                break
            if self._proc.poll() is not None:
                rest = fd.read() if fd else ""
                raise RuntimeError(f"Hexagon agent exited before ready:\n{rest}")
            r, _, _ = select.select([fd], [], [], min(0.2, remaining))
            if not r:
                continue
            line = fd.readline()
            if line and "AGENT_READY" in line:
                return
        raise RuntimeError("Hexagon agent did not report ready in time")

    def _connect(self, timeout: float = 10.0) -> socket.socket:
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                s = socket.create_connection(("127.0.0.1", self.host_port), timeout=2.0)
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                return s
            except OSError:
                time.sleep(0.05)
        raise RuntimeError("could not connect to Hexagon agent socket")

    def close(self):
        try:
            self._sock.sendall(b"\x02")  # shutdown
            self._sock.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            self._sh(f"pkill -f {self._exe_name}")
        except Exception:  # noqa: BLE001
            pass
        try:
            self._proc.terminate()
        except Exception:  # noqa: BLE001
            pass
        subprocess.run(self._base + ["forward", "--remove", f"tcp:{self.host_port}"], capture_output=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ---- invoke -----------------------------------------------------------
    def _recvn(self, n: int) -> bytes:
        chunks, got = [], 0
        while got < n:
            c = self._sock.recv(n - got)
            if not c:
                raise RuntimeError("Hexagon agent closed the connection")
            chunks.append(c)
            got += len(c)
        return b"".join(chunks)

    def invoke(self, inputs: list[np.ndarray], changed=None) -> list[np.ndarray]:
        """Run once.  *inputs* are the input arrays in input-param order.

        *changed* is the set of input indices to actually transmit this call;
        omitted inputs reuse the resident copy left on the device by a prior call
        (e.g. a fixed weight).  Default = all inputs.  The first call for a given
        resident buffer must include it.
        """
        n_in = sum(1 for pl in self.plans if not pl.is_output and not pl.is_scalar)
        changed = set(range(n_in)) if changed is None else set(changed)
        mask = 0
        for i in changed:
            mask |= (1 << i)
        self._sock.sendall(bytes([1, mask]))  # op=run, then the changed-mask
        in_i = 0
        for pl in self.plans:
            if pl.is_output or pl.is_scalar:
                continue
            if in_i in changed:
                self._sock.sendall(np.ascontiguousarray(inputs[in_i]).tobytes())
            in_i += 1
        # Status byte first: the agent sends 0 on success, non-zero if the kernel
        # invoke failed.  On failure no output bytes follow, so we must read this
        # before the output loop or _recvn() would block forever.
        if self._recvn(1)[0] != 0:
            raise RuntimeError("Hexagon agent: on-device kernel run failed")
        outs: list[np.ndarray] = []
        for pl in self.plans:
            if not pl.is_output:
                continue
            p = self.params[pl.index]
            buf = self._recvn(pl.nbytes)
            arr = np.frombuffer(buf, dtype=np.dtype(str(p.dtype))).reshape([int(s) for s in p.shape])
            outs.append(arr.copy())
        return outs
