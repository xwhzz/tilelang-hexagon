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

import atexit
import contextlib
import os
import select
import socket
import subprocess
import time

import numpy as np
from tvm import DataType

from . import fastrpc
from .device import _DEFAULT_ADSP_DIRS

# Seconds one call to an agent may take, its transfers included.
CALL_TIMEOUT = 600.0


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
        self.plans = fastrpc._plan(self.params, self.result_idx)
        self._base = [adb] + (["-s", serial] if serial else [])
        self.workdir = workdir
        self.device_port = device_port
        self.host_port = host_port  # adb picks one when None

        self._sh(f"mkdir -p {workdir}")
        self._push(skel_path)
        exe = self._push(agent_path)
        self._sh(f"chmod 755 {exe}")
        self._exe_name = os.path.basename(exe)

        if host_port is None:
            # adb picks a free host port: on a shared host, a fixed one may be taken.
            done = subprocess.run(self._base + ["forward", "tcp:0", f"tcp:{self.device_port}"], check=True, capture_output=True, text=True)
            self.host_port = int(done.stdout.split()[-1])
        else:
            subprocess.run(self._base + ["forward", f"tcp:{self.host_port}", f"tcp:{self.device_port}"], check=True, capture_output=True)

        adsp = os.pathsep.join([workdir, *_DEFAULT_ADSP_DIRS])
        # The directory is the session's own: it goes when the agent exits.
        cmd = f"cd {workdir} && LD_LIBRARY_PATH={workdir} ADSP_LIBRARY_PATH={adsp} ./{self._exe_name} {self.device_port}; rm -rf {workdir}"
        self._proc = subprocess.Popen(self._base + ["shell", cmd], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        self._wait_ready(ready_timeout)
        self._sock = self._connect()
        # __del__ isn't guaranteed to run at interpreter shutdown, so register an
        # explicit exit hook — otherwise the on-device agent leaks: it keeps
        # holding the full VTCM, starving the next agent into silent zeros.
        atexit.register(self.close)

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
                # A call may move hundreds of MB or run long: not the connect's timeout.
                s.settimeout(CALL_TIMEOUT)
                return s
            except OSError:
                time.sleep(0.05)
        raise RuntimeError("could not connect to Hexagon agent socket")

    def close(self):
        if getattr(self, "_closed", False):
            return
        self._closed = True
        with contextlib.suppress(Exception):
            atexit.unregister(self.close)
        with contextlib.suppress(Exception):
            self._sock.sendall(b"\x02")  # shutdown
            self._sock.close()
        # The agent exits after closing its FastRPC session, which releases
        # its VTCM. Wait for that, so that the next session (another worker's,
        # once a device lock is released) finds the DSP free.
        try:
            self._proc.wait(timeout=5)
        except Exception:
            with contextlib.suppress(Exception):
                # Match exe name AND port so concurrent adapters for the same
                # kernel (distinct ports, see HexagonKernelAdapter) don't kill
                # each other.
                self._sh(f"pkill -f '{self._exe_name} {self.device_port}'")
            with contextlib.suppress(Exception):
                self._proc.terminate()
                self._proc.wait(timeout=5)
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
            mask |= 1 << i
        in_i = 0
        payloads = []
        for pl in self.plans:
            if pl.is_output or pl.is_scalar:
                continue
            if in_i in changed:
                data = np.ascontiguousarray(inputs[in_i]).tobytes()
                # A short or long payload would desynchronize the wire.
                if len(data) != pl.nbytes:
                    raise ValueError(f"Hexagon agent: input {in_i} has {len(data)} bytes, the kernel takes {pl.nbytes}")
                payloads.append(data)
            in_i += 1
        # op=run, the changed-mask, then the inputs, in one write: adb relays
        # each write as a packet, and small packets wait out delayed ACKs.
        self._sock.sendall(b"".join([bytes([1, mask]), *payloads]))
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
            if DataType(str(p.dtype)).bits < 8:
                # Packed sub-byte elements come back as their bytes.
                outs.append(np.frombuffer(buf, dtype=np.uint8).copy())
                continue
            arr = np.frombuffer(buf, dtype=np.dtype(str(p.dtype))).reshape([int(s) for s in p.shape])
            outs.append(arr.copy())
        return outs
