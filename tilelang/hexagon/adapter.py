"""Adapter that runs a tilelang-generated Hexagon kernel on a real device.

Flow: take the generated kernel C source + its :class:`KernelParam`s, emit a
FastRPC project (:mod:`._fastrpc`), build it with the Hexagon SDK
(:mod:`.build`), deploy the skel + host driver over adb (:mod:`.device`), and
return a torch-callable.  Each call marshals input tensors to ``.bin`` files,
pushes them, runs the host driver (which FastRPC-invokes the cDSP kernel), and
pulls the outputs back.  Simple one-shot transport — correctness first; a
persistent low-latency agent is an M3 concern.
"""

from __future__ import annotations

import itertools
import os
import re
import tempfile

import numpy as np
import torch

from tilelang.jit.adapter.base import BaseKernelAdapter

from . import _fastrpc
from .build import build_cmake_fastrpc
from .device import AdbConnection, HexagonConnection
from .env import HexagonSDK

# The kernel entry is the non-static generated function.  Hexagon kernels return
# int32_t so the FastRPC skel can propagate failures; older/source kernels may
# still be void.  Skip static helpers such as `<name>_worker`.
_KERNEL_RE = re.compile(r"^\s*(?:void|int|int32_t)\s+(\w+)\s*\(", re.MULTILINE)


def _kernel_name(source: str) -> str:
    m = _KERNEL_RE.search(source)
    if not m:
        raise RuntimeError("could not find the kernel function name in generated source")
    return m.group(1)


class HexagonKernelAdapter(BaseKernelAdapter):
    """Build + deploy a generated Hexagon kernel and call it on-device."""

    # Process-wide sequence so concurrent adapters get distinct agent ports and
    # device workdirs — otherwise they collide on tcp:9777 and the shared dir.
    _port_seq = itertools.count()

    def __init__(
        self,
        params,
        result_idx,
        target,
        kernel_source: str,
        *,
        kernel_name: str | None = None,
        connection: HexagonConnection | None = None,
        sdk: HexagonSDK | None = None,
        dsp_arch: str = "v79",  # SM8750/Hexagon v79 hardware (see env.py)
        verbose: bool = False,
        workdir: str | None = None,
        use_agent: bool = True,
    ) -> None:
        self.kernel_source = kernel_source
        self.kernel_name = kernel_name or _kernel_name(kernel_source)
        self.target = target
        self.verbose = verbose
        self.dsp_arch = dsp_arch
        self.sdk = (sdk or HexagonSDK(dsp_arch=dsp_arch)).validate()
        self._param_list = list(params)
        # Legalize negative output indices (e.g. out_idx=-1) up front: _deploy()
        # consumes self._result_idx before BaseKernelAdapter.__init__ runs, so we
        # can't rely on the base's self.result_idx here.
        n = len(self._param_list)
        ri = list(result_idx) if isinstance(result_idx, (list, tuple)) else ([] if result_idx is None else [result_idx])
        self._result_idx = [i if i >= 0 else n + i for i in ri]
        self._workdir = workdir or tempfile.mkdtemp(prefix="tl_hexagon_")
        self._conn = connection
        self._use_agent = use_agent
        self._agent = None
        self._deploy()
        # BaseKernelAdapter wires self.func = self._convert_torch_func()
        super().__init__(mod=None, params=params, result_idx=result_idx)

    # ---- build + deploy ---------------------------------------------------
    def _connection(self) -> HexagonConnection:
        if self._conn is None:
            serials = AdbConnection.list_devices()
            if not serials:
                raise RuntimeError("no authorized adb device found (check `adb devices`)")
            self._conn = AdbConnection(serial=serials[0])
        return self._conn

    def _deploy(self) -> None:
        proj, iface = _fastrpc.write_project(
            self._workdir, self.kernel_name, self.kernel_source, self._param_list, self._result_idx
        )
        self._iface = iface
        if self.verbose:
            print(f"[hexagon] FastRPC project: {proj} (iface={iface})")
        artifacts = build_cmake_fastrpc(proj, iface, sdk=self.sdk, dsp_arch=self.dsp_arch, verbose=self.verbose)
        conn = self._connection()
        conn.push(artifacts.dsp_skel)
        self._exe_remote = conn.push(artifacts.host_exe)
        if self.verbose:
            print(f"[hexagon] deployed skel + host to {conn.workdir}")
        # Prefer the persistent agent (one held session, ~30x lower per-call
        # latency) unless the kernel has scalar params (not yet marshaled over
        # the agent); otherwise fall back to the one-shot host driver.
        has_scalar = any(p.is_scalar() for p in self._param_list)
        if self._use_agent and artifacts.agent_exe and not has_scalar:
            from .agent import HexagonAgentSession
            idx = next(HexagonKernelAdapter._port_seq)
            port = 10000 + (os.getpid() * 13 + idx) % 40000
            self._agent = HexagonAgentSession(
                artifacts.dsp_skel, artifacts.agent_exe, self._param_list, self._result_idx,
                serial=getattr(conn, "serial", None),
                device_port=port,
                workdir=f"/data/local/tmp/tilelang_hex_agent_{os.getpid()}_{idx}",
            )
            if self.verbose:
                print("[hexagon] persistent agent transport active")

    def close(self):
        if getattr(self, "_agent", None) is not None:
            self._agent.close()
            self._agent = None

    def __del__(self):
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass

    def _agent_forward(self, *ins):
        params, result_idx = self._param_list, set(self._result_idx)
        it = iter(ins)
        in_arrays = []
        for i, p in enumerate(params):
            if i in result_idx:
                continue
            in_arrays.append(next(it).detach().to("cpu").contiguous().numpy())
        out_arrays = self._agent.invoke(in_arrays)
        # invoke() yields outputs in ascending param-index (plan) order; the
        # public contract is self._result_idx order (matches the one-shot path),
        # so reorder — matters when out_idx isn't ascending.
        by_param = dict(zip(sorted(self._result_idx), out_arrays))
        outs = [torch.from_numpy(np.ascontiguousarray(by_param[i])) for i in self._result_idx]
        return outs[0] if len(outs) == 1 else outs

    # ---- call -------------------------------------------------------------
    def _convert_torch_func(self):
        params = self._param_list
        result_idx = set(self._result_idx)
        if self._agent is not None:
            return self._agent_forward
        conn = self._connection()

        def forward(*ins):
            # Materialize args in kernel-param order; allocate outputs.
            values = [None] * len(params)
            it = iter(ins)
            for i, p in enumerate(params):
                if i in result_idx:
                    values[i] = torch.empty(
                        tuple(int(s) for s in p.shape), dtype=p.torch_dtype(), device="cpu"
                    )
                else:
                    t = next(it)
                    values[i] = t.detach().to("cpu").contiguous()

            # Build argv (kernel-param order): buffer -> pushed file; scalar -> int.
            argv = []
            pulls = []  # (local_path, remote_name, out_tensor)
            for i, p in enumerate(params):
                if p.is_scalar():
                    argv.append(str(int(values[i].item())))
                    continue
                fname = f"{self._iface}_p{i}.bin"
                local = os.path.join(self._workdir, fname)
                if i in result_idx:
                    pulls.append((local, fname, values[i]))
                else:
                    arr = values[i].numpy()
                    with open(local, "wb") as f:
                        f.write(arr.tobytes())
                    conn.push(local, fname)
                argv.append(fname)

            rc, out = conn.run_host_binary(self._exe_remote, argv, timeout=120)
            if rc != 0:
                raise RuntimeError(f"Hexagon device run failed (rc={rc}):\n{out}")

            # Pull outputs back into the allocated tensors.
            for local, fname, tensor in pulls:
                conn.pull(os.path.join(conn.workdir, fname), local)
                with open(local, "rb") as f:
                    buf = f.read()
                arr = np.frombuffer(buf, dtype=np.dtype(str(tensor.dtype).replace("torch.", ""))).reshape(tensor.shape)
                tensor.copy_(torch.from_numpy(arr.copy()))

            outs = [values[i] for i in self._result_idx]
            return outs[0] if len(outs) == 1 else outs

        return forward
