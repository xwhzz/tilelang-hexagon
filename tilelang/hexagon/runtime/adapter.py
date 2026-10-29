"""tilelang.compile's adapter for Hexagon kernels.

On Android, a compiled kernel is a :class:`~.session.Session` of one kernel:
its skel, and an agent on the phone that holds its FastRPC session, reached
through ``adb forward``. A call sends the kernel's inputs and scalars and
returns its outputs, in ``out_idx`` order. ``close()``, or ``close_all()``
between tests, ends the session and frees the device; the next call starts
another, from the cached build.

On a Linux BSP (``TILELANG_HEXAGON_HLOS=linux``) there is no agent: each call
runs the kernel's one-shot host driver (:class:`~.linux_runtime.OneShotKernel`).
"""

from __future__ import annotations

import contextlib
import re
import weakref

from tilelang.hexagon.target import hexagon_arch
from tilelang.jit.adapter.base import BaseKernelAdapter

from .config import hlos
from .env import HexagonSDK

# The kernel entry is the generated function that returns its int32 status;
# static helpers such as `<name>_worker` are not.
_KERNEL_RE = re.compile(r"^\s*(?:void|int|int32_t)\s+(\w+)\s*\(", re.MULTILINE)


def _kernel_name(source: str) -> str:
    m = _KERNEL_RE.search(source)
    if not m:
        raise RuntimeError("could not find the kernel function name in generated source")
    return m.group(1)


class HexagonKernelAdapter(BaseKernelAdapter):
    """A generated Hexagon kernel, built and called on the device."""

    # Adapters of this process, for close_all().
    _live: weakref.WeakSet = weakref.WeakSet()

    def __init__(
        self,
        params,
        result_idx,
        target,
        kernel_source: str,
        *,
        sdk: HexagonSDK | None = None,
        dsp_arch: str | None = None,
        verbose: bool = False,
    ) -> None:
        self.kernel_source = kernel_source
        self.kernel_name = _kernel_name(kernel_source)
        self.target = target
        self.verbose = verbose
        # The target names the DSP architecture; HEXAGON_DSP_ARCH only seeds
        # its default during target normalization.
        dsp_arch = dsp_arch or hexagon_arch(target)
        self.sdk = (sdk or (HexagonSDK(dsp_arch=dsp_arch) if dsp_arch else HexagonSDK())).validate()
        self.dsp_arch = dsp_arch or self.sdk.dsp_arch
        self._param_list = list(params)
        n = len(self._param_list)
        ri = list(result_idx) if isinstance(result_idx, (list, tuple)) else ([] if result_idx is None else [result_idx])
        self._result_idx = [i if i >= 0 else n + i for i in ri]
        # The kernel on the device (a session, or a Linux one-shot kernel),
        # and the callable that runs it.
        self._device = None
        self._run = None
        # Up now, so that a build error surfaces at compile time.
        self._open()
        HexagonKernelAdapter._live.add(self)
        super().__init__(mod=None, params=params, result_idx=result_idx)

    def _open(self):
        """The callable that runs the kernel on the device, which it starts
        unless it is up."""
        if self._device is None:
            if hlos() == "linux":
                from .linux_runtime import OneShotKernel

                self._device = OneShotKernel(
                    self.kernel_name,
                    self.kernel_source,
                    self._param_list,
                    self._result_idx,
                    sdk=self.sdk,
                    dsp_arch=self.dsp_arch,
                    verbose=self.verbose,
                )
                self._run = self._device
            else:
                from .session import Session

                lowered = [(self.kernel_name, self.kernel_source, self._param_list, self._result_idx)]
                self._device = Session.of_lowered(lowered, sdk=self.sdk, dsp_arch=self.dsp_arch, verbose=self.verbose)
                self._run = self._device.kernels[0]
        return self._run

    def close(self) -> None:
        """Ends the kernel's session and releases the device; a call starts another."""
        device, self._device, self._run = getattr(self, "_device", None), None, None
        if device is not None:
            device.close()

    @classmethod
    def close_all(cls) -> None:
        """Closes every adapter of this process (e.g. between tests)."""
        for adapter in list(cls._live):
            adapter.close()

    def __del__(self):
        with contextlib.suppress(Exception):
            self.close()

    def _convert_torch_func(self):
        return lambda *args: self._open()(*args)
