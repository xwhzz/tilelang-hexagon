"""Build, deploy and run Hexagon kernels.

Sessions (:mod:`.session`) run kernels built with the Hexagon SDK
(:mod:`.env`, :mod:`.build`). A resident device agent holds the FastRPC
connections and device buffers.

On Android and Linux, :class:`~.adapter.HexagonKernelAdapter` connects sessions
to the shared JIT layer: each compiled kernel uses a session containing one
kernel. :class:`~.linux_runtime.OneShotKernel` provides a Linux compatibility
path for one-shot execution.

Callers must serialize all operations on a session, including buffer operations.
"""
