"""Execution side of the Hexagon backend: build, deploy and run kernels.

Kernels run in sessions (:mod:`.session`): their skel, built with the Hexagon
SDK (:mod:`.env`, :mod:`.build`), and an agent on the device that holds their
FastRPC session and resident buffers. :class:`~.adapter.HexagonKernelAdapter`
connects this to the shared JIT layer: a compiled kernel is a session of one.
On a Linux BSP (:mod:`.linux_runtime`), the adapter uses the resident Linux agent
and session. :class:`.linux_runtime.OneShotKernel` remains available as a
compatibility path.
"""
