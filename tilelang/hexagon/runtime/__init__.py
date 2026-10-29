"""Execution side of the Hexagon backend: build, deploy and run kernels.

Kernels run in sessions (:mod:`.session`): their skel, built with the Hexagon
SDK (:mod:`.env`, :mod:`.build`), and an agent on the device that holds their
FastRPC session and resident buffers. :class:`~.adapter.HexagonKernelAdapter`
connects this to the shared JIT layer: a compiled kernel is a session of one.
On a Linux BSP (:mod:`.linux_runtime`), a kernel runs one call at a time.
"""
