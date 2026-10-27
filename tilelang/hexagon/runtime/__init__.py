"""Execution side of the Hexagon backend: build, deploy and run kernels.

The Hexagon execution backend turns generated kernel source into a FastRPC
project (:mod:`.fastrpc`), builds it with the Hexagon SDK (:mod:`.env`,
:mod:`.build`), deploys it to a device (:mod:`.device`) and invokes it, either
one call at a time or through a persistent on-device agent (:mod:`.agent`).
:class:`~.adapter.HexagonKernelAdapter` connects this to the shared JIT layer.

The SDK, build and transport modules use only the standard library.
"""

from __future__ import annotations

from .build import FastRPCArtifacts, HexagonBuildError, build_cmake_fastrpc
from .device import AdbConnection, HexagonConnection, HexagonConnectionError
from .env import HexagonEnvError, HexagonSDK

__all__ = [
    "HexagonSDK",
    "HexagonEnvError",
    "HexagonConnection",
    "AdbConnection",
    "HexagonConnectionError",
    "build_cmake_fastrpc",
    "FastRPCArtifacts",
    "HexagonBuildError",
]
