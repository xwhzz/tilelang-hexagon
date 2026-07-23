"""tilelang Hexagon NPU backend — build/deploy/execute harness (layer 5).

This subpackage owns the *remote-execution* side of the Hexagon backend: SDK and
toolchain discovery (:mod:`.env`), the swappable on-device transport
(:mod:`.device`), and FastRPC build orchestration (:mod:`.build`).  It is
deliberately import-light (stdlib only) and independent of tilelang's native
library, so the deploy/run path can be developed and tested on its own.

The codegen / pass-pipeline / adapter wiring (layers 2-4) lands in sibling
modules and in ``src/hexagon`` as those milestones are implemented.
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
