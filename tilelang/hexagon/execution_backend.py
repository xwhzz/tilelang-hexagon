from __future__ import annotations

from tilelang.backend.execution_backend import ExecutionBackendSpec

from .runtime.config import deployment_cache_key

# HexagonKernelAdapter builds the generated device source with the Hexagon SDK,
# deploys it over FastRPC and launches it on the device. It needs neither host
# codegen nor in-process device compilation.
EXECUTION_BACKENDS = [
    ExecutionBackendSpec("hexagon", runtime_cache_key=deployment_cache_key),
]
