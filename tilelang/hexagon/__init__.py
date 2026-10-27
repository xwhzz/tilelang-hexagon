"""Qualcomm Hexagon NPU backend.

* :mod:`.language`: the Hexagon dialect, ``from tilelang.hexagon import language as T``.
* :mod:`.intrinsics`: Crouton layouts and HVX/HMX instruction emitters.
* :mod:`.op`: Hexagon tile-op implementations.
* :mod:`.pipeline`, :mod:`.codegen`, :mod:`.backend`: lowering and registration.
* :mod:`.runtime`: the execution backend (FastRPC build, deploy and run).
"""

from . import op  # noqa: F401
from . import backend as backend  # noqa: F401
