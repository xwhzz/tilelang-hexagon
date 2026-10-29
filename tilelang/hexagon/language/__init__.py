"""Hexagon language dialect: common TileLang plus Hexagon extensions.

``T.Kernel`` accepts ``num_workers``, ``T.dma_copy``/``T.dma_wait`` submit
explicit DMA transfers, and ``T.reduce_*`` reduce shared buffers and fragments
in place.
"""

from __future__ import annotations

from tilelang.language.common import *  # noqa: F401,F403
from tilelang.language.common import __all__ as _COMMON_ALL

from .dma import *  # noqa: F401,F403
from .dma import __all__ as _DMA_ALL

# The Hexagon dialect shadows T.Kernel with a version that accepts
# num_workers, and the reductions with in-place ones.
from .kernel import *  # noqa: F401,F403
from .kernel import __all__ as _KERNEL_ALL
from .reduce_op import *  # noqa: F401,F403
from .reduce_op import __all__ as _REDUCE_ALL

__tilelang_dialect__ = "hexagon"
__all__ = tuple(dict.fromkeys((*_COMMON_ALL, *_KERNEL_ALL, *_REDUCE_ALL, *_DMA_ALL)))

del _COMMON_ALL, _DMA_ALL, _KERNEL_ALL, _REDUCE_ALL
