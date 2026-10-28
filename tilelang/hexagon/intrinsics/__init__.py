"""Hexagon instruction emitters and native layouts.

* :mod:`.crouton`: Crouton layouts of HMX operands in VTCM.
* :mod:`.hmx`: :class:`~.hmx.HMXIntrinEmitter` over the ``T.hmx`` atoms.
* :mod:`.panel`: panel layouts of the integer GEMM operands HVX ``vrmpy``
  reads, and the host packing of weights into them.

Quantized weight formats are model-level, not hardware primitives: a model
packs its codes into a panel layout itself (``examples/hexagon/lfm2``).
"""
