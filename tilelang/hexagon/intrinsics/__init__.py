"""Hexagon instruction emitters and native layouts.

* :mod:`.crouton`: Crouton layouts of HMX operands in VTCM.
* :mod:`.hmx`: :class:`~.hmx.HMXIntrinEmitter` over the ``T.hmx`` atoms.
* :mod:`.panel`: panel layouts of the integer GEMM operands HVX ``vrmpy``
  reads, and the host packing of weights into them.

Quantized weight formats are model-level, not hardware primitives; the ggml
Q4_0/Q8_0 helpers live in ``examples/hexagon/quantize``.
"""
