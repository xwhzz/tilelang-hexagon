"""ggml quantization formats used by the Hexagon examples.

* :mod:`.q4_0`: HVX atoms that dequantize ggml-hexagon's native 576-byte Q4_0
  tiles into HMX weight Crouton tiles.
* :mod:`.q8_0`: the staged Q8_0 tile layout of ggml-hexagon and its HVX
  ``32x1`` dot atom.

These describe model weight formats rather than hardware primitives, so they
live with the kernels that use them, not in the backend. The atoms lower to
``tl_templates/hexagon/{qmatmul,qgemv}.h``.
"""
