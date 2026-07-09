# Writing high-performance DSL kernels on Hexagon (quantized matmul & the HVX rules)

This captures what it takes to express a **fully-DSL quantized matmul** (q4_0 dequant +
HMX gemm) in tilelang on Hexagon — the codegen support that makes the DSL lower to full
HVX, the non-obvious rules, the DSL-driven HMX K-loop, and an honest read on where the
performance actually lands. Companion to `edge_tilelang_vs_cuda.md` (why edge is hard) and
`llama_cpp_inference_flow.md` (the runtime).

## 1. The essential HVX rule: the narrowest dtype must fill a whole register

**HVX has no sub-register vector ops — every instruction works on a full 128-byte
register.** This one fact drives everything:

- A loop mixing a wide and a narrow dtype (e.g. fp16 output + uint8 weight) must vectorize
  at a width where the **narrowest** dtype is a whole number of 128-byte registers. uint8
  needs ≥128 lanes (128 B); fp16 needs ≥64 lanes; the binding constraint is the narrowest.
- If the width leaves a dtype sub-register (uint8 at 64 lanes = 64 B = half a register),
  the emitted op (`uint8_t64 & ...`) lowers to a full-register op that **over-reads and
  faults on device**.

The codegen enforces this (`src/transform/loop_vectorize.cc`, Plan): for Hexagon it uses a
**1024-bit** vector width (was 128-bit, a GPU inheritance) and then requires the final
vector size to be a multiple of `1024 / min_dtype_bits`; it bumps to that width when the
loop extent and buffer accesses allow, else **scalarizes (safe)**. Backing this,
`src/tl_templates/hexagon/common.h`'s `vec_type` is a native clang `ext_vector` (so
`a * b`, `a & mask` become real HVX ops, not scalar struct loops), and `PrintType` emits
vector spellings up to 128 lanes. Zero regression (matmul/rmsnorm/flash-attn pass); any
pointwise fp16 DSL kernel (silu/gelu/gating/bias/residual) now vectorizes to full HVX.

Practical: **DSL guidance** — for a loop that touches uint8, use `T.vectorized(≥128)` (a
multiple of the narrowest dtype's register lanes). A clean fp16 map (x*x) went 20 ms → 1.4 ms.

## 2. Writing the q4_0 dequant in the DSL (not a hand template)

The nibble decode is plain DSL arithmetic that lowers to HVX — see `examples/hexagon/
example_qmatmul.py`. Two non-obvious rules beyond §1:

```python
q  = T.Cast("int16", qcm[j, n])                        # widen to int16 FIRST
lo = (q & T.Cast("int16", 0xF)) - T.Cast("int16", 8)   # then mask/shift on int16
hi = (q >> T.Cast("int16", 4)) - T.Cast("int16", 8)    # int16 consts -> no int32 promotion
B[2*j, n]   = T.Cast("float16", lo) * scb[2*j, n]
B[2*j+1, n] = T.Cast("float16", hi) * scb[2*j+1, n]
```
- **Widen to int16 before the bitwise** — uint8 bitwise below full-register width faults;
  int16 at 64 lanes is a full register.
- **Keep the mask/shift constants int16** — `int16 & 0xF(int)` promotes to int32 in C,
  which caps the vector width to `1024/32 = 32`.
- **Pre-pack the weight column-major** (`qcm[K/2][N]`, `scb[K][N]`) so the dequant store is
  contiguous over N (no interleave/shuffle). The interleave/quant is *codec*, not a
  `layout` — it's computation, expressed here as DSL vector ops (like CUDA composes `mma`
  intrinsics, not scalar loops the vectorizer must magically widen).

Result: bit-exact vs numpy, and the DSL dequant **matches the hand-written HVX intrinsic
template** (~1.4 ms/512K weights) — so the template is not needed; the DSL generates it.

## 3. The fused matmul, and the DSL-driven HMX K-loop

`example_qmatmul.py` fuses dequant + `T.gemm` in one kernel: the dequantized weight lands in
VTCM (`alloc_shared`) and never hits DDR. `T.gemm` needs `clear_accum=True` on Hexagon (the
HMX accumulator must be `mxclracc`'d, else NaN).

For **K-streaming** (dequant one K-tile at a time, interleaved with the MAC so it hides under
the MAC — like ggml-hexagon), the HMX K-loop primitives are exposed in
`src/tl_templates/hexagon/hmx.h`: `tl_hexagon_hmx_{open,clear,mac,store,close,pack_a,pack_b,
unpack_c}`. A DSL kernel can drive `clear → (dequant B-tile; pack; MAC) ×K → store` itself.
**Status: the structure is device-validated correct** (a DSL-driven K-loop matmul and an
interleaved-dequant q4_0 matmul both pass, rel 2e-4). The perf/hiding optimization
(whole-register 64-feature dequant slices vs the 32-feature HMX tile; double-buffering;
worker-pool overlap) is **not yet done** — see §4.

## 4. Honest performance reality (read this before integrating)

- **Measure compute, not single-call wall-clock.** A Hexagon kernel's single `kernel()`
  call is dominated by FastRPC **marshaling** of the inputs; we twice misdiagnosed perf from
  it (a phantom "transposed-pack 55 ms" that was really marshaling). Always amortize
  (reps, or resident buffers) before reading a number as "compute".
- **HMX accumulates K in high precision.** The K-loop MACs each K-tile into the accumulator
  without clearing; fp16 conversion happens only at `cvt.hf = acc` readout. So "K can't be
  split" is wrong — K streams; the accumulator holds the running sum in high precision.
  (One physical accumulator, though, so K-parallel across workers isn't possible.)
- **A single q4_0 matmul is a parity ceiling.** The backend already fuses HVX dequant into
  the HMX MAC (dequant hidden under MAC). Our current fused kernel *materializes* the whole
  fp16 weight then gemms — the dequant is a serial, exposed pass (~0.75 GB/s materialize,
  slow), so integrating it as-is would **regress**, not win. K-streaming (§3, incomplete)
  hides the dequant → reaches **parity**. **Beating stock needs subgraph fusion** (e.g. FFN
  gate/up → SwiGLU → down keeping intermediates in VTCM) — the thing the backend does per-op
  with DDR round-trips. That is the real win and a separate, graph-level effort.
- In llama.cpp specifically, the intercept is the HMX path (`hmx_mm_2d_f32`); decode (M=1) is
  a GEMV on the HVX path and is **not** touched — so a single-op swap only affects prefill.

## 5. What's here / what's next

Device-validated, in-tree: the whole-register codegen fix (§1), the native `vec_type`, the
DSL q4_0 dequant + fused matmul (`example_qmatmul.py`), the exposed HMX K-loop primitives,
and the K-streaming *structure* (correct). Next, in priority order: (a) finish K-streaming
perf (hide the dequant → parity), or (b) subgraph fusion (→ beat stock); (c) make the
one-time weight repack a generated op; (d) a `T.dequantize` tile op wrapping the pattern.
