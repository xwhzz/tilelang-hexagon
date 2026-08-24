# The Hexagon tilelang backend — summary & state

Goal: **author high-performance operators in the tilelang DSL and run edge LLMs (LFM2)
on the Qualcomm Hexagon NPU**, by swapping tilelang-generated kernels into llama.cpp's
`ggml-hexagon` backend. This is the top-level narrative; the detailed docs are linked at
the end.

Hexagon is **not a GPU**: a VLIW DSP + HVX (1024-bit / 128-byte vector unit, ~6 HW
threads) + HMX (one fp16 matrix engine, 32×32 Crouton tiles) + VTCM (8 MB software-managed
SRAM) + DMA. No SIMT, no automatic cache, no `mma` instruction. Most of the work below is
the consequence of that gulf.

## The stack (bottom to top)

| layer | what | where | status |
|---|---|---|---|
| **Codegen** | whole-register HVX vectorization — 1024-bit width + "narrowest dtype fills a register" rule + native `ext_vector` `vec_type` | `src/transform/loop_vectorize.cc`, `src/tl_templates/hexagon/common.h`, `codegen_hexagon.cc` | ✅ committed, zero regression |
| **DSL kernels** | fully-DSL q4_0 dequant (matches hand HVX) + fused dequant→VTCM→`T.gemm`→HMX matmul; compact resident scales | `examples/hexagon/example_qmatmul*.py` | ✅ device-correct (rel 2.5e-4) |
| **HMX abstraction** | `HMXIntrinEmitter` — native Crouton layouts, 32x32x32 MAC atoms, and the implicit-accumulator protocol used by explicit kernels and `T.gemm` lowering | `tilelang/hexagon/{hmx_intrin,gemm_hmx}.py` | emitter and native-layout `T.gemm` device-correct |
| **Runtime integration** | bridge (ride host VTCM+HMX), op registry (`tl_op_ctx`/`tl_dispatch`), the ggml intercept, embeddable-kernel emit | `src/tl_templates/hexagon/tl_{bridge,embed}.h`, `examples/hexagon/llama_cpp_integration/` | ✅ mechanism done |
| **Device** | LFM2-1.2B on the NPU, tilelang q4_0 matmul A/B'd inside the model | (below) | ✅ verified end-to-end |

## Device result — LFM2-1.2B on the NPU (A/B on this box)

Built llama.cpp `4fc4ec55` + the ggml-hexagon backend from a bare host (Android NDK +
Hexagon SDK 6.6, no Docker), applied the integration, flipped the `tl_mm_enabled` toggle
(rebuild skel, redeploy). Two versions of the op:

| version | prefill (short) | prefill (long, warm) | decode | output |
|---|---|---|---|---|
| stock | ~96–130 t/s | ~530 t/s | ~24–29 | coherent |
| tilelang, hand-C scalar dequant | 0.7 t/s | — | ~24 | coherent |
| **tilelang, generated whole-register dequant** | 4.3 t/s (repack-bound) | **537.7 t/s** | ~26 | coherent |

Flipping only the toggle, **both arms coherent** → the tilelang HMX op is **active AND
correct** inside LFM2's forward pass. The **fast path reaches PARITY** with the backend's
hand-tuned q4_0 matmul (537.7 vs 529.9 t/s at the same long prompt) — the predicted ceiling.
The generated whole-register dequant fixed the scalar hand-C (0.7 → parity); the short-prompt
4.3 is the **one-time repack** (ggml tile → column-major) charged to a single prefill, which
in real deployment belongs at model load. Decode is unchanged: the intercept is the HMX
prefill path (`hmx_mm_2d_f32`); decode (M=1) is a **HVX GEMV** the HMX-only intercept doesn't
touch. (Gotcha found: HVX whole-register loads from `malloc`'d DDR need ≥128B alignment —
`memalign(256,…)` for the repack cache, else silent garbage.)

## Why the Hexagon DSL looks different from CUDA (the hard-won findings)

1. **HVX has no sub-register ops.** Every instruction is a full 128-byte register, so a
   mixed-dtype loop must vectorize where the *narrowest* dtype fills whole registers (uint8
   needs ≥128 lanes), else it over-reads and faults. This one fact drove the codegen fix.
2. **HMX has no `mma` and no addressable accumulator.** Loading the a/b Crouton tiles
   (`mxmem`) *is* the MAC, into a single invisible per-core accumulator. The
   `HMXIntrinEmitter` therefore exposes `clear -> mma_atom/mma_tile -> convert -> store`.
   The `T.gemm` lowering receives only the three VTCM A/B/C buffers and composes those atoms
   plus its internal state; explicit K-streaming kernels can still interleave
   producer work with `mma_atom`. The honest CUDA divergence is the invisible accumulator
   (no `C_local` fragment).
   Result stores target each native C tile directly. The shared-memory planner propagates
   role-specific alignment (activation/output 2 KB, weight 128 B, config 256 B), and each
   32x32 FP16 output tile advances by exactly 2 KB.
3. **The Crouton pack is inherent.** `GemmHMX.infer_layout` now makes its VTCM operands
   native Crouton. A single logical `T.copy` can lower from a strided DDR matrix region
   directly to Crouton VTCM (and back); the helper uses HVX Crouton pack/unpack on the
   common 64-wide path and scalar fallbacks for transposed/non-64-wide forms. A future
   DMA lowering can stage the same explicit row-major region, or directly transfer
   prepacked Crouton DDR tiles.
4. **Tile ops must be `@T.macro`.** Buffer uses inside a plain Python helper are invisible
   to the VTCM liveness/arena pass, which then aliases `alloc_shared` buffers → silent
   garbage (not a crash). Emitter methods build+return a nested `@T.macro` (the tensor-core
   pattern) so the uses are visible.
5. **Measure compute, not single-call wall-clock.** A standalone kernel's `kernel()` call
   is dominated by FastRPC input marshaling; real compute only shows up with resident
   weights — i.e. *inside the model*. Twice we misdiagnosed perf from the marshal artifact.
6. **A single q4_0 matmul is a parity ceiling.** The backend already fuses HVX dequant into
   the HMX MAC. Matching it (K-streaming, whole-register dequant) reaches parity; **beating
   stock needs subgraph fusion** (keep FFN/attention intermediates in VTCM) or a
   fewer-byte quant — not re-implementing its q4_0 matmul.

## What's next

- **Fast device op (→ parity)** — **DONE** (task #30): the tilelang generated whole-register
  dequant + a one-time cached repack (ggml tile → column-major `qcm[K/2][N]` + `sc[K/32][N]`)
  reach **parity** with the backend (537.7 vs 529.9 t/s, coherent). Remaining polish: move the
  repack to **model load** (so cold prefill isn't repack-bound), and generate per-shape fully-DSL
  fused kernels for full coverage (today the whole-register *dequant* is generic over K/N; the
  fully-fused DSL kernel is fixed-shape).
- **Beat stock (→ win)** — subgraph fusion (own a fused FFN/attention block in one PD via
  the registry), or a quant format with fewer weight bytes. This is where tilelang's
  composability pays off, and it reuses the whole stack above.
- **Perf-hide the K-streaming dequant** — task #29: whole-register 64-feature dequant slices
  vs the 32-feature HMX tile + double-buffer, so the dequant hides under the MAC.

## Reproduce the device A/B (bare host, no Docker)

```bash
# env: Android NDK + Hexagon SDK 6.6 (HEXAGON_TOOLS_ROOT ends at .../19.0.07, no /Tools)
export ANDROID_NDK_ROOT=.../android-ndk-r25c HEXAGON_SDK_ROOT=.../Hexagon_SDK/6.6.0.0 \
       HEXAGON_TOOLS_ROOT=.../Hexagon_SDK/6.6.0.0/tools/HEXAGON_Tools/19.0.07
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp && git checkout 4fc4ec55
cp docs/backend/snapdragon/CMakeUserPresets.json .
cmake --preset arm64-android-snapdragon-release -B build-snap -DGGML_OPENCL=OFF
cmake --build build-snap --target llama-cli htp-v79 -j$(nproc)          # baseline

TL=/path/to/tilelang-hexagon                                            # integrate
cp $TL/examples/hexagon/llama_cpp_integration/tl_ggml_matmul.cc ggml/src/ggml-hexagon/htp/
git apply $TL/examples/hexagon/llama_cpp_integration/ggml-hexagon.patch # -I points at $TL
# A/B: set `int tl_mm_enabled` 0 or 1 in tl_ggml_matmul.cc, then:
cmake --build build-snap --target htp-v79 -j$(nproc)
adb push build-snap/ggml/src/ggml-hexagon/libggml-htp-v79.so /data/local/tmp/llamahtp/
adb shell "cd /data/local/tmp/llamahtp && LD_LIBRARY_PATH=. \
  ADSP_LIBRARY_PATH=.:/vendor/lib/rfsa/adsp:/dsp \
  ./llama-cli -m LFM2-1.2B-Q4_0.gguf --device HTP0 -ngl 99 -n 8 -st -p 'The capital of France is'"
```

## Detailed docs

- [`edge_tilelang_vs_cuda.md`](edge_tilelang_vs_cuda.md) — why swapping a tilelang op onto an edge runtime is harder than a CUDA callable.
- [`hexagon_dsl_kernels.md`](hexagon_dsl_kernels.md) — the HVX whole-register rule, the DSL q4_0 dequant, and the `HMXIntrinEmitter`.
- [`llama_cpp_integration.md`](llama_cpp_integration.md) — the bridge/registry/intercept, the embeddable-kernel emit, and the honest reflection on what's clean.
- [`lfm2_1.2b_compute_graph.md`](lfm2_1.2b_compute_graph.md) — the device-traced compute graph (shapes/types/placement).
- [`llama_cpp_reproduce.md`](llama_cpp_reproduce.md) / [`llama_cpp_inference_flow.md`](llama_cpp_inference_flow.md) — from-zero commands + the ggml-hexagon call chain.
