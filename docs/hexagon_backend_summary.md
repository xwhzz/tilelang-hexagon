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
| **HMX abstraction** | `HMXIntrinEmitter` — instruction-atom emitter (`pack_a/pack_b`=ldmatrix, `mma`=mxmem+MAC, `store`=stmatrix, `clear`=mxclracc), the Hexagon analog of tilelang's `TensorCoreIntrinEmitter` | `tilelang/hexagon/hmx_intrin.py` | ✅ device-correct K-streaming |
| **Runtime integration** | bridge (ride host VTCM+HMX), op registry (`tl_op_ctx`/`tl_dispatch`), the ggml intercept, embeddable-kernel emit | `src/tl_templates/hexagon/tl_{bridge,embed}.h`, `examples/hexagon/llama_cpp_integration/` | ✅ mechanism done |
| **Device** | LFM2-1.2B on the NPU, tilelang q4_0 matmul A/B'd inside the model | (below) | ✅ verified end-to-end |

## Device result — LFM2-1.2B on the NPU (A/B on this box)

Built llama.cpp `4fc4ec55` + the ggml-hexagon backend from a bare host (Android NDK +
Hexagon SDK 6.6, no Docker), applied the integration, and flipped the `tl_mm_enabled`
toggle (rebuild skel, redeploy):

| arm | prefill | decode | output |
|---|---|---|---|
| stock | ~96–130 t/s | ~24–29 t/s | coherent |
| **tilelang op active** | **0.7 t/s** | ~24 t/s | coherent ("…Paris. Paris") |

Flipping only the toggle swings prefill with **both arms coherent** → the tilelang HMX op
is provably **active AND correct** inside LFM2's forward pass. Decode is unchanged: the
intercept is the HMX prefill path (`hmx_mm_2d_f32`); decode (M=1) is a **HVX GEMV** the
HMX-only intercept doesn't touch. The 0.7 is the current op's **scalar** per-call dequant
(~99% of the cost) — a correctness proof, not the fast path (see "what's next").

## Why the Hexagon DSL looks different from CUDA (the hard-won findings)

1. **HVX has no sub-register ops.** Every instruction is a full 128-byte register, so a
   mixed-dtype loop must vectorize where the *narrowest* dtype fills whole registers (uint8
   needs ≥128 lanes), else it over-reads and faults. This one fact drove the codegen fix.
2. **HMX has no `mma` and no addressable accumulator.** Loading the a/b Crouton tiles
   (`mxmem`) *is* the MAC, into a single invisible per-core accumulator. So a matmul is
   authored as instruction atoms (`clear → pack → mma → store`) around that implicit acc —
   an `HMXIntrinEmitter`, **not** a monolithic `T.gemm` — which is what lets a dequant
   interleave with the MAC (K-streaming). Modelled on tilelang's `TensorCoreIntrinEmitter`;
   the one honest divergence is the invisible accumulator (no `C_local` fragment).
3. **The Crouton pack is inherent.** Operands must be Crouton for `mxmem`, but the whole-
   register dequant must be row-major — so a row-major→Crouton HVX pack sits between them.
   `GemmHMX` deliberately keeps operands row-major + packs at runtime, because a Crouton-
   layout *DSL* store is scalar (would scalarize the dequant).
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

- **Make the device op fast (→ parity)** — task #30: replace the hand-C scalar dequant with
  the tilelang **generated** whole-register kernel (`emit_embeddable.py` already emits it),
  fed by a **one-time, cached** repack of ggml's tile format → the kernel's compact
  column-major format (`qcm[K/2][N]` + `sc[K/32][N]`), plus per-shape kernels + dispatch by
  manifest. The embeddable-kernel emit is done; the repack + shape-family wiring is the work.
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
