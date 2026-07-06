# tilelang kernel embedded in llama.cpp (ggml-hexagon) — reference glue

The concrete **llama.cpp-side glue** that runs a tilelang q4_0 matmul *inside*
llama.cpp's `ggml-hexagon` NPU backend, via the reusable embedding API. It's the
worked example behind [`docs/llama_cpp_integration.md`](../../../docs/llama_cpp_integration.md)
— read that first; it explains the *why* (the two execution modes, the VTCM/HMX
bridge, the op registry) and reflects honestly on how general/clean this is.

The **reusable** half lives in the tilelang repo and is what you build on:
`src/tl_templates/hexagon/tl_embed.h` (C-safe op-context + op registry) and
`tl_bridge.h` (bind the tilelang runtime to a host skel's VTCM + ride its HMX
lock). This directory is only the host-specific glue.

## Files

| file | what |
|---|---|
| `tl_ggml_matmul.cc` | the integration TU: the op registry table + the q4_0 matmul op as a `tl_op_desc` (`matches`/`run`) that self-registers. Goes in `ggml/src/ggml-hexagon/htp/`. |
| `ggml-hexagon.patch` | the two upstream-file edits: a generic intercept in `matmul-ops.c` (`hmx_mm_2d_f32` → build a `tl_op_ctx`, call `tl_dispatch`) and the `CMakeLists.txt` additions (the `.cc` + `-I` the tilelang templates + `-Wno-unused-function`). |

Base: upstream **ggml-org/llama.cpp @ `4fc4ec5`** (the `GGML_HTP` Hexagon backend).
The patch is anchored to that tree; on a newer llama.cpp re-apply the two edits by
hand (they're small — see the patch).

## Apply

```bash
LCPP=/path/to/llama.cpp            # ggml-org/llama.cpp with the Hexagon backend
TL=/path/to/tilelang-hexagon       # this repo

cp tl_ggml_matmul.cc "$LCPP/ggml/src/ggml-hexagon/htp/"
git -C "$LCPP" apply /abs/path/to/ggml-hexagon.patch
# The patch hardcodes an -I to the tilelang templates; edit CMakeLists.txt so
# include_directories(...) points at "$TL/src" and "$TL/src/tl_templates/hexagon".
```

## Build + run + A/B

Build the Hexagon backend as usual (see llama.cpp `docs/backend/snapdragon/`), then
rebuild the DSP skel so it picks up the new TU:

```bash
# env: Android NDK + Hexagon SDK 6.6.x; HEXAGON_TOOLS_ROOT ends at .../19.0.07 (NO /Tools)
ninja -C build-snap htp-v79            # or your DSP arch target
adb push build-snap/ggml/src/ggml-hexagon/libggml-htp-v79.so /data/local/tmp/llamahtp/
adb shell "cd /data/local/tmp/llamahtp && LD_LIBRARY_PATH=. \
  ADSP_LIBRARY_PATH=.:/vendor/lib/rfsa/adsp:/dsp \
  ./llama-cli -m LFM2-1.2B-Q4_0.gguf --device HTP0 -ngl 99 -n 8 -st -p 'The capital of France is'"
```

**A/B**: `tl_ggml_matmul.cc` has `int tl_mm_enabled` (default `0` = all stock). Flip
to `1` and rebuild to route eligible q4_0 matmuls through the tilelang op. Verified
on LFM2-1.2B (Hexagon v79): flipping the toggle swings **prefill 156 → 0.8 t/s with
both outputs coherent** — proving the tilelang op is active *and* correct.

## Caveats (this is a correctness proof, not a fast path)

- **Slow by construction.** Scalar per-call dequant + M padded 1→32; ~99% of the
  time is the dequant. Only the *prefill* (HMX-path) matmuls are routed — decode
  (M=1) uses the backend's HVX GEMV, which this HMX-only intercept doesn't touch.
  See the perf/parity discussion in the doc.
- **Absolute `-I` paths** in the CMake edit (smell #5 in the doc) — point them at
  your tilelang checkout.
- **Op body is hand-C** that *calls* the tilelang HMX gemm runtime (`hmx.h`), not a
  `tilelang.compile` artifact. Closing that gap (an embeddable-kernel codegen
  target) is the main remaining work — see the doc's roadmap.
