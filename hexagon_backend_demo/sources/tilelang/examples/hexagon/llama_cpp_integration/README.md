# TileLang kernels embedded in llama.cpp / ggml-hexagon

Two reference integrations are kept here:

- **Q8_0 decode (current design):** replace the HVX `32x1` dot atom after ggml has
  quantized the activation and DMA-staged a weight tile into VTCM. TileLang owns the
  instruction atom; ggml retains DMA, VTCM, and six-worker scheduling.
- **Q4_0 prefill (HMX abstraction reference):** intercept `hmx_mm_2d_f32` and run
  a TileLang-scheduled HVX dequant atom plus instruction-level HMX atoms. The
  staged `M=32` path reaches stock performance parity on the tested model; shape
  coverage remains experimental.

The Q8 rationale and measured model A/B are in
[`docs/tilelang_hexagon_q8_design.md`](../../../docs/tilelang_hexagon_q8_design.md).

## Q8 Files

| file | purpose |
|---|---|
| `emit_qgemv_q8_0.py` | compile fixed K=2048/8192 TileLang PrimFuncs into embeddable C bodies and manifests |
| `kernel_qgemv_q8_0_k*.c` | generated `extern "C"` Hexagon kernels, with no FastRPC/session wrapper |
| `tl_ggml_qgemv.cc` | op registry and Q8 descriptor; dispatches to the generated kernels |
| `ggml-hexagon-q8.patch` | add the CMake source/include and a fallback-safe seam in `tiled_vec_dot_q8_0_32x1` |

The reusable implementation is in:

- `tilelang/hexagon/qgemv.py`: layout ABI + `Q8GemvIntrinEmitter`;
- `src/tl_templates/hexagon/qgemv.h`: v79 HVX signed-int8 dot atom;
- `src/tl_templates/hexagon/tl_embed.h`: C-safe embedding context/registry.

## Generate

```bash
/home/xwh/miniforge3/envs/tl/bin/python emit_qgemv_q8_0.py
```

The generated manifests record the exact source/staged layout sizes. Regenerate after
changing the emitter; do not hand-edit generated kernels.

## Apply Q8 Integration

Base: ggml-org/llama.cpp `4fc4ec5` with the experimental Hexagon backend.

```bash
LCPP=/path/to/llama.cpp
TL=/path/to/tilelang-hexagon

cp tl_ggml_qgemv.cc \
   kernel_qgemv_q8_0_k2048.c \
   kernel_qgemv_q8_0_k2048_nobias.c \
   kernel_qgemv_q8_0_k8192.c \
   kernel_qgemv_q8_0_k8192_nobias.c \
   "$LCPP/ggml/src/ggml-hexagon/htp/"
git -C "$LCPP" apply "$TL/examples/hexagon/llama_cpp_integration/ggml-hexagon-q8.patch"

/home/xwh/.local/cmake/bin/cmake --build /path/to/build-snap --target htp-v79 -j8
adb push /path/to/build-snap/ggml/src/ggml-hexagon/libggml-htp-v79.so \
  /data/local/tmp/llamahtp/
```

`TL_QGEMV_ENABLED` defaults to `1` in the example TU. Compile with
`-DTL_QGEMV_ENABLED=0`, or deploy an unpatched stock skel, for the fallback side of an
A/B. Unsupported dtype/K/tails return `-1` from the registry and execute stock code.

## Validate

Standalone atom correctness:

```bash
cd /path/to/tilelang-hexagon
/home/xwh/miniforge3/envs/tl/bin/python examples/hexagon/example_qgemv_q8_0.py --k 2048
/home/xwh/miniforge3/envs/tl/bin/python examples/hexagon/example_qgemv_q8_0.py --k 8192
```

Model A/B:

```bash
adb shell "cd /data/local/tmp/llamahtp && LD_LIBRARY_PATH=. \
  ADSP_LIBRARY_PATH=.:/vendor/lib/rfsa/adsp:/dsp \
  ./llama-bench -m LFM2-1.2B-Q8_0.gguf -dev HTP0 -ngl 99 \
  -p 1024 -n 128 -r 3"
```

Measured on Hexagon v79 with the 65536-row `lm_head` allowed on HTP:

| skel | pp1024 | tg128 |
|---|---:|---:|
| TileLang Q8 atom | 3300.53 ± 63.94 | 35.67 ± 0.37 |
| stock Q8 atom | 3331.28 ± 20.96 | 35.46 ± 0.17 |

This is integration and performance parity, not a speedup claim. The stock path already
uses 2D DMA, multi-buffer prefetch, VTCM, and six workers. The next useful atom is
`32x2`, which covers the dominant fused gate+up path.

## Q4 Explicit-HMX Prefill

This path is the reference for the low-level Q4/HMX abstraction. The TileLang
kernel owns the K loop and emits
`acquire/load_bias/clear/mma/convert/store/release` atoms instead of calling
`T.gemm`. `Q4HMXIntrinEmitter.dequant_tile` is deliberately one hardware-sized
operation: it expands one native 576-byte Q4_0 tile into one 32x32 FP16 weight
Crouton. The implementation in `src/tl_templates/hexagon/qmatmul.h` exposes the
scale loads, low/high nibble extraction, subtraction by 8, FP16 multiply,
interleave, and final HVX stores. `T.Layout` defines the activation, weight, and
output Crouton address maps. There is no row-major FP16 weight matrix, `pack_b`,
or persistent qcm/scale cache.

The optimized integration uses three generated stages:

1. Pack the FP32 activation into its Crouton once per `M=32` call.
2. Dequantize a caller-selected linear Q4 tile range into B Croutons.
3. Run the explicit HMX protocol and unpack FP32 output with a runtime row stride.

llama.cpp remains the scheduler and resource owner. It lends TileLang its existing
ordered DMA queue and worker pool through `tl_op_ctx`; TileLang does not create a
second persistent pool or acquire another FastRPC/VTCM session. A 2D DMA copies
each 576-byte raw tile into a 640-byte aligned VTCM row. Six workers then run
disjoint dequant ranges before the single HMX stage. For `N=256,K=8192`, A, B, C,
bias, and both raw weight stages remain below the device's 8 MB VTCM limit.

| file | purpose |
|---|---|
| `emit_embeddable.py` | emit complete baseline `M=32,N={128,512}` kernels and manifests |
| `emit_staged_qmatmul.py` | emit the `M=32,N=256` pack/dequant/compute stages |
| `kernel_qmatmul_hmx_atoms_*.cc` | generated full-kernel fallback family for K=2048/8192 |
| `kernel_qmatmul_hmx_staged_*.cc` | generated staged fast path for K=2048/8192 |
| `tl_ggml_matmul.cc` | shape dispatch, VTCM plan, 2D DMA, parallel dequant callback, HMX lock, and fallback |
| `ggml-hexagon.patch` | `hmx_mm_2d_f32` registry seam and CMake changes |

Regenerate the checked-in kernels without connecting a device:

```bash
/home/xwh/miniforge3/envs/tl/bin/python emit_embeddable.py --n 128 --k 2048
/home/xwh/miniforge3/envs/tl/bin/python emit_embeddable.py --n 128 --k 8192
/home/xwh/miniforge3/envs/tl/bin/python emit_embeddable.py --n 512 --k 2048
/home/xwh/miniforge3/envs/tl/bin/python emit_embeddable.py --n 512 --k 8192
/home/xwh/miniforge3/envs/tl/bin/python emit_staged_qmatmul.py --k 2048
/home/xwh/miniforge3/envs/tl/bin/python emit_staged_qmatmul.py --k 8192
```

The wrapper consumes ggml-hexagon's native `[N/32][K/32][576]` Q4_0 tiles. The
staged path is restricted to exactly `M=32`, `N` divisible by 256, and
`K=2048/8192`. Positive M values divisible by 32 can use the checked full-kernel
family in 32-row slices; nonmatching shapes or any failed stage return `-1` to the
stock backend. Bias/fused cases remain stock. This fixed-M guard matters because
the staged output ABI only writes 32 rows.

Apply the Q4 integration to a clean llama.cpp `4fc4ec5` tree:

```bash
LCPP=/path/to/llama.cpp
TL=/path/to/tilelang-hexagon
export TILELANG_SOURCE_DIR="$TL"

cp tl_ggml_matmul.cc \
   kernel_qmatmul_hmx_atoms_*.cc \
   kernel_qmatmul_hmx_staged_*.cc \
   "$LCPP/ggml/src/ggml-hexagon/htp/"
git -C "$LCPP" apply "$TL/examples/hexagon/llama_cpp_integration/ggml-hexagon.patch"
```

The patch is intentionally fallback-safe.  Its
`GGML_HEXAGON_TILELANG_Q4_HMX` CMake option defaults to `OFF`; configure with
`-DGGML_HEXAGON_TILELANG_Q4_HMX=ON` only after capturing the stock build for A/B
comparison.  `GGML_HEXAGON_TILELANG_SOURCE_DIR` may be passed explicitly instead
of exporting `TILELANG_SOURCE_DIR`.

The generated translation units are compiled with `-O2 -fno-lto`: keeping
their IR out of the whole-skeleton LTO link avoids multi-gigabyte linker memory
growth. The original 64-lane nibble expression lowered to `<64 x i32>` and made
Hexagon SelectionDAG codegen exceed 1 GB before failing. The native tile atom
keeps four independent 32-lane low/high chains and interleaves only after FP16
conversion, so codegen stays bounded while the compiler can overlap HVX work.

The corrected `cpos(k,n)` weight layout makes the raw
`[16 K-pairs][32 channels]` quant order naturally match one 64-lane Crouton
vector: each byte expands to adjacent low/high K lanes, so no matrix transpose is
required. The TileLang kernel makes the tile traversal and Crouton addresses
visible; the instruction atom keeps all four 128-byte quant vectors live, then
runs their extract, shuffle, unpack, multiply, and store chains. This is both a
native HVX scheduling boundary and a tractable lowering boundary for LLVM.

Standalone `32x128x512` and `32x512x512` device checks pass with relative errors
`4.4e-4` and `4.0e-4`. Deterministic `llama-cli` generation with prompt
`The capital of France is`, temperature 0, and seed 123 produced the same text as
stock: `The capital of France is Paris. Paris`.

LFM2-1.2B Q4_0 `llama-bench -p 32` on the same v79 device:

| v79 skel | pp32 |
|---|---:|
| full-kernel N=512 serial baseline | 127.39 tok/s |
| staged TileLang Q4/HMX, 5 runs | 616.78 +/- 29.77 tok/s |
| stock Q4_0, 5 runs | 604.36 +/- 36.28 tok/s |
| final rebuilt staged skel, 3 runs | 643.00 +/- 6.53 tok/s |

The staged result and stock are in the same performance tier; run-to-run variance
overlaps, so this is not evidence of a stable speedup. Operator profiles show the
main 2048x8192 matmul at 0.375-0.392 ms for staged versus 0.385-0.408 ms for
stock, down from about 5.02 ms in the serial baseline. `pp64` also completes
correctly, but intentionally uses the slower full-kernel fallback (`146.65 tok/s`
in a one-run guard check). Keep `GGML_HEXAGON_TILELANG_Q4_HMX=OFF` by default
until the schedule covers larger M and more fused shapes.

The Q4 integration and Q8 integration each define the registry owner, so do not compile
both TUs into one skel without first moving the registry table into a shared TU.
