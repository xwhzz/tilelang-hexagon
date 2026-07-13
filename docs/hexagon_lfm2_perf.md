# LFM2-1.2B on the Qualcomm Hexagon NPU — performance analysis & operator mapping

A complete, reproducible study of **LFM2-1.2B** running on the Qualcomm Hexagon NPU: how
fast it is (vs nexa/GenieX), which operators run where (HMX / HVX / scalar), what the
`npu-mobile` weights actually are, and every command to reproduce it. Companion deep-dive:
[`hexagon_profiling.md`](hexagon_profiling.md); interactive views:
[`hexagon_timeline.html`](hexagon_timeline.html) and
[`lfm2_perfetto_trace.json`](lfm2_perfetto_trace.json).

> **Note on an earlier revision.** A first pass of this profile undercounted: the parser's
> op-name regex (`\w+`) silently dropped ggml-hexagon's **fused** ops (`MUL_MAT+MUL_MAT`,
> `MUL_MAT+ADD`, `RMS_NORM+MUL` — the `+` broke the match), losing ~39 % of ops and
> inverting the hot-spot conclusion. All numbers below are the corrected full set (7994
> ops, 500 ms). The fix: match `[\w+]+`.

## 1. Setup

| | |
|---|---|
| **Device** | OnePlus PJZ110 · **Snapdragon 8 Elite (SM8750**, platform `sun`) · Android 15 |
| **NPU** | Hexagon **v79** — 6 HW threads, 6 HVX, **1 HMX**, 8 MB VTCM |
| **Model** | LFM2-1.2B — 16 layers (`n_embd` 2048, `n_ff` 8192, vocab 65536); **hybrid**: layers `[2,5,8,10,12,14]` attention, other 10 gated short-conv |
| **Runtime** | llama.cpp `4fc4ec55`, ggml-hexagon backend (stock skel `libggml-htp-v79.so`, `tl_mm_enabled=0`), Hexagon SDK 6.6.0.0 / Tools 19.0.07 / NDK r25c |

## 2. Performance — llama.cpp vs nexa @ 1024-token input

`llama-bench -p 1024 -n 128` (prefill 1024 tokens, then generate 128), 3 repeats, on `HTP0`:

| runtime | weights | prefill (pp1024) | decode (tg128) |
|---|---|---:|---:|
| **nexa / GenieX** (their reported figure, QNN NPU) | **8-bit (w8a16)** | **3618.4 t/s** | **69.5 t/s** |
| **llama.cpp ggml-hexagon** | **Q4_0 (4-bit)** | **3174.6 t/s** | **35.6 t/s** |
| llama.cpp ggml-hexagon | F16 (16-bit) | 2174.0 t/s | 22.1 t/s |

- **Prefill**: nexa **1.14×** llama.cpp-Q4_0 — close.
- **Decode**: nexa **1.95×** llama.cpp-Q4_0 — nearly 2×.

**The decode gap is the real story, and it is *not* weight bandwidth.** nexa's weights are
**8-bit (2× the bytes** of Q4_0's 4-bit), yet it decodes 2× faster. If decode were
weight-streaming-bound the 4-bit model would win. So llama.cpp's decode is bound by
something else — the HVX GEMV compute + 4-bit dequant overhead + per-op dispatch — and
nexa's QNN path (whole-graph compile, static scheduling, better kernel/engine use) beats it.
F16 is slowest (no HMX win at `ne1=1`, and 3.4× the bytes of Q4_0).

*Caveat: the nexa number is their published figure (we could not run nexa on-device — its
license failed to validate). llama.cpp numbers are measured here. The quantizations differ
(8-bit vs 4-bit), so this compares shipping configs, not identical math.*

## 3. What are the `npu-mobile` weights? — 8-bit, not bf16

The `NexaAI/LFM2-1.2B-npu-mobile` package on the device holds `weights-1-2.nexa` (730 MB) +
`weights-2-2.nexa` (537 MB) = **1.27 GB for 1.17 B params = 1.08 bytes/param = 8.7 bits/param**.

That rules out bfloat16 (which would be **2.34 GB**). It is **8-bit weights (w8a16** — 8-bit
weights, 16-bit activations, the standard QNN LLM export), with the ~0.7 extra bits being
per-channel scales + any fp16 embedding/norm tensors. The `.nexa` container is
compressed/opaque (only the magic `NEXAG` and QNN tensor names like `past_conv_8_out` are
readable), so the exact QNN datatype can't be extracted on-device, but the byte budget is
decisive: **not bf16, ~8-bit.**

## 4. Operator mapping — which ops run where

Captured with the backend's per-op profiler (`GGML_HEXAGON_PROFILE=2` — usec + cycles + 8
hardware PMU counters per op) over a prefill + 32-token decode. **7994 op instances, 500.3
ms of packed HTP compute.**

### 4a. NPU vs CPU

Everything in the transformer runs **on the NPU (HTP)**. The `supports-op` log + actual
`execute-op`/`profile-op` show these run on HTP:

`MUL_MAT` (and fused `MUL_MAT+MUL_MAT`, `MUL_MAT+ADD`, `MUL_MAT+MUL_MAT+MUL_MAT`),
`FLASH_ATTN_EXT`, `RMS_NORM+MUL`, `SSM_CONV`, `SWIGLU`, `CONCAT`, `CPY`, `ROPE`, `SET_ROWS`,
`MUL`, `ADD`, `SCALE`, `GET_ROWS`.

`PERMUTE` / `RESHAPE` / `VIEW` / `NONE` are **zero-cost layout ops** (no compute — they
reinterpret tensor metadata). Only the very final sampling runs on the CPU. **ggml-hexagon
fuses aggressively** (`OPFUSION=1`): `ffn_gate+ffn_up`, `ffn_down+residual-add`, the QKV
triple, and `rms_norm+weight-mul` are each a single fused HTP op.

### 4b. HMX vs HVX vs scalar — the answer to "what runs on scalar"

By the `kparams` kernel tag **and** the on-silicon `HVX_ACTIVE` PMU counter (event `0x100`;
ratio = HVX-active-cycles / total-cycles, can exceed 1 across the 6-thread worker-pool):

| engine | share | ops | evidence |
|---|---:|---|---|
| **HMX** (matrix) | **5.2 %** | batched **prefill** matmul (`hmx-tiled`) + attn (`hmx-pipe`) only | kparams `hmx-*` |
| **HVX** (vector) | **~94 %** | everything else | `hvx-tiled`/`hvx` + HVX_ACTIVE |
| **scalar** | **≈0 %** | none doing arithmetic | — |

Per-op, HVX_ACTIVE splits HVX into two regimes:

| op | HVX_ACTIVE/cyc | engine reading |
|---|---:|---|
| `MUL_MAT+MUL_MAT` (ffn gate+up) | 1.91× | HVX, saturated, multi-threaded |
| `CPY` | 2.01× | HVX, multi-threaded |
| `MUL_MAT+MUL_MAT+MUL_MAT` (QKV) | 1.88× | HVX, multi-threaded |
| `MUL_MAT+ADD` (ffn down) | 1.74× | HVX, multi-threaded |
| `MUL_MAT` (shortconv proj) | 1.73× | HVX, multi-threaded |
| `CONCAT` | 1.69× | HVX, multi-threaded |
| `FLASH_ATTN_EXT` | 1.07× | HVX |
| `SWIGLU` | 0.74× | HVX |
| `RMS_NORM+MUL` | 0.28× | HVX (light) |
| `SSM_CONV` | 0.21× | HVX (light) + memory |
| `ROPE` | 0.15× | overhead/memory-bound |
| `MUL` / `ADD` / `SCALE` | 0.05–0.11× | overhead-bound (tiny) |
| `SET_ROWS` (scatter) | 0.06× | memory |
| `GET_ROWS` (gather) | 0.00× | pure memory (scalar-thread indexing) |

**Answer to "what runs on scalar":** essentially **nothing runs arithmetic on the scalar
unit.** The only non-HVX/non-HMX work is memory movement — `GET_ROWS` (embedding gather) and
`SET_ROWS` (KV/conv-cache scatter) are scalar-thread-driven memory ops with ~0 HVX activity.
The tiny elementwise ops (`MUL`/`ADD`/`SCALE`/`ROPE`) *do* issue HVX vector instructions
(source uses `Q6_V*` intrinsics) but are **overhead-bound** — there's too little data for the
vector math to register against fixed per-op dispatch + memory latency. That fixed overhead
is what fusion removes (and ggml already fuses the big ones).

### 4c. Where the time goes (500.3 ms total)

| block | share | ops |
|---|---:|---|
| **FFN** | **~66 %** | `ffn_gate+ffn_up` 41.8 % + `ffn_down+add` 22.5 % + `SWIGLU` 1.6 % |
| **short-conv** | **~15 %** | `in_proj` 10.1 % + `out_proj` 3.5 % + `SSM_CONV` 1.1 % |
| **attention** | **~11 %** | `FLASH_ATTN` 5.4 % + QKV 3.0 % + `attn_output` 2.2 % |
| norm / glue | ~7 % | `RMS_NORM+MUL` 1.3 % + `CONCAT` 2.2 % + `CPY`/`ROPE`/rows/etc. |

**The FFN dominates (~two-thirds of compute)**, and it is *already fused* by ggml
(gate+up, down+add). The short-conv block is **not** fused — `in_proj → SSM_CONV →
out_proj` runs as three separate ops because the conv sits between the projections.

## 5. Implications for tilelang

1. **HMX is 95 % idle.** The matrix engine only fires on batched prefill; all of decode is
   HVX GEMV (`ne1=1`). The decode ceiling is memory-bound GEMV + dequant on HVX — exactly
   where nexa's QNN path is 2× faster.
2. **The FFN subgraph is already fused by ggml** (gate+up, down+add). Re-fusing it in
   tilelang wins little; the matmul *kernel* (HVX GEMV / 4-bit dequant) is the lever there.
3. **The short-conv block is the open fusion target** (~15 %, un-fused): a tilelang kernel
   that keeps `in_proj → SSM_CONV → out_proj` intermediates in VTCM would remove two
   DDR round-trips per conv layer, every token.
4. Single-op matmul parity is done; the gap to nexa (2× decode) is a **runtime/scheduling**
   gap (whole-graph compile, kernel quality), not an op-coverage gap.

## 6. Reproduce

Device deploy dir: `/data/local/tmp/llamahtp/` (skel = stock `libggml-htp-v79.so`,
models `LFM2-1.2B-{Q4_0,F16}.gguf`, `llama-bench`, `llama-cli`).

### Performance (§2)

```bash
adb shell "cd /data/local/tmp/llamahtp && LD_LIBRARY_PATH=. \
  ADSP_LIBRARY_PATH=.:/vendor/lib/rfsa/adsp:/dsp \
  ./llama-bench -m LFM2-1.2B-Q4_0.gguf -dev HTP0 -ngl 99 -p 1024 -n 128 -r 3"
# repeat with -m LFM2-1.2B-F16.gguf
```

### Per-op profile with PMU (§4)

```bash
adb shell "cd /data/local/tmp/llamahtp && LD_LIBRARY_PATH=. \
  ADSP_LIBRARY_PATH=.:/vendor/lib/rfsa/adsp:/dsp \
  GGML_HEXAGON_PROFILE=2 GGML_HEXAGON_VERBOSE=1 \
  ./llama-cli -m LFM2-1.2B-Q4_0.gguf --device HTP0 -ngl 99 -n 32 -st --verbose \
     -p '<a ~35-token prompt>' 2>&1" | tr -d '\r' > prof_pmu.log

# GGML_HEXAGON_PROFILE: 1=usec+cycles, 2=+8 PMU counters. --verbose REQUIRED
# (profile-op uses GGML_LOG_DEBUG). Log is non-ISO ASCII -> parse with grep -a / latin1.
# CRITICAL: op names can be fused ("MUL_MAT+MUL_MAT") -> match [\w+]+ , not \w+ .

python3 examples/hexagon/profiling/profile_lfm2.py prof_pmu.log      # per-op + role + engine tables
python3 examples/hexagon/profiling/timeline_export.py prof_pmu.log timeline.json   # nsys timeline data
python3 examples/hexagon/profiling/chrome_trace_export.py prof_pmu.log trace.json  # Perfetto JSON
```

### NPU-vs-CPU op split (§4a)

```bash
adb shell "cd /data/local/tmp/llamahtp && LD_LIBRARY_PATH=. \
  ADSP_LIBRARY_PATH=.:/vendor/lib/rfsa/adsp:/dsp GGML_HEXAGON_VERBOSE=1 \
  ./llama-cli -m LFM2-1.2B-Q4_0.gguf --device HTP0 -ngl 99 -n 2 -st --verbose -p Hi 2>&1" \
  | tr -d '\r' | grep -a 'supports-op'     # yes = HTP-accepted, no = CPU fallback
```

### PMU event decode (§4b)

ggml programs 8 Hexagon PMU events `{0x3, 0x111, 0x100, 0x105, 0x240, 0x256, 0x7D, 0x8C}`
(`opt_pmu_evt` in `ggml-hexagon.cpp`). Counter **[2] = `0x100` = `HVX_ACTIVE`**, [0] = `0x3`
= `COMMITTED_PKT_ANY` (names from Hexagon SDK `libs/itrace`, raw-opcode table). The `pmu[...]`
array in each `profile-op` line is in that counter order.

### Weight type (§3)

```bash
# on the phone's nexa cache (installed via Termux/proot):
adb shell "run-as com.termux sh -c 'ls -la <cache>/NexaAI/LFM2-1.2B-npu-mobile/*.nexa'"
# weights-*.nexa total / 1.17e9 params -> bytes/param  (1.08 = 8-bit, not bf16's 2.0)
```

### Interactive views

- `hexagon_timeline.html` — drag-zoom nsys-style timeline of all 7994 ops (engine lanes).
- `lfm2_perfetto_trace.json` — drag into [ui.perfetto.dev](https://ui.perfetto.dev): 3
  engine tracks + HVX_ACTIVE/committed-pkt/AXI counter graphs, per-op args.
