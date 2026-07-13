# LFM2-1.2B per-op profiling on Hexagon NPU

Detailed operator-level profile of **LFM2-1.2B-Q4_0** running on llama.cpp's
`ggml-hexagon` backend, captured with the backend's built-in per-op profiler
(`GGML_HEXAGON_PROFILE`). This tells us exactly which operators dominate — the input
to deciding what tilelang should target. Companion to
[`hexagon_backend_summary.md`](hexagon_backend_summary.md).

## Test setup

| | |
|---|---|
| **Device** | OnePlus PJZ110, **Snapdragon 8 Elite (SM8750**, platform `sun`), Android 15 |
| **CPU** | arm64-v8a, prime core 4.32 GHz |
| **NPU** | Hexagon **Arch v79** — 6 HW threads, 6 HVX, **1 HMX**, 8 MB VTCM (`HTP0 hwinfo`) |
| **Model** | LFM2-1.2B-Q4_0, 661 MiB, 1.17 B params. 16 layers, `n_embd` 2048, `n_ff` 8192, `n_head` 32 / `n_head_kv` 8, vocab 65536 |
| **Architecture** | **hybrid**: only layers `[2,5,8,10,12,14]` are attention (`n_head_kv=[0,0,8,0,0,8,…]`); the other 10 are **gated short convolution** (`shortconv`) |
| **Runtime** | llama.cpp `4fc4ec55`, ggml-htp-v79 skel (stock, `tl_mm_enabled=0`), built with Hexagon SDK 6.6.0.0 / Tools 19.0.07 / NDK r25c |
| **Workload** | ~35-token prompt (prefill) + 32 generated tokens (decode), `GGML_HEXAGON_PROFILE=2` (PMU) |

## Headline: per-op-type latency

4904 op instances, **145.9 ms total HTP compute** (prefill + 32 decode tokens):

| op | total | % | count | avg µs | avg cycles |
|---|---:|---:|---:|---:|---:|
| **MUL_MAT** | 79.5 ms | **54.5 %** | 748 | 106.3 | 225 754 |
| **FLASH_ATTN_EXT** | 26.8 ms | **18.4 %** | 210 | 127.7 | 271 782 |
| CONCAT | 10.8 ms | 7.4 % | 350 | 30.9 | 66 502 |
| SWIGLU | 7.9 ms | 5.4 % | 559 | 14.1 | 30 792 |
| SSM_CONV | 5.7 ms | 3.9 % | 350 | 16.4 | 35 704 |
| CPY | 3.7 ms | 2.5 % | 350 | 10.5 | 23 408 |
| ROPE | 3.3 ms | 2.3 % | 420 | 7.8 | 17 872 |
| SET_ROWS | 2.9 ms | 2.0 % | 420 | 7.0 | 16 574 |
| MUL | 2.7 ms | 1.9 % | 700 | 3.9 | 9 257 |
| GET_ROWS / ADD / SCALE | <1 % each | | | | |

**MUL_MAT + FLASH_ATTN = 73 %.** Everything else is small.

## MUL_MAT broken down by weight tensor — the real hot spot

This is the non-obvious result. Within MUL_MAT, the FFN projections are **not** the top —
the **short-conv projections dominate**:

| weight role | total | % of MUL_MAT | count | avg µs | dims (w × act) |
|---|---:|---:|---:|---:|---|
| **shortconv.in_proj** | 50.8 ms | **63.8 %** | 350 | 145.0 | `2048×6144 · 2048×1` |
| **shortconv.out_proj** | 17.4 ms | **21.8 %** | 350 | 49.6 | `2048×2048 · 2048×1` |
| ffn_up | 5.1 ms | 6.4 % | 15 | 340.1 | `2048×8192 · 2048×35` |
| ffn_gate | 5.1 ms | 6.4 % | 15 | 338.0 | `2048×8192 · 2048×35` |
| attn_q / attn_v / attn_k | <1 % each | | | | |

**85 % of all matmul time is the two short-conv projections** (`in_proj` produces the
6144-wide B·C·x gate/conv input; `out_proj` maps back to 2048). They're small per-call
but run **every layer × every token**, so they accumulate. The FFN projections have high
per-call cost but only 6 FFN-bearing layers × few calls in this trace.

## Which engine ran each op — HMX is idle, ~87 % is HVX

The profiler's `kparams` field only tags a kernel variant for the ops that *have* a
kernel-selection struct (`htp_mm_kernel_params`) — i.e. **matmul and flash-attn**, which
report `hvx-tiled` / `hmx-tiled` / `hmx-pipe`. Every other op prints `----`. **A blank
`kparams` does not mean the op ran on the scalar unit** — the ggml-hexagon HTP kernels for
add/mul/swiglu/conv/concat/cpy/rope are all HVX-vectorized (`hvx_*` / `Q6_V*` intrinsics on
the 6-thread worker-pool). Grouping by kernel tag gives three lanes:

| lane | total | % of compute | ops | what |
|---|---:|---:|---:|---|
| **HVX · matmul/attn** | 86.9 ms | **59.5 %** | 884 | `hvx-tiled` GEMV + `hvx` flash-attn |
| **HVX · elementwise** | 39.6 ms | **27.1 %** | 3946 | add/mul/swiglu/conv/concat/cpy/rope — untagged, still HVX |
| **HMX · matrix** | 19.5 ms | **13.4 %** | 74 | `hmx-tiled`/`hmx-pipe`, batched prefill only |

**The HMX matrix engine — the thing the NPU is built around — carries only 13 % of the
work; ~87 % is HVX.** HMX only fires on the batched (`ne1=35`) prefill matmuls; decode's
short-conv projections are `ne1=1` GEMV that route to `hvx-tiled` and never touch HMX.

### Hardware-PMU verification (on-silicon counters, not source-reading)

`PROFILE=2` also captures 8 Hexagon **PMU hardware counters** per op. Decoding the events
ggml programs (`{0x3, 0x111, 0x100, 0x105, 0x240, 0x256, 0x7D, 0x8C}`), **counter[2] =
`0x100` = `HVX_ACTIVE`** (HVX-active cycles; confirmed against the SDK's raw-opcode table
and `itrace` example). Per-op `HVX_ACTIVE / total-cycles`:

| op | HVX_ACTIVE / cycle | reading |
|---|---:|---|
| CPY | **2.01×** | HVX-bound, worker-pool spread across threads |
| MUL_MAT | **1.73×** | HVX-bound, multi-threaded |
| CONCAT | **1.69×** | HVX-bound |
| FLASH_ATTN_EXT | **1.07×** | HVX-bound |
| SWIGLU | 0.74× | mostly HVX |
| SSM_CONV | 0.21× | HVX + memory/setup |
| ROPE / ADD / MUL / SCALE / rows | 0.00–0.15× | **overhead-bound**, not scalar-compute |

(ratio > 1 because counters sum across the up-to-6 worker-pool HVX contexts.) The heavy ops
genuinely saturate HVX. The tiny glue ops read low **not because they run on a scalar ALU**
— the source proves HVX intrinsics — but because they're **overhead-bound**: FastRPC
dispatch + memory latency + address setup dwarf the handful of vector packets. That fixed
per-op cost is precisely what operator **fusion** eliminates, which sharpens the fusion
argument rather than pointing at a scalar bottleneck.

## Proper NPU profilers (beyond ggml's built-in one)

ggml-hexagon's `GGML_HEXAGON_PROFILE` is homegrown (its ambiguous `kparams` field is what
mislabeled the "scalar" lane above). The Qualcomm stack ships real profilers; which one
applies depends on the runtime:

| tool | granularity | output | applies to our LFM2/llama.cpp path? |
|---|---|---|---|
| **itrace** (Hexagon SDK `libs/itrace`) | per-DSP-section or sampled, PMU events | **Chrome-trace / Perfetto JSON** timeline | Partially — see below; its *sampling* mode misattributes on Hexagon |
| **Snapdragon Profiler** | system-level DSP/HMX/HVX util, clocks, thermal | GUI/CSV, real-time over ADB | **Yes**, for utilization/clock/thermal — not per-op graph timing |
| **QNN `qnn-profile-viewer`** (`--profiling_level detailed`) | per-op on HTP, cycle-accurate | text/CSV | **No** — QNN-graph only (our Qwen3-4B QNN path, not llama.cpp) |
| **Qualcomm AI Hub** profiling job | per-layer, official, unit breakdown | web report | **No** — needs an AI-Hub-compilable model (LFM2 unsupported) |
| **ETM** (`HAP_user_etm_enable`, `GGML_HEXAGON_ETM`) | cycle-accurate instruction trace | binary trace, heavy post-processing | possible but heavyweight |

Notably ggml's default event set has **no HMX counter** — to measure HMX occupancy directly
we'd add an HMX event to `opt_pmu_evt`.

### itrace attempted on-device — and why in-context per-op beats it

We deployed itrace's full runtime to the device (all prebuilt for our exact target:
`android_aarch64` host libs + `hexagon_toolv19_v79` DSP skel + libperfetto/libprotobuf) and
drove it via the **zero-recompile automated constructor** (`LD_PRELOAD=libitrace_constructor.so`
+ `itrace_config.txt`). It got most of the way: loaded, parsed the config, and **identified
every event** (`HVX_ACTIVE=0x80cc`, `COMMITTED_PKT_ANY`, `AXI_*`) on the CDSP domain — then
**null-deref SIGSEGV** (`fault addr 0x0`) at the first setup action *after* config parse,
because the constructor runs at **library-load time (before `main`)**, before the process's
FastRPC subsystem is initialized. The clean fix is explicit itrace init after `main` (a
llama.cpp rebuild linking libitrace).

But the deeper finding made that rebuild not worth it: **Hexagon PMU counters (`upmucnt`) are
per-hardware-thread.** itrace's periodic-sampling mode runs its reader in its *own* PD/thread,
so it samples the counters of threads that *aren't* running ggml's ops — wrong attribution.
ggml's `PROFILE=2` reads the same counters **in-context, around each op, on the executing
thread** — the *correct* per-op methodology, and the data we already have. So itrace's
sampling would be strictly worse here; itrace's value is a CPU+DSP *system* view, not per-op.

### Deliverable: a real Perfetto trace from the in-context PMU

[`lfm2_perfetto_trace.json`](lfm2_perfetto_trace.json) is the **Chrome-trace / Perfetto JSON**
itrace's `ITRACE_JSON_FILE` would emit — but built from ggml's correctly-attributed per-op PMU
capture (`chrome_trace_export.py`). **Open it at [ui.perfetto.dev](https://ui.perfetto.dev)**
(drag the file) or `chrome://tracing`. It has all 4904 ops as duration bars on three engine
tracks (HMX / HVX-matmul / HVX-elementwise) with per-op args (layer, shape, dtype, kernel,
cycles, `HVX_ACTIVE`, committed packets), plus three **hardware counter line-graphs**
(`HVX_ACTIVE`, `committed_pkts`, `AXI_write_req`) sampled per op — richer than itrace's 500 µs
sampling because it's per-op, not time-sampled.

## Interactive timeline

[`hexagon_timeline.html`](hexagon_timeline.html) is an nsys-style trace viewer for this
capture (open the file, or the published artifact): every one of the 4904 op instances is
a bar placed in execution order at its measured latency, colored by op type, split into
**HMX / HVX-matmul / HVX-elementwise** lanes so the idle HMX lane is visible at a glance. Zoom/pan, a
phase ribbon marking the 36 forward passes (2 warmup + 1 prefill@35tok + 31 decode@1tok),
a minimap, and per-bar tooltips (layer, role, shape, dtype). Built from the same log by
`timeline_export.py`. The timeline is **packed by compute time** — the per-op `start`
cycle-counter is a per-core free-running counter (6 HW threads + op batching), not a
shared clock, so ~half the timestamps run backwards and can't place ops on a wall-clock;
laying measured durations end-to-end in ggml execution order gives the true compute
sequence without host-logging gaps.

## Timeline

Op instances in execution order, 20 buckets (`#` = relative time in bucket):

```
[    0-  245]   6855us  MUL_MAT   ##########          <- prefill (batched FFN, HMX)
[  245-  490]  25902us  MUL_MAT   ####################  <- prefill peak
[  490-  735]   9404us  MUL_MAT   ##############
[  735- 4900]  ~6100us each       #########            <- 32 decode steps, steady
```

Prefill is a short compute-heavy burst (HMX-tiled FFN matmuls); the 32 decode steps are a
long steady tail of ~6.1 ms each, dominated by the per-layer short-conv GEMVs on HVX.

## Top single-instance costs

```
728us FLASH_ATTN_EXT  cache_k_l8    64:35:32 x 64:256:8   (prefill attention, 8 kv-heads)
707us FLASH_ATTN_EXT  cache_k_l2    ...
359us MUL_MAT         blk.8.ffn_gate  2048:8192 x 2048:35 (prefill FFN, HMX)
348us MUL_MAT         blk.10.ffn_up   2048:8192 x 2048:35
```

## What this means for tilelang

1. **MUL_MAT is the top target (54 %) — but the volume is short-conv GEMV on HVX, not
   HMX FFN.** Our tilelang q4_0 matmul targets the HMX path; that's only ~10 % of MUL_MAT
   time here. The bigger prize is the `hvx-tiled` GEMV that runs every layer every token.
2. **The HMX matrix engine is idle ~90 % of the time** (only the batched prefill FFN uses
   it). Decode is HVX-bound. A win needs to speed up the HVX GEMV path or find work for HMX.
3. **FFN is a fusable subgraph** (`ffn_gate` + `ffn_up` → SWIGLU → `ffn_down`, 6.4+6.4+5.4 %),
   and the short-conv block (`in_proj` → SSM_CONV → `out_proj`, 63.8+3.9+21.8 % of MM/conv)
   is an even bigger fusable unit — both keep intermediates in VTCM instead of DDR.
4. Confirms the strategic read: single-op parity isn't the win; **subgraph fusion is** —
   and the biggest subgraph is LFM2's short-conv block, not the FFN.

## Reproduce

```bash
# --- device/model already deployed at /data/local/tmp/llamahtp/ ---
# (skel = stock ggml-htp-v79, tl_mm_enabled=0)

adb shell "cd /data/local/tmp/llamahtp && LD_LIBRARY_PATH=. \
  ADSP_LIBRARY_PATH=.:/vendor/lib/rfsa/adsp:/dsp \
  GGML_HEXAGON_PROFILE=2 GGML_HEXAGON_VERBOSE=1 \
  ./llama-cli -m LFM2-1.2B-Q4_0.gguf --device HTP0 -ngl 99 -n 32 -st --verbose \
     -p '<a ~35-token prompt>' 2>&1" | tr -d '\r' > prof_pmu.log

# GGML_HEXAGON_PROFILE: 1=usec+cycles, 2=+8 PMU counters
# --verbose is REQUIRED (profile-op uses GGML_LOG_DEBUG); the log is
# non-ISO ASCII, so parse/grep with `grep -a` / latin1.

python3 profile_lfm2.py prof_pmu.log     # (in scratchpad-llamacpp/)
```

`profile_lfm2.py` parses each `profile-op OP|names|dims|types|strides|kparams|usec N
cycles C start S mhz M pmu [...]` line and emits: per-op-type latency, MUL_MAT-by-weight,
kernel(HMX/HVX) split, a timeline, and the slowest instances.

### NPU hwinfo banner (for the record)

```
ggml-hex: Hexagon Arch version v79
ggml-hex: HTP0 hwinfo: threads 6, hvx 6, hmx 1, vtcm 8 MB
ggml-hex: HTP0 op batching: n-bufs 16 n-tensors 7168 n-ops 1024 vmem 3355443200
```
