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

## Which engine ran each op — HMX is mostly idle

The `kparams` field records the kernel that ran. The split is striking:

| op | kernels used |
|---|---|
| **MUL_MAT** | **`hvx-tiled` 64.3 ms (×680)**, `hmx-tiled` 15.3 ms (×68) |
| FLASH_ATTN_EXT | `hvx` 22.6 ms (×204), `hmx-pipe` 4.2 ms (×6) |
| SSM_CONV | scalar 5.7 ms | SWIGLU | scalar 7.9 ms |

**~80 % of matmul time runs on HVX (the vector unit), not HMX (the matrix engine).**
Reason: the short-conv projections are decode-time **GEMV** (activation `ne1=1`), which
ggml-hexagon routes to an HVX tiled kernel — the single HMX unit only handles the
batched (`ne1=35`) prefill FFN matmuls (`hmx-tiled`, 15.3 ms). So **the matrix engine is
mostly idle during decode**; decode is HVX-bound on the short-conv GEMVs.

Totalled across **all** ops (not just MUL_MAT), the engine split is the single sharpest
number in this profile:

| engine | total | % of compute | ops |
|---|---:|---:|---:|
| **HVX** (vector) | 86.9 ms | **59.5 %** | 884 |
| **scalar / DSP** | 39.6 ms | **27.1 %** | 3946 |
| **HMX** (matrix) | 19.5 ms | **13.4 %** | 74 |

**The HMX matrix engine — the thing the NPU is built around — carries only 13 % of the
work.** ~87 % of inference is vector + scalar. Any tilelang win has to move the needle on
the HVX GEMV path, or find a way to route the short-conv projections onto HMX.

## Interactive timeline

[`hexagon_timeline.html`](hexagon_timeline.html) is an nsys-style trace viewer for this
capture (open the file, or the published artifact): every one of the 4904 op instances is
a bar placed in execution order at its measured latency, colored by op type, split into
**HMX / HVX / scalar** lanes so the idle HMX lane is visible at a glance. Zoom/pan, a
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
