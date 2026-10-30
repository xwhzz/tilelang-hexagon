# How LFM2 and LFM3 got fast on Hexagon

This is the record of the optimization work on the LFM2-1.2B example, and then
on LFM3-1.7B in fp16 ([below](#lfm3-17b-in-fp16)): what each step bought,
measured on the device; what was tried and did not help; where the time goes
now; and what is left. The [README](README.md) describes the kernels as they
are.

All numbers are from the v79 cDSP of a Snapdragon 8 Elite (SM8750) phone,
batch 1, with the bundle of #106 (4-bit projections with an fp16 scale per 64
columns, an 8-bit tied head: 685.2 MB streamed a token). The DSP runs at about
2.06 GHz under a session's clock vote. MB and GB are 10^6 and 10^9 bytes.
"#N" is a pull request of this repository.

## Where it stands

| | decode, tg128 | prefill, pp1024 |
|---|---:|---:|
| **this example** | **80.1-81.2 t/s** (12.3-12.5 ms a token) | **176.2 ms, 5813 t/s** |
| roofline | 83 t/s (DMA, 57 GB/s) | about 129 ms (the HMX work alone) |
| NexaML on the NPU, LFM2.5-1.2B, in [Liquid AI's model card](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct) | 82 t/s | 4391 t/s (1K tokens) |
| nexa's demo app (NexaSDK, QNN NPU), measured on this phone | 65-68 t/s | 1.1-1.5k t/s (40 tokens) |
| llama.cpp ggml-hexagon Q4_0, measured on this phone | 35.6 t/s | 3175 t/s |

The decode runs at 96-98% of the DMA roofline. The prefill's GEMM phases are
close to the HMX limit; most of what remains is HVX work that HMX waits for.

The model card measures NexaML on a ROG Phone 9 Pro, which has the same chip,
with a 1K-token prompt and 100 new tokens, and does not say how it times
them. LFM2.5-1.2B has LFM2-1.2B's architecture and other weights. Against
those numbers, this example decodes 1-2% slower and prefills 1.3x faster.

Quality: WikiText-2 perplexity on the device is 16.684, against 16.683 for
the host reference with the device's arithmetic and 16.29 for bf16; the gap
to bf16 is the quantization's. The prefill's keys and values match the
decode's, a token a step, to 4.3e-3 at most, and the tokens generated after a
prefill equal the decode's.

## How it was measured

- **Benchmarks.** `bench.py`: the median of 5 runs. A run is one kernel
  call, timed on the phone around its FastRPC call
  (`Session.last_run_seconds`): the call's overhead is included; tokenization
  and the round trip to the PC that drives the phone are not. Two runs of the
  same kernels differ by about 1.5% on the phone: pp1024 measured 176.2 ms in
  one and 173.5 ms in another.
- **A/B in one session.** For a change of a few percent, the old and the new
  kernel run in one session, alternating, and the median of 7 or 9 runs
  compares them. Two identical kernels differ by up to 2 ms (1%) at pp1024
  this way, so smaller gains need the cycle profile below.
- **Cycles per phase.** A copy of the kernel with a `HAP_perf_get_pcycles`
  mark after each phase and each barrier, summed per worker. The workers'
  totals agree to 0.1%, and runs agree to about 0.5%: the steadiest
  measure.
- **Correctness gates.**
  - The tiny model (`tiny.py`) against the reference, in the device tests.
  - On LFM2-1.2B, the prefill's KV cache against the decode's (a token a
    step) and the 16 tokens generated after it.
  - `ppl.py` on the device against `quantize.py --eval` on the host.

## Decode: at the memory roofline

A decode step reads every weight once: at 685.2 MB a token, it is bound by
how fast weights reach the DSP. DMA tops out near 57 GB/s from 3 workers on,
whatever the block size or depth (83 t/s for this bundle). HVX loads alone reach about
37 GB/s, and DMA and HVX together 58-63. So the decode streams the weights by
DMA and keeps everything else out of the stream's way.

- **The GEMV** (#97, #99). A block's GEMV multiplies the int16-quantized input
  by the 4-bit codes with `vrmpy`, exactly in int32 per group of 64 columns,
  and weighs the sums by the scales into fp32 once per group. One fused call
  per block is 1.99x faster than one `T.gemm` per group plus an epilogue
  (int4 128 x 2048 on one thread: 16807 -> 8436 cycles). A GEMV of
  2048 -> 6144 streams 50.9 GB/s on 6 workers.
- **GEMMs out of line** (#99). Inlined at every call site, the GEMMs of a
  megakernel ran from a cold instruction cache in every phase. As functions of
  their own, eight chained GEMVs on 4 workers went from 572 to 335 µs
  (31 -> 53 GB/s), against 326 µs for the DMA alone.
- **One call per generation** (#107). The loop over tokens runs on the DSP.
  Four worker programs meet at `T.sync_grid()` between a token's phases. Each
  owns matching shares of every layer, so nothing else needs a barrier. Each
  worker's weights stream by DMA through a ring of VTCM slots three blocks
  ahead, across phases and from one token into the next.
- **Sessions** (#101). The bundle, the KV cache and the conv state stay on the
  device. Coalescing the agents' messages took a standalone call from 46 to
  4 ms, and a session call with resident arguments takes 2.4 ms.
- **Compile time.** Lowering the decode megakernel took 438 s before layout
  inference stopped at its first root on one thread (#105), then 49 s. A
  session now starts from the cached lowered source: 36.5 -> 2.0 s (#130).

The result, unchanged since #107 within noise: 81.2 t/s (12.3 ms a token),
55.6 GB/s of weights, 98% of the DMA roofline. What is left is in the
bundle's size, not the kernel (see [What is next](#what-is-next)).

## Prefill: from 12.6 s to 176 ms

Before #128 the prefill ran one decode step a token: at 12.3 ms a step,
about 12.6 s for 1024 tokens. The HMX prefill runs a prompt in chunks of 128 rows. Each block of
128 weight rows is dequantized once by HVX for all 128 rows and multiplied on
HMX, and the four workers take turns at HMX a 32 x 32 output tile at a time.

| step | pp1024 | measured as |
|---|---:|---|
| HMX prefill under development | 280 ms | bench |
| #127 heavy maps unroll by two, short rows unroll (SwiGLU 126 -> 56 cycles a vector) | 243 ms | bench |
| #128 the prefill lands | 217.3 ms (4713 t/s) | bench |
| #135 attention in blocks of 128 keys, not 64 | 217.4 -> 207.9 ms | bench |
| #134 a pool GEMM's epilogue runs under the worker's own MACs | 207.9 -> 198.5 ms | bench |
| #137 fp16 exp, exp2 and sigmoid in qf16, a register at a time | 198.9 -> 189.4 ms | bench |
| #140 worker programs bind their buffers at entry (no aliasing) | 191.6 -> 189.6 ms | A/B |
| #141 the norms' weights in VTCM | 189.6 -> 188.7 ms | A/B |
| #150 guarded loads select instead of branching | 190.8 -> 188.2 ms | A/B |
| #151 the FFN's parts meet once each, two slots of hidden units | 188.2 -> 183.8 ms | A/B |
| #154 keys and values by DMA into VTCM, a block ahead; softmax in base 2 | 184.6 -> 180.8 ms | A/B |
| #163 the attention's scores stay in HMX's fp16 tiles | 373.7M -> 370.3-371.5M cycles | profile |
| #166 the RoPE tables' rows in VTCM | 370.3M -> 367.4M cycles | profile |
| #167 programs waiting at a barrier pause between polls | 367.5M -> 365.6M cycles | profile |
| today | 176.2 ms (5813 t/s) | bench |

Each A/B pair ran in a session of its own, so its absolute times do not
compare across rows; the bench rows do. The "profile" rows are cycles per
worker.

Some steps made the kernel fit or compile rather than run faster:

- **VTCM scratch** (#115). Operators' VTCM scratch is reused across call
  sites: a 3-layer prefill's worker arena went from 5.5 MB to 0.96 MB.
- **Lowering time** (#117, #123). The vectorizer's legalization analyzes each
  loop in its own scope: with 16 layers it took 138 s of a 226 s lowering,
  and now takes 5.4 s. With the block-scaled GEMM out of line, the 16-layer
  source went from 22K to 8.3K lines.
- **Shared activations** (#113). VTCM buffers shared by the programs of a
  launch hold a chunk's activations once, not once per worker.
- **One loop over the layers** (#128). The layers are a loop at run time that
  reads each layer's kind from a table: 1K lines of C, lowered in 3 s.

### What each step was about

- **HMX's turns** (#134). A worker held HMX for its tile's MACs and did
  nothing meanwhile; its epilogue and its next slice of dequantization came
  after. Running the previous tile's epilogue under its own MACs took 4
  workers from 8.96 to 8.55 cycles per 32 x 32 x 32, against an HMX rate of
  8.1.
- **Math in fp16** (#137). Softmax and SwiGLU results are fp16, which HMX takes.
  Computing exp, exp2 and sigmoid in qf16 on 64 lanes instead of widening to
  two fp32 registers took exp from 46.1 to 20.4 cycles a register, exp2 from
  42.1 to 14.2 and sigmoid from 67.2 to 28.7. Each stays within its fp16
  result's precision.
- **Software pipelining** (#127, #140, #150). LLVM pipelines an HVX loop only
  when three conditions hold:
  - **One basic block.** The bounds checks became selects (#150).
  - **Provably distinct pointers.** Worker lambdas captured buffers by
    reference, and an alignment test captured their pointers. Binding them at
    entry as distinct objects (#140) took the norms' scaling from 12.4M to
    8.2M cycles.
  - **An initiation interval of at most 27.** Heavy maps (sigmoid's chain is
    47) unroll by two (#127).
- **DDR out of the HVX loops** (#141, #154, #166). An HVX load from DDR goes
  through L2 and stalls a pipelined loop on each miss. What a chunk reads
  repeatedly now comes into VTCM first:
  - the norms' weights: 8.2M -> 5.9M cycles;
  - the blocks of keys and values, by DMA: 9.5M -> 2.9M;
  - the chunk's rows of the RoPE tables: 7.2M -> 3.2M.
- **Fewer barriers** (#151). A barrier makes the workers start their next
  phase together, and HMX then waits out their first dequantization. With two
  slots of hidden units, a layer has 9 barriers instead of 12.
- **Fewer conversions** (#163). The scores are fp16 when HMX stores them. They
  now stay in HMX's tiles, where the mask, the row maxima and exp2 work, and
  P goes straight into the tiles P V reads. The scores' epilogue went from
  6.0M to 0.7M cycles, and P's copy disappeared.

## Where the prefill's time goes now

Cycles per worker at 1024 tokens (365.6M in all, about 177 ms). Each GEMM
phase's HMX bound is its 32 x 32 x 32 multiplies at 8.1 cycles each, for the
four workers together:

| phase | Mcycles | HMX bound | HMX busy |
|---|---:|---:|---:|
| FFN w1 / w3, and SwiGLU | 163.1 | 135.9 | 83% |
| FFN w2 | 69.5 | 68.0 | 98% |
| short conv in_proj | 38.4 | 31.8 | 83% |
| out_proj | 20.0 | 17.0 | 85% |
| q, k, v, and the q/k norms and RoPE | 15.6 | 9.6 | 61% |
| attention | 32.0 | 3.6 | 11% |
| RMSNorm | 14.4 | | |
| barriers | 11.6 | | |
| embedding, stream prologues | 1.1 | | |
| **total** | **365.6** | **265.9** | **73%** |

- **The HMX bound.** All of the prefill's HMX work takes 266M cycles. If
  everything else hid behind it, pp1024 would take about 129 ms (7900 t/s).
- **SwiGLU does not hide** in the w1/w3 phase. Removing it took that phase
  from 163.1M to 134.6M cycles. By an estimate from the loops' instruction
  counts, a round of four block pairs (one a worker) needs about 65k packets
  of HVX work: each worker's dequantization about 10k, SwiGLU 5k, epilogues
  1.3k. That is against 68k cycles of HMX, so the core's issue slots are
  nearly full.
- **The attention is HVX work.** Of its 32M cycles, exp2 takes 12.8M and the
  row sums 4.4M; HMX takes under 4M.
- **The barriers are mostly the first worker waiting.** It finishes its GEMM
  phases ahead of the others (w1/w3: 154M against 166M cycles), and waits
  28M cycles in all, the others 5-7M each.

## What did not work

Each of these was measured on the device and reverted:

| tried | measured |
|---|---|
| Dequantizing a pool GEMM's weights under the worker's own MACs (with #134) | slows the MACs: 9.8 cycles per 32 x 32 x 32, against 8.55 |
| `T.gemm` out of line, as `T.gemm_blockscaled` is (#149) | small calls 4-6% slower; kept inline |
| exp2 in place over VTCM, writing P over the scores (#163) | 15.9M cycles, against 11.6M into another buffer |
| The softmax's row sums by HMX, from 32 columns of ones after V | as costly as the Crouton reduction they replace, in more VTCM |
| SwiGLU over HMX's fp16 tiles instead of fp32 rows | w1/w3 163.1M -> 168.8M cycles: the map costs more over tiles (32.6M against 27.5M), though it has fewer instructions; why is not known (VTCM bank conflicts are a guess) |
| Two output tiles a worker's turn at HMX | GEMM phases 10% slower: the epilogue no longer runs under the MACs |
| Pausing in the DMA queue's wait, as #167 does at barriers | K/V copies 8% slower: completions are seen late |
| Splitting fused row maps into per-row loops (SplitVectorRows) | no gain at pp1024; a plain scale map 8 -> 10 cycles a register |
| `-pipeliner-max-mii` raised for heavy maps | no effect; unrolling by two (#127) works |
| Staggering the workers' SwiGLU | slower |
| Fusing the norm's scaling with its pack into `a`'s tiles | a map into a region of a Crouton buffer runs element by element (3.3G cycles); see below |

## What the hardware taught us

- **HMX.** About 8.1 cycles per 32 x 32 x 32 fp16 multiply at its best, plus
  a fixed cost per output tile (clear, convert, store). Workers share one HMX
  a tile at a time, and each worker's HVX work must fit in the others' turns.
- **Shared issue slots.** The cDSP's hardware threads share the core's issue
  slots. One thread issues a packet every two cycles at most, and with four
  workers on HVX a loop runs 2-4x slower than its schedule alone: what counts
  is the HVX work of all the workers together. A thread that spins takes
  slots from the others too, so waiters pause (#167).
- **VTCM and L2.** HVX loads from VTCM keep pipelined loops running; loads
  from DDR through L2 stall them. Small tables that a chunk reads repeatedly
  belong in VTCM. A map that wrote the buffer it read cost about 40% more than
  one between two buffers (exp2 over the scores, #163).
- **The pipeliner.** It needs one basic block, pointers it can tell apart,
  and an initiation interval of at most 27 cycles. The compiler now provides
  the first two (#150, #140); heavy maps unroll by two (#127).
- **VTCM capacity.** The chunk's shared activations take 2.5 MB and each
  worker's arena about 1.3 MB: 7.7 of the 8 MB. This holds a chunk at 128 rows,
  and so the rows each dequantized weight block serves. 256 rows would halve
  the dequantization per token, but do not fit.
- **The instruction cache.** A megakernel that inlines its GEMMs at every call
  site runs cold code in every phase, so the GEMMs are functions of their own
  (#99, #123).

## What is next

Three directions change the model's numerics or its bundle. Each needs a
decision:

- **W4A8 prefill.** With activations quantized to int8, the GEMMs could run
  on HMX's integer path, which would raise the HMX bound if its int8 rate
  beats its fp16 rate (not measured here). But the 4-bit weights' scales per
  64 columns do not fit an integer accumulation over all of K: it would mean
  requantizing the weights per channel, at some cost in quality.
- **A 4-bit head.** The 8-bit tied head streams 134 MB a token. At 4 bits the
  decode would be about 11% faster by bandwidth, for about 0.7% more
  perplexity (estimated in the quantization study, not built).
- **An fp16 KV cache.** The decode reads 25 MB of fp32 keys and values a token
  at 1024 positions. In fp16, by estimate, the decode would be 2-4% faster at
  1-2K positions and the prefill about 1% faster.

Within W4A16, each of these would give about 0.5-1%:

- **Norm fusion.** Fuse the RMSNorm's passes: a sum-of-squares reduction, and
  the scaling written straight into `a`'s tiles. The second needs
  `PhysicalOrderLoopLayout` (`src/transform/lower_tile_op.cc`) to take a
  region of a Crouton buffer: it takes whole buffers only, and a map into a
  region falls back to a scalar loop.
- **A cheaper causal mask.** The mask over the score tiles computes each
  lane's row and column (1.7M cycles); with a precomputed tile of `j - i` it
  would be a load, a compare and a select a register.
- **A leaner fp16 sigmoid.** Within the sigmoid's own clamp, the exp2 inside
  it need not clamp again.

## LFM3-1.7B in fp16

LFM3-1.7B runs unquantized: its projections in fp16, in a bundle of 3463.4 MB,
of which a decoded token reads 3461.9 MB. The numbers are from the same phone
and the same measurements as above, with `bench.py` taking the median of 3
runs. #179 tracks this work.

### LFM3: where it stands

| | decode, tg128 | prefill, pp1024 |
|---|---:|---:|
| **this example** | **16.6-16.7 t/s** (60 ms a token) | **248.8 ms, 4116 t/s** |
| roofline | 16.7 t/s (3461.9 MB a token at the DMA's 57.7 GB/s) | about 200 ms (each chunk of 256 rows streams 2835 MB of weights at the DMA's rate; HMX alone would take 174 ms) |

The decode reads its weights at the DMA's top rate, and HVX loads straight
from DDR do not raise it (see below). The prefill's GEMM phases run near both
limits at once: at about 55 GB/s, and with HMX at about 9.7 cycles per
32 x 32 x 32 multiply, against 8.1 at its best. A chunk of 256 rows is what
VTCM holds; the rest of the time is HVX work between the GEMM phases.

Quality: WikiText-2 perplexity on the device is 15.8834 over 16 windows of
1024 tokens, against 15.8812 for the fp32 reference (`model.py`) on the host;
15.8814 with the device's fp16 projection inputs emulated. After a chat
prompt, prefilled, the first 64 greedy tokens match the checkpoint's own code
(transformers 5.22, fp32).

### LFM3's prefill: from 320.7 to 248.8 ms

The first four steps are #189's.

| step | pp1024 |
|---|---:|
| the first version: the residual and the branch outputs in DDR, rows moved by HVX | 320.7 ms |
| the residual's rows by DMA; each residual add fused with the next norm; the last layer's FFN skipped | 308.9 ms |
| the branch outputs in VTCM, the GEMMs adding into them; attention in blocks of 64 keys; masks only where needed | 274.6 ms |
| the residual's DMA two slots deep; the queries by DMA in HMX's tiles (#189) | 270.9 ms |
| the KV cache in HMX's tiles, so keys and values reach HMX's operands by DMA with no pack; blocks of 64 rows against 128 keys; the gates by head; the sandwich norm's sums in the GEMMs' epilogues (#191) | 259.2 ms |
| the residual moved with no cache maintenance (`T.annotate_dma_only`, #193) | 254.2 ms |
| ... and bypassing the L2, as other transfers do (#195) | 250.1 ms |
| the residual adds' batches in pairs, so that each loop's slot is a constant; the sandwich scales as vectors (#197) | 248.8 ms |

### LFM3: where the prefill's time goes

pp1024, a copy of the kernel with `HAP_perf_get_pcycles` marks after each
phase and barrier (Mcycles per worker, the mean of 4; ms at 2.06 GHz):

| phase | Mcycles | ms |
|---|---:|---:|
| the GEMM phases: in_proj and qkv, out_proj, w1/w3 with SwiGLU, w2 | 423.3 | 205 |
| attention | 41.9 | 20 |
| the residual adds and norms | 33.5 | 16 |
| barriers | 15.9 | 8 |
| the chunk's start: embedding, RoPE rows, the first norm | 7.3 | 3.5 |
| total | 521.7 | 253 |

Inside attention (Mcycles, the slowest worker): exp2 and the row sums 13.5,
waiting for keys and values 8.1, the row blocks' prologue and epilogue 7.0,
P V on HMX 6.2, the mask, the row max and alpha 4.5, the rescale 2.6, q K^T on
HMX 0.6.

The barriers are mostly the first worker waiting, as in LFM2's prefill: it
finishes its GEMM phases ahead of the others (w1/w3: 204.9M against 213.2M
cycles).

### LFM3: what did not work

| tried | measured |
|---|---|
| HVX loads from DDR beside the DMA, in the same workers: a quarter or an eighth of each worker's panels | 49.1 and 42.8 GB/s in all, against 53.4 by DMA alone (probe of 256 MiB of weights) |
| 2 or 4 more workers multiplying their panels straight from DDR | 43.3-52.4 GB/s in all |
| `l2fetch` of the next panel before HVX loads it from DDR | 22.0 GB/s, against 22.4 without |
| Unrolling every float map by two, not only the heavy ones (#127) | the residual add's loop alone: 14.2 -> 10.3 cycles a vector; the whole prefill: unchanged (527.6 against 529.2 Mcycles) |
| The decode's attention on the KV cache's tiles (#191) | tg128 unchanged, in an A/B in one sitting: 16.53 against 16.49 t/s |

### LFM3: what the hardware taught us

- **DMA's cache maintenance.** A transfer's DDR side is kept in step with the
  DSP's caches: a load writes the source's lines back first, and a store
  writes the destination's lines back and drops them after. That costs about
  a cycle every 16 bytes: a 32 KB transfer's writeback took 2,000 cycles to
  submit, and a store's invalidate more when it was retired. A buffer that
  only DMA moves needs none of it (`T.annotate_dma_only`, #193).
- **DMA and the L2.** Transfers bypass the DSP's L2. Through it, 4 workers'
  DMA loads read DDR at 30 GB/s, against 50 bypassing it, whatever the block
  size (16-128 KB) or the transfers in flight (1-4) (#195).
- **Constant slots.** The residual adds' loops ran 24% faster with their
  slot of the two-slot ring a constant (the batches in pairs, #197) than with
  the run-time `rb % 2`: the add 16.0M against 12.2M cycles a worker. Yet
  `hexagon-clang` pipelined both with the same initiation interval (6), so the
  schedule does not explain it, and the cause is not established. A probe that
  seemed to show the run-time slot stopping the pipeliner also had its buffers
  unmarked as exclusive, which is enough on its own.
- **The decode's ceiling is the DMA's.** DMA alone read 53-58 GB/s of weights;
  HVX loads from DDR beside it lowered the total.
- **VTCM capacity.** At 256 rows a chunk takes 7.82 of the 8 MiB. 384 rows do
  not fit.

### LFM3: what is next

- **Attention's HVX work.** exp2 and the row sums take 13.5M cycles a worker
  at pp1024. Waiting for keys and values takes 8.1M, much of it their loads'
  writeback: the decode and the prefill write the cache with HVX, so its
  transfers keep the maintenance.
- **The embedding.** A token's row is a column of the head's tiles, which the
  DMA gathers as 1024 rows of 4 bytes: 7.3M cycles a worker at pp1024.
