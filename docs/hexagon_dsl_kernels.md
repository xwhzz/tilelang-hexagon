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

The vectorizer enforces this from two target attributes (`src/hexagon/target.cc`):
`max_vector_bits` = 1024 sets the vector width, and `min_vector_bits` = 1024 requires the
final vector size to be a multiple of `1024 / min_dtype_bits`
(`src/transform/loop_vectorize.cc`, Plan). It bumps to that width when the loop extent and
buffer accesses allow, else **scalarizes (safe)**. An access need not start on a register
boundary (`supports_unaligned_vector_access`): the codegen prints the vector accesses whose
offsets it proves multiples of their alignment as aligned `vmem`, and the others as
`tl_hvx_vec_loadu` / `tl_hvx_vec_storeu` (`hvx_lanes.h`), `vmemu`. On v79 a map over a
64x512 fp16 sub-tile that reads or writes rows off register boundaries runs in 3.6K to 5.1K
cycles, as fast as an aligned one, where it was a scalar loop of 3.7M to 4.0M. Backing this,
`src/tl_templates/hexagon/common.h`'s `vec_type` is a native clang `ext_vector` (so
`a * b`, `a & mask` become real HVX ops, not scalar struct loops), and `PrintType` emits
vector spellings up to 128 lanes. fp16/fp32 vector arithmetic compiles to a qfloat op
plus a conversion: `-mhvx-ieee-fp` would select single IEEE `.hf`/`.sf` instructions, but
the SM8750 (v79) cDSP returns zeros for them, although hexagon-sim computes them correctly.
HVX has no divide, and clang scalarizes a vector `/`, so the codegen prints float vector
division as `tl_hvx_div` (`hvx_math.h`): a times the Newton-Raphson reciprocal of b, in fp32.
The fp16/fp32 math intrinsics (`T.exp`, `T.exp2`, `T.log`, `T.log2`, `T.sqrt`, `T.rsqrt`,
`T.tanh`, `T.sigmoid`, `T.erf`, `T.abs`) become Hexagon builtins before layout inference
(`LowerMath`), so their loops vectorize, and print as the `hvx_math.h` functions. A scalar
runs the same 32-lane kernel on a splat, so a result never depends on whether its loop
vectorized, and fp16 computes in fp32 and rounds once. Vector `T.min`/`T.max` print as
`tl_vec_min`/`tl_vec_max` (`common.h`), one HVX `vmin`/`vmax` for every lane type. HVX fp16
arithmetic flushes subnormal results (below 2^-14) to zero.

The functions aim at fp32 accuracy. Under the `tl.enable_fast_math` pass config, `LowerMath`
picks fast variants instead (`tl_hvx_<fn>_fast`), and a float division becomes
`tl_hvx_div_fast`: fp16-grade accuracy for finite inputs in the functions' domains. They keep
their chains in qf32, take two Newton-Raphson steps, use lower-degree polynomials and skip
special values and subnormals. On v79, in cycles per fp32 register, with the maximum error
over the inputs (relative, or absolute for log, tanh, sigmoid and erf):

| function | accurate | error | fast | error |
|---|---|---|---|---|
| exp | 36.1 | 3.8e-6 | 22.1 | 7.6e-5 |
| exp2 | 28.1 | 3.6e-6 | 20.1 | 7.6e-5 |
| log, log2 | 92-94 | 6e-7 | 30.1 | 2.5e-6 |
| sqrt | 98.1 | 1.9e-7 | 24.0 | 4.7e-6 |
| rsqrt | 90.1 | 1.6e-7 | 18.0 | 4.8e-6 |
| tanh | 66.2 | 1.5e-6 | 38.1 | 3.7e-5 |
| sigmoid | 54.2 | 7.9e-7 | 32.1 | 1.8e-5 |
| erf | 176.1 | 1.5e-6 | 44.1 | 1.1e-4 |
| a / b | 26.1 | 2.4e-7 | 16.0 | 1.1e-5 |

Comparisons of vectors give lane masks, integers of the compared lanes' width, and a vector
`T.Select` prints as `tl_vec_select` (`common.h`), one `vcmp` and one `vmux`; `vec_type` also
has `-`, `~` and `!`, and `&&`/`||` on masks. The target declares `supports_vector_predicate`,
so the vectorizer keeps a select whose condition varies across lanes (leaky ReLU, `T.abs` of
integers), and it turns an `T.if_then_else` into a select when both branches are safe to
evaluate: no side effects but loads, each provably in bounds. A causal mask
(`T.if_then_else(j > i, -inf, s[i, j])`) therefore vectorizes, with the lane indices as a
`tl_vec_ramp`, while a branch that may load past its buffer stays guarded. A comparison of
float lanes hides them from LLVM (`tl_vec_opaque`): it would fold a symmetric range check or
an abs-like select into `llvm.fabs`, which hexagon-clang 19 scalarizes for HVX vectors.

Two passes let LLVM's software pipeliner overlap the iterations of these maps; a map it
does not pipeline issues about one instruction per packet. `BroadcastFromRegisters`
handles a scalar broadcast per row (`S[i, j] - rmax[i]`, `x[i, j] * inv[i]`): such a row
loop runs in strips of one register's worth of rows; a strip loads its rows' scalars
into a register once (`tl_hvx_load_lanes`) and each row broadcasts its lane from it
(`tl_hvx_broadcast_lane`, a rotate and a delta-network copy), instead of a scalar VTCM
load at the head of every iteration. `MarkExclusiveBuffers` gives each VTCM buffer that
shares no memory with another a pointer of its own (`tl_vtcm_exclusive`, which LLVM
treats as a fresh allocation): the buffers of an arena are otherwise one object to LLVM,
so a map that reads one buffer and writes another runs one iteration at a time.

`T.alloc_fragment` gives the tile program's one hardware thread a tile in VTCM, laid out
row-major in the same arena as its shared buffers (`fragment_scope` on the target).
`T.copy`, `T.fill` and `T.Parallel` work on fragments as on shared buffers. `T.reduce_*`
(sum, abssum, max, min, absmax and the bit operations, any axis, clear or accumulate)
reads and writes shared buffers and fragments in place. Reducing the innermost axis is
one `tl_hvx_reduce_rows` call (`hvx_reduce.h`): each row accumulates into one register,
and a transpose tree (`vdeal` + combine) merges the registers of a block of rows into one
register whose lane i is row i, which HVX stores; no value passes through a scalar
register. Other axes accumulate blocks of outputs in registers. Float sums accumulate in
fp32; max and min stay in fp16 for an fp16 source when that is exact.

Practical: **DSL guidance** — for a loop that touches uint8, use `T.vectorized(≥128)` (a
multiple of the narrowest dtype's register lanes). A clean fp16 map (x*x) went 20 ms → 1.4 ms.

## 2. Writing the q4_0 dequant in the DSL (not a hand template)

The nibble decode is plain DSL arithmetic that lowers to HVX — see `examples/hexagon/
example_qmatmul.py`. Two non-obvious rules beyond §1:

```python
q = T.Cast("int16", qcm[j, n])  # widen to int16 FIRST
lo = (q & T.Cast("int16", 0xF)) - T.Cast("int16", 8)  # then mask/shift on int16
hi = (q >> T.Cast("int16", 4)) - T.Cast("int16", 8)  # int16 consts -> no int32 promotion
B[2 * j, n] = T.Cast("float16", lo) * scb[2 * j, n]
B[2 * j + 1, n] = T.Cast("float16", hi) * scb[2 * j + 1, n]
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
VTCM (`alloc_shared`) and never hits DDR.

`T.gemm(A, B, C)` runs on HMX when A and B are fp16 and C is fp16 or fp32, all three in VTCM
(shared buffers or fragments), M, N and K are multiples of 32, and each operand is a matrix
of whole 32x32 tiles that starts on a tile: a whole buffer, a sub-matrix of a larger one, or
one matrix of a stack. Transposes and `clear_accum`, also as an expression such as `k == 0`,
combine freely.

Any other GEMM of row-major matrices runs on HVX (`GemmHVX`, `tl.hexagon.GemmHVXRejection`):
fp16 operands into an fp16 or fp32 C, fp32 operands into an fp32 C, any M, N and K, at most
one operand transposed. The whole call is one `tl_hvx_gemm` (`hvx_gemm.h`): outer products
when B's rows run along N, A's elements broadcast from a register of its row staged on the
stack (scalar loads from VTCM are slow), and dot products when `transpose_B` puts both
operands' rows along K, each output's register of partial sums summed by a transpose tree.
Products accumulate in qf32 registers over the call's K and reach C once. On v79 the outer
products run at 17-19 MAC per cycle, the dot products at 10-12, about 1000x the scalar loop,
and a decode GEMV that streams its weights by DMA moves 40-44 GB/s (`example_gemv.py`). v79's
qf32 adds of two zeros step the encoding's exponent up until it reads as a huge number, so
accumulators combine with each other and with C as IEEE fp32, and a result flushes the
denormals a chain of zero products converts to. A GEMM neither unit runs (both operands
transposed, other types, an operand another operator gave a layout that is not row-major)
runs as a scalar loop, and the compiler logs why.

An integer `T.gemm(A, B, C, transpose_B=True)` also runs on HVX, exactly: an int8, uint8 or
int16 A times an 8- or 4-bit B (`int8`, `uint8`, `T.int4`, `T.uint4`) into an int32 C. It is
one `tl_hvx_gemm_int` call (`hvx_gemm.h`). `vrmpy` multiplies the 4 bytes of each 32-bit lane
of a register by the 4 bytes of a scalar and adds them up, so B, in VTCM, takes the panel
layout (`tilelang.hexagon.intrinsics.panel`): each register holds 32 rows of B, a row's lane
its next 4 (8-bit) or 8 (4-bit) columns, so one register is 32 outputs' partial dot products.
`T.gemm` gives a VTCM B this layout itself, also one a pipeline multi-buffers. A's row reaches
the scalars through the stack, regrouped by the columns a lane holds. An int16 a is 256 hi + lo,
its signed high and unsigned low bytes, so it takes two `vrmpy`; a signed B's bytes read as
unsigned with the sign bit flipped, and the call subtracts the offset times the row's sum.

Weights go to the device packed in the panel layout (`pack_panel`). A parameter annotated
with it, `T.annotate_layout({W: make_hvx_panel_layout(W)})`, arrives in that order, and a
`T.copy` between two panel buffers moves whole panels by DMA, also outside a pipeline: the
region covers whole panels of one matrix, rows in multiples of 32 and columns in multiples of
the lane. 4-bit buffers are addressed by whole registers only: copied and multiplied, never
read element by element.

Group-quantized weights take `T.gemm_blockscaled(A, B, C, SFA, SFB, transpose_B=True, ...)`:
the integer GEMM above into an fp32 C, each group's int32 sums weighed by an fp32 factor of
A per `sf_a_granularity_k` columns (`[M, K / ga]`, or one per row) and an fp16 factor of B
per `sf_b_granularity_k` columns, group-major (`[K / gb, N]`). It is one
`tl_hvx_gemm_int_scaled` call: the sums leave the vrmpy accumulators once per group, into
fp32 accumulators, and a signed B's offset comes off each group with that group's sum of A.
A signed B takes an int16 A, in groups of whole registers (64 columns). On v79 the w4a16
GEMV (`example_gemv_w4a16.py`), which also quantizes x to int16 with one dynamic scale,
computes 30 GB/s of weights on one worker, against 18 (groups of 64) to 23 (128) for one
`T.gemm` per group and a `T.Parallel` epilogue, and streams 2048 -> 8192 at 44-50 GB/s with
4 workers (DMA tops out near 57 GB/s).

The HVX GEMMs are functions of their own, not inlined (`TL_GEMM` in `hvx_gemm.h`): the calls
of one shape share their code. A kernel that multiplies in many places, a megakernel whose
phases each call a GEMV, otherwise outgrows the instruction cache; on v79 a chain of eight
2048 -> 2048 w4a16 GEMVs streamed across 4 workers went from 31 to 53 GB/s.

Each 32x32 output tile reduces the call's K in the HMX accumulator and converts to fp16: HMX
accumulates within a call, and HVX across calls. An fp16 C keeps the HMX output layout, and a
call that overwrites it takes the tiles directly. Every other C takes each tile through a
staging tile in VTCM, which `tl_crouton_tile_store` (`crouton.h`) writes or adds into C:

- an fp32 C stays row-major, where widening a row-pair register by 1.0 gives the pair's two
  rows as fp32 registers;
- an fp16 C adds in its own Crouton tiles;
- a C of another layout is written element by element.

Two staging tiles alternate, so each tile's epilogue runs on HVX while HMX multiplies the
next tile, and the call holds the HMX accumulator throughout. In a worker pool, whose
workers share HMX (`LowerKernelLaunch` marks their gemms), each staged output tile is a
critical section of its own instead: a worker takes HMX, sets its scales, multiplies,
converts and stores the tile, releases HMX, and runs the tile's epilogue while another
worker multiplies. On v79, six independent 128 x 128 x 512 K loops into fp32 C run
2.2x (BK = 256) to 3.6x (BK = 64) faster on six workers than when calls took turns, and
5.4x to 7.2x faster than on one. A loop of calls costs one HVX pass over C per call, which
the K inside a call does not. On v79, one call over K = 1024 runs at 8.1 cycles per
32x32x32 multiply, the HMX rate. A loop of calls into an fp32 C (`example_gemm.py`) runs at:

| K per call (BK) | 64 | 256 | 512 |
|---|---|---|---|
| cycles per 32x32x32 multiply | 114 | 27 | 12.7 |

So give each call as much K as VTCM holds. Each call's product is rounded to fp16 before HVX
adds it, so an fp32 C carries the error of a sum of K / BK fp16 numbers.

For explicit scheduling, HMX is driven at **instruction-atom granularity** via
`tilelang/hexagon/intrinsics/hmx.py`'s `HMXIntrinEmitter` — the Hexagon analog of tilelang's
`TensorCoreIntrinEmitter` (`tilelang/cuda/intrinsics/macro/mma_macro_generator.py`). Like the
tensor-core emitter, it turns the matmul's tiling config into `@T.macro` atoms so either an
explicit kernel or `T.gemm` lowering can compose the operation:

| tensor core | `HMXIntrinEmitter` | instruction |
|---|---|---|
| shared-memory fragment layout | `activation_layout` / `weight_layout` | A `[M,K]` / B `[K,N]` (including storage transposes) in VTCM Crouton |
| `mma_atom(...,m,n,k)` | one 32x32x32 MAC atom | `{activation=mxmem; weight=mxmem}` |
| one K step of `mma` | `mma_tile(...,m,n)` | the complete K-loop for one output tile |
| accumulator store layout | `output_layout` / `store` | direct `mxmem=cvt` into a 2 KB-aligned native C `[M,N]` tile |
| `T.clear(C_local)` | `clear` | `mxclracc` |

**Honest divergence from tensor cores:** HMX has no `mma` instruction and NO addressable
accumulator fragment — loading the a/b Crouton tiles is what triggers the MAC, into a single
invisible per-core accumulator. Consequently `mma_tile()` reduces K for exactly one `(m,n)` tile;
the caller must complete `clear -> mma_tile -> convert -> store` before selecting another output
tile. The 32x32 atom uses `cpos(r,c)=(r//2)*64+c*2+r%2`; `repeat` constructs the A/C
`[mt,*t,cpos]` and B `[nt,kt,cpos]` tile orders, while `expand` adds staging dimensions.
`T.Layout` only rewrites addresses: packing, operand-specific allocation alignment, and the HMX
protocol remain explicit. Whole-buffer A/B dependencies are carried to `store` because HMX may
consume `mxmem` operands asynchronously. Session/power/VTCM ownership and cold-session
accumulator-read setup stay with the embedding runtime.

The shared-memory planner propagates each native VTCM requirement from its HMX operand:
activation/output 2 KB, weight 128 B, and scale/bias config 256 B. Because every 32x32 FP16
Crouton output tile is exactly 2 KB, `store` writes an output tile at its native address in
C, and `store_at` at any other 2 KB boundary, such as a staging tile. Operand origins
(`a_origin`, `b_origin`, `output_origin`) are the logical indices of each operand's first
element, so the atoms address sub-matrices and stacked matrices.

`GemmHMX.lower` takes only `T.gemm(A, B, C)`. Layout inference assigns native Crouton layouts
to A and B (and to an fp16 C without a layout), and lowering checks that they have them,
while it owns the dependency-only HMX states, the internal scale/bias block, the staging
tiles and the loop over output tiles. It emits the same explicit atoms. Kernels may use one logical `T.copy` to move any
region of a strided matrix, in DDR or VTCM, fp16 or fp32, to or from a Crouton buffer: the
lowering passes the region's origin and the matrix strides to one `tl_crouton_copy`, which
converts fp32 on the way. Row pairs of a row-contiguous matrix move as HVX registers
(`vshuff`/`vdeal`), with masked stores for partial tiles at the region's edges; packing a
column-contiguous fp16 matrix (a transposed weight in row-major storage) goes through a
32x32 word transpose in registers; other strides go element by element.

HVX code works on Crouton buffers directly, so chained gemms and the maps between them
(the flash attention softmax on `S`, an FFN gate `silu(h1) * h3`) convert no layouts.
A Crouton layout's physical shape is `[tiles, 16 row pairs, 64 lanes]`, and a
single-thread `T.Parallel` that writes every element of a buffer with a layout iterates
in that buffer's physical order (`LowerTileOp`): a loop over tiles and their row pairs,
vectorized over the 64 lanes of one register, so Crouton operands of the same layout are
whole-register loads and stores. A row operand `r[i]` then reads a window of two
elements per register, `[r0, r1, r0, r1, ...]`, and a column operand `c[j]` 32 elements
each repeated twice; the vectorizer accepts such window loads on Hexagon
(`vector_window_loads`), `LowerWindowLoads` lowers them to `tl_hvx_load_replicate` and
`tl_hvx_load_repeat`, and `BroadcastFromRegisters` loads a tile's 32 row values once and
broadcasts each row pair's group of two lanes. `T.reduce_*` of a whole Crouton matrix
along its rows or columns is one `tl_crouton_reduce` call (`crouton.h`), with the
transpose trees of the row-major reductions: fp32 widening splits a row pair register
into its two rows, and an fp16 max or min deals the pair's halfwords apart.

fp32 register sums in these trees are spelled out as `sf + sf` through qf32 and back
(`tl_reduce_combine`), as the `hvx_math.h` adds are. hexagon-clang 19 miscompiled long
chains of generic fp32 vector adds, and of qf32 values kept across permutes and spills,
in the Crouton row sums: a row lost one tile's sum, or turned to inf.

DMA is a separate row-major DDR↔VTCM operation and does not perform the Crouton
permutation. `dma.h` exposes raw instructions, descriptor builders, synchronous
copy helpers, and a caller-owned linked queue through explicit extern calls;
`T.dma_copy` / `T.dma_wait` expose a managed per-callback FIFO for explicit
manual DMA schedules. With `T.Kernel(num_workers=...)`, every tile program owns
its descriptors and FIFO in its worker's private VTCM arena and drains the FIFO
before it ends, including error cleanup. A worker must issue and complete its own queue on the
same QuRT thread; managed and raw/foreign DMA may not be mixed. Concurrent DDR
stores must own disjoint cache lines, including the lines around partial rows.
Scalar/vector producers are published before DMA reads, and DMA stores finish
cache maintenance before completion. Small source flushes use `dccleana` plus one
`syncht` per transfer, preserving the QuRT flush semantics. A load from a kernel
argument the kernel never writes skips the flush (`MarkUnwrittenDMASources`,
`TL_HEXAGON_DMA_DDR_TO_VTCM_UNWRITTEN`): the DSP gets its arguments with clean
caches. That flush was most of a transfer's cost on v79. With it, two 128-byte rows
in flight moved at 3.3 GB/s and 16 KB blocks at 23 GB/s; without it, 14.6 and
34 GB/s. This path is device-tested on v79; it does not imply independent physical
engines or arbitrary nested dispatch.

The programs of one launch can run in phases. `T.sync_grid()` waits until every tile
program of the launch reaches it, and what a program stored before it, from HVX or
scalar code, is visible to all programs after it. Such programs run all at once, one
per worker, so the grid may have at most `num_workers` programs: `MaterializeGrid`
rejects a larger grid, and the launch fails with `TL_ERR_GRID` if fewer workers start.
A waiting worker spins and keeps its thread and HVX context, so a barrier costs a few
cache-line transfers rather than a semaphore wake-up: 81 cycles across 2 programs and
191 across 6 on v79. The barrier does not wait for DMA. A program's queue spans it, so
a `T.dma_copy` submitted before the barrier for the next phase overlaps the phase
change; data other programs read after the barrier must be complete before it.

`T.Pipelined` makes the copies of a loop from DDR asynchronous: the target declares
`supports_async_copy` and names the functions that build the pipeline's commit and
wait (`async_copy_commit`, `async_copy_wait`), so the shared pipeline injector
commits a stage's copies as one group of the tile program's DMA queue and waits for
them before their consumers. A pipelined `T.copy` from DDR becomes the descriptors
of a DMA transfer, when DMA moves it faster than HVX loads: one contiguous block,
or rows of at least 512 bytes (`DMAOutpacesHVX`). Other copies run at once in
their stage. Into an HMX operand, which DMA cannot fill in Crouton order, the copy
goes to a row-major staging buffer that the pipeline multi-buffers, and an HVX pack
in the gemm's stage fills the operand (`StageHMXOperandCopies`). On v79, a
128 x 128 x 2048 K loop of fp32-accumulated gemms runs 1.6x (BK = 64) to 3.1x
(BK = 256) faster than with serial copies (`example_gemm.py`), and a GEMV streams
8 MB of fp16 weights at 40-44 GB/s (`example_gemv.py`). A transfer costs about 150
cycles to submit and as many to retire, so blocks of a few KB that serial HVX
copies read from L2 gain little: flash attention's 8 KB K and V blocks pipelined
match serial copies at SEQ 1024 and run 22% faster at 4096, but 12% slower at
256. An explicit `T.dma_copy` stays a manual schedule and is rejected inside
`T.Pipelined`.

The shared planner (`PipelinePlanning`) runs a loop `num_stages` deep when every
copy comes after every statement that reads a copied buffer, as in `copy A; copy B;
gemm`, and one stage deeper otherwise: when copies feed different statements, as
flash attention's K and V do, or when each staged HMX operand's transfer feeds its
own pack. So flash attention double-buffers with `num_stages=1`, and a K loop of
gemms with staged operands at `num_stages=2` prefetches two iterations ahead, which
pays off for small blocks (BK = 64: 14% faster than two deep).
The native-layout
`T.gemm` path is device-validated on v79: a staged
32x128x128 q4 case has relative error 0.000351 and the fused-copy 256x256x256 FP16 path has
maximum absolute error 0.0009766.

**Status: device-validated correct** (atom-granular K-streaming, single- and multi-N-tile, rel
2–4e-4). The perf/hiding optimization (whole-register 64-feature dequant slices vs the
32-feature HMX tile — the per-tile dequant is scalar today; double-buffering; worker-pool
overlap) is **not yet done** — see §4.

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
