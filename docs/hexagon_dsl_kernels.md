# Writing high-performance DSL kernels on Hexagon

How DSL constructs lower on Hexagon and what makes them fast: the HVX register rule, GEMMs
on HMX and HVX (floating point, integer and block-scaled), pipelines that stream by DMA, and
programs that meet at `T.sync_grid()`. Companion to [`hexagon_backend.md`](hexagon_backend.md),
the backend's architecture. The kernels of `examples/hexagon/` use what is described here;
`examples/hexagon/lfm/` composes it into whole models.

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
`tl_hvx_div_fast`, or a multiply by the reciprocal of a constant divisor: fp16-grade accuracy
for finite inputs in the functions' domains. They keep
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

On fp16 lanes the fast exp, exp2 and sigmoid compute in qf16, 64 lanes a register,
instead of two fp32 registers: in cycles per fp16 register, with the error of the fp16
results (relative, absolute for sigmoid), against the fp32 kernels on the same lanes:

| function | in fp32 | qf16 | error |
|---|---|---|---|
| exp | 46.1 | 20.4 | 6.2e-4 |
| exp2 | 42.1 | 14.2 | 6.7e-4 |
| sigmoid | 67.2 | 28.7 | 9.2e-4 |

exp reduces x - k ln 2 in two parts, so that rounding x / ln 2 to fp16 costs nothing, and
exp and exp2 give 0 below their range (a softmax masks with -inf). A map that rounds its
result to fp16 anyway, a SwiGLU into an fp16 activation or a softmax into an fp16 P,
computes in fp16 to take them (`T.Cast("float16", ...)`).

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

Three passes shape these maps for LLVM's software pipeliner; a map it does not pipeline
runs one chain of dependent instructions at a time, about one instruction per packet.

- `BroadcastFromRegisters` handles a scalar broadcast per row (`S[i, j] - rmax[i]`,
  `x[i, j] * inv[i]`): such a row loop runs in strips of one register's worth of rows; a
  strip loads its rows' scalars into a register once (`tl_hvx_load_lanes`) and each row
  broadcasts its lane from it (`tl_hvx_broadcast_lane`, a rotate and a delta-network
  copy), instead of a scalar VTCM load at the head of every iteration. A row of at most 8
  vectors then unrolls, so that the loop over the strip's rows is the innermost, which
  LLVM pipelines: on v79, `exp(s - m[i])` over 128 rows of 2 fp32 vectors went from 79 to
  27 cycles a vector. Compute a row's factor once, in a loop over the rows
  (`inv[i] = T.rsqrt(ss[i] * (1 / D) + eps)`), rather than in the map, where each vector
  recomputes it: an RMSNorm's scaling went from 155 to 22 cycles a vector.
- `MarkExclusiveBuffers` gives each VTCM buffer that shares no memory with another a
  pointer of its own (`tl_exclusive`, which LLVM treats as a fresh allocation): the
  buffers of an arena are otherwise one object to LLVM, so a map that reads one buffer
  and writes another runs one iteration at a time. A worker program's lambda takes the
  kernel's arguments (when they do not alias) and its `global.vtcm` buffers the same
  way, bound at its entry: read through the lambda's captures, LLVM loads them again
  after a store to any buffer whose pointer a call or an alignment test has seen, and
  the loop does not pipeline. The LFM2 prefill's RMSNorm scaling went from 12.4M to
  8.2M cycles at 1024 tokens.
- `InterleaveMaps`: a map that computes an HVX math function runs a chain of a few dozen
  instructions per vector, which can exceed the initiation interval LLVM pipelines
  (`-pipeliner-max-mii`). Its loop prints unrolled by two (`#pragma clang loop
  unroll_count(2)`), so that an iteration interleaves two chains: SwiGLU's
  `x * sigmoid(x) * y` over 128 x 128 fp32 went from 126 to 56 cycles a vector.

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

## 2. GEMMs, and the pipelined K loop

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
and a decode GEMV that streams fp16 weights by DMA moves 40-44 GB/s. v79's
qf32 adds of two zeros step the encoding's exponent up until it reads as a huge number, so
accumulators combine with each other and with C as IEEE fp32, and a result flushes the
denormals a chain of zero products converts to. A GEMM neither unit runs (both operands
transposed, other types, an operand another operator gave a layout that is not row-major)
runs as a scalar loop, and the compiler logs why.

A B in the HMX weight layout (`make_hmx_weight_layout`, transposed with `transpose_B`) runs
on HVX too, as one `tl_hvx_gemm_weight` call, for an fp16 A [M, K] and a row-major C, B's
region whole tiles: the copy of a model's weights that HMX multiplies in its prefill serves
its decode's GEMVs. A register of a tile holds two rows of K for its 32 columns, interleaved,
so A's two elements, broadcast as one word, multiply it into the even row's products and the
odd row's, each in its column's lane: no reduction across lanes. Up to four panels of 32
columns share each broadcast. On v79, for a [1, 2048] A from VTCM, a worker multiplies 10.3
weights a cycle on one panel and 14.8 on four, against 10.9 for the dot products and 1.9
and 13.7 for the outer products of a row-major B.

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

With an fp16 A, `T.gemm_blockscaled` runs on HMX, for prompts rather than single tokens:

- **Operands.**
  - B is the same 4-bit matrix in the panel layout.
  - SFB is the same group-major fp16 factors.
  - SFA is fp32, one factor per row (`[M, K / ga]` with ga >= K), weighing C's rows.
  - C is an fp32 row-major or fp16 Crouton matrix.
- **Dequantization.** HVX expands B 32 rows at a time into a column of fp16 weight tiles (`tl_crouton_dequant4`, `crouton.h`). A panel register of 32 rows by 8 columns of K becomes exactly four registers of the weight tiles:
  - `vlut16` looks the nibbles up as fp16 (on v79 it reads entry i of its table at halfword 2i);
  - `vshuffe`/`vshuffo` pair the columns;
  - a qf16 multiply applies the group's factors.

  This runs at about 18 weights per cycle on one thread, bit-exact with torch's fp16 products. IEEE `vmpy(hf, hf)` gives wrong results on v79.
- **Multiply.** HMX multiplies every row tile of A by the column, so each weight dequantizes once per call, however many rows A has. Two columns alternate, and a slice of the next column dequantizes alongside each tile's MACs.
- **Out of line.** The whole call is one `tl_hexagon_hmx_gemm_blockscaled` (`hmx_gemm.h`), so the calls of one configuration share their code, as the HVX GEMMs' do below. Within it the dequantization is a function of its own: inlined into the call's loop it competes for registers and spills its lookup table.

On v79, M x 128 x 2048 runs at:

| Programs | Cycles per 32x32x32 (HMX rate: 8.1), M = 256 | M = 128 |
|---|---|---|
| one | 14.1 | 22.6 |
| a pool of 4, sharing HMX | 8.45 | 8.55 |

One program is bound by its HVX work: a tile's slice of the next column and its epilogue take longer than its MACs. In a pool, a worker dequantizes while another holds HMX, and while it holds HMX itself, the previous tile's epilogue runs under its own MACs (for K of 32 tiles or more). Its dequantization does not: under the MACs, it slows them (4 workers, 128 rows: 9.8).

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
- a C of another layout takes the product from a row-major matrix of its own, element by
  element.

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

The call is one `tl_hexagon_hmx_gemm` (`hmx_gemm.h`), the driver `T.gemm_blockscaled`
shares: its tile loop takes the weight tiles of an output column from a source, B's tiles
in VTCM for `T.gemm` or the dequantized panels for `T.gemm_blockscaled`. HMX has no
addressable accumulator, a single one per core that loading the activation and weight
tiles feeds, so the driver completes each output tile (clear, multiply its K, convert,
store) before the next. The 32x32 atom uses `cpos(r,c)=(r//2)*64+c*2+r%2`; `repeat`
constructs the A/C `[mt,*t,cpos]` and B `[nt,kt,cpos]` tile orders. The shared-memory
planner aligns the call's VTCM operands as HMX needs them, which the builtin declares:
activation and output tiles on 2 KB, weight tiles on 128 B, and the scale/bias config on
256 B.

`GemmHMX.lower` takes `T.gemm(A, B, C)`. Layout inference assigns native Crouton layouts to A
and B (and to an fp16 C without a layout), and lowering checks that they have them. The call
gets each operand's first tile and the tiles between A's rows or B's columns, so sub-matrices
and stacked matrices need no copies. Kernels may use one logical `T.copy` to move any
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
permutation. Between two buffers of one Crouton layout, `T.dma_copy` moves whole tiles,
which follow each other in the tile order (a weight's K, an activation's columns): each run
of a region's tiles is a row of one 2D descriptor, or the runs are one 1D transfer when they
follow each other on both sides, as one panel of weights does. `dma.h` exposes raw instructions, descriptor builders, synchronous
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
34 GB/s. A DDR buffer that only DMA moves, in every kernel of the session that
reaches it, can be declared so: `T.annotate_dma_only(X)` (`MarkDMACacheDirections`).
The DSP then holds none of its lines, and its loads and stores skip the maintenance
(`TL_HEXAGON_DMA_DDR_TO_VTCM_UNWRITTEN`, `TL_HEXAGON_DMA_VTCM_TO_DDR_UNCACHED`).
Like every transfer, they bypass the L2: through it, 4 workers' 32 KB loads read
DDR at 30 GB/s rather than 57. The compiler rejects any other use of the
buffer in the kernel; it cannot check the session's other kernels. On v79 a 32 KB
transfer's writeback took about 2,000 cycles to submit, and a store's invalidate
more when it was retired: LFM3's prefill, which moves its residual stream between
DDR and VTCM a few rows at a time, spends 29% fewer cycles on its residual adds
with the stream declared DMA-only. A DDR buffer that the DSP writes and DMA reads,
such as a KV cache, can instead be declared write-through: `T.annotate_write_through(X)`.
The compiler then writes back each region written right after its write
(`InsertWriteThroughWritebacks`): after a `T.copy` into the buffer, a `T.Parallel`
loop that stores into it, or another store. Of a Crouton buffer it writes back the
whole tiles the region touches, a run of tiles at a time. Its DMA loads then skip
the source's writeback (`TL_HEXAGON_DMA_DDR_TO_VTCM_UNWRITTEN`), which pays where a
region is read many times after one write. As with DMA-only buffers, every kernel
of the session that writes the buffer must declare it. This path is
device-tested on v79; it does not imply independent physical engines or arbitrary
nested dispatch.

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
8 MB of fp16 weights at 40-44 GB/s. A transfer costs about 150
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

## 3. Measuring

- **Measure compute, not one call's wall clock.** A standalone kernel's call is dominated
  by moving its arguments over FastRPC and `adb`. Time inside the kernel
  (`HAP_perf_get_pcycles`), repeat the work, or keep the arguments resident in a session
  (`tilelang.hexagon.runtime.session`) before reading a number as compute.
- **HMX accumulates K in high precision.** Each K tile accumulates without a clear, and
  the conversion to fp16 happens only when the accumulator is read out, so K streams.
  There is one physical accumulator, so K cannot be split across workers.
