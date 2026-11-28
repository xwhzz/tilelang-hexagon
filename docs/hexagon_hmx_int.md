# Integer HMX

This note records what the HMX of v73 and v79 offers beyond fp16, what was
measured on the devices, and how TileLang maps it: `T.gemm` of a `uint8` A by
an `int8` B into an exact `int32` C (#30).

## What the SDK exposes

`hmx_hexagon_protos.h` (Hexagon Tools 19.0.07) lists these operand types:

| port | types |
|---|---|
| activation | `hf` (fp16), `ub` (unsigned 8-bit), `f8` |
| weight | `hf`, `b` (signed 8-bit), `n` (4-bit, also `:2x`), `c`, `sc`, `sm`, `sbit`, `ubit`, `f8` |
| conversion | `cvt.hf`, `cvt.ub` (one byte of the accumulator), `cvt.uh` (`:2x1`, `:2x2`), `cvt.f8` |

The activation port is unsigned for integers: a signed operand goes through a
zero point. fp8 is not pursued here (it compiles for v79 but not for v73).
4-bit weights (`weight.n`) multiply 4-bit codes directly, but one accumulation
has one scale per output column, so they suit per-channel 4-bit weights, not
the per-group scales of the LFM2 bundle.

## Measured rates

A probe of the bare instructions: single thread, operands in VTCM, a
64-tile-deep (K = 2048) accumulation per output block, cycles per 32 x 32 x 32
multiply-accumulate. "With output" includes the conversions and stores of each
block (one for fp16, two bytes for int8).

| | v79 (phone) | v73 (SA8650) |
|---|---:|---:|
| fp16, MACs | 7.79 | 15.60 |
| fp16, with output | 8.00 | 15.87 |
| int8, 32 columns (`weight.b`), MACs | 3.34 | 2.90 |
| int8, 32 columns, with output | 3.42 | 3.16 |
| int8, 64 columns (`weight.b:deep`), MACs | 2.07 | 1.96 |
| int8, 64 columns, with output | 2.17 | 2.13 |

Int8 runs 2.3x (32 columns) to 3.7x (64 columns) as fast as fp16 on v79, and
5x to 7.5x on v73, whose fp16 rate is half v79's.

## Layouts and conversion (verified bit-exact on both devices)

- **Activation tile**: 64 rows x 32 K of `uint8`, 2048 bytes; element
  `(r, k)` at `(r // 4) * 128 + k * 4 + r % 4`: one HVX register holds four
  rows. A deep load reads consecutive tiles along K (range `2048 n - 1`).
- **Weight tile** (`weight.b`): 32 K x 32 columns of `int8`, 1024 bytes;
  element `(k, n)` at `(k // 4) * 128 + n * 4 + k % 4`. This is the 8-bit
  **panel layout** that HVX's `vrmpy` already reads (`tilelang/hexagon/intrinsics/panel.py`):
  a panel's 32 rows of B are consecutive weight tiles along K.
- **64 columns** (`weight.b:deep`): each K step reads 2048 bytes, the tiles of
  two consecutive 32-row panels of B at the same K position. Their products
  go to two accumulators. Convert bytes 0, 1, 2 and 3 with retain bits
  1, 1, 1 and 0 for each accumulator in turn. The final conversion clears
  the active accumulator and selects the other one.
- **Output**: `cvt.ub` takes one byte of each accumulator, after adding a
  per-column bias and shifting by a per-column amount, both from a bias bank
  (`bias = mxmem2`, 256 bytes: 32 exponent words, then 32 int32 biases). v79
  holds four banks and selects one in the conversion's control word; v73 has
  one, reloaded before each conversion. Exponent `e` selects bits
  `[24 - e, 32 - e)`: 24, 16, 8 and 0 give the four bytes. With control bit 1
  set, the byte wraps instead of saturating, and the four bytes are the exact
  two's-complement int32 (checked at K = 2048 with full-range operands). The
  output tile is stored spatially, in the activation tile layout, with store
  range `0xF00` (the fp16 range `0x700` writes only its first 32 rows).
- llama.cpp (ggml-hexagon) keeps 16 bits instead: two bytes of
  `(acc + bias) >> p`, with `p` from each column's bound and the bias folding
  in its activation zero point. That is also bit-exact as specified, but drops
  `p` low bits.

## `T.gemm` on integer HMX

`T.gemm(A, B, C, transpose_B=True)` runs on HMX when:

- A is `uint8` `[M, K]` in VTCM, in the 8-bit activation layout (an 8-bit
  `CroutonLayout`: 64 x 32 tiles, four rows a register);
- B is `int8` `[N, K]` in VTCM, in ordinary or paired 8-bit panels;
- C is `int32`, row-major, in VTCM;
- M is a multiple of 64, and N and K of 32, and each region starts on a tile.

C is exact: `C (+)= A @ B^T` in int32 (each call wraps like int32 arithmetic,
which a K below 2^31 / (255 * 128) = 65793 never reaches). It is the same
result as the HVX path (`tl_hvx_gemm_int`); other integer GEMMs stay there,
under its own rules (whole panels of B).

With whole 64-row pairs of B and whole 32-column K tiles, layout inference
prefers paired panels for an unconstrained B. This lets HMX compute 64 output
columns per weight load. An explicit ordinary panel layout keeps the
32-column path, including when N is divisible by 64. An N of 32 modulo 64,
or a region starting halfway through a pair, uses ordinary panels. A buffer
annotated as paired must be used on whole pairs.

For DDR weights, annotate B with `make_hvx_panel_layout(B, paired=True)`
and pass `pack_panel(weights, 8, paired=True)` at the call site. Use the
default `paired=False` in both helpers for ordinary HMX or HVX panels.
A copy from annotated DDR weights carries that layout into unannotated
VTCM staging. DMA moves one or more complete pairs and can slice K on
32-column boundaries. Pairing changes the byte order at each K step and
preserves the number of bytes in the buffer.

A call accumulates each 64 x 32 or 64 x 64 output tile over its whole K.
It converts each 32-column half into four byte planes. Double buffering
lets HVX join the previous tile's planes into rows of C while HMX multiplies
the next tile. In a worker pool one tile occupies the HMX unit per turn,
and HVX runs the epilogue after releasing it.

The following device measurements used ordinary **32-column int8** weights.
The figures are cycles per 32 x 32 x 32 over 50 `T.gemm` calls with operands
in VTCM (fp16 into fp32 C, int8 into int32 C).

| M x N x K | v79 fp16 | v79 int8 | v73 fp16 | v73 int8 |
|---|---:|---:|---:|---:|
| 128 x 256 x 2048 | 8.35 | 3.76 | 16.34 | 3.77 |
| 256 x 256 x 2048 | 8.29 | 3.65 | 16.30 | 3.65 |
| 128 x 256 x 8192 | 8.12 | 3.56 | 16.08 | 3.20 |

Why A is unsigned: HMX has no signed activation. A signed A would need its
buffer to hold `a ^ 0x80` instead of `a`, a value encoding that TileLang's
layouts (index maps) do not express. Quantized activations usually carry a
zero point anyway. For `A = q + 128`, `A @ B^T = q @ B^T + 128 * colsum(B)`;
the caller subtracts the column sums, which are constants for weights.

## A per-column shift into int16

An overwritten integer accumulator can feed an ordinary parallel map:

```python
ci = T.alloc_fragment((M, N), "int32")
T.gemm(a, b, ci, transpose_B=True, clear_accum=True)
for i, j in T.Parallel(M, N):
    y[i, j] = T.Cast("int16", (ci[i, j] + bias[j]) >> shift[j])
```

Use int32 column vectors in VTCM and an int16 destination with contiguous columns.
The map must immediately follow the GEMM, cover its entire output region and be
the sole reader of the private accumulator. A shared accumulator also qualifies.
A local used only by this pair can take fragment ownership. Accumulating into an
existing C, reading C elsewhere or using row-dependent metadata keeps the
ordinary GEMM and map. Stored int16 rows use the destination's leading dimension.

For a shift `p` from 0 through 16, the exponents `24 - p` and `16 - p`
select consecutive bytes of the biased sum. Both conversions use wrap mode,
so int32 addition wraps and narrowing keeps the low 16 shifted bits. The
caller can include a zero-point correction or a rounding offset in the bias.
The compiler adds only the bias supplied by the map.

Use shift counts from 0 through 31 for int32. The kernel checks them once
per 32- or 64-column band. If any lane falls
outside that range, HMX emits the four exact bytes for that band. HVX performs
the original word add, arithmetic right shift and truncation in registers.
For example, a shift of 31 preserves the sign of the wrapped sum. Both paths
write the int16 destination directly and need at most one tile of four-plane
scratch, independent of M and N.

On v73, each byte conversion reloads the unit's single bank. On v79,
ordinary panels use two banks for the two-byte path and paired panels use
four. Workers reload their banks after acquiring the shared HMX unit and
join the planes after releasing it. An exclusive caller can join one
two-byte result while HMX multiplies its next row tile.

## Using it for a quantized prefill

LFM2's prefill GEMMs keep fp16 activations today and dequantize the 4-bit
weights to fp16 tiles on HVX (`tl_crouton_dequant4`). An integer prefill
quantizes each activation row to uint8 and requantizes each 4-bit weight group
into int8 per output channel (the group scales do not fit one accumulation).
Both are lossy; the model-level study and the measured prefill are in the LFM2
example's documentation.
