# LFM2 and LFM3 on Hexagon

LFM2-1.2B (`LiquidAI/LFM2-1.2B`; LFM2.5-1.2B has the same architecture) on the
v79 cDSP, every operation a TileLang kernel, at batch 1. LFM3-1.7B
(`LiquidAI/LFM3-1.7B`) runs unquantized
([#179](https://github.com/xwhzz/tilelang-hexagon/issues/179)): its reference,
its fp16 bundle, its decode and its prefill are here (see [LFM3](#lfm3)). The
other sections are LFM2's.

| File | |
|---|---|
| `model.py` | LFM2 and LFM3 in plain PyTorch, on explicit decode state: the reference. |
| `quantize.py` | LFM2's GPTQ, its 4-bit bundle, and the bundle's perplexity with the device's arithmetic. |
| `convert.py` | The fp16 bundle of an LFM2 or LFM3 checkpoint: its projections in HMX's Crouton tiles. |
| `bundle.py` | The bundle file: one header and every tensor the kernels read, in their layouts. |
| `stream.py` | The weight streams of the kernels (4-bit panels or fp16 Crouton tiles), and the decode's w4a16 GEMV blocks. |
| `decode.py` | The decode kernel: every token of a generation in one call; and `Decoder`, which holds a bundle's kernels. |
| `decode_fp16.py` | The fp16 bundle's decode kernel, LFM2's or LFM3's. |
| `prefill_fp16.py` | The fp16 bundle's prefill kernel: a prompt in chunks of 256 rows on HMX, in one call. |
| `prefill.py` | The prefill kernel: a prompt in chunks of 128 rows on HMX, in one call. |
| `text.py` | The checkpoint's tokenizer and chat template, on the host, without transformers. |
| `generate.py` | Chat on the device: the reply streams as it is generated. |
| `bench.py` | Prompt processing and token generation on the device, in the style of llama-bench. |
| `ppl.py` | The bundle's perplexity on the device. |
| `tiny.py` | Small random LFM2 and LFM3 models with the real ones' structures, for tests. |
| `OPTIMIZATION.md` | How the decode and the prefill got fast: each step's measured gain, what did not help, where the time goes, and what is left. |

## Weights

Projections are 4-bit, with an fp16 scale per 64 columns (GPTQ, layer by
layer). The tied embedding and LM head are 8-bit, one scale per row. Each
projection's input is rounded to int16 with one scale per row, as on the
device. WikiText-2 perplexity (40 windows of 1024 tokens):

| | ppl |
|---|---:|
| bf16 | 16.29 |
| bundle, device arithmetic (`quantize.py --eval`) | 16.683 |
| bundle, on the device (`ppl.py`) | 16.684 |

`ppl.py` scores the same windows on the device, with the decode, every token
forced (`Decoder.score`).

The bundle streams 685.2 MB (10^6 bytes) per token, of 819.8 MB on disk (the
embedding is read one row a token).

    python quantize.py --model LFM2-1.2B --out lfm2.bundle
    python quantize.py --model LFM2-1.2B --eval lfm2.bundle

A projection is stored in slices of 2048 columns, so `ffn.w2` [2048, 8192] is
four [2048, 2048] slices, and every block that a worker streams, 128 rows of a
slice, is one contiguous 128 KB with its 8 KB of scales.

## Decode

One kernel call generates the tokens of a generation (`Decoder.generate`): the
loop over tokens runs on the DSP. Four worker programs meet at
`T.sync_grid()` between a token's phases, four a layer and one after the LM
head. Each worker owns matching shares of every layer, so nothing else needs
a barrier:

- its slice of the model dimension: the rows of each projection that writes
  the residual, and the conv channels;
- its query heads and the two KV heads they share;
- its slice of the FFN's hidden units, in both w1 and w3, so SwiGLU is local;
- its slice of the vocabulary, whose argmax it proposes.

Weights stream by DMA through a ring of VTCM slots per worker, three blocks
ahead of the one it multiplies, across phases and from one token into the
next. A block's GEMV is one `T.gemm_blockscaled` call: the int16 input times
the 4-bit codes on HVX, exactly in int32 per group, weighed by the scales into
fp32.

The bundle, the KV cache and the conv state stay on the device in one session
(`tilelang.hexagon.runtime.session`), each of the bundle's tensors in a
resident buffer of its own (`bundle.load`).

On v79, with the bundle above (685.2 MB a token), decode runs at **81 t/s,
12.3 ms a token** over 140 steps (a 13-token prompt and 128 new tokens): it
streams 55.7 GB/s, 98% of the 57 GB/s that DMA reaches on the DSP. Its greedy
tokens match `BundleModel`'s, the reference with the device's arithmetic,
token for token.

The head also samples on the device: at a temperature, a sample is the argmax
of the logits over the temperature plus Gumbel noise, from a hash of (seed,
position, token) that a decode call computes in rounds of add, rotate and xor
(hexagon-clang 19 miscompiles a vector product shifted right, so it multiplies
nothing). And for every forced token it writes the log-probability the step
gives it, the scores of `Decoder.score`.

## Prefill

`Decoder.prefill` runs a prompt in chunks of 128 rows, all in one kernel call
(`prefill.py`), on HMX, and `Decoder.generate` prefills the prompt but its last
token, which the decode's first step runs. The four workers keep the decode's
shares of every projection, and a chunk's activations stay in VTCM that the
workers share (`global.vtcm`): its residual, the normalized input of a layer's
GEMMs, and their output or a part of the FFN's hidden units.

- **GEMMs.** Each block a worker streams, 128 rows of a projection, is one
  `T.gemm_blockscaled` by the whole chunk: HVX dequantizes the block's 4-bit
  weights once for all 128 rows, and the workers take turns at HMX, each
  dequantizing and running its epilogues while another multiplies.
- **Epilogues.** They fuse the short conv's gates and its causal conv over time,
  SwiGLU, the q/k norms and RoPE, and the residual adds, into the residual
  itself. The kernel has fast math, and SwiGLU and the softmax compute in fp16,
  which their results are.
- **Attention.** Per KV head, against the KV cache in blocks of 128 positions with
  an online softmax: HMX multiplies the scores and P V, and a block of keys and
  values packs once for the four query heads it serves. The scores stay in the
  fp16 tiles HMX stores them in, where the mask, the row maxima and exp2 work,
  and P goes straight into the tiles P V reads.
- **Layers.** One loop at run time, which reads each layer's kind from a table:
  the kernel's code does not grow with the depth.

On v79, each prompt one kernel call, timed on the phone around its FastRPC
call:

| prompt | ms | t/s |
|---:|---:|---:|
| 40 | 23.2 | 1728 |
| 128 | 21.0 | 6101 |
| 256 | 41.6 | 6157 |
| 512 | 86.6 | 5915 |
| 1024 | 176.2 | 5813 |
| 2048 | 373.6 | 5482 |

The prefill's keys and values match the decode's, a token a step, to 4.3e-3
at most (the last attention layer's keys), and the tokens generated after it
match the decode's.

## On the device

    adb push lfm2.bundle /data/local/tmp/lfm2.bundle
    python generate.py lfm2.bundle /data/local/tmp/lfm2.bundle LFM2-1.2B --prompt "What is 17 times 23?"
    python bench.py lfm2.bundle /data/local/tmp/lfm2.bundle --pp 512,1024 --tg 128
    python ppl.py lfm2.bundle /data/local/tmp/lfm2.bundle LFM2-1.2B --text wikitext2_test.txt

The device reads the bundle's tensors from its copy, and the host needs only
the header: a copy of the file's first `data_offset` bytes
(`bundle.read_header`) serves as the first argument.

`generate.py` renders the checkpoint's chat template and tokenizes on the
host (it needs `tokenizers` and `jinja2`), prefills the prompt and streams the
reply, a decode call per 16 tokens; `--temperature` and `--seed` sample.
Greedy, on v79:

    > What is 17 times 23?
    To find the product of 17 and 23, we can use the standard multiplication
    method or leverage the distributive property of multiplication over addition.
    ...
    Therefore, 17 times 23 is 391.
    [prompt 17 tokens, prefill 21 ms; 105 tokens, decode 80.8 t/s; 1.59 s in all]

    > Translate to French: The weather is nice today.
    Le temps est agréable aujourd'hui.
    ...

`bench.py` reports the median of 5 runs. A run is one kernel call, timed on
the phone around its FastRPC call: the call's overhead is included;
tokenization and the round trip to the PC that drives the phone are not.

| test | ms | t/s |
|---|---:|---:|
| pp128 | 21.0 | 6101 |
| pp512 | 86.6 | 5915 |
| pp1024 | 176.2 | 5813 |
| tg128 | 1597.1 | 80.1 |

For reference, on the same chip: [Liquid AI's model
card](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct) gives NexaML on
the NPU 4391 t/s prefill and 82 t/s decode for LFM2.5-1.2B (LFM2-1.2B's
architecture, other weights), with a 1K-token prompt and 100 new tokens,
without saying how they are timed. On this phone, nexa's demo app decodes at
65-68 t/s and prefills a 40-token prompt at 1.1-1.5k t/s, and llama.cpp's
Hexagon backend (Q4_0, `llama-bench -p 1024 -n 128`) runs pp1024 at 3175 t/s
and tg128 at 35.6 t/s.

## LFM3

LFM3-1.7B has 24 layers: 13 short convs of 7 taps, 6 of full attention and 5
of sliding attention over the last 256 positions, each followed by a SwiGLU
FFN of 7680 hidden units. Each branch's output is normed too (sandwich norm),
and each head's attention output is gated by a sigmoid of the layer's input.
`model.py` matches Liquid AI's implementation (the checkpoint's own code, on
transformers 5): on a 700-token prompt, which spans the sliding window, its
logits are within 1.5e-4 of it (relative 4.5e-6), and every argmax agrees.

Its weights stay unquantized. `convert.py` writes them in fp16, which moves
0.029 % of them, by at most 3e-8; on the bundle's weights, the reference's
logits are within 1.7e-4 of the official ones (relative 5.0e-6).

    python convert.py --model LFM3-1.7B --out lfm3.bundle

Each projection is stored as the HMX weight operand it is, in Crouton tiles,
so that the prefill's HMX and the decode's HVX multiply the same copy, and
the parts that one phase multiplies (B, C and x; w1 and w3; a KV head's
queries, key, value and gates) interleave by panels of 32 channels. The
bundle is 3463.4 MB, and a decoded token reads 3461.9 MB of it: at the DMA's
57 GB/s, 16.5 tokens/s at most.

One session holds it: the DSP's 4 GB of addresses fit its tensors as a buffer
each, allocated the largest first (1440 MiB for the FFN's w1 and w3 down to
137.5 MiB for the attention's projections), where seven buffers of 512 MiB
are as many as map. On the phone, the 14 buffers are allocated in 0.4 s and
read from the device's copy in 0.6 s.

### LFM3's decode

`decode_fp16.py` decodes the fp16 bundle as `decode.py` decodes the 4-bit one:
4 workers, a token in four barriers a layer, each worker's share of every
projection streaming by DMA a panel of 32 channels at a time and multiplied on
HVX straight from the HMX weight layout (`T.gemm` of a weight-layout B). A
worker owns one KV head of LFM3 with its four query heads and their gates.
Every worker keeps the residual stream in fp32 and adds each branch's whole
output, normed over all of it (the sandwich norm needs the workers' sums of
squares, which they write beside their slices). Projections take their inputs
in fp16. The KV cache is fp16, in HMX's tiles (keys as the transposed weight
operand of q K^T, values as the weight operand of P V), which the decode's
attention multiplies on HVX as they are and the prefill's moves into HMX's
operands by DMA.

On v79 (`bench.py`, tg128, timed on the phone around the kernel's FastRPC
call), LFM3-1.7B decodes at **16.7 tokens/s, 60.0 ms a token**: 3461.9 MB a
token at 57.7 GB/s, the DMA's top rate. After a chat prompt, its first 64
greedy tokens match the checkpoint's own code (transformers 5.22, fp32) one
for one.

### LFM3's prefill

`prefill_fp16.py` runs a prompt in chunks of 256 rows, all in one kernel call,
as `prefill.py` runs LFM2's. The bundle's fp16 tiles are HMX's weight operand
as they are: each worker streams the decode's share of every projection, a
panel of 32 channels a block, and multiplies it by the whole chunk on HMX,
with nothing to dequantize.

- **VTCM.** A chunk's activations stay in VTCM the workers share:
  - the normalized input and the hidden units, which HMX reads;
  - the branch outputs, in fp32, into which the output projections' GEMMs write
    and the FFN's parts add, and which LFM3's sandwich norm normalizes over each
    row before the residual add.

  The residual itself is in DDR, a worker's rows moving by DMA two slots deep.
  Only DMA moves it (`T.annotate_dma_only`), so its transfers skip the cache
  maintenance a DDR buffer the DSP also reads or writes needs.
- **Attention** runs per query head, in blocks of 64 rows against 128 keys at a
  time, with an online softmax in base 2. Blocks of keys and values come from
  the cache into HMX's operands by DMA, a block ahead, with no pack; so do the
  queries, which go to DDR in HMX's activation tiles.
- **The sandwich norm's** row sums of squares come from the output
  projections' GEMMs: each panel's epilogue adds its share, while HVX waits
  for the weights anyway. The residual add then reads each branch output once,
  and squares the new rows for the next norm as it makes them.
- **The last layer's** output projection and FFN feed nothing a prefill leaves
  (the decode runs the prompt's last token), so it skips them.

On v79 (`bench.py`, median of 3, timed on the phone around the kernel's FastRPC
call):

| test | ms | t/s |
|---|---:|---:|
| pp256 | 60.5 | 4230 |
| pp512 | 121.9 | 4199 |
| pp1024 | 248.8 | 4116 |
| pp2048 | 522.8 | 3917 |

A chunk streams 2835 MB of weights: for 1024 tokens, 205 ms of GEMM phases at
55 GB/s, near the DMA's top rate, while HMX's multiplies alone would take
174 ms at 8.1 cycles a 32 x 32 x 32 op. A cycle profile puts the rest of pp1024 in four
places:
- attention, mostly HVX: exp2, the row sums and the output's rescaling (20 ms);
- the residual adds and norms, HVX passes over each chunk's rows (16 ms);
- the barriers (8 ms);
- the embedding, each token's row gathered from the head's tiles as 1024 DMA
  rows of 4 bytes (3.5 ms).

[OPTIMIZATION.md](OPTIMIZATION.md#lfm3-17b-in-fp16) has how each step got
here and what did not work.

After a chat prompt, prefilled, the first 64 greedy tokens still match the
checkpoint's own code.

    python bench.py lfm3.head /data/local/tmp/lfm3.bundle --pp 512,1024 --tg 128

### LFM3 on the device

    python generate.py lfm3.head /data/local/tmp/lfm3.bundle LFM3-1.7B --prompt "What is 17 times 23?"
    python ppl.py lfm3.head /data/local/tmp/lfm3.bundle LFM3-1.7B --text wikitext2_test.txt --windows 16

LFM3's chat template opens the reply with a `<think>` section, which the model
closes before it answers; `--no-think` asks for the answer alone. Greedy, on
v79:

    > What is 17 times 23?
    We need to compute 17 times 23. That's a simple multiplication. 17 * 23. We
    can do 17 * 20 = 340, and 17 * 3 = 51, sum = 391. ...
    Thus final answer: \boxed{391}.</think>To find the product of 17 and 23, we
    can multiply directly:
    ...
    Thus, the result is 391.
    [prompt 18 tokens, prefill 63 ms; 197 tokens, decode 16.7 t/s; 12.96 s in all]

With `--no-think`:

    > Translate to French: The weather is nice today.
    Le temps est beau aujourd'hui.
    [prompt 22 tokens, prefill 63 ms; 7 tokens, decode 16.6 t/s; 1.09 s in all]

The prefill runs whole chunks of 256 rows, so a short prompt takes as long as
256 tokens do.

WikiText-2 perplexity, 16 windows of 1024 tokens (BOS and 1023 tokens of the
text, each window scored token by token):

| | ppl |
|---|---:|
| the reference (`model.py`) in fp32 on the bundle's weights, on the host | 15.8812 |
| the same, each projection's input rounded to fp16 as on the device | 15.8814 |
| the bundle, on the device (`ppl.py`) | 15.8834 |

The device scores every token with the decode (`Decoder.score`). Besides the
fp16 inputs of its projections, its attention multiplies the fp16 KV cache by
queries and probabilities rounded to fp16, which the host reference keeps in
fp32.
