# LFM2 on Hexagon

LFM2-1.2B (`LiquidAI/LFM2-1.2B`; LFM2.5-1.2B has the same architecture) on the
v79 cDSP, every operation a TileLang kernel, at batch 1.

| File | |
|---|---|
| `model.py` | The model in plain PyTorch, on explicit decode state: the reference. |
| `quantize.py` | GPTQ, the device bundle, and the bundle's perplexity with the device's arithmetic. |
| `bundle.py` | The bundle file: one header and every tensor the kernels read, in their layouts. |
| `stream.py` | The weight stream of a decode step, and its w4a16 GEMV blocks. |
| `decode.py` | The decode kernel: every token of a generation in one call. |
| `bench_decode.py` | Decode on the device: tokens per second and the greedy tokens. |
| `tiny.py` | A small random LFM2 with the same structure, for tests. |

## Weights

Projections are 4-bit, with an fp16 scale per 64 columns (GPTQ, layer by
layer). The tied embedding and LM head are 8-bit, one scale per row. Each
projection's input is rounded to int16 with one scale per row, as on the
device. WikiText-2 perplexity (40 windows of 1024 tokens):

| | ppl |
|---|---:|
| bf16 | 16.29 |
| bundle, device arithmetic | 16.68 |

The bundle streams 653.5 MB per token, of 781.8 MB on disk (the embedding is
read one row a token).

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
(`tilelang.hexagon.runtime.session`).

On v79, with the bundle above (653.5 MB a token), decode runs at **81 t/s,
12.3 ms a token** over 140 steps (a 13-token prompt and 128 new tokens), 94%
of the DMA roofline (57 GB/s); nexa measured 65-68 t/s on the same phone. Its
greedy tokens match `BundleModel`'s, the reference with the device's
arithmetic, token for token.

    adb push lfm2.bundle /data/local/tmp/lfm2.bundle
    python bench_decode.py lfm2.bundle /data/local/tmp/lfm2.bundle --prompt 1,1098,5706,803 --new 128
