# LFM2 on Hexagon

LFM2-1.2B (`LiquidAI/LFM2-1.2B`; LFM2.5-1.2B has the same architecture) on the
v79 cDSP, every operation a TileLang kernel, at batch 1.

| File | |
|---|---|
| `model.py` | The model in plain PyTorch, on explicit decode state: the reference. |
| `quantize.py` | GPTQ, the device bundle, and the bundle's perplexity with the device's arithmetic. |
| `bundle.py` | The bundle file: one header and every tensor the kernels read, in their layouts. |
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
