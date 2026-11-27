"""Perplexity of the bundle on the device: windows of WikiText-2 (BOS and 1023
tokens of the text), each scored by the decode with every token forced
(``Decoder.score``), the windows that ``quantize.py --eval`` scores on the
host.

    python ppl.py <bundle> <bundle on the device> <checkpoint dir> --text wikitext2_test.txt --windows 40

For suffix-only perplexity conditioned on prefill, add `--prefill-prefix 512`.
Each window then prefills its first 512 tokens (including BOS), and decode
scores each target following the first token of the suffix.

The checkpoint directory needs only the tokenizer's files.
"""

import argparse
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decode import Decoder  # noqa: E402
from text import Text  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("bundle")
    p.add_argument("device_bundle")
    p.add_argument("checkpoint", help="the directory of tokenizer.json")
    p.add_argument("--text", required=True, help="the evaluation text, e.g. WikiText-2's test split")
    p.add_argument("--windows", type=int, default=40)
    p.add_argument("--prefill-prefix", type=int, default=0, help="prefill this many tokens of each window before scoring its suffix")
    p.add_argument("--prefill-rows", type=int, help="chunk rows for prefix prefill (default: Decoder's auto)")
    p.add_argument("--w8a8-w13", action="store_true", help="use online W8A8 for the 4-bit LFM2 prefill gate/up projection")
    args = p.parse_args()
    step = 1023
    if args.windows <= 0:
        p.error("--windows must be positive")
    if not 0 <= args.prefill_prefix < step:
        p.error(f"--prefill-prefix must be in [0, {step - 1}]")
    if args.prefill_rows is not None and args.prefill_rows <= 0:
        p.error("--prefill-rows must be positive")
    if not args.prefill_prefix and (args.w8a8_w13 or args.prefill_rows is not None):
        p.error("--w8a8-w13 and --prefill-rows require a positive --prefill-prefix")
    text = Text(args.checkpoint)
    ids = text.encode(open(args.text).read())
    windows = min(args.windows, len(ids) // step)
    if not windows:
        p.error(f"--text must encode at least {step} tokens")
    bos = text.tokenizer.token_to_id(text.bos_token)
    rows = None if args.prefill_prefix == 0 else ("auto" if args.prefill_rows is None else args.prefill_rows)
    decoder = Decoder(args.bundle, args.device_bundle, prefill_rows=rows, w8a8_w13=args.w8a8_w13)
    nll, count = 0.0, 0
    t0 = time.time()
    try:
        for i in range(windows):
            scores = decoder.score([bos] + ids[i * step : (i + 1) * step], prefill_prefix=args.prefill_prefix)
            nll -= sum(scores)
            count += len(scores)
            print(f"window {i + 1}: ppl {math.exp(nll / count):.4f} ({time.time() - t0:.0f} s)", flush=True)
    finally:
        decoder.close()
    if args.prefill_prefix:
        print(
            f"WikiText-2 prefill-conditioned suffix ppl: {math.exp(nll / count):.4f} over {count} scored tokens "
            f"({args.prefill_prefix} prefilled tokens per window)"
        )
    else:
        print(f"WikiText-2 ppl on the device: {math.exp(nll / count):.4f} over {count} tokens")


if __name__ == "__main__":
    main()
