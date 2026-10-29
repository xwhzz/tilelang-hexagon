"""Perplexity of the bundle on the device: windows of WikiText-2 (BOS and 1023
tokens of the text), each scored by the decode with every token forced
(``Decoder.score``), the windows that ``quantize.py --eval`` scores on the
host.

    python ppl.py <bundle> <bundle on the device> <checkpoint dir> --text wikitext2_test.txt --windows 40

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
    args = p.parse_args()
    text = Text(args.checkpoint)
    ids = text.encode(open(args.text).read())
    bos = text.tokenizer.token_to_id(text.bos_token)
    decoder = Decoder(args.bundle, args.device_bundle, prefill_rows=None)
    step, nll, count = 1023, 0.0, 0
    t0 = time.time()
    try:
        for i in range(min(args.windows, len(ids) // step)):
            scores = decoder.score([bos] + ids[i * step : (i + 1) * step])
            nll -= sum(scores)
            count += len(scores)
            print(f"window {i + 1}: ppl {math.exp(nll / count):.4f} ({time.time() - t0:.0f} s)", flush=True)
    finally:
        decoder.close()
    print(f"WikiText-2 ppl on the device: {math.exp(nll / count):.4f} over {count} tokens")


if __name__ == "__main__":
    main()
