"""LFM2 on the device in the style of llama-bench: prompt processing (ppN, a
random prompt of N tokens through the prefill) and token generation (tgN, N
tokens after a one-token prompt, greedy or sampled at `--temperature`), each
the median of `--runs` runs. A run is timed on the phone around the kernel's
FastRPC call (`Session.last_run_seconds`).

    python bench.py <bundle> <bundle on the device> --pp 512,1024 --tg 128

The bundle is read on the host for its header and on the device for its
tensors (push it there once). Without a prefill kernel (the fp16 bundle's
decoder, for now), ppN runs the prompt through the decode, a token a step.
"""

import argparse
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decode import Decoder  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("bundle")
    p.add_argument("device_bundle")
    p.add_argument("--pp", default="512,1024", help="comma-separated prompt lengths")
    p.add_argument("--tg", default="128", help="comma-separated generation lengths")
    p.add_argument("--temperature", type=float, default=0.0, help="sample the generated tokens (0: greedy)")
    p.add_argument("--runs", type=int, default=5)
    args = p.parse_args()
    pp = [int(v) for v in args.pp.split(",") if v]
    tg = [int(v) for v in args.tg.split(",") if v]
    t0 = time.time()
    decoder = Decoder(args.bundle, args.device_bundle)
    print(f"session ready in {time.time() - t0:.0f} s")
    torch.manual_seed(0)
    prompt = torch.randint(1000, decoder.c.vocab, (max(pp + [1]),)).tolist()

    def median(run) -> float:
        times = []
        for _ in range(args.runs):
            decoder.reset()
            run()
            times.append(decoder.session.last_run_seconds())
        return statistics.median(times)

    def prefill(tokens):
        if decoder.prefill_rows:
            decoder.prefill(tokens)
        else:
            decoder.run(tokens, 0, len(tokens), len(tokens))

    print("| test | ms | t/s |\n|---|---:|---:|")
    try:
        for n in pp:
            s = median(lambda n=n: prefill(prompt[:n]))
            print(f"| pp{n} | {s * 1e3:.1f} | {n / s:.1f} |", flush=True)
        for n in tg:
            s = median(lambda n=n: decoder.run([decoder.c.bos], 0, n, 1, args.temperature))
            print(f"| tg{n} | {s * 1e3:.1f} | {n / s:.1f} |", flush=True)
    finally:
        decoder.close()


if __name__ == "__main__":
    main()
