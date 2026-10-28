"""LFM2 prefill on the device: tokens per second by prompt length.

    python bench_prefill.py <bundle> <bundle on the device> --lengths 128,512,1024

The prompts are random token ids. The bundle is read on the host for its
header and on the device for its tensors (push it there once).
"""

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decode import Decoder  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("bundle")
    p.add_argument("device_bundle")
    p.add_argument("--lengths", default="128,512,1024", help="comma-separated prompt lengths")
    p.add_argument("--runs", type=int, default=3)
    args = p.parse_args()
    lengths = [int(v) for v in args.lengths.split(",")]
    t0 = time.time()
    decoder = Decoder(args.bundle, args.device_bundle)
    print(f"session ready in {time.time() - t0:.0f} s")
    torch.manual_seed(0)
    prompt = torch.randint(1000, decoder.c.vocab, (max(lengths),)).tolist()
    try:
        for n in lengths:
            best = None
            for _ in range(args.runs):
                decoder.reset()
                decoder.prefill(prompt[:n])
                dsp = decoder.session.last_run_seconds()
                best = dsp if best is None else min(best, dsp)
            print(f"pp{n}: {best * 1000:.1f} ms on the DSP: {n / best:.0f} t/s")
    finally:
        decoder.close()


if __name__ == "__main__":
    main()
