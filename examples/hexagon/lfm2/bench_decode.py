"""LFM2 decode on the device: tokens per second, and the greedy tokens.

    python bench_decode.py <bundle> <bundle on the device> --prompt 1,1098,5706 --new 128

The bundle is read on the host for its header and on the device for its
tensors (push it there once). Token ids in and out; decode them with the
checkpoint's tokenizer.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decode import Decoder  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("bundle")
    p.add_argument("device_bundle")
    p.add_argument("--prompt", required=True, help="comma-separated token ids")
    p.add_argument("--new", type=int, default=128, help="tokens to generate")
    p.add_argument("--runs", type=int, default=3)
    args = p.parse_args()
    prompt = [int(v) for v in args.prompt.split(",")]
    t0 = time.time()
    decoder = Decoder(args.bundle, args.device_bundle)
    print(f"session ready in {time.time() - t0:.0f} s")
    steps = len(prompt) + args.new - 1
    try:
        for run in range(args.runs):
            decoder.reset()
            tokens, _ = decoder.run(prompt, 0, steps, len(prompt))
            dsp = decoder.session.last_run_seconds()
            print(
                f"run {run}: {steps} steps in {dsp * 1000:.0f} ms on the DSP: {steps / dsp:.1f} t/s ({dsp / steps * 1000:.2f} ms a token)"
            )
        print("generated:", ",".join(str(t) for t in tokens[len(prompt) :]))
    finally:
        decoder.close()


if __name__ == "__main__":
    main()
