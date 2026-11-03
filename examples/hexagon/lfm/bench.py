"""LFM2 on the device in the style of llama-bench: prompt processing (ppN, a
random prompt of N tokens through the prefill) and token generation (tgN, N
tokens after a one-token prompt, greedy or sampled at `--temperature`), each
the median of `--runs` runs. A run is timed on the phone around the kernel's
FastRPC calls (`Decoder.last_run_seconds`). With `--depth D`, each test runs
after D positions of context, a random prompt prefilled outside its timing
(llama-bench's -d): tgN @ dD decodes at positions D on.

    python bench.py <bundle> <bundle on the device> --pp 512,1024 --tg 128
    python bench.py <bundle> <bundle on the device> --max-pos 36864 --pp 32768 --tg 0
    python bench.py <bundle> <bundle on the device> --max-pos 36864 --depth 32768 --pp 0 --tg 128
    python bench.py <bundle> <bundle on the device> --max-pos 262144 --depth 245760 --pp 16384 --tg 0

The bundle is read on the host for its header and on the device for its
tensors (push it there once). Without a prefill kernel (`--prefill-rows
none`), ppN runs the prompt through the decode, a token a step. The fp16
bundle runs over as many DSP processes as its context needs (`--processes
auto`), its layers split among them (decode_fp16.plan_stages).
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
    p.add_argument("--prefill-rows", default="auto", help="the prefill's chunk (Decoder's prefill_rows)")
    p.add_argument("--max-pos", type=int, default=4096, help="the KV cache's positions (Decoder's max_pos)")
    p.add_argument("--depth", type=int, default=0, help="positions of context before each test, prefilled untimed")
    p.add_argument("--processes", default="auto", help="the DSP processes of the fp16 bundle (Decoder's processes)")
    args = p.parse_args()
    pp = [int(v) for v in args.pp.split(",") if int(v or 0)]
    tg = [int(v) for v in args.tg.split(",") if int(v or 0)]
    t0 = time.time()
    named = {"auto": "auto", "none": None}
    rows = named[args.prefill_rows] if args.prefill_rows in named else int(args.prefill_rows)
    processes = args.processes if args.processes == "auto" else int(args.processes)
    decoder = Decoder(
        args.bundle, args.device_bundle, max_pos=args.max_pos, max_tokens=args.max_pos, prefill_rows=rows, processes=processes
    )
    stages = " ".join(f"[{st.layers.start}, {st.layers.stop})" for st in getattr(decoder, "stages", []))
    print(f"session ready in {time.time() - t0:.0f} s" + (f"; the layers' stages: {stages}" if stages else ""))
    torch.manual_seed(0)
    depth = args.depth
    prompt = torch.randint(1000, decoder.c.vocab, (depth + max(pp + [1]),)).tolist()

    def prefill(tokens, pos0=0):
        if decoder.prefill_rows:
            decoder.prefill(tokens, pos0)
        else:
            decoder.run(tokens, pos0, len(tokens), len(tokens))

    def median(run) -> float:
        times = []
        for _ in range(args.runs):
            decoder.reset()
            if depth:
                prefill(prompt[:depth])
            run()
            times.append(decoder.last_run_seconds())
        return statistics.median(times)

    at = f" @ d{depth}" if depth else ""

    print("| test | ms | t/s |\n|---|---:|---:|")
    try:
        for n in pp:
            s = median(lambda n=n: prefill(prompt[depth : depth + n], depth))
            print(f"| pp{n}{at} | {s * 1e3:.1f} | {n / s:.1f} |", flush=True)
        for n in tg:
            s = median(lambda n=n: decoder.run([decoder.c.bos], depth, n, 1, args.temperature))
            print(f"| tg{n}{at} | {s * 1e3:.1f} | {n / s:.1f} |", flush=True)
    finally:
        decoder.close()


if __name__ == "__main__":
    main()
