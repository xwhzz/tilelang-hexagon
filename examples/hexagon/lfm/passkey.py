"""Passkey retrieval on the device: a random five-digit key in filler text of
about `--tokens` tokens, at `--depth` of it, which the model is asked for at
the end (greedy, without LFM3's <think> section). It checks that a long
context reaches the decode whole: the prefill's chunks, the KV cache, and
the stages over DSP processes when it takes several.

    python passkey.py <bundle> <bundle on the device> <checkpoint dir> --max-pos 262144 --tokens 261000
    python passkey.py <bundle> <bundle on the device> <checkpoint dir> --max-pos 49152 --tokens 48000 --processes 1,2

With several `--processes`, each runs the same prompt in a fresh session, and
their replies and last logits are compared. The checkpoint directory needs
only the tokenizer's files.
"""

import argparse
import os
import random
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decode import Decoder  # noqa: E402
from text import Text  # noqa: E402

TASK = "There is an important info hidden inside a lot of irrelevant text. Find it and memorize it. I will quiz you about the important information there.\n\n"
FILLER = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "
QUESTION = "\n\nWhat is the pass key? Answer with the number only."


def prompt_ids(text: Text, tokens: int, key: int, depth: float) -> list:
    """The chat's ids, about `tokens` of them: the task, the filler with the
    key's sentence at `depth` of it, and the question."""
    needle = f"The pass key is {key}. Remember it. {key} is the pass key. "
    per = len(text.encode(FILLER))
    fixed = len(text.chat([{"role": "user", "content": TASK + needle + QUESTION}], enable_thinking=False))
    reps = max(0, (tokens - fixed) // per)
    before = int(reps * depth)
    content = TASK + FILLER * before + needle + FILLER * (reps - before) + QUESTION
    return text.chat([{"role": "user", "content": content}], enable_thinking=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("bundle")
    p.add_argument("device_bundle")
    p.add_argument("checkpoint", help="the directory of tokenizer.json and chat_template.jinja")
    p.add_argument("--max-pos", type=int, default=262144)
    p.add_argument("--tokens", type=int, default=261000, help="the prompt's length, about")
    p.add_argument("--depth", type=float, default=0.1, help="where the key sits, a fraction of the filler")
    p.add_argument("--processes", default="auto", help="comma-separated: the DSP processes of each run (Decoder's processes)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--new", type=int, default=12, help="the tokens of the reply")
    args = p.parse_args()
    text = Text(args.checkpoint)
    key = random.Random(args.seed).randint(10000, 99999)
    ids = prompt_ids(text, args.tokens, key, args.depth)
    assert len(ids) + args.new <= args.max_pos, f"a prompt of {len(ids)} tokens and its reply pass --max-pos {args.max_pos}"
    print(f"key {key}; prompt {len(ids)} tokens; the key at {args.depth:.0%} of the filler", flush=True)
    runs = []
    for procs in args.processes.split(","):
        procs = procs if procs == "auto" else int(procs)
        t0 = time.time()
        decoder = Decoder(args.bundle, args.device_bundle, max_pos=args.max_pos, max_tokens=args.max_pos, processes=procs)
        stages = " ".join(f"[{st.layers.start}, {st.layers.stop})" for st in decoder.stages)
        print(f"processes {procs}: session ready in {time.time() - t0:.0f} s; the layers' stages: {stages}", flush=True)
        try:
            timings, out = [], []
            for part in decoder.stream(ids, args.new, chunk=args.new, timings=timings):
                out += part
            logits = torch.frombuffer(bytearray(decoder.logits.read()), dtype=torch.float32).clone()
        finally:
            decoder.close()
        prefill = sum(s for kernel, _, s in timings if kernel == "prefill")
        steps = sum(n for kernel, n, _ in timings if kernel == "decode")
        decode = sum(s for kernel, _, s in timings if kernel == "decode")
        reply = text.decode(out)
        print(
            f"processes {procs}: reply {reply!r} ({'found' if str(key) in reply else 'NOT found'}); "
            f"prefill of {len(ids) - 1} tokens {prefill:.1f} s ({(len(ids) - 1) / prefill:.0f} t/s); "
            f"decode {steps / decode:.2f} t/s",
            flush=True,
        )
        runs.append((procs, out, logits))
    for procs, out, logits in runs[1:]:
        diff = (logits - runs[0][2]).abs().max().item()
        print(
            f"processes {procs} against {runs[0][0]}: tokens {'the same' if out == runs[0][1] else 'DIFFERENT'}; logits within {diff:.3g}"
        )


if __name__ == "__main__":
    main()
