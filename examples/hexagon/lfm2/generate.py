"""Chat with LFM2 on the device. The tokenizer and the chat template run on
the host (``text.py``); the prefill, the decode and the sampling run on the
device, and the reply shows as it comes, a decode call per 16 tokens.

    python generate.py <bundle> <bundle on the device> <checkpoint dir> --prompt "What is the capital of France?"
    python generate.py <bundle> <bundle on the device> <checkpoint dir> --chat

Each `--prompt` (it repeats) is answered from a fresh conversation; `--chat`
holds one conversation from stdin, a turn a line. `--temperature` samples, with
`--seed` (greedy at 0, the default). The checkpoint directory needs only the
tokenizer's files.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decode import Decoder  # noqa: E402
from text import Detokenizer, Text  # noqa: E402


def reply(decoder: Decoder, text: Text, messages: list, args) -> str:
    """The assistant's next message, printed as it comes, then the speeds."""
    prompt = text.chat(messages)
    shown, ids, timings = Detokenizer(text), [], []
    t0 = time.perf_counter()
    for part in decoder.stream(prompt, args.max_new, args.temperature, args.seed, timings=timings):
        stop = text.eos in part
        part = part[: part.index(text.eos)] if stop else part
        ids += part
        print(shown.add(part), end="", flush=True)
        if stop:
            break
    wall = time.perf_counter() - t0
    pre = sum(s for kernel, _, s in timings if kernel == "prefill")
    steps = sum(n for kernel, n, _ in timings if kernel == "decode")
    dec = sum(s for kernel, _, s in timings if kernel == "decode")
    print(
        f"\n[prompt {len(prompt)} tokens, prefill {pre * 1e3:.0f} ms; {len(ids)} tokens, decode {steps / dec:.1f} t/s; {wall:.2f} s in all]"
    )
    return text.decode(ids)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("bundle")
    p.add_argument("device_bundle")
    p.add_argument("checkpoint", help="the directory of tokenizer.json and chat_template.jinja")
    p.add_argument("--prompt", action="append", help="a message to answer, from a fresh conversation (repeats)")
    p.add_argument("--chat", action="store_true", help="a conversation from stdin, a turn a line")
    p.add_argument("--system", help="a system message")
    p.add_argument("--max-new", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    assert args.prompt or args.chat, "--prompt or --chat"
    text = Text(args.checkpoint)
    t0 = time.time()
    decoder = Decoder(args.bundle, args.device_bundle)
    print(f"[session ready in {time.time() - t0:.0f} s]", file=sys.stderr)
    messages = [{"role": "system", "content": args.system}] if args.system else []
    try:
        for prompt in args.prompt or []:
            print(f"> {prompt}")
            reply(decoder, text, messages + [{"role": "user", "content": prompt}], args)
        if args.prompt:
            return
        for line in sys.stdin:
            if not line.strip():
                continue
            messages.append({"role": "user", "content": line.strip()})
            messages.append({"role": "assistant", "content": reply(decoder, text, messages, args)})
    finally:
        decoder.close()


if __name__ == "__main__":
    main()
