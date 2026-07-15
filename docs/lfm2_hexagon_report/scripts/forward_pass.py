#!/usr/bin/env python3
"""Reconstruct LFM2-1.2B's full forward pass from a ggml-hexagon PROFILE=2 log.

Walks one decode token's ops in execution order, assigns each to a layer, and prints:
  - the exact op sequence of a short-conv layer and an attention layer (the two archetypes)
  - per-layer totals (conv vs attention layers)
  - the whole-forward-pass breakdown, incl. what falls to the CPU (lm_head)

Usage: python3 forward_pass.py <prof_pmu.log>
"""
import re
import sys
from collections import defaultdict

LOG = sys.argv[1] if len(sys.argv) > 1 else "/tmp/prof_pmu.log"
ATTN_LAYERS = {2, 5, 8, 10, 12, 14}          # LFM2: the rest are gated short-conv
N_LAYERS = 16

pat = re.compile(
    r"profile-op ([\w+]+)\|([^|]*)\|([^|]*)\|([^|]*)\|[^|]*\|([^|]*)\|usec (\d+) cycles (\d+)")

ops = []
for line in open(LOG, encoding="latin1"):
    if "profile-op" not in line:
        continue
    m = pat.search(line)
    if not m:
        continue
    op, names, dims, types, kparams, usec, cyc = m.groups()
    ops.append({"op": op, "names": names, "dims": dims.split(" -> ")[0].strip(),
                "dt": types.split(" -> ")[0].strip(),
                "kern": (kparams.strip() or "-"), "usec": int(usec), "cyc": int(cyc)})


def act_ne1(dims):
    p = dims.split(" x ")
    if len(p) < 2:
        return 1
    n = re.findall(r"\d+", p[1])
    return int(n[1]) if len(n) > 1 else 1


# --- isolate one decode token (ne1==1) -------------------------------------
# a new forward pass starts when the layer index resets; carry layer forward
def layer_of(names):
    m = re.search(r"blk\.(\d+)\.", names) or re.search(r"cache_[krv]_l(\d+)", names) \
        or re.search(r"_l(\d+)\b", names)
    return int(m.group(1)) if m else None


cur = None
for o in ops:
    L = layer_of(o["names"])
    if L is not None:
        cur = L
    o["L"] = cur if cur is not None else -1

# decode region = after the last batched matmul
last_pref = max(i for i, o in enumerate(ops)
                if o["op"].startswith("MUL_MAT") and act_ne1(o["dims"]) > 1)
dec = ops[last_pref + 1:]

# split decode into tokens: a token boundary is where layer goes back to 0 after being high
toks, cur_tok = [], []
prev_L = -1
for o in dec:
    if o["L"] == 0 and prev_L >= N_LAYERS - 3 and cur_tok:
        toks.append(cur_tok); cur_tok = []
    cur_tok.append(o)
    if o["L"] >= 0:
        prev_L = o["L"]
if cur_tok:
    toks.append(cur_tok)
# pick a full representative token (median length)
toks = [t for t in toks if len(t) > 100]
tok = sorted(toks, key=len)[len(toks) // 2]
total = sum(o["usec"] for o in tok)

print(f"# LFM2-1.2B 前向过程(一个 decode token)")
print(f"# {len(tok)} 个 NPU 算子, DSP 计算合计 {total/1000:.2f} ms\n")


def show_layer(L, title):
    seg = [o for o in tok if o["L"] == L]
    if not seg:
        return
    t = sum(o["usec"] for o in seg)
    print(f"── {title}(blk.{L}) — {len(seg)} 算子, {t/1000:.2f} ms ──")
    for o in seg:
        w = o["names"].split(" x ")[0].strip()
        w = re.sub(r"blk\.\d+\.", "", w)[:30]
        eng = "HMX" if o["kern"].startswith("hmx") else ("HVX" if o["kern"].startswith("hvx") else "vec/mem")
        print(f"   {o['usec']:>4}us  {o['op']:<24}{eng:<8}{w:<30}{o['dims']}")
    print()


# the two archetypes
conv_L = next(l for l in range(N_LAYERS) if l not in ATTN_LAYERS)
attn_L = min(ATTN_LAYERS)
show_layer(conv_L, "短卷积层")
show_layer(attn_L, "注意力层")

# --- per-layer totals ------------------------------------------------------
print("── 每层耗时 ──")
per = defaultdict(int)
for o in tok:
    per[o["L"]] += o["usec"]
conv_t = sum(v for k, v in per.items() if 0 <= k < N_LAYERS and k not in ATTN_LAYERS)
attn_t = sum(v for k, v in per.items() if k in ATTN_LAYERS)
n_conv = sum(1 for k in per if 0 <= k < N_LAYERS and k not in ATTN_LAYERS)
n_attn = sum(1 for k in per if k in ATTN_LAYERS)
for L in range(N_LAYERS):
    if L in per:
        kind = "attn" if L in ATTN_LAYERS else "conv"
        bar = "█" * int(per[L] / max(per.values()) * 30)
        print(f"   blk.{L:<2} {kind}  {per[L]:>5}us  {bar}")
print(f"\n   短卷积层 ×{n_conv}: {conv_t/1000:.2f} ms ({100*conv_t/total:.0f}%)  平均 {conv_t/max(n_conv,1)/1000:.2f} ms/层")
print(f"   注意力层 ×{n_attn}: {attn_t/1000:.2f} ms ({100*attn_t/total:.0f}%)  平均 {attn_t/max(n_attn,1)/1000:.2f} ms/层")
other = total - conv_t - attn_t
print(f"   层外(embedding 等): {other/1000:.2f} ms ({100*other/total:.0f}%)")
