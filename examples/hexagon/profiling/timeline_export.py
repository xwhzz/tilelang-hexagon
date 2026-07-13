#!/usr/bin/env python3
"""Reconstruct an nsys-style execution timeline from ggml-hexagon PROFILE=2 logs.

Each `profile-op` line carries a DSP `start` cycle-timestamp, duration (usec/cycles),
the kernel/engine (kparams), and tensor names (-> layer index + role). We unwrap the
32-bit cycle counter, convert to microseconds, tag each op with engine + layer + phase,
and emit a compact JSON the HTML timeline renders as zoomable colored bars.

Usage: python3 timeline_export.py <logfile> <out.json>
"""
import json
import re
import sys

LOG = sys.argv[1] if len(sys.argv) > 1 else "/tmp/prof_pmu.log"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/tmp/timeline.json"

pat = re.compile(
    r"profile-op (\w+)\|([^|]*)\|([^|]*)\|([^|]*)\|([^|]*)\|([^|]*)\|"
    r"usec (\d+) cycles (\d+) start (\d+) mhz ([\d.]+)(?: pmu \[([\d,]+)\])?"
)

ENGINE = [  # kparams prefix -> (engine lane, pretty)
    ("hmx", "HMX"),
    ("hvx", "HVX"),
]


def engine_of(kparams):
    k = kparams.strip().split()[0] if kparams.strip() and kparams != "----" else ""
    for pre, name in ENGINE:
        if k.startswith(pre):
            return name, k
    return "scalar/DSP", (k or "scalar")


def layer_of(names):
    # blk.N.  |  cache_k_lN  |  _lN
    m = re.search(r"blk\.(\d+)\.", names) or re.search(r"_l(\d+)\b", names) \
        or re.search(r"cache_[krv]_l(\d+)", names)
    return int(m.group(1)) if m else -1


def role_of(op, names):
    w = names.split(" x ")[0].strip()
    if op == "MUL_MAT":
        r = re.sub(r"blk\.\d+\.", "", w).replace(".weight", "")
        return r
    return op.lower()


rows = []
for line in open(LOG, encoding="latin1"):
    if "profile-op" not in line:
        continue
    m = pat.search(line)
    if not m:
        continue
    op, names, dims, types, strides, kparams, usec, cyc, start, mhz, pmu = m.groups()
    eng, kern = engine_of(kparams)
    rows.append({
        "op": op, "eng": eng, "kern": kern, "L": layer_of(names),
        "role": role_of(op, names),
        "dims": dims.split(" -> ")[0].strip(),
        "dt": types.split(" -> ")[0].strip(),
        "usec": int(usec), "cyc": int(cyc), "start": int(start), "mhz": float(mhz),
        "pmu": [int(x) for x in pmu.split(",")] if pmu else [],
    })

print(f"parsed {len(rows)} op instances")

# The per-op `start` cycle-counter is a per-core free-running counter (6 HW threads +
# op batching), so it is NOT a global monotonic clock — ~half the ops appear to go
# backwards. The log ORDER, however, is faithful ggml execution order (each op's profile
# is dumped as its response is processed, in graph-topological order). So we build the
# timeline by PACKING measured durations back-to-back in execution order: a true
# compute timeline (no host-logging gaps, no cross-core clock skew).
WRAP = 1 << 32
acc = 0
prev = None
raw_bad = 0
for r in rows:  # keep a best-effort real start too (diagnostic only)
    s = r["start"]
    if prev is not None and s < prev - WRAP // 4:
        acc += WRAP
    if prev is not None and (acc + s) < prev_abs - 1:
        raw_bad += 1
    prev = s
    prev_abs = acc + s

t = 0.0
for r in rows:
    r["t"] = t          # packed start (us)
    t += r["usec"]
span_pack = t
span_real = None
print(f"start-counter non-monotonic steps: {raw_bad}/{len(rows)} (per-core counter, unusable as clock)")
print(f"packed compute span (sum of durations, = timeline length): {span_pack/1000:.1f} ms")

# --- detect prefill -> decode boundary: prefill has batched MUL_MAT (act ne1>8) ---
def act_ne1(dims):
    parts = re.split(r"\s*x\s*", dims)
    if len(parts) < 2:
        return 1
    nums = re.findall(r"(\d+)", parts[1])
    return int(nums[1]) if len(nums) > 1 else 1

pref_end = 0
for i, r in enumerate(rows):
    if r["op"] == "MUL_MAT" and act_ne1(r["dims"]) > 8:
        pref_end = i
prefill_ops = pref_end + 1
print(f"prefill: ops[0:{prefill_ops}]  decode: ops[{prefill_ops}:{len(rows)}]")

# per-op-type + engine totals (packed)
from collections import defaultdict
tot = defaultdict(lambda: [0, 0])
eng_tot = defaultdict(lambda: [0, 0])
for r in rows:
    tot[r["op"]][0] += r["usec"]; tot[r["op"]][1] += 1
    eng_tot[r["eng"]][0] += r["usec"]; eng_tot[r["eng"]][1] += 1
print("\nby engine (packed us):")
for e, (u, n) in sorted(eng_tot.items(), key=lambda x: -x[1][0]):
    print(f"  {e:<12}{u:>8} us  {100*u/span_pack:>5.1f}%  x{n}")

# --- compact export: intern op names, roles, dims, dtypes ---
op_names = sorted({r["op"] for r in rows})
roles = sorted({r["role"] for r in rows})
dims_tab = sorted({r["dims"] for r in rows})
dt_tab = sorted({r["dt"] for r in rows})
engs = ["HMX", "HVX", "scalar/DSP"]
oi = {v: i for i, v in enumerate(op_names)}
ri = {v: i for i, v in enumerate(roles)}
di = {v: i for i, v in enumerate(dims_tab)}
ti = {v: i for i, v in enumerate(dt_tab)}
ei = {v: i for i, v in enumerate(engs)}

recs = []
for r in rows:
    recs.append([
        round(r["t"], 2),           # 0 start us (real)
        r["usec"],                  # 1 duration us
        oi[r["op"]],                # 2 op idx
        ei[r["eng"]],               # 3 engine idx
        r["L"],                     # 4 layer (-1 unknown)
        ri[r["role"]],              # 5 role idx
        di[r["dims"]],              # 6 dims idx
        ti[r["dt"]],                # 7 dtype idx
    ])

# --- step boundaries: a new forward pass starts at layer-0 GET_ROWS ---
step_idx = [i for i, r in enumerate(rows) if r["op"] == "GET_ROWS" and r["L"] == 0]
if not step_idx or step_idx[0] != 0:
    step_idx = [0] + step_idx
steps = []
for si, i0 in enumerate(step_idx):
    i1 = step_idx[si + 1] if si + 1 < len(step_idx) else len(rows)
    toks = 1
    for r in rows[i0:i1]:
        if r["op"] == "MUL_MAT":
            toks = max(toks, act_ne1(r["dims"]))
    steps.append([round(rows[i0]["t"], 1), toks])  # [packed_start_us, token_count]
print(f"steps: {len(steps)}  token-counts: {[s[1] for s in steps][:6]}...{[s[1] for s in steps][-3:]}")

out = {
    "meta": {
        "device": "OnePlus PJZ110 · Snapdragon 8 Elite (SM8750) · Hexagon v79",
        "npu": "6 threads, 6 HVX, 1 HMX, 8 MB VTCM",
        "model": "LFM2-1.2B-Q4_0 (16 layers; attn=[2,5,8,10,12,14], rest short-conv)",
        "runtime": "llama.cpp 4fc4ec55 ggml-hexagon (stock skel), GGML_HEXAGON_PROFILE=2",
        "workload": "~35-tok prefill + 32 decode tokens",
        "n_ops": len(rows),
        "span_pack_ms": round(span_pack / 1000, 1),
        "prefill_ops": prefill_ops,
    },
    "op_names": op_names, "roles": roles, "dims": dims_tab, "dtypes": dt_tab,
    "engines": engs,
    "steps": steps,
    "recs": recs,
}
json.dump(out, open(OUT, "w"), separators=(",", ":"))
import os
print(f"\nwrote {OUT}  ({os.path.getsize(OUT)//1024} KB, {len(recs)} recs)")
