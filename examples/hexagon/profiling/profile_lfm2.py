#!/usr/bin/env python3
"""Parse ggml-hexagon GGML_HEXAGON_PROFILE=2 per-op logs into a detailed report:
per-op-type latency, per-instance breakdown, timeline, and the PMU counters.

Log line format (host GGML_LOG_DEBUG, needs --verbose):
  ... ggml-hex: HTP0 profile-op OP|names|dims|types|strides|kparams|usec N cycles C start S mhz M pmu [8 ints]

Usage: python3 profile_lfm2.py <logfile>
"""
import re
import sys
from collections import defaultdict

LOG = sys.argv[1] if len(sys.argv) > 1 else "/tmp/prof_pmu.log"

# PMU counter names for opt_profile==2 (order per ggml-hexagon skel; the first few are the
# load-bearing ones for a matmul: packets, HVX pkts, stalls, etc.)
PMU_NAMES = ["p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7"]

pat = re.compile(
    r"profile-op ([\w+]+)\|([^|]*)\|([^|]*)\|([^|]*)\|([^|]*)\|([^|]*)\|"
    r"usec (\d+) cycles (\d+) start (\d+) mhz ([\d.]+)(?: pmu \[([\d,]+)\])?"
)

ops = []
for line in open(LOG, encoding="latin1"):
    if "profile-op" not in line:
        continue
    m = pat.search(line)
    if not m:
        continue
    op, names, dims, types, strides, kparams, usec, cyc, start, mhz, pmu = m.groups()
    ops.append({
        "op": op, "names": names, "dims": dims, "types": types, "kparams": kparams,
        "usec": int(usec), "cycles": int(cyc), "start": int(start), "mhz": float(mhz),
        "pmu": [int(x) for x in pmu.split(",")] if pmu else None,
    })

if not ops:
    print("no profile-op lines found (need GGML_HEXAGON_PROFILE + --verbose, grep -a)")
    sys.exit(1)

total_us = sum(o["usec"] for o in ops)

# --- 1. per-op-type aggregate ---
agg = defaultdict(lambda: {"us": 0, "cyc": 0, "n": 0})
for o in ops:
    a = agg[o["op"]]
    a["us"] += o["usec"]; a["cyc"] += o["cycles"]; a["n"] += 1

print(f"# LFM2 per-op profile  ({len(ops)} op instances, total {total_us} usec = {total_us/1000:.1f} ms HTP compute)\n")
print(f"{'op':<16}{'total_us':>10}{'%':>6}{'count':>7}{'avg_us':>8}{'avg_cyc':>10}")
print("-" * 57)
for op, a in sorted(agg.items(), key=lambda x: -x[1]["us"]):
    print(f"{op:<16}{a['us']:>10}{100*a['us']/total_us:>5.1f}%{a['n']:>7}{a['us']/a['n']:>8.1f}{a['cyc']//a['n']:>10}")

# --- 2. MUL_MAT broken down by weight role (in_proj / gate / up / down / out / attn) ---
print("\n# MUL_MAT by weight tensor role")
role = defaultdict(lambda: {"us": 0, "n": 0, "dims": ""})
for o in ops:
    if o["op"] != "MUL_MAT":
        continue
    w = o["names"].split(" x ")[0].strip()
    r = re.sub(r"blk\.\d+\.", "", w).replace(".weight", "")
    role[r]["us"] += o["usec"]; role[r]["n"] += 1; role[r]["dims"] = o["dims"].split(" -> ")[0]
mm_us = sum(v["us"] for v in role.values())
print(f"{'role':<22}{'total_us':>10}{'%MM':>6}{'count':>7}{'avg_us':>8}  dims(w x act)")
print("-" * 78)
for r, v in sorted(role.items(), key=lambda x: -x[1]["us"]):
    print(f"{r:<22}{v['us']:>10}{100*v['us']/mm_us:>5.1f}%{v['n']:>7}{v['us']/v['n']:>8.1f}  {v['dims']}")

# --- 3. kernel-type distribution (kparams: hvx-tiled / hmx / etc.) ---
print("\n# kernel implementation used (kparams field)")
kern = defaultdict(lambda: {"us": 0, "n": 0})
for o in ops:
    k = o["kparams"].split()[0] if o["kparams"].strip() and o["kparams"] != "----" else "(none)"
    kern[k]["us"] += o["usec"]; kern[k]["n"] += 1
for k, v in sorted(kern.items(), key=lambda x: -x[1]["us"]):
    print(f"  {k:<20}{v['us']:>10} us  {100*v['us']/total_us:>5.1f}%   x{v['n']}")

# --- 4. timeline: bucket by wall-clock, show phase transition prefill->decode ---
print("\n# timeline (op instances in capture order, 20 buckets)")
nb = 20
per = max(1, len(ops) // nb)
for b in range(0, len(ops), per):
    chunk = ops[b:b + per]
    cus = sum(o["usec"] for o in chunk)
    top = defaultdict(int)
    for o in chunk:
        top[o["op"]] += o["usec"]
    dominant = max(top.items(), key=lambda x: x[1])[0]
    bar = "#" * int(40 * cus / max(sum(o["usec"] for o in ops[i:i+per]) for i in range(0, len(ops), per)))
    print(f"  [{b:>5}-{b+len(chunk):>5}] {cus:>6}us  {dominant:<14} {bar}")

# --- 5. the single most expensive instances ---
print("\n# top 10 slowest single op instances")
for o in sorted(ops, key=lambda x: -x["usec"])[:10]:
    print(f"  {o['usec']:>5}us  {o['op']:<14} {o['names'][:40]:<40} {o['dims'].split(' -> ')[0]}")
