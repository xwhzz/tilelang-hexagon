#!/usr/bin/env python3
"""Emit ggml-hexagon PROFILE=2 per-op data as a Chrome-trace / Perfetto JSON —
the same format itrace's ITRACE_JSON_FILE produces, but from ggml's in-context
per-op PMU reads (the correct per-op attribution on Hexagon, whose PMU counters
are per-hardware-thread).

Open the output at https://ui.perfetto.dev (drag the .json) or chrome://tracing.

Tracks (tids):
  0  HMX (matrix)          - ops the backend ran on the matrix engine
  1  HVX matmul/attn       - hvx-tiled GEMV + flash-attn
  2  HVX elementwise       - add/mul/swiglu/conv/concat/cpy/rope (untagged, HVX)
Counter tracks (line graphs): HVX_ACTIVE, committed_pkts, AXI_wr  (per op)

Usage: python3 chrome_trace_export.py <prof_pmu.log> <out.json>
"""
import json
import re
import sys

LOG = sys.argv[1] if len(sys.argv) > 1 else "/tmp/prof_pmu.log"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/tmp/lfm2_perfetto_trace.json"

# ggml opt_pmu_evt order -> our counter meaning
PMU_COMMITTED_PKT, PMU_HVX_ACTIVE, PMU_AXI_WR = 0, 2, 4

pat = re.compile(
    r"profile-op (\w+)\|([^|]*)\|([^|]*)\|([^|]*)\|([^|]*)\|([^|]*)\|"
    r"usec (\d+) cycles (\d+) start (\d+) mhz ([\d.]+)(?: pmu \[([\d,]+)\])?"
)


def engine_lane(op, kparams):
    k = kparams.strip().split()[0] if kparams.strip() and kparams != "----" else ""
    if k.startswith("hmx"):
        return 0
    if k.startswith("hvx"):
        return 1
    return 2  # untagged elementwise, still HVX


def layer_of(names):
    m = re.search(r"blk\.(\d+)\.", names) or re.search(r"_l(\d+)\b", names) \
        or re.search(r"cache_[krv]_l(\d+)", names)
    return int(m.group(1)) if m else -1


rows = []
for line in open(LOG, encoding="latin1"):
    if "profile-op" not in line:
        continue
    m = pat.search(line)
    if not m:
        continue
    op, names, dims, types, strides, kparams, usec, cyc, start, mhz, pmu = m.groups()
    rows.append({
        "op": op, "lane": engine_lane(op, kparams), "L": layer_of(names),
        "role": re.sub(r"blk\.\d+\.", "", names.split(" x ")[0].strip()).replace(".weight", ""),
        "dims": dims.split(" -> ")[0].strip(), "dt": types.split(" -> ")[0].strip(),
        "kern": kparams.strip() or "-",
        "usec": int(usec), "cyc": int(cyc),
        "pmu": [int(x) for x in pmu.split(",")] if pmu else [0] * 8,
    })

print(f"parsed {len(rows)} ops")

LANE_NAMES = {0: "HMX (matrix)", 1: "HVX matmul/attn", 2: "HVX elementwise"}
PID = 1
events = []

# process + thread (track) metadata
events.append({"ph": "M", "pid": PID, "name": "process_name", "args": {"name": "LFM2-1.2B-Q4_0 · Hexagon v79 (SM8750)"}})
for tid, nm in LANE_NAMES.items():
    events.append({"ph": "M", "pid": PID, "tid": tid, "name": "thread_name", "args": {"name": nm}})

# packed compute timeline: durations back-to-back in execution order
t = 0.0
for r in rows:
    events.append({
        "ph": "X", "pid": PID, "tid": r["lane"], "ts": round(t, 3), "dur": r["usec"],
        "name": r["op"] + (f' {r["role"]}' if r["op"] == "MUL_MAT" else ""),
        "args": {
            "layer": (f'blk.{r["L"]}' if r["L"] >= 0 else "-"),
            "role": r["role"], "shape": r["dims"], "dtype": r["dt"], "kernel": r["kern"],
            "usec": r["usec"], "cycles": r["cyc"],
            "HVX_ACTIVE(cyc)": r["pmu"][PMU_HVX_ACTIVE],
            "HVX_ACTIVE/cycle": round(r["pmu"][PMU_HVX_ACTIVE] / r["cyc"], 2) if r["cyc"] else 0,
            "committed_pkts": r["pmu"][PMU_COMMITTED_PKT],
            "AXI_write_req": r["pmu"][PMU_AXI_WR],
        },
    })
    # counter tracks (line graphs) sampled at each op's start
    events.append({"ph": "C", "pid": PID, "ts": round(t, 3), "name": "HVX_ACTIVE (cyc/op)",
                   "args": {"HVX_ACTIVE": r["pmu"][PMU_HVX_ACTIVE]}})
    events.append({"ph": "C", "pid": PID, "ts": round(t, 3), "name": "committed_pkts (/op)",
                   "args": {"committed_pkts": r["pmu"][PMU_COMMITTED_PKT]}})
    events.append({"ph": "C", "pid": PID, "ts": round(t, 3), "name": "AXI_write_req (/op)",
                   "args": {"AXI_write_req": r["pmu"][PMU_AXI_WR]}})
    t += r["usec"]

trace = {
    "traceEvents": events,
    "displayTimeUnit": "ns",
    "metadata": {
        "source": "ggml-hexagon GGML_HEXAGON_PROFILE=2 (in-context per-op PMU)",
        "device": "OnePlus PJZ110 · Snapdragon 8 Elite SM8750 · Hexagon v79",
        "note": "packed compute timeline (per-HW-thread cycle counters are not a shared "
                "clock; durations laid end-to-end in ggml execution order). Counters are "
                "real on-silicon PMU: HVX_ACTIVE=0x100, committed=0x3, AXI_wr=0x42.",
        "total_ms": round(t / 1000, 1), "n_ops": len(rows),
    },
}
json.dump(trace, open(OUT, "w"), separators=(",", ":"))
import os
print(f"wrote {OUT}  ({os.path.getsize(OUT)//1024} KB, {len(events)} events, {t/1000:.1f} ms)")
print("open at https://ui.perfetto.dev (drag the file) or chrome://tracing")
