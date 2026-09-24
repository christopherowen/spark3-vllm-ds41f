#!/usr/bin/env python3
"""Compare arms: mean of two passes per point, pass range, LRU, memory.

usage: summarize.py RUNS_DIR LABEL [LABEL...]   (first label is the reference)
"""
import json, sys
from pathlib import Path

root, labels = Path(sys.argv[1]), sys.argv[2:]
points = [f"{case}-c{c}" for case in ("prose", "code") for c in (1, 2, 4, 8)]


def arm(label):
    out = {}
    for point in points + ["prose-c1-repeat", "code-c1-repeat"]:
        vals = []
        for p in (1, 2):
            f = root / f"{label}-p{p}" / f"{point}.json"
            if f.exists():
                j = json.loads(f.read_text())
                vals.append((j["aggregate_e2e_tps"], j["mean_ttft_s"]))
        out[point] = vals
    lru = []
    for p in (1, 2):
        f = root / f"{label}-p{p}" / "lru.json"
        if f.exists():
            runs = json.loads(f.read_text())["runs"]
            lru.append(f"{sum(r.get('verdict') == 'pass' for r in runs)}/{len(runs)}")
    mem = root / f"{label}-mem.json"
    return out, lru, json.loads(mem.read_text()) if mem.exists() else {}


data = {label: arm(label) for label in labels}
ref = data[labels[0]][0]
print(f"{'point':16s}" + "".join(f"{l:>30s}" for l in labels))
for point in points:
    row = f"{point:16s}"
    base = [v for v, _ in ref[point]] + [v for v, _ in ref.get(point.replace('c1', 'c1-repeat'), [])] if point.endswith("c1") else [v for v, _ in ref[point]]
    base_mean = sum(base) / len(base) if base else None
    for label in labels:
        vals = [v for v, _ in data[label][0][point]]
        if point.endswith("c1"):
            vals += [v for v, _ in data[label][0][point.replace("c1", "c1-repeat")]]
        if not vals:
            row += f"{'-':>30s}"
            continue
        mean = sum(vals) / len(vals)
        delta = "" if label == labels[0] or not base_mean else f" {100 * (mean / base_mean - 1):+5.1f}%"
        row += f"{mean:9.1f} [{min(vals):6.1f}-{max(vals):6.1f}]{delta:>8s}"
    print(row)
for label in labels:
    _, lru, mem = data[label]
    print(f"{label}: LRU {lru} min MemAvailable {mem}")
