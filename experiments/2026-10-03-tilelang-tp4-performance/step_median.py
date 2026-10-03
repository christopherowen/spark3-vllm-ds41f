"""Median per-step main-stream time by kernel (name, grid, block) over steady 6-row decode steps."""
import gzip, json, statistics, sys
from collections import Counter, defaultdict
events = json.load(gzip.open(sys.argv[1], "rt"))["traceEvents"]
k = sorted((e for e in events if e.get("cat") == "kernel" and e.get("ph") == "X"), key=lambda e: e["ts"])
main = Counter(e["args"].get("stream") for e in k).most_common(1)[0][0]
seq = [e for e in k if e["args"]["stream"] == main]
marks = [i for i, e in enumerate(seq) if "ConcatAndCacheGlmNextMla" in e["name"]]
steps = []
for i in range(0, len(marks) - 40, 40):
    a, b = marks[i], marks[i + 40]
    rows = seq[a]["args"]["grid"][0]  # cache writer grid = rows in the step
    if rows != 6:
        continue
    per = defaultdict(float)
    for e in seq[a:b]:
        n = e["name"]
        key = (n if n.startswith(("main_kernel", "per_token", "swiglu", "norm_forward")) else n[:240], str(e["args"]["grid"]), str(e["args"]["block"]))
        per[key] += e["dur"] / 1000
    wall = (seq[b - 1]["ts"] + seq[b - 1]["dur"] - seq[a]["ts"]) / 1000
    steps.append((wall, sum(per.values()), per))
print(f"6-row steps: {len(steps)}; median wall {statistics.median(s[0] for s in steps):.2f} ms; median main-stream busy {statistics.median(s[1] for s in steps):.2f} ms")
keys = set().union(*(s[2] for s in steps))
med = {key: statistics.median(s[2].get(key, 0.0) for s in steps) for key in keys}
json.dump({"|".join(k): v for k, v in med.items()}, open(sys.argv[2], "w"))
for key, v in sorted(med.items(), key=lambda kv: -kv[1])[:int(sys.argv[3])]:
    print(f"  {v:6.3f} ms  {key[1]:14s} {key[2]:14s} {key[0][:70]}")
