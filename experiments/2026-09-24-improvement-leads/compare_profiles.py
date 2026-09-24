#!/usr/bin/env python3
"""Compare rank-0 decode profiles: GPU time per decode step and big GEMM kernels.

usage: compare_profiles.py LABEL=DIR [LABEL=DIR ...]
"""
import collections, glob, gzip, json, os, sys


def load(directory):
    trace = json.load(gzip.open(glob.glob(os.path.join(directory, "*.trace.json.gz"))[0]))
    events = [e for e in trace["traceEvents"] if e.get("ph") == "X"]
    kernels = [e for e in events if e.get("cat") == "kernel"]
    steps = [e for e in events if e["name"].startswith("execute_context_0")]
    return kernels, steps


for arg in sys.argv[1:]:
    label, directory = arg.split("=", 1)
    kernels, steps = load(directory)
    total = sum(e["dur"] for e in kernels) / 1e3
    big = collections.defaultdict(list)
    for e in kernels:
        if "dense_gemm" in e["name"].lower() and e["dur"] > 150:
            big[round(e["dur"] / 50) * 50].append(e["dur"])
    allreduce = sum(e["dur"] for e in kernels if "roce" in e["name"].lower()) / 1e3
    print(f"{label}: decode steps {len(steps)}, kernel time {total:.1f} ms "
          f"({total / max(len(steps), 1):.2f} ms/step), RoCE {allreduce:.1f} ms")
    for bucket in sorted(big):
        durations = big[bucket]
        print(f"   dense GEMM ~{bucket} us: {len(durations)} calls, "
              f"{sum(durations) / 1e3:.1f} ms total")
