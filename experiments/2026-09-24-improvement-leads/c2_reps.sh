#!/bin/bash
# usage: c2_reps.sh LABEL   (dgx1, against the running service: five prose and
# five code runs at concurrency 2, 256 output tokens, alternating cases)
set -u
cd ~/projects/spark3-vllm-ds41f
B=experiments/2026-09-20-upstream-main-rebase/benchmark_serving.py
OUT=/tmp/claude-runs/$1-c2reps; mkdir -p "$OUT"
python3 $B --base-url http://127.0.0.1:8000 --case prose --concurrency 2 --output-tokens 64 --output "$OUT/warmup.json" > /dev/null
for i in 1 2 3 4 5; do
  for case in prose code; do
    python3 $B --base-url http://127.0.0.1:8000 --case $case --concurrency 2 --output-tokens 256 --output "$OUT/$case-$i.json" > /dev/null
  done
done
python3 - "$OUT" <<'PY'
import json, glob, statistics, sys
for case in ("prose", "code"):
    v = [json.load(open(f))["aggregate_e2e_tps"] for f in sorted(glob.glob(f"{sys.argv[1]}/{case}-*.json"))]
    print(f"{case} c2: mean {statistics.mean(v):.1f} sd {statistics.stdev(v):.1f} n {len(v)} {[round(x,1) for x in v]}")
PY
