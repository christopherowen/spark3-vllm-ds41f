#!/bin/bash
# usage: run_arm.sh <profile>   (runs on dgx1 against the local API)
# Serving matrix (prose/code, c1-c8, 256 tokens) then the five-repeat LRU gate.
set -u
P=$1; OUT=/tmp/claude-runs/$P; mkdir -p "$OUT"
cd ~/projects/spark3-vllm-ds41f
B=experiments/2026-09-20-upstream-main-rebase/benchmark_serving.py
Q=experiments/2026-09-23-consolidated-quality/probe_quality.py
U=http://127.0.0.1:8000
minmem() { awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo; }
echo "start $(date -u +%FT%TZ) mem=$(minmem)GiB" | tee "$OUT/log.txt"
python3 $B --base-url $U --case prose --concurrency 1 --output-tokens 64 --output "$OUT/warmup.json" >/dev/null
for case in prose code; do
  for c in 1 2 4 8; do
    python3 $B --base-url $U --case $case --concurrency $c --output-tokens 256 --output "$OUT/$case-c$c.json" >/dev/null
    echo "$case c$c rc=$? mem=$(minmem)GiB" | tee -a "$OUT/log.txt"
  done
done
python3 $B --base-url $U --case prose --concurrency 1 --output-tokens 256 --output "$OUT/prose-c1-repeat.json" >/dev/null
python3 $B --base-url $U --case code --concurrency 1 --output-tokens 256 --output "$OUT/code-c1-repeat.json" >/dev/null
python3 $Q --url $U --profile "$P" --repeats 5 --output "$OUT/lru.json" | tee -a "$OUT/log.txt"
echo "end $(date -u +%FT%TZ) mem=$(minmem)GiB" | tee -a "$OUT/log.txt"
