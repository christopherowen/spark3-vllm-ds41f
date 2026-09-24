#!/bin/bash
# usage: measure.sh LABEL   (on dgx1 against the running cluster: two serving
# matrices plus the LRU gate via run_arm.sh, with per-node minimum memory)
set -u
cd ~/projects/spark3-vllm-ds41f
D=experiments/2026-09-24-kk-speed-tuning
mkdir -p /tmp/claude-runs
bash $D/sample_mem.sh "/tmp/claude-runs/$1-mem.json" &
S=$!
for pass in 1 2; do bash $D/run_arm.sh "$1-p$pass" > /dev/null 2>&1; done
kill -TERM $S; wait $S
cat "/tmp/claude-runs/$1-mem.json"
grep -hE "/5:" "/tmp/claude-runs/$1-p1/log.txt" "/tmp/claude-runs/$1-p2/log.txt" | sort | uniq -c
