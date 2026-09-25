#!/bin/bash
# usage: memtrace.sh OUT   (on a node) until killed:
#   M  MemAvailable (KiB) every 0.2 s
#   P  the five largest processes by RSS (KiB) every second
#   I  every /proc/meminfo field (KiB) every second
#   B  the Normal zone's watermark boost (pages) every second
#   G  per-process CUDA memory (MiB) every 2 s
out=$1
i=0
while :; do
  now=$(date +%T.%N | cut -c1-12)
  printf '%s M %s\n' "$now" "$(awk '/MemAvailable/{print $2}' /proc/meminfo)" >> "$out"
  if [ $((i % 5)) -eq 0 ]; then
    ps -eo rss=,comm= --sort=-rss | head -5 | awk -v t="$now" '{printf "%s P %s %s\n", t, $1, $2}' >> "$out"
    printf '%s I %s\n' "$now" "$(awk '{sub(":", "", $1); printf "%s=%s ", $1, $2}' /proc/meminfo)" >> "$out"
    printf '%s B %s\n' "$now" "$(awk '/zone +Normal/{n=1} n&&/boost/{print $2; exit}' /proc/zoneinfo)" >> "$out"
  fi
  if [ $((i % 10)) -eq 0 ]; then
    printf '%s G %s\n' "$now" "$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null | tr '\n' ' ')" >> "$out"
  fi
  i=$((i + 1)); sleep 0.2
done
