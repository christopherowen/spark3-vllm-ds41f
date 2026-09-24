#!/bin/bash
# usage: sample_mem.sh OUT_JSON   (runs on dgx1 until killed; records the minimum
# MemAvailable per node, sampled every 2 s)
set -u
OUT=$1
m1=999999999 m2=999999999 m3=999999999
finish() {
  awk -v a="$m1" -v b="$m2" -v c="$m3" 'BEGIN {
    printf "{\"dgx1_min_gib\": %.2f, \"dgx2_min_gib\": %.2f, \"dgx3_min_gib\": %.2f}\n",
      a / 1048576, b / 1048576, c / 1048576 }' > "$OUT"
  exit 0
}
trap finish TERM INT
while true; do
  a=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
  b=$(ssh -o ConnectTimeout=5 192.168.0.2 "awk '/MemAvailable/{print \$2}' /proc/meminfo" 2>/dev/null || echo 999999999)
  c=$(ssh -o ConnectTimeout=5 192.168.2.2 "awk '/MemAvailable/{print \$2}' /proc/meminfo" 2>/dev/null || echo 999999999)
  [ "$a" -lt "$m1" ] && m1=$a
  [ "$b" -lt "$m2" ] && m2=$b
  [ "$c" -lt "$m3" ] && m3=$c
  sleep 2 & wait $!
done
