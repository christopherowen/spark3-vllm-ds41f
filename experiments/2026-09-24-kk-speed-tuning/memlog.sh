#!/bin/bash
# usage: memlog.sh OUT   (on dgx1 until killed: UTC time and MemAvailable GiB
# for dgx1, dgx2, dgx3 every 10 s)
set -u
while true; do
  a=$(awk '/MemAvailable/{printf "%.2f", $2/1048576}' /proc/meminfo)
  b=$(ssh -o ConnectTimeout=5 192.168.0.2 "awk '/MemAvailable/{printf \"%.2f\", \$2/1048576}' /proc/meminfo" 2>/dev/null || echo NA)
  c=$(ssh -o ConnectTimeout=5 192.168.2.2 "awk '/MemAvailable/{printf \"%.2f\", \$2/1048576}' /proc/meminfo" 2>/dev/null || echo NA)
  echo "$(date -u +%H:%M:%S) $a $b $c" >> "$1"
  sleep 10 & wait $!
done
