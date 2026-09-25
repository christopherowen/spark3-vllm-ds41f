#!/bin/bash
# usage: memtrace.sh OUT   (on a node) MemAvailable every 0.2 s and the five
# largest processes by RSS every second, until killed.
out=$1
i=0
while :; do
  printf '%s M %s\n' "$(date +%T.%N | cut -c1-12)" "$(awk '/MemAvailable/{print $2}' /proc/meminfo)" >> "$out"
  if [ $((i % 5)) -eq 0 ]; then
    ps -eo rss=,comm= --sort=-rss | head -5 | awk -v t="$(date +%T)" '{printf "%s P %s %s\n", t, $1, $2}' >> "$out"
  fi
  i=$((i + 1)); sleep 0.2
done
