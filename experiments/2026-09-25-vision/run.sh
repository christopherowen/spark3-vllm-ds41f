#!/bin/bash
# usage: run.sh [--idle SECONDS]   (on dgx1, deployment checkout at this
# experiment's commit)
# Starts the vision arm in place of whatever is running (no rollback), with a
# memory trace on dgx1 that runs until the checks finish, then checks images
# and text if it comes up.
set -u
IDLE=0
[ "${1:-}" = --idle ] && IDLE=$2
cd ~/projects/spark3-vllm-ds41f
E=experiments/2026-09-25-vision
# A watermark boost hides up to 0.87 GiB from MemAvailable on dgx1 (README).
if [ "$(cat /proc/sys/vm/watermark_boost_factor)" != 0 ]; then
  echo "refusing: vm.watermark_boost_factor is not 0 on $(hostname)" >&2
  exit 1
fi
TRACE=/tmp/vision-memtrace-$(date -u +%H%M%S).txt
bin/spark3 cluster stop --remove --apply >/dev/null 2>&1
bin/spark3 --cluster-config $E/cluster-vision.json cluster stop --remove --apply >/dev/null 2>&1
$E/memtrace.sh "$TRACE" & TRACER=$!
bin/spark3 --cluster-config $E/cluster-vision.json cluster start --replace --apply | grep -v 'docker run'
started=${PIPESTATUS[0]}
mark() { echo "$(date +%T.%N | cut -c1-12) X $*" >> "$TRACE"; }
trap 'kill $TRACER' EXIT
echo "memory trace: $TRACE"
mark "start exit $started"
[ "$started" -eq 0 ] || exit 1
bin/spark3 --cluster-config $E/cluster-vision.json doctor --live || exit 1
mark "doctor done"
python3 $E/vision_check.py || exit 1
mark "vision check done"
if [ "$IDLE" -gt 0 ]; then
  # Trace the idle service instead of benchmarking it.
  sleep "$IDLE"
  mark "idle done"
  exit 0
fi
bin/spark3 --cluster-config $E/cluster-vision.json bench --compare none --output results/private/bench/vision || exit 1
mark "bench done"
echo "vision arm ready"
