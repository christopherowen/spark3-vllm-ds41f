#!/bin/bash
# usage: run_arm.sh ARM LABEL [bench options...]   (on dgx1, from the deployment checkout)
# Stops the promoted service and every arm, starts cluster-ARM.json under the
# launcher's memory guards, then benchmarks it into results/private/bench/dsp-LABEL.
# The arm stays up; the next run_arm.sh (or restore.sh) stops it.
set -eu
cd ~/projects/spark3-vllm-ds41f
E=experiments/2026-09-26-dspark-policy
arm=$1
label=$2
shift 2
for config in config/cluster.json $E/cluster-*.json; do
  bin/spark3 --cluster-config "$config" cluster stop --remove --apply >/dev/null 2>&1 || true
done
for _ in $(seq 1 60); do
  avail=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
  [ "$avail" -ge 100 ] && break
  sleep 5
done
mkdir -p cache/kkref/dspark-trace
echo "$(date -u +%FT%TZ) start $arm as $label (dgx1 MemAvailable ${avail} GiB)"
bin/spark3 --cluster-config "$E/cluster-$arm.json" cluster start --replace --apply | grep -v 'docker run'
bin/spark3 --cluster-config "$E/cluster-$arm.json" doctor --live || true
# --allow-mismatch: doctor --live reports only the nvidia-drm modeset host
# setting (display), which does not touch inference.
bin/spark3 --cluster-config "$E/cluster-$arm.json" bench --allow-mismatch \
  --output "results/private/bench/dsp-$label" "$@"
