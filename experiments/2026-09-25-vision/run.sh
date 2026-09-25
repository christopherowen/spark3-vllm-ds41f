#!/bin/bash
# usage: run.sh   (on dgx1, deployment checkout at this experiment's commit)
# Replaces the promoted service with the vision arm, checks images and text,
# and restores the promoted service if any step fails.
set -u
cd ~/projects/spark3-vllm-ds41f
E=experiments/2026-09-25-vision
restore() {
  bin/spark3 --cluster-config $E/cluster-vision.json cluster stop --remove --apply
  bin/spark3 cluster start --replace --apply | grep -v 'docker run'
  bin/spark3 doctor --live
  echo "restored the promoted service"
}
bin/spark3 cluster stop --remove --apply || exit 1
bin/spark3 --cluster-config $E/cluster-vision.json cluster start --replace --apply | grep -v 'docker run'
[ "${PIPESTATUS[0]}" -eq 0 ] || { restore; exit 1; }
bin/spark3 --cluster-config $E/cluster-vision.json doctor --live || { restore; exit 1; }
python3 $E/vision_check.py || { restore; exit 1; }
bin/spark3 --cluster-config $E/cluster-vision.json bench --compare none --output results/private/bench/vision || { restore; exit 1; }
echo "vision arm ready"
