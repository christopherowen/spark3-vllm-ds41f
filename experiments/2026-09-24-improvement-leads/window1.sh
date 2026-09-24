#!/bin/bash
# Maintenance window 1: dense MXFP8 GEMM plan sweeps at the DS4.1 TP3 shapes.
# Run on dgx1 from a clean checkout of main. Stops the promoted service, runs
# B12X's offline plan sweep in a memory-capped container under a host
# watchdog, then restarts the promoted service and checks it.
set -u
cd ~/projects/spark3-vllm-ds41f
D=experiments/2026-09-24-improvement-leads
OUT="$PWD/$D/runs/window1"
NAME=spark3-gemm-sweep
IMAGE=$(jq -r .container.image config/cluster.json)
mkdir -p "$OUT"

bin/spark3 cluster stop --apply
for _ in $(seq 1 60); do
  avail=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
  [ "$avail" -ge 100 ] && break
  sleep 5
done
[ "$avail" -ge 100 ] || { echo "refusing: only ${avail} GiB available after stop"; exit 1; }

(
  while sleep 0.5; do
    a=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
    if [ "$a" -lt 10 ]; then
      echo "watchdog: MemAvailable ${a} GiB, killing $NAME"
      docker kill "$NAME" >/dev/null 2>&1
    fi
  done
) &
WATCHDOG=$!

docker run --rm --name "$NAME" --gpus=all --ipc=host --memory=24g --memory-swap=24g \
  -v "$PWD/cache:/cache" -v "$OUT:/out" \
  -e PYTHONPATH=/opt/spark3/candidate/vllm:/opt/spark3/candidate/b12x \
  -e B12X_COMPILE_CACHE_DIR=/cache/kkref/jit/b12x/compile \
  -e B12X_CUTE_COMPILE_CACHE_DIR=/cache/kkref/jit/b12x/cute \
  -e CUTE_DSL_CACHE_DIR=/cache/kkref/jit/cute-dsl \
  -e CUTE_DSL_ARCH=sm_121a -e B12X_DENSE_SPLITK_TURBO=1 \
  --entrypoint bash "$IMAGE" -c '
    cd /opt/spark3/candidate/b12x
    for shape in "wqa_wkv 1792 5120" "wq_b 12288 1280" "shared_gate_up 1536 5120" \
                 "shared_down 5120 768" "indexer_wq_b 4096 1280" "engram_wkv 25600 6144" \
                 "dspark_main_proj 5120 15360" "dspark_ctx_kv 512 5120"; do
      set -- $shape
      echo "== $1 N=$2 K=$3"
      python3 benchmarks/autotune_dense_mxfp8_plan.py --name "$1" --n "$2" --k "$3" \
        --m-list 4,32 --tile-k-list 128 --output "/out/$1.tsv" || echo "sweep failed: $1"
    done'
rc=$?
kill "$WATCHDOG" 2>/dev/null

bin/spark3 cluster start --apply
bin/spark3 doctor --live
echo "sweep exit $rc; results in $OUT"
