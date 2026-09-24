#!/bin/bash
# Run bench_ds41_moe_decode.py for tiny decode on and off, on dgx1 with the
# cluster stopped. A host watchdog kills the benchmark below 10 GiB available.
set -u
cd ~/projects/spark3-vllm-ds41f
E=experiments/2026-09-23-karmic-kraken-reference
NAME=spark3-moe-decode-bench
IMAGE=$(jq -r .container.image $E/cluster-kkref-notiny.json)
if docker ps --format '{{.Names}}' | grep -q dsv41; then
  echo "refusing: a dsv41 container is running"; exit 1
fi
(
  while sleep 0.5; do
    avail=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
    if [ "$avail" -lt 10 ]; then
      echo "watchdog: MemAvailable ${avail} GiB, killing $NAME"
      docker kill "$NAME" >/dev/null 2>&1; exit 0
    fi
    docker ps --format '{{.Names}}' | grep -q "^$NAME\$" || [ -f /tmp/$NAME.started ] || continue
    docker ps --format '{{.Names}}' | grep -q "^$NAME\$" || exit 0
  done
) &
WATCHDOG=$!
touch /tmp/$NAME.started
O=$PWD/$E/overlay-b12x
docker run --rm --name "$NAME" --gpus=all --ipc=host --memory=24g --memory-swap=24g \
  -v "$PWD/cache:/cache" -v "$PWD/$E:/exp:ro" \
  -v "$O/tiny_decode.py:/opt/spark3/candidate/b12x/b12x/moe/_shared/kernels/tiny_decode.py:ro" \
  -v "$O/fused_moe_impl.py:/opt/spark3/candidate/b12x/b12x/moe/fused_moe/_impl.py:ro" \
  -v "$O/fused_moe_preparation.py:/opt/spark3/candidate/b12x/b12x/moe/fused_moe/_preparation.py:ro" \
  -e PYTHONPATH=/opt/spark3/candidate/vllm:/opt/spark3/candidate/b12x \
  -e B12X_COMPILE_CACHE_DIR=/cache/kkref/jit/b12x/compile \
  -e B12X_CUTE_COMPILE_CACHE_DIR=/cache/kkref/jit/b12x/cute \
  -e CUTE_DSL_CACHE_DIR=/cache/kkref/jit/cute-dsl -e TRITON_CACHE_DIR=/cache/kkref/jit/triton \
  -e XDG_CACHE_HOME=/cache/kkref/jit -e CUTE_DSL_ARCH=sm_121a \
  -e B12X_DENSE_SPLITK_TURBO=1 -e B12X_AUTOTUNE=0 \
  --entrypoint bash "$IMAGE" -c '
    for t in 1 0; do B12X_W4A8_TINY_DECODE=$t python3 /exp/bench_ds41_moe_decode.py "$@"; done' _ "$@"
rc=$?
rm -f /tmp/$NAME.started
kill $WATCHDOG 2>/dev/null
exit $rc
