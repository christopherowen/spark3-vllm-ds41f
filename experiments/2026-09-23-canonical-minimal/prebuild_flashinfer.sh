#!/bin/bash
# usage: prebuild_fi.sh <image> <cache-subdir>   (runs on a Spark host; capped container)
set -euo pipefail
IMG=$1; SUB=$2
C=~/projects/spark3-vllm-ds41f/cache
mkdir -p "$C/$SUB"
avail=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
[[ $avail -ge 64 ]] || { echo "refusing: only ${avail} GiB available"; exit 1; }
docker run --rm --gpus all --memory=24g --memory-swap=24g --cpus=8 \
  -e FLASHINFER_WORKSPACE_BASE=/cache/$SUB -e MAX_JOBS=4 \
  -v "$C":/cache:rw --entrypoint /bin/bash "$IMG" -c '
set -e
mkdir -p /cache/'"$SUB"'/.cache
[ -d /cache/'"$SUB"'/.cache/flashinfer ] || cp -a /root/.cache/flashinfer /cache/'"$SUB"'/.cache/ 2>/dev/null || true
python3 -c "
import time; t=time.time()
from flashinfer.sampling import get_sampling_module
m=get_sampling_module(); print(\"sampling module ready\", type(m).__name__, round(time.time()-t,1), \"s\")
"
find /cache/'"$SUB"'/.cache/flashinfer -name "*.so" | sed "s|/cache/||"
'
echo "$(hostname) done; MemAvailable $(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo) GiB"
