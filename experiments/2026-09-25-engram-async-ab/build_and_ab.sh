#!/bin/bash
# usage: build_and_ab.sh   (on dgx1, deployment checkout at this experiment's commit)
# Stops the promoted service, builds and smoke-tests the r3 image, copies it
# to the other nodes, runs the off/on/off/on A/B, then restores the promoted
# service. Any build or distribution failure restores the service and stops.
set -u
cd ~/projects/spark3-vllm-ds41f
E=experiments/2026-09-25-engram-async-ab
TAG=vllm-ds41f-kkref:01f1b874c774-r3
log() { echo "$(date -u +%FT%TZ) $*"; }
restore() { log "restore"; $E/restore.sh; log "restore exit $?"; }

log "stop promoted service"
bin/spark3 cluster stop --remove --apply || { restore; exit 1; }
for _ in $(seq 1 60); do
  avail=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
  [ "$avail" -ge 100 ] && break
  sleep 5
done
log "build prepare"
bin/spark3 build prepare || { restore; exit 1; }
log "build image $TAG"
bin/spark3 build image --apply --tag "$TAG" || { restore; exit 1; }
local_id=$(docker image inspect "$TAG" --format '{{.Id}}')
for host in dgx2 dgx3; do
  log "copy image to $host"
  docker save "$TAG" | ssh "$host" docker load || { restore; exit 1; }
  remote_id=$(ssh "$host" docker image inspect "$TAG" --format '{{.Id}}')
  [ "$remote_id" = "$local_id" ] || { log "image ID differs on $host: $remote_id"; restore; exit 1; }
done
log "image $local_id on all nodes"
for arm in off on off on; do
  log "arm $arm"
  $E/run_arm.sh "$arm" quality,decode
  log "arm $arm exit $?"
done
restore
