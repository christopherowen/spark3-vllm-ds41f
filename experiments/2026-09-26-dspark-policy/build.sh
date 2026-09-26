#!/bin/bash
# usage: build.sh   (on dgx1, deployment checkout at this experiment's commit, service stopped)
# Builds the r4a candidate image with bin/spark3 build and copies it to the
# other nodes, checking that every node holds one image ID.
set -eu
cd ~/projects/spark3-vllm-ds41f
TAG=vllm-ds41f-kkref:01f1b874c774-r4a
log() { echo "$(date -u +%FT%TZ) $*"; }
log "build prepare"
bin/spark3 build prepare
log "build image $TAG"
bin/spark3 build image --apply --tag "$TAG"
local_id=$(docker image inspect "$TAG" --format '{{.Id}}')
for host in dgx2 dgx3; do
  log "copy image to $host"
  docker save "$TAG" | ssh "$host" docker load
  remote_id=$(ssh "$host" docker image inspect "$TAG" --format '{{.Id}}')
  [ "$remote_id" = "$local_id" ] || { log "image ID differs on $host: $remote_id"; exit 1; }
done
log "image $local_id on all nodes"
