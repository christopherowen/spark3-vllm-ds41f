#!/usr/bin/env bash
set -euo pipefail

if (($# != 1)); then
  echo "usage: $0 <local-image>" >&2
  exit 2
fi

repo_root=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
image=$1
name=spark3-memguard-smoke
unit=${name}-memguard.service
config=tests/fixtures/memguard-kill-cluster.json

cleanup() {
  sudo -n systemctl stop "$unit" >/dev/null 2>&1 || true
  docker rm --force "$name" >/dev/null 2>&1 || true
}
trap cleanup EXIT HUP INT TERM

cleanup
docker image inspect "$image" >/dev/null
docker run \
  --detach \
  --pull=never \
  --name "$name" \
  --network=none \
  --memory=64m \
  --memory-swap=64m \
  --entrypoint=/bin/sleep \
  "$image" \
  300 >/dev/null

if "$repo_root/scripts/memguard-start.sh" "$config" startup running; then
  echo "memguard unexpectedly accepted the deliberately impossible threshold" >&2
  exit 1
fi

state=$(docker inspect --format '{{.State.Running}} {{.State.ExitCode}}' "$name")
if [[ "$state" != "false 137" ]]; then
  echo "memguard did not SIGKILL the smoke container: $state" >&2
  exit 1
fi

echo "memguard kill path passed: $name exited 137"
