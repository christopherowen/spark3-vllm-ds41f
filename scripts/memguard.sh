#!/usr/bin/env bash
set -euo pipefail

: "${CONTAINER_NAME:?CONTAINER_NAME is required}"
: "${THRESHOLD_GIB:?THRESHOLD_GIB is required}"
: "${INTERVAL:?INTERVAL is required}"

if ! [[ "$THRESHOLD_GIB" =~ ^[1-9][0-9]*$ ]]; then
  echo "THRESHOLD_GIB must be a positive integer" >&2
  exit 2
fi
if ! [[ "$INTERVAL" =~ ^[1-9][0-9]*$ ]]; then
  echo "INTERVAL must be a positive integer" >&2
  exit 2
fi

threshold_kib=$((THRESHOLD_GIB * 1024 * 1024))
while [[ "$(docker inspect --format '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null || true)" == "true" ]]; do
  available_kib=$(awk '$1 == "MemAvailable:" { print $2 }' /proc/meminfo)
  if [[ -z "$available_kib" ]]; then
    logger -t spark3-memguard "cannot read MemAvailable; leaving $CONTAINER_NAME running"
    exit 1
  fi
  if ((available_kib < threshold_kib)); then
    logger -t spark3-memguard \
      "stopping $CONTAINER_NAME: MemAvailable ${available_kib} KiB below ${threshold_kib} KiB"
    docker stop --time 5 "$CONTAINER_NAME" >/dev/null
    exit 1
  fi
  sleep "$INTERVAL"
done
