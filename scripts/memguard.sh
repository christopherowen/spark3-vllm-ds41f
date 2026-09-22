#!/usr/bin/env bash
set -euo pipefail

: "${CONTAINER_NAME:?CONTAINER_NAME is required}"
: "${THRESHOLD_GIB:?THRESHOLD_GIB is required}"
: "${INTERVAL:?INTERVAL is required}"
: "${MEMGUARD_PHASE:=steady}"
: "${WAIT_FOR_CONTAINER_SECONDS:=0}"
: "${READY_FILE:=}"

if ! [[ "$THRESHOLD_GIB" =~ ^[1-9][0-9]*$ ]]; then
  echo "THRESHOLD_GIB must be a positive integer" >&2
  exit 2
fi
if ! [[ "$INTERVAL" =~ ^[0-9]+([.][0-9]+)?$ ]] || [[ "$INTERVAL" =~ ^0+([.]0+)?$ ]]; then
  echo "INTERVAL must be a positive number" >&2
  exit 2
fi
if ! [[ "$WAIT_FOR_CONTAINER_SECONDS" =~ ^[0-9]+$ ]]; then
  echo "WAIT_FOR_CONTAINER_SECONDS must be a non-negative integer" >&2
  exit 2
fi

threshold_kib=$((THRESHOLD_GIB * 1024 * 1024))
deadline=$((SECONDS + WAIT_FOR_CONTAINER_SECONDS))
observed_running=false
ready=false

while true; do
  inspectable=true
  if ! running=$(docker inspect --format '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null); then
    inspectable=false
    running=
  fi
  if [[ "$inspectable" == "false" && "$observed_running" == "true" ]]; then
    logger -t spark3-memguard \
      "cannot inspect $CONTAINER_NAME after startup during $MEMGUARD_PHASE; failing closed"
    docker kill --signal KILL "$CONTAINER_NAME" >/dev/null 2>&1 || true
    exit 1
  fi
  if [[ "$running" == "true" ]]; then
    observed_running=true
  elif [[ "$observed_running" == "true" ]]; then
    exit 0
  elif ((SECONDS >= deadline)); then
    logger -t spark3-memguard \
      "$CONTAINER_NAME did not start within ${WAIT_FOR_CONTAINER_SECONDS}s; leaving no unbounded watcher"
    exit 1
  fi

  available_kib=$(awk '$1 == "MemAvailable:" { print $2 }' /proc/meminfo)
  if [[ -z "$available_kib" ]]; then
    logger -t spark3-memguard \
      "cannot read MemAvailable during $MEMGUARD_PHASE; failing closed"
    if [[ "$running" == "true" ]]; then
      docker kill --signal KILL "$CONTAINER_NAME" >/dev/null
    fi
    exit 1
  fi
  if ((available_kib < threshold_kib)); then
    if [[ "$running" == "true" ]]; then
      logger -t spark3-memguard \
        "killing $CONTAINER_NAME during $MEMGUARD_PHASE: MemAvailable ${available_kib} KiB below ${threshold_kib} KiB"
      docker kill --signal KILL "$CONTAINER_NAME" >/dev/null
    else
      logger -t spark3-memguard \
        "refusing to wait for $CONTAINER_NAME during $MEMGUARD_PHASE: MemAvailable ${available_kib} KiB below ${threshold_kib} KiB"
    fi
    exit 1
  fi

  if [[ "$ready" == "false" ]]; then
    if [[ -n "$READY_FILE" ]]; then
      : >"$READY_FILE"
    fi
    ready=true
  fi
  sleep "$INTERVAL"
done
