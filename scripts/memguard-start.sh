#!/usr/bin/env bash
set -euo pipefail

repo_root=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
config_relative=${1:-config/cluster.json}
phase=${2:-steady}
mode=${3:-running}
if [[ "$config_relative" = /* ]]; then
  echo "cluster config must be relative to the repository root" >&2
  exit 1
fi
config_path=$(realpath "$repo_root/$config_relative")
case "$config_path" in
  "$repo_root"/*) ;;
  *)
    echo "cluster config must remain inside the repository" >&2
    exit 1
    ;;
esac
if [[ ! -f "$config_path" ]]; then
  echo "cluster config does not exist: $config_relative" >&2
  exit 1
fi

name=$(jq -er '.container.name' "$config_path")
case "$phase" in
  startup)
    guard_gib=$(jq -er '.host.startup_memguard_gib' "$config_path")
    interval=$(jq -er '.host.startup_memguard_interval_seconds' "$config_path")
    wait_seconds=$(jq -er '.host.startup_memguard_wait_seconds' "$config_path")
    ;;
  steady)
    guard_gib=$(jq -er '.host.memguard_gib' "$config_path")
    interval=$(jq -er '.host.memguard_interval_seconds' "$config_path")
    wait_seconds=0
    ;;
  *)
    echo "memguard phase must be startup or steady" >&2
    exit 2
    ;;
esac
unit="${name}-memguard.service"

case "$mode" in
  running)
    if [[ "$(docker inspect --format '{{.State.Running}}' "$name" 2>/dev/null || true)" != "true" ]]; then
      echo "refusing to arm memguard: container $name is not running" >&2
      exit 1
    fi
    wait_seconds=0
    ;;
  prelaunch)
    if [[ "$phase" != "startup" ]]; then
      echo "prelaunch mode is valid only for the startup memguard" >&2
      exit 2
    fi
    ;;
  *)
    echo "memguard mode must be running or prelaunch" >&2
    exit 2
    ;;
esac

runtime_dir=${XDG_RUNTIME_DIR:-/tmp}
if [[ ! -d "$runtime_dir" || ! -w "$runtime_dir" ]]; then
  runtime_dir=/tmp
fi
ready_file=$(mktemp "$runtime_dir/${name}-memguard.XXXXXX")
rm -f -- "$ready_file"

# A collected transient unit may still have a failed record from a prior stop.
sudo -n systemctl stop "$unit" >/dev/null 2>&1 || true
sudo -n systemctl reset-failed "$unit" >/dev/null 2>&1 || true
sudo -n systemd-run --quiet \
  --unit="${name}-memguard" \
  --collect \
  --property=Type=simple \
  --property="User=$(id -un)" \
  --property="WorkingDirectory=$repo_root" \
  --property=OOMScoreAdjust=-1000 \
  --property=MemoryMin=64M \
  --property=CPUWeight=1000 \
  --setenv="THRESHOLD_GIB=$guard_gib" \
  --setenv="INTERVAL=$interval" \
  --setenv="CONTAINER_NAME=$name" \
  --setenv="MEMGUARD_PHASE=$phase" \
  --setenv="WAIT_FOR_CONTAINER_SECONDS=$wait_seconds" \
  --setenv="READY_FILE=$ready_file" \
  "$repo_root/scripts/memguard.sh"

for _ in {1..50}; do
  if [[ -f "$ready_file" ]]; then
    rm -f -- "$ready_file"
    echo "$phase memguard active for $name (threshold ${guard_gib} GiB, interval ${interval}s, mode $mode)"
    exit 0
  fi
  if ! sudo -n systemctl is-active --quiet "$unit"; then
    echo "memguard exited before completing its initial protected sample" >&2
    exit 1
  fi
  sleep 0.1
done

sudo -n systemctl stop "$unit" >/dev/null 2>&1 || true
echo "memguard did not confirm its initial protected sample" >&2
exit 1
