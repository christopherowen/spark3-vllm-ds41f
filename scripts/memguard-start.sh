#!/usr/bin/env bash
set -euo pipefail

repo_root=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
config_relative=${1:-config/cluster.json}
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
guard_gib=$(jq -er '.host.memguard_gib' "$config_path")
interval=$(jq -er '.host.memguard_interval_seconds' "$config_path")
unit="${name}-memguard.service"

if [[ "$(docker inspect --format '{{.State.Running}}' "$name" 2>/dev/null || true)" != "true" ]]; then
  echo "refusing to arm memguard: container $name is not running" >&2
  exit 1
fi

# A collected transient unit may still have a failed record from a prior stop.
sudo -n systemctl stop "$unit" >/dev/null 2>&1 || true
sudo -n systemctl reset-failed "$unit" >/dev/null 2>&1 || true
sudo -n systemd-run --quiet \
  --unit="${name}-memguard" \
  --collect \
  --property=Type=simple \
  --property="User=$(id -un)" \
  --property="WorkingDirectory=$repo_root" \
  --setenv="THRESHOLD_GIB=$guard_gib" \
  --setenv="INTERVAL=$interval" \
  --setenv="CONTAINER_NAME=$name" \
  "$repo_root/scripts/memguard.sh"

echo "memguard active for $name (threshold ${guard_gib} GiB, interval ${interval}s)"
