#!/usr/bin/env bash
set -euo pipefail

: "${CONTAINER_NAME:?CONTAINER_NAME is required}"
: "${OUTPUT_PATH:?OUTPUT_PATH is required}"
: "${SAMPLE_INTERVAL:=0.5}"

if ! [[ "$SAMPLE_INTERVAL" =~ ^[0-9]+([.][0-9]+)?$ ]] \
  || [[ "$SAMPLE_INTERVAL" =~ ^0+([.]0+)?$ ]]; then
  echo "SAMPLE_INTERVAL must be a positive number" >&2
  exit 2
fi

samples=0
minimum_available_kib=0
maximum_swap_used_kib=0
started_at=$(date +%s)

finish() {
  local elapsed=$(( $(date +%s) - started_at ))
  printf '# samples=%s min_available_kib=%s max_swap_used_kib=%s elapsed_seconds=%s\n' \
    "$samples" "$minimum_available_kib" "$maximum_swap_used_kib" "$elapsed" \
    >>"$OUTPUT_PATH"
  logger -t spark3-startup-telemetry \
    "$CONTAINER_NAME samples=$samples min_available_kib=$minimum_available_kib max_swap_used_kib=$maximum_swap_used_kib elapsed_seconds=$elapsed" \
    || true
}
trap finish EXIT

printf 'epoch_ms\tmem_available_kib\tswap_used_kib\tcontainer_state\n' >"$OUTPUT_PATH"
while true; do
  read -r available_kib swap_total_kib swap_free_kib < <(
    awk '
      $1 == "MemAvailable:" { available = $2 }
      $1 == "SwapTotal:" { swap_total = $2 }
      $1 == "SwapFree:" { swap_free = $2 }
      END { print available, swap_total, swap_free }
    ' /proc/meminfo
  )
  swap_used_kib=$((swap_total_kib - swap_free_kib))
  state=$(docker inspect \
    --format '{{.State.Running}}:{{.State.ExitCode}}:{{.State.OOMKilled}}' \
    "$CONTAINER_NAME" 2>/dev/null || printf 'absent')
  printf '%s\t%s\t%s\t%s\n' \
    "$(date +%s%3N)" "$available_kib" "$swap_used_kib" "$state" \
    >>"$OUTPUT_PATH"

  samples=$((samples + 1))
  if ((minimum_available_kib == 0 || available_kib < minimum_available_kib)); then
    minimum_available_kib=$available_kib
  fi
  if ((swap_used_kib > maximum_swap_used_kib)); then
    maximum_swap_used_kib=$swap_used_kib
  fi
  sleep "$SAMPLE_INTERVAL"
done
