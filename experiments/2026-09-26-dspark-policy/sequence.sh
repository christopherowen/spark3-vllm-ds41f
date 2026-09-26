#!/bin/bash
# usage: sequence.sh   (on dgx1, deployment checkout at this experiment's commit)
# Runs the timed arms in ABAB order, then the gathered-bias arm, then the
# five-draft trace (replay data only, few samples). Each arm is left up
# until the next one stops it.
set -u
cd ~/projects/spark3-vllm-ds41f
E=experiments/2026-09-26-dspark-policy
CASES=prose,code,prose-nothink,code-nothink,json-nothink
log() { echo "$(date -u +%FT%TZ) $*"; }
for step in base:b1 marginal:m1 base:b2 marginal:m2 topk:t1; do
  arm=${step%%:*}
  label=${step##*:}
  log "arm $arm ($label)"
  $E/run_arm.sh "$arm" "$label" --suites quality,decode --decode-cases "$CASES"
  log "arm $arm ($label) exit $?"
done
log "arm k5trace"
$E/run_arm.sh k5trace k5trace --suites quality,decode --decode-cases "$CASES" \
  --min-samples 2 --max-samples 2 --compare none
log "arm k5trace exit $?"
