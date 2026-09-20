# Experiment title

## Hypothesis

State one falsifiable performance or memory claim.

## Intended delta

List the one variable changed from the promoted baseline. Link the upstream
issue, pull request, commit, or local patch when source changes.

## Quality and safety gates

Define output-integrity checks, memory floor, request-failure tolerance, and
rollback trigger before running the candidate.

## Workloads

Record exact prompt/input identities, concurrency, warm/cold state, repetition
count, and client command. Save native receipts under `runs/`.

## Acceptance criteria

State the minimum TTFT/TPS/memory effect and acceptable variance in advance.
