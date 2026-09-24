# Methodology

## State model

Every fact belongs to one of four states:

- **Observed:** read from a running node and retained in an immutable manifest.
- **Promoted:** accepted desired state in `config/`.
- **Candidate:** an isolated change under `experiments/`.
- **Historical:** a previous immutable baseline retained for comparison.

Observed state is not automatically promoted. A live hotfix must be captured as an
experiment and deliberately promoted or rolled back.

## Experiment layout

Use `experiments/YYYY-MM-DD-short-name/` with:

- `README.md`: hypothesis, one changed variable, risks, and acceptance criteria;
- `base.json`: exact repository commit, upstream pins, image digest, and baseline;
- `candidate.json`: complete delta from promoted configuration;
- `runs/`: native raw receipts, including unsuccessful runs;
- `results.json`: normalized per-run metrics and counts;
- `decision.md`: promote, reject, retain for more evidence, or inconclusive.

Do not compare a new candidate only with a historical number collected under a
different client, prompt, model load, or cluster state. Save sufficient identity to
repeat both sides.

## Minimum benchmark matrix

The stable suite must cover:

1. single-stream code and prose decode;
2. concurrency 1, 2, 4, and the promoted normal Strix concurrency;
3. cold prefill at representative short, 32K, 64K, and long-context sizes;
4. prefix-cache replay where applicable;
5. the four-long-context admission target;
6. an end-to-end Strix workload with fixed scope and tool policy;
7. minimum available host memory, swap movement, KV use, OOMs, allocation retries,
   request failures, and output-integrity gates.

`bin/spark3 bench` implements items 1-5 and 7; the Strix workload remains
manual. Report TTFT, per-stream and aggregate TPS, total wall time, prompt/decode token
counts, and variability across complete runs. Performance is not accepted at the
expense of model quality or silent request rejection.

Every target-hardware startup must be fail-closed. Pre-arm the configured
startup memory guard before each container launch, verify it remains active
through API readiness, and reject the run on any driver allocation failure or
guard stop. Do not repeat a coordinated launch after management-path loss until
the failure is isolated with bounded tests and the owner explicitly authorizes
new hardware work.

## Promotion

A promotion commit must:

1. update `config/`;
2. add a new immutable baseline manifest;
3. link the accepted experiment and all native receipts;
4. update upstream pins or patch series when source changed;
5. prove all three ranks use the same content-addressed image;
6. pass `bin/spark3 doctor --live` after coordinated deployment;
7. add the deployed service's complete `bin/spark3 bench` report as
   `manifests/benchmarks/<baseline>.json`, the reference later runs compare with.

Rollback is a new coordinated deployment of the previous promoted commit. It is
not an ad hoc reconstruction from shell history.
