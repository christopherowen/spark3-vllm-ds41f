# spark3-vllm-ds41f

Reproducible Docker/vLLM deployment, tuning, and benchmarks for DeepSeek V4.1
Flash on a switchless three-node DGX Spark fabric.

This repository is being promoted from a forensic capture of the running cluster
into its only operational source of truth. Until the transition checklist is
complete, files are explicit about whether they describe observed state, desired
state, or an experiment.

## Current baseline

The active baseline is recorded in
[manifests/baselines/2026-09-25-karmic-kraken-r3.json](manifests/baselines/2026-09-25-karmic-kraken-r3.json):

- three DGX Spark nodes using tensor parallelism 3;
- direct dual ConnectX-7 paths between every pair of nodes;
- Local Inference Lab's `integration/karmic-kraken-beta` vLLM (plus Engram
  projection sharding and asynchronous Engram row patches) and B12X (plus the switchless RoCEnante patch),
  with B12X attention, linear, MoE, and mHC kernels;
- DeepSeek V4.1 Flash native FP8/FP4 weights, unchanged;
- DSpark speculative decoding with three draft tokens and block rejection,
  full CUDA graphs for decode batches up to 32 tokens;
- B12X W4A8 tiny decode disabled (`B12X_W4A8_TINY_DECODE=0`): it omits the
  model's SwiGLU clamp and caused the incoherence seen in earlier images;
- 160,000-token per-request limit, eight admitted sequences, and 933,168 KV
  tokens in a 2 GiB-per-rank cache;
- one concurrent prefill, 4,096 batched tokens, and fail-closed 5 GiB startup
  and 3 GiB steady memory guards;
- FlashInfer autotune disabled (`--no-enable-flashinfer-autotune`): only the
  sampler uses FlashInfer, and the pass saved no configs.

One content-addressed image runs on all three nodes and passes the LRU
coherence gate 5/5; see [Performance](#performance).

The machine-readable desired configuration is [config/cluster.json](config/cluster.json).
To reproduce the deployment on your own three Sparks, follow
[docs/replicate.md](docs/replicate.md).

## Performance

Current baseline, measured with `bin/spark3 bench` from dgx1: prose and code
prompts, temperature 0, reasoning on, 256 output tokens.

| Prompt | Streams | Aggregate tok/s | Per-stream decode tok/s | First token |
|---|---:|---:|---:|---:|
| prose | 1 | 41.6 | 43.2 | 0.25 s |
| prose | 2 | 70.5 | 37.6 | 0.37 s |
| prose | 4 | 102.4 | 28.4 | 0.48 s |
| prose | 8 | 150.4 | 20.6 | 0.56 s |
| code | 1 | 50.6 | 52.9 | 0.24 s |
| code | 2 | 83.6 | 44.7 | 0.34 s |
| code | 4 | 120.9 | 33.3 | 0.50 s |
| code | 8 | 173.0 | 24.3 | 0.58 s |

| Other measurements | |
|---|---|
| Quality gate (fixed LRU task, 5 repeats) | 5/5 |
| Single-stream decode step | about 51 ms; 1.2 accepted drafts per step on prose, 1.7 on code |
| Cold prefill | 2K 3.8k, 32K 4.1k, 64K 4.0k, 128K 3.8k tok/s |
| Prefix-cache replay, 32K prompt | 7.75 s cold, 0.26 s warm |
| Four concurrent 64K contexts | all admitted without preemption, peak KV use 22%, 12.1 tok/s per stream |
| KV capacity | 933,168 tokens in 2 GiB per rank (5.8 full 160K contexts) |
| Host memory headroom under load | dgx1 at least 5.8 GiB MemAvailable (3 GiB guard) |

The quick default takes three or four samples per decode point, about ±4-9%
at 95% confidence. Single boots of one configuration vary by about 3%.
Reports: [decode](manifests/benchmarks/2026-09-25-karmic-kraken-r3.json),
[prefill, prefix cache, and admission](manifests/benchmarks/2026-09-25-karmic-kraken-r3-capacity.json).
See [Benchmarking](#benchmarking) to reproduce them.

## Repository contract

There are three deliberately separate kinds of state:

1. `manifests/baselines/` contains immutable observations and benchmark identity.
2. `config/` contains the promoted desired state used to render launches.
3. `experiments/` contains isolated candidates and evidence before promotion.

An experiment never becomes the baseline merely because it is running. Promotion
requires reproducible measurements, an explicit decision, and a commit updating
the desired configuration and baseline record together.

## Commands

All commands are run from the repository root.

```sh
bin/spark3 doctor
bin/spark3 doctor --live
bin/spark3 status
bin/spark3 render dgx1
bin/spark3 cluster sync
bin/spark3 cluster start --replace
bin/spark3 cluster stop
bin/spark3 bench
bin/spark3 upstream list
bin/spark3 upstream prepare vllm
bin/spark3 upstream prepare b12x
bin/spark3 build prepare
bin/spark3 build check
bin/spark3 build image
bin/spark3 build image --apply
bin/spark3 build smoke
scripts/host-recovery check
scripts/host-recovery apply
```

`doctor`, `status`, `render`, and every cluster command without `--apply` are
read-only. `bench` only sends API requests; see [Benchmarking](#benchmarking).
`build prepare` writes only under ignored `.work/build/`; see
[docker/README.md](docker/README.md). `cluster sync` fetches a published commit and detaches every clean node
checkout at that exact revision; it never copies a working tree or ignored files.
A commit counts as published when a branch on `origin` contains it. The
promoted configuration sets `deployment.branch: main`, so production deploys
only from `main`; experiment configurations omit the field, so they stay
deployable after their branch is merged and deleted.
The only cleanliness exception is a repository-local writable runtime mount
declared in `cluster.json` (currently `cache/`), which is preserved in place and
never enters Git.
`cluster start` preflights all three ranks, pre-arms a protected host-local
startup memory guard, waits for its first successful memory sample, and only
then starts workers before the head. Readiness fails if a guard exits or
available host memory crosses its threshold. Only a healthy API switches every
node to the lower steady-state guard. Mutating operations require an explicit
`--apply`; replacing an existing
service additionally requires `--replace`, and an experiment with
`deployment.launch_enabled=false` refuses mutation locally.

The promoted configuration is launch-enabled on `main`. Experiment
configurations set `launch_enabled` for themselves.

`scripts/` holds the memory guard that `cluster start` installs on each node
and host recovery, which has not yet moved into `bin/spark3`. Experiments keep
their own scripts in their directories.

Host management-plane recovery is versioned separately under `host/recovery/`.
It arms the existing hardware watchdog and protects SSH/Tailscale without
restarting Docker or the inference service. The incident evidence and exact
policy boundary are documented in [docs/recovery.md](docs/recovery.md).

## Benchmarking

`bin/spark3 bench` measures the live service from the head node and writes one
self-contained report to `results/private/bench/<UTC time>/bench.json`
(`--output` to change). By default it is a quick check of about six minutes:
the quality gate and every decode point with three or four samples each.
`bin/spark3 bench --full` runs every suite below and samples each decode
point to its precision target, about 35 minutes; use it for experiments that
need to resolve small differences. Before sending anything it runs the `doctor --live`
comparison and refuses a cluster that differs from its configuration. It also
records the configuration hash, each node's image ID and checkout, and the
client commit.

Suites of the full run, in order (`--suites` selects a subset):

- `quality`: the fixed LRU request five times at temperature 0; all five must
  pass, or the run stops before measuring anything.
- `decode`: prose and code prompts at concurrency 1, 2, 4, and 8, 256 output
  tokens, reasoning on, temperature 0, and the same prompts and metric
  (aggregate completion tokens per wall second) as every published baseline.
  Output text still differs from run to run, so each point is a random
  draw. Points are sampled in shuffled rounds after a discarded warmup round.
  With `--full`, each point keeps sampling until its 95% confidence interval
  is within `--precision` (2%) of the mean, between `--min-samples` (6) and
  `--max-samples` (40) samples; the quick default takes 3-4.
  Single boots of one configuration differ by about 3%, because adaptive
  verification profiles its costs at startup, so effects smaller than that
  need several boots per arm.
- `sampled`: DSpark accepted drafts per step at temperature 1.0 from the
  engine counters, 32 requests per case at concurrency 1 and 4.
- `prefill`: cold prefill at 2K, 32K, 64K, and 128K tokens, three unique
  uncached prompts each.
- `prefix`: a 32K prompt followed by two identical replays, reporting cold and
  warm TTFT and the cache hit rate.
- `admission`: four concurrent 64K-token contexts; all four must run at once
  without preemption.

Measurements stay clean and safe:

- A sample starts only when the engine reports no running or waiting requests.
- A sample that overlaps anyone else's request is repeated. Overlap shows up
  in the engine's request counters and peak running count.
- A per-node memory monitor stops the run, cancelling in-flight requests, if
  MemAvailable falls within 1 GiB of the steady memory guard.

The report is still written if the run stops early.

With a reference run (by default
`manifests/benchmarks/<promoted_baseline>.json`, or `--compare PATH`), each
decode and prefill point shows its percent change with a Welch 95% interval.
The command exits non-zero on a failed quality or admission check, any failed
request, an early stop, or a point that is significantly slower by more than
`--tolerance` (3%). A promotion adds its reference run to
`manifests/benchmarks/`.

## Upstreams

Canonical upstreams, contribution forks, tracking branches, pinned commits, and
applied upstream fixes live in [upstreams.lock.json](upstreams.lock.json). Local
changes are kept as ordered patch series rather than edits to copied source
trees. `upstream` always means the canonical project; `origin` is Christopher's
fork when one exists; this deployment repository is neither. See
[docs/upstreams.md](docs/upstreams.md). Repositories that are useful for ideas
but are not build inputs are kept separately in
[docs/inspiration.md](docs/inspiration.md).

## Transition status

`bin/spark3 build` builds the promoted image from the pinned sources in
`upstreams.lock.json` and the patch series ([docker/README.md](docker/README.md)),
and one content digest runs on all three nodes.

Canonical vLLM main is parked; the serving sources are Local Inference Lab's
`integration/karmic-kraken-beta` vLLM and B12X with the local patch series
([docs/upstreams.md](docs/upstreams.md)).
