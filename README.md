# spark3-vllm-ds41f

Reproducible Docker/vLLM deployment, tuning, and benchmarks for DeepSeek V4.1
Flash on a switchless three-node DGX Spark fabric.

This repository is being promoted from a forensic capture of the running cluster
into its only operational source of truth. Until the transition checklist is
complete, files are explicit about whether they describe observed state, desired
state, or an experiment.

## Current baseline

The active baseline was promoted on 2026-09-24
([manifests/baselines/2026-09-24-karmic-kraken-nofiat.json](manifests/baselines/2026-09-24-karmic-kraken-nofiat.json);
[speed tuning](experiments/2026-09-24-kk-speed-tuning/decision.md)):

- three DGX Spark nodes using tensor parallelism 3;
- direct dual ConnectX-7 paths between every pair of nodes;
- Local Inference Lab's `integration/karmic-kraken-beta` vLLM (unchanged) and
  B12X (plus the switchless RoCEnante patch), with B12X attention, linear, MoE,
  and mHC kernels;
- DeepSeek V4.1 Flash native FP8/FP4 weights, unchanged;
- DSpark speculative decoding with three draft tokens, full CUDA graphs for
  decode batches up to 32 tokens;
- B12X W4A8 tiny decode disabled (`B12X_W4A8_TINY_DECODE=0`): it omits the
  model's SwiGLU clamp and caused the incoherence seen in earlier images;
- 160,000-token per-request limit, eight admitted sequences, and 933,168 KV
  tokens in a 2 GiB-per-rank cache;
- one concurrent prefill, 4,096 batched tokens, and fail-closed 5 GiB startup
  and 3 GiB steady memory guards;
- FlashInfer autotune disabled (`--no-enable-flashinfer-autotune`): only the
  sampler uses FlashInfer, and the pass saved no configs.

One content-addressed image runs on all three nodes. It passes the LRU
coherence gate 5/5 and is faster than the 2026-09-20 image at every point of
the serving matrix; see
[the qualifying experiment](experiments/2026-09-23-karmic-kraken-reference/README.md)
and its [decision](experiments/2026-09-23-karmic-kraken-reference/decision.md).

The machine-readable desired configuration is [config/cluster.json](config/cluster.json).
The previous baseline's live evidence is
[manifests/baselines/2026-09-20-live.json](manifests/baselines/2026-09-20-live.json).

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
bin/spark3 upstream list
bin/spark3 upstream prepare vllm
bin/spark3 upstream prepare b12x
bin/spark3 upstream prepare flashinfer
bin/spark3 upstream prepare cutlass
scripts/fetch-cutlass-dsl-wheels
scripts/stage-build-contexts
scripts/host-recovery check
scripts/host-recovery apply
bin/spark3 build check
bin/spark3 build render
```

`doctor`, `status`, `render`, and every cluster command without `--apply` are
read-only. `cluster sync` fetches a published commit and detaches every clean node
checkout at that exact revision; it never copies a working tree or ignored files.
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

The promoted configuration is launch-enabled on `main` since the 2026-09-24
karmic-kraken promotion (owner decision). Experiment configurations enable
launch only on their own branches.

The compatibility helpers in `scripts/` are thin wrappers around these commands.
They contain no independent topology, credentials, or launch logic.

Host management-plane recovery is versioned separately under `host/recovery/`.
It arms the existing hardware watchdog and protects SSH/Tailscale without
restarting Docker or the inference service. The incident evidence and exact
policy boundary are documented in [docs/recovery.md](docs/recovery.md).

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

The promoted image is built reproducibly from pinned sources by
[prepare-sources](experiments/2026-09-23-karmic-kraken-reference/prepare-sources)
and [build-candidate](experiments/2026-09-23-karmic-kraken-reference/build-candidate),
and one content digest runs on all three nodes. `bin/spark3 build render` and
[docker/Dockerfile](docker/Dockerfile) still describe the earlier
reconstruction and have not been migrated to that recipe.

Canonical vLLM main is parked. The
[upstream-main rebase](experiments/2026-09-20-upstream-main-rebase/README.md)
and [canonical-minimal](experiments/2026-09-23-canonical-minimal/README.md)
experiments record that work; both shared the tiny-decode defect found later.
