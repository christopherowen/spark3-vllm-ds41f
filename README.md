# spark3-vllm-ds41f

Reproducible Docker/vLLM deployment, tuning, and benchmarks for DeepSeek V4.1
Flash on a switchless three-node DGX Spark fabric.

This repository is being promoted from a forensic capture of the running cluster
into its only operational source of truth. Until the transition checklist is
complete, files are explicit about whether they describe observed state, desired
state, or an experiment.

## Current baseline

The active baseline was captured on 2026-09-20:

- three DGX Spark nodes using tensor parallelism 3;
- direct dual ConnectX-7 paths between every pair of nodes;
- vLLM with B12X attention, linear, MoE, and mHC kernels;
- DeepSeek V4.1 Flash native FP8/FP4 weights, unchanged;
- DSpark speculative decoding with three draft tokens;
- 160,000-token per-request limit, eight admitted sequences, and 498,145 KV
  tokens in the measured 3 GiB-per-rank cache;
- one concurrent prefill, 8,192 batched tokens, and a 4 GiB host-memory guard.

The immutable baseline retains the observed 4 GiB guard. Current orchestration
uses a fail-closed 5 GiB startup threshold sampled every quarter-second, then switches
to the promoted 3 GiB steady threshold only after API readiness;
`config/cluster.json` owns both desired values.

The machine-readable desired configuration is [config/cluster.json](config/cluster.json).
The original live evidence is
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

Both tracked cluster configurations are launch-disabled during the physical
recovery pause. Re-enabling either one requires an explicit reviewed config
change after the hosts have been power-cycled and inspected.

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

The active launch can now be rendered, deployed, and audited. The forensic
source reconstruction is not yet complete: a follow-up comparison found local
DS4.1 model changes in the live image beyond the captured indexer override. The
recorded vLLM patch series therefore proves only that override, not the entire
live serving tree.

Rather than keep reconstructing a divergent release branch, the
[upstream-main rebase experiment](experiments/2026-09-20-upstream-main-rebase/README.md)
starts from current upstream architecture and carries each still-required
change as a separate commit. Neither that experiment nor
[docker/Dockerfile](docker/Dockerfile) has been built or qualified on a Spark.
The current live tag still resolves to node-local image IDs and remains the
promoted runtime until a scheduled qualification produces one tested content
digest for all three nodes.
