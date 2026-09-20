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
bin/spark3 upstream list
bin/spark3 upstream prepare vllm
bin/spark3 upstream prepare b12x
bin/spark3 upstream prepare flashinfer
bin/spark3 upstream prepare cutlass
scripts/fetch-cutlass-dsl-wheels
scripts/stage-build-contexts
bin/spark3 build check
bin/spark3 build render
```

`doctor` is read-only. `render` prints the exact Docker command without executing
it. The initial repository intentionally has no cluster-mutating `up` command;
that will be added only after the image build and deployment are reproducible.

## Upstreams

Canonical upstreams, contribution forks, tracking branches, pinned commits, and
applied upstream fixes live in [upstreams.lock.json](upstreams.lock.json). Local
changes are kept as ordered patch series rather than edits to copied source
trees. `upstream` always means the canonical project; `origin` is Christopher's
fork when one exists; this deployment repository is neither. See
[docs/upstreams.md](docs/upstreams.md).

## Transition status

The active runtime can now be rendered and audited. Fresh pinned vLLM and B12X
checkouts plus their ordered patch series reproduce every modified live source
file byte-for-byte. The image reconstruction now pins the official base digest,
FlashInfer, CUTLASS, and every CuTe DSL wheel, but has not yet been built and
qualified on a Spark. The current live tag still resolves to node-local image
IDs. Critical serving source is identical; rank 0 only omits vLLM's unused
`benchmarks/` package. The next milestone is to qualify
[docker/Dockerfile](docker/Dockerfile), push one content digest, and deploy that
exact digest to all three nodes.
