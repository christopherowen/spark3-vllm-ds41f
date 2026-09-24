# Image reconstruction

The promoted image (`vllm-ds41f-kkref:01f1b874c774-r1`, baseline
2026-09-24-karmic-kraken) is not built from this directory. Its recipe is
`experiments/2026-09-23-karmic-kraken-reference/`: `prepare-sources` checks out
the pinned Local Inference Lab trees and applies the B12X series, and
`build-candidate` builds its `Dockerfile` on the vLLM nightly base. The rest of
this file describes the earlier reconstruction, which `bin/spark3 build render`
still renders.

`docker/Dockerfile` is the deterministic successor to the node-local image
chain captured on 2026-09-20. It starts at the official vLLM image digest and
accepts only named contexts prepared from `upstreams.lock.json`:

| Build context | Canonical upstream | Local treatment |
|---|---|---|
| `vllm-source` | `vllm-project/vllm` | pinned commit plus `patches/vllm/series` |
| `b12x-source` | `local-inference-lab/b12x` | pinned commit, recorded upstream fix, then local series |
| `flashinfer-source` | `flashinfer-ai/flashinfer` | pinned `v0.7.0rc1` source and submodules |
| `cutlass-source` | `NVIDIA/cutlass` | pinned `v4.4.2` source |
| `cutlass-dsl-wheels` | NVIDIA packages on PyPI | 4.6.2 ARM64 wheels verified by SHA-256 |

Prepare the inputs at deterministic ignored paths:

```sh
bin/spark3 upstream prepare vllm
bin/spark3 upstream prepare b12x
bin/spark3 upstream prepare flashinfer
bin/spark3 upstream prepare cutlass
scripts/fetch-cutlass-dsl-wheels
scripts/stage-build-contexts
bin/spark3 build check
```

Staging atomically replaces each context after checking both staged and unstaged
diffs. It strips Git metadata and Python bytecode before Docker sees the source.
`build check` rejects source paths not recorded in the source manifest, unknown
wheels, and any content-or-mode difference between a prepared tree and its staged
context.

Render a local build or registry push without executing it:

```sh
bin/spark3 build render
bin/spark3 build render --push
```

The rendered tag includes the deployment commit as well as the vLLM and B12X
pins, so two repository states cannot silently reuse one tag. The build is a
**reconstruction candidate**, not yet the promoted image. Do not
deploy it until it builds on a Spark, passes import/schema checks, serves the
quality probes, reproduces the baseline benchmark, and has one pushed OCI digest
used by all three ranks. The captured live service remains authoritative during
that transition.

## Build resource policy on DGX Spark

Build the current-head candidate with
`experiments/2026-09-23-latest-head-rebase/build-candidate` after preparing its
pinned sources. Run `check` first; run `build` only on an idle Spark with the
DS4.1 container stopped. With no model weights resident, compilation can use
much more of the 128 GiB host than a serving process can. The candidate build
selects up to eight compile jobs on a 20-CPU Spark, each with two NVCC threads;
it reserves four CPUs, budgets 8 GiB per job, and never selects more than ten
jobs. It requires 64 GiB `MemAvailable` at start and cancels if a one-second
sample falls below a 24 GiB host reserve. A host lock prevents overlapping
candidate builds. The older two-job setting was chosen for caution, not
because the previous builds exhausted host memory: the
recorded native rebuild stayed above 100 GiB available.

Keep the build policy separate from inference startup. Once the image is built,
verify its imports and exact image identity, then use the coordinated launch
with its existing 5 GiB startup and 3 GiB steady guards. After an aborted
build, confirm the Docker build has stopped and memory has recovered before
starting a model. Record the chosen job count, minimum observed available
memory, elapsed build time, and image digest in the experiment receipt so
future resource changes can be compared without changing model quality gates.
