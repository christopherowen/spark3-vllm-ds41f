# Image reconstruction

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

Staging strips Git metadata and Python bytecode before Docker sees the source.
`build check` then compares a content-and-mode digest of every staged tree with
its prepared source, preventing a stale or hand-edited context from entering the
image.

Render a local build or registry push without executing it:

```sh
bin/spark3 build render
bin/spark3 build render --push
```

The build is a **reconstruction candidate**, not yet the promoted image. Do not
deploy it until it builds on a Spark, passes import/schema checks, serves the
quality probes, reproduces the baseline benchmark, and has one pushed OCI digest
used by all three ranks. The captured live service remains authoritative during
that transition.
