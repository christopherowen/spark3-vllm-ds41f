# Kernel backends

The top-level `kernel_backend` field of a cluster profile selects the library
that runs the DS4.1 model compute kernels: attention (target and DSpark
drafter), linear layers and MoE. Omitting it selects `b12x`.

| Backend | Status | Kernel sources |
| --- | --- | --- |
| `b12x` | promoted: `config/cluster.json` and both [page-size profiles](memory-profiles.md) | B12X in the promoted image |
| `tilelang` | [candidate](../experiments/2026-10-03-tilelang-kernels/README.md) for TP3 and TP4, not built | TileLang 0.1.15 with `patches/tilelang`, DeepSeek's TileKernels 2.0.0, and the DS4.1 TileLang kernels as an extra vLLM patch |

## What it does not select

Collectives and checkpoint loading are outside the policy. Both backends keep
RoCEnante and NCCL for tensor-parallel collectives (`fabric.transport`,
`B12X_ROCE_*`, `VLLM_ENABLE_ROCE_ALLREDUCE`; see
[switchless-topology.md](switchless-topology.md)), and the B12X checkpoint
loader (`--load-format b12x`, `VLLM_PLUGINS=b12x_loader`). B12X therefore
remains in every image.

## Required settings

`doctor` requires the serve arguments and environment to match the selection:

| Setting | `b12x` | `tilelang` |
| --- | --- | --- |
| `--attention-backend` | `B12X` | `TILELANG` |
| `--linear-backend` | `b12x` | `tilelang` |
| `--moe-backend` | `b12x` | `tilelang` |
| `attention_backend` in `--speculative-config` | `B12X` | `TILELANG` |
| `VLLM_DS41_KERNEL_BACKEND` | absent or `b12x` | `tilelang` |
| `TILELANG_CACHE_DIR` | not required | required, under `/cache/` |

The settings stay explicit in the profile; nothing is derived at render time.
vLLM selects B12X when `VLLM_DS41_KERNEL_BACKEND` is absent, so the promoted
profiles neither set it nor `kernel_backend`, and their launch commands are
unchanged by the policy. A profile that sets the variable must set it to its
backend, and a TileLang profile must set it. In vLLM the variable selects the
DS4.1 kernel family, and explicit attention, linear and MoE backends must
agree with it.

## Selecting TileLang in a candidate

A TileLang profile, launch disabled until qualified, needs:

1. `kernel_backend: tilelang` and the settings above;
2. an `upstreams_config` lock that lists the `tilelang` and `tile_kernels`
   sources and a vLLM series with the DS4.1 TileLang kernel patch;
3. a source manifest whose vLLM record carries capability `tilelang-kernels`,
   with records for both sources (TileLang's includes its submodule commits);
4. `container.expected_labels` entries `local.spark3.tilelang.tree`,
   `local.spark3.tile_kernels.tree` and `local.spark3.vllm.tree` equal to the
   manifest's trees.

Doctor rejects a TileLang profile that lacks any of these. A B12X profile needs
none of them, and a lock that lists neither source builds exactly as before.
The two sources are built together; a lock listing only one is rejected.

The candidate has one profile per topology, each the B12X configuration of that
topology with only the policy, image and sources changed: `tp3` mirrors the
promoted TP3 profile and `tp4` the TP4 candidate. Its tuning catalog
(`experiments/2026-10-03-tilelang-kernels/profiles.json`) repeats the B12X
catalog's transport settings, so `bin/spark3 tuning --profiles-config <catalog>
create tp3|tp4` materializes TileLang configurations exactly as the B12X ones
are. A TP4 profile keeps its own pinned DSpark cost directory: the curves are
keyed by shapes, not by kernel family.

## Image

For a lock that lists both sources, `build prepare` also fetches TileLang with
its submodules, nested ones included, at the commits the patched tree records,
compares them with the manifest, and exports `tilelang-source` and
`tile_kernels-source` contexts without Git metadata. `build image` then selects
the `runtime-tilelang` stage of `docker/Dockerfile`. It builds wheels of both
from those contexts, replaces the base image's `tilelang` 0.1.12 (vLLM's pin)
and installs TileKernels without dependencies, then fails the build unless:

- every installed file matches the built wheels and the versions match the lock;
- every requirement of both packages already holds;
- every consumer of `apache-tvm-ffi` and `tilelang` accepts the installed
  version, including the candidate vLLM's `requirements/cuda.txt`.

No shared package moves back to suit a consumer ([upstreams.md](upstreams.md)).
The GPU smoke also imports `tilelang` (checking its version), `tile_kernels`
and, once the vLLM manifest record has the capability,
`vllm.models.deepseek_v4_1.tilelang`. TileLang's JIT cache lives at
`TILELANG_CACHE_DIR` on the shared `/cache` mount.

## Promotion

TileLang is promoted like any candidate (`AGENTS.md`): an immutable image,
qualification against the B12X baseline (quality gate, determinism, decode,
prefill and admission), and the owner's acceptance. The promotion commit moves
both sources into `upstreams.lock.json` and its source manifest, the kernel
patch into `patches/vllm/series`, sets `kernel_backend` and the table's
settings in `config/cluster.json` and both page-size profiles together, and
adds the new baseline.
