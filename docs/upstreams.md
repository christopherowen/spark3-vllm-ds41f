# Upstreams and contribution flow

`upstreams.lock.json` is the promoted machine-readable authority. An experimental
cluster profile may select a separate repository-relative lock through
`upstreams_config`; build preparation, doctor and image builds use that lock.
This lets candidates add patches without changing the promoted image identity.
The current production chain is:

```text
local-inference-lab/vllm integration/karmic-kraken-beta @ 04c30fa9
        (canonical base vllm-project/vllm @ 0f8fa53a)
        + patches/vllm/series (Engram projection TP padding,
                               asynchronous Engram rows, DSML tool
                               parameters, multimodal block hashes;
                               dead verification rows; sequence-
                               parallel prefill; one-read custom-op
                               defaults; other DSpark tools off by
                               default)
                    \
local-inference-lab/b12x integration/karmic-kraken-beta @ f8069b2c
        + patches/b12x/series (switchless RoCEnante routing, CuTe DSL 4.7.1,
          top-k position ties, TMA stage-release fences)
                     ---- vllm-ds41f-kkref:04c30fa98e79-r5o (sha256:288fc5bd…)
                    /
NVIDIA/nccl v2.30.7-1 @ 73cf1122
        + patches/nccl/series (IB send-path fence), replacing the base
          image's nvidia-nccl-cu13 libnccl.so.2

deepseek-ai/DeepSeek-V4.1-Flash @ dba1be0a (unchanged; TP3 head padding is in
        the serving source, so no config overlay is mounted)
```

The official vLLM nightly at `af1c0149` supplies the CUDA/runtime base, including
FlashInfer 0.6.18.post1; only vLLM's two stable CUDA extensions are rebuilt for
SM121 against CUTLASS v4.7.1. Canonical vLLM main is parked while the Local
Inference Lab integration branch is the serving source.

## Source map

| Lock key | Canonical upstream | Contribution fork | Purpose |
|---|---|---|---|
| `vllm` | `local-inference-lab/vllm` (`integration/karmic-kraken-beta`) | `christopherowen/vllm` | serving source; canonical `vllm-project/vllm` main is parked |
| `vllm_base_image` | `docker.io/vllm/vllm-openai` | not applicable | official CUDA/Torch/native-extension foundation |
| `b12x` | `local-inference-lab/b12x` (`integration/karmic-kraken-beta`) | not created yet | DS4.1 kernels and RoCEnante transport |
| `nccl` | `NVIDIA/nccl` (tag `v2.30.7-1`) | not created yet | the base image's NCCL version, rebuilt for SM121 with `patches/nccl` |
| `flashinfer` | `flashinfer-ai/flashinfer` | not created yet | not a promoted build input (the runtime base supplies 0.6.18.post1) |
| `cutlass` | `NVIDIA/cutlass` | not created yet | SM121 stable-extension headers at `v4.7.1` |
| `cutlass_dsl` | NVIDIA packages on PyPI | not applicable | SHA-256-locked ARM64 CuTe DSL wheel set |
| `model` | `deepseek-ai/DeepSeek-V4.1-Flash` | not applicable | unchanged native model weights and tokenizer |
| `tilelang` | `tile-ai/tilelang` (`main`) | `christopherowen/tilelang` | TileLang kernel backend candidate only: TileLang 0.1.15 with `patches/tilelang`, built with its submodules |
| `tile_kernels` | `deepseek-ai/TileKernels` (`main`) | not applicable | TileLang kernel backend candidate only: DeepSeek's TileLang kernels, unchanged |

`tilelang` and `tile_kernels` are optional sources: a lock builds them only
when it lists them, and the promoted lock does not. The
[TileLang candidate](../experiments/2026-10-03-tilelang-kernels/README.md)
lists them; see [kernel-backends.md](kernel-backends.md). The TileLang patches
are the first three commits of the fork's `deepseek-v41-sm120` branch; its
later commits are TileLang examples, not build inputs.

The URL and full immutable revision in `upstreams.lock.json`, not a directory
name or Docker tag, define identity. The `tracking_ref` says which canonical
branch to inspect for future pulls; it does not float the build.

## Refreshing an upstream

Prepare an isolated tree:

```sh
bin/spark3 upstream prepare vllm
bin/spark3 upstream prepare b12x
bin/spark3 --cluster-config experiments/2026-10-03-tilelang-kernels/cluster.json upstream prepare tilelang
```

The helper clones the canonical repository as `upstream`, adds Christopher's fork
as `origin` when one exists, checks out the full pinned SHA, initializes pinned
submodules where required, applies recorded upstream commits, and then applies
the local series. Worktrees live under ignored `.work/upstreams/`.

To evaluate a newer upstream, create a branch in that prepared tree, rebase the
small patch series, build a new immutable image tag, and record it as an experiment.
Do not advance `upstreams.lock.json` merely because a newer commit exists.

Shared runtime packages move forward, never back. CuTe DSL is used by vLLM,
quack-kernels, B12X, FlashInfer and the cuDNN frontend; the locked wheel set in
`requirements/cutlass-dsl-aarch64.txt` must satisfy every one of them and must
not be older than the version the vLLM base image installs. When a component
pins an older version, patch the component's pin forward and qualify it rather
than downgrading the others. The image build checks each consumer's declared
requirement and fails on a mismatch. The same rule covers `apache-tvm-ffi`,
which TileLang and B12X's CuTe DSL stack share, and `tilelang` itself: the
TileLang image checks every installed consumer and the candidate vLLM's
`requirements/cuda.txt`, which must move its `tilelang==0.1.12` pin forward.

## Contributing back

One conceptual local patch should become one upstream branch and pull request.
Preserve tests and benchmark evidence alongside it. After acceptance:

1. advance the pinned upstream revision to include the merged commit;
2. remove the redundant local patch from `series`;
3. rebuild once and distribute the same digest;
4. reproduce the accepted baseline before promotion.

The B12X contribution fork does not yet exist, so its `contribution_remote` is
currently `null`. Add the fork URL only when creating the first contribution.
The same rule applies to FlashInfer and CUTLASS: do not invent a fork URL or use
this deployment repository as their `origin`.

The GLM switchless repository is recorded as a research reference, not an
upstream. Ideas may be reimplemented and measured, but its topology- and
model-specific patches do not enter our patch stack implicitly.

The complete research-reference watchlist, review questions, and adoption gates
are maintained in [inspiration.md](inspiration.md). Entries under `references` in
`upstreams.lock.json` are deliberately excluded from build preparation.
