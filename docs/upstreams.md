# Upstreams and contribution flow

`upstreams.lock.json` is the machine-readable authority. The current chain is:

```text
local-inference-lab/vllm integration/karmic-kraken-beta @ 01f1b874
        (canonical base vllm-project/vllm @ 0f8fa53a)
        + patches/vllm/series (Engram projection TP padding)
                    \
                     vllm-ds41f-kkref:01f1b874c774-r2 (sha256:a95eb867…)
                    /
local-inference-lab/b12x integration/karmic-kraken-beta @ 0f846212
        + patches/b12x/series (switchless RoCEnante routing)

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
| `flashinfer` | `flashinfer-ai/flashinfer` | not created yet | not a promoted build input (the runtime base supplies 0.6.18.post1) |
| `cutlass` | `NVIDIA/cutlass` | not created yet | SM121 stable-extension headers at `v4.7.1` |
| `cutlass_dsl` | NVIDIA packages on PyPI | not applicable | SHA-256-locked ARM64 CuTe DSL wheel set |
| `model` | `deepseek-ai/DeepSeek-V4.1-Flash` | not applicable | unchanged native model weights and tokenizer |

The URL and full immutable revision in `upstreams.lock.json`, not a directory
name or Docker tag, define identity. The `tracking_ref` says which canonical
branch to inspect for future pulls; it does not float the build.

## Refreshing an upstream

Prepare an isolated tree:

```sh
bin/spark3 upstream prepare vllm
bin/spark3 upstream prepare b12x
```

The helper clones the canonical repository as `upstream`, adds Christopher's fork
as `origin` when one exists, checks out the full pinned SHA, initializes pinned
submodules where required, applies recorded upstream commits, and then applies
the local series. Worktrees live under ignored `.work/upstreams/`.

To evaluate a newer upstream, create a branch in that prepared tree, rebase the
small patch series, build a new immutable image tag, and record it as an experiment.
Do not advance `upstreams.lock.json` merely because a newer commit exists.

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
