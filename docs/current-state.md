# Captured current state

Captured 2026-09-20 from the three running containers.

| Setting | Active value |
|---|---:|
| Tensor parallel ranks | 3 |
| Maximum model length | 160,000 tokens |
| Maximum sequences | 8 |
| Maximum parallel prefills | 1 |
| Batched-token budget | 8,192 |
| Long-prefill threshold | 4,096 |
| Explicit KV memory | 3 GiB per rank |
| Reported KV capacity | 498,145 tokens |
| Full 160K concurrency | 3.11x |
| DSpark depth | 3 |
| Container memory limit | 112 GiB |
| Host memguard threshold | 3 GiB, armed after JIT/API readiness |
| Async scheduling | disabled |
| Reasoning | enabled |

The eight-sequence limit controls admission count; it does not imply eight complete
160K KV windows. The aggregate live cache is approximately 498K tokens.

The three images carry matching critical vLLM and B12X source. dgx2 and dgx3 are
byte-identical. dgx1 has a different image ID because its image omits the unused
`vllm/benchmarks/` package; serving-critical source hashes match. This is acceptable
as captured evidence but not as the final deployment method.

## Known transition gaps

- Do not promote the incomplete release-branch image reconstruction; retain it
  only as forensic evidence until the upstream-main candidate supersedes it.
- Build once and distribute a single OCI digest to all ranks.
- Replace the incomplete release-branch reconstruction with the isolated
  upstream-main candidate in `experiments/2026-09-20-upstream-main-rebase/`.
- Keep the upstream-native allocator/model qualification separate from the
  B12X/RoCEnante performance overlay. The current prepared source carries do
  not yet include vLLM's DS4.1 B12X or RoCEnante consumer adapters.
- Qualify the tracked model configuration and indexer artifacts under the
  deterministic `/home/swank/projects/spark3-vllm-ds41f/` node path.
- Exercise the coordinated deployment rollback path during a scheduled restart.
- Establish a frozen benchmark baseline for this exact concurrency-tuned runtime.

The B12X patch stack and the mounted vLLM indexer override have been applied to
fresh pinned checkouts and match their recorded live files. A later full-tree
comparison found additional live modifications under the DS4.1 model package,
plus `vllm/models/deepseek_v4/nvidia/model.py`, that are not represented by the
vLLM patch series. `manifests/sources/2026-09-20-active-source.json` now states
that narrower verification scope explicitly.

The reconstruction additionally pins the official vLLM image digest, FlashInfer
`v0.7.0rc1`, CUTLASS `v4.4.2`, and the five CuTe DSL 4.6.2 wheel hashes. Its
Dockerfile remains a candidate until a build and functional/benchmark
qualification are attached as an experiment.
