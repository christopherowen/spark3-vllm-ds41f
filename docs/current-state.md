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

- Build and qualify the committed image reconstruction on a Spark.
- Build once and distribute a single OCI digest to all ranks.
- Qualify the tracked model configuration and indexer artifacts under the
  deterministic `/home/swank/projects/spark3-vllm-ds41f/` node path.
- Exercise the coordinated deployment rollback path during a scheduled restart.
- Establish a frozen benchmark baseline for this exact concurrency-tuned runtime.

The vLLM and B12X patch stacks have now been applied to fresh pinned checkouts and
match every modified live file byte-for-byte. The hashes are recorded in
`manifests/sources/2026-09-20-active-source.json`.

The reconstruction additionally pins the official vLLM image digest, FlashInfer
`v0.7.0rc1`, CUTLASS `v4.4.2`, and the five CuTe DSL 4.6.2 wheel hashes. Its
Dockerfile remains a candidate until a build and functional/benchmark
qualification are attached as an experiment.
