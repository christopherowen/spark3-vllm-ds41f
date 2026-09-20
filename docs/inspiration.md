# Improvement watchlist

This document tracks repositories worth reviewing for ideas that may improve
TTFT, decode throughput, context capacity, memory use, startup behavior, or
operational reliability on the three-node switchless DGX Spark cluster.

These are **references, not dependencies**. A repository appearing here does
not authorize copying its launch scripts, source tree, image, defaults, or model
weights into the promoted deployment. Machine-readable URLs live under
`references` in `upstreams.lock.json`; canonical build inputs remain under
`sources` in that file.

## Canonical upstreams to watch first

| Repository | Ownership here | What to watch |
|---|---|---|
| [vllm-project/vllm](https://github.com/vllm-project/vllm) | Canonical serving upstream | DeepSeek V4.1 support, distributed execution, speculative decoding, sparse attention, multimodal support, memory planning, scheduler changes, and ARM64/Blackwell fixes. Prefer merged upstream functionality over a local patch. |
| [local-inference-lab/b12x](https://github.com/local-inference-lab/b12x) | Canonical kernel/transport upstream | DS4.1 kernels, QSA/indexer work, mHC, DSpark, RoCEnante collectives and topology, memory reductions, Spark-specific fixes, issues, and pull requests. |
| [flashinfer-ai/flashinfer](https://github.com/flashinfer-ai/flashinfer) | Canonical sparse-attention upstream | Native sparse-attention behavior, SM121 support, JIT/autotuning, cache correctness, and release compatibility. |
| [NVIDIA/cutlass](https://github.com/NVIDIA/cutlass) | Canonical kernel foundation | SM121/CuTe DSL correctness and performance changes relevant to the pinned B12X and vLLM builds. |

Pinned revisions and contribution remotes are documented in
[upstreams.md](upstreams.md). This watchlist never overrides those pins.

## Reference implementations

| Repository | Review it for | Adoption boundary |
|---|---|---|
| [MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks) | Multi-Spark SGLang launch and tuning, native-weight performance, engram handling, startup behavior, and its published benchmark methodology. | Use as the principal performance comparison, then reproduce each promising mechanism independently in an experiment. Do not infer parity from differently shaped workloads. |
| [MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks) | Memory layout, cache/offload, scheduling, and the magnitude of performance available from a highly specialized two-node implementation. | EXL3 changes numerical quality and is not eligible for the native-weight baseline. Only quality-neutral techniques should move forward. |
| [MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark) | DSpark implementation changes, draft/verification behavior, JIT and CUDA-graph choices, and Spark-specific performance commits. | Confirm model and topology applicability, then prefer the equivalent upstream vLLM change or a small upstreamable patch. |
| [jakejharris/jspark3-deepseek](https://github.com/jakejharris/jspark3-deepseek/tree/v2.0.1) | Three-node orchestration, network topology, benchmark scripts, context/concurrency choices, and operational ergonomics. | Compare against tag `v2.0.1`; do not import mutable scripts or configuration without recording an exact source revision and experiment. |
| [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) | Container construction and newer combinations of CUDA, NCCL, PyTorch, FlashInfer, and vLLM on DGX Spark. | Version novelty alone is not a promotion reason. Rebuild from pinned canonical sources and qualify correctness, memory, TTFT, and TPS together. |
| [jnardiello/GLM-5.3-Flash-FP8-4-DGX-Spark-Switchless](https://github.com/jnardiello/GLM-5.3-Flash-FP8-4-DGX-Spark-Switchless) | Switchless multi-rail topology, rank/interface mapping, launch discipline, collective profiling, and four-Spark operational lessons. | Separate generally useful topology ideas from GLM-, TP4-, and four-node-specific assumptions before testing on DS4.1 TP3. |
| [osolmaz/oomwrap](https://github.com/osolmaz/oomwrap) | Process-level memory-pressure supervision and failure handling. | Compare its semantics with `scripts/memguard.sh`; adopt it only if it demonstrably improves protection without interfering with JIT startup or adding opaque policy. |

## Questions to revisit

Use this list when reviewing new commits, issues, and pull requests:

1. Can TTFT improve without reducing output quality, context capacity, or normal
   four-request admission?
2. Can aggregate and per-stream decode TPS improve under the same prompt and
   output-token workload?
3. Can model, compiler, or host-memory use fall enough to expand KV cache without
   moving native weights to a lower-quality representation?
4. Can RoCEnante use both physical rails more effectively, and which collective
   sizes should remain on NCCL?
5. Are DeepSeek CED/DSpark draft and verification paths using all upstream
   correctness and performance fixes?
6. Can JIT, CUDA-graph, and autotuning work be made deterministic and reusable
   while leaving enough unguarded memory for first startup?
7. Are newer CUDA, NCCL, PyTorch, FlashInfer, CUTLASS, or vLLM combinations
   materially better on SM121, rather than merely newer?
8. Can image support return without compromising the promoted text baseline?

## From idea to promotion

For every candidate idea:

1. Record the repository, exact commit or tag, source link, license/provenance,
   and the specific mechanism—not just a configuration value.
2. Check canonical vLLM, B12X, FlashInfer, and CUTLASS first for the same or a
   newer implementation.
3. Create one experiment under `experiments/YYYY-MM-DD-short-name/` with the
   promoted commit as its base and one intended causal change.
4. Run the repository benchmark matrix, including quality gates, memory
   observations, four-request admission, and the Strix workload where relevant.
5. Reject changes that trade away native-weight quality unless a separate,
   explicitly named quality/performance profile is requested.
6. Prefer a small patch suitable for its canonical upstream. Carry it locally
   only while the upstream contribution is pending or when the upstream cannot
   accept a Spark-specific boundary.
7. Promote only through a reviewed config/manifest commit and coordinated
   deployment. Never deploy directly from a watched repository.

## Review log

| Date | Scope | Result |
|---|---|---|
| 2026-09-20 | Initial watchlist from the DS4.1 three-Spark investigation | Classified canonical upstreams and seven reference repositories; no new runtime change was promoted. |
