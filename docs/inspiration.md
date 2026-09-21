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

## Official DeepSeek V4.1 references

| Source | Use it for | Adoption boundary |
|---|---|---|
| [DeepSeek V4.1 Flash release note](https://www.deepseek.com/en/news/deepseek-v4-1-flash/) | Official release identity, 552B CED architecture summary, 8B-prefill/16B-decode activation split, multimodal status, and the claimed one-quarter HBM/eighth persistent-cache footprint relative to V4 Flash. | Treat release claims as architecture expectations, not evidence that this three-Spark deployment realizes them. Measure effective cache bytes per token and workload results locally. |
| [DeepSeek V4.1 Flash technical report](https://arxiv.org/abs/2609.19969) ([model-repository PDF](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/dba1be0a40aa45a94ad051997016db3960a90277/DeepSeek_V41_Tech_Report.pdf)) | Primary source for CED, CSA2 Full/Reindex/Reuse geometry, FP4 global KV, SWA Bounded Replay, Engram, mHC, DSpark, multimodal behavior, context limits, and evaluation settings. | Architecture and quality invariants come from the paper. Its accelerator-scale results are not TP3 DGX Spark targets, and serving shortcuts must not change the native checkpoint's arithmetic without an explicit quality profile. |
| [vLLM DeepSeek V4.1 Flash recipe](https://github.com/vllm-project/recipes/blob/main/models/deepseek-ai/DeepSeek-V4.1-Flash.yaml) | Upstream-supported model flags, reasoning/tool parsers, DSpark configuration, text-only versus vision tradeoffs, memory notes, and current validation status. | It is a moving operational reference for larger datacenter accelerators, not a drop-in Spark launch. Reconcile every flag with the pinned vLLM revision and TP3/SM121 constraints before testing. |
| [deepseek-ai/FlashMLA](https://github.com/deepseek-ai/FlashMLA) | Official V4.1 prefill/decode kernels, FP8/FP4 cache semantics, fused paths, and performance baselines for sparse MLA. | Use it to check semantics and identify upstream kernel opportunities. The promoted SM121 path may use FlashInfer or B12X, so do not replace it merely because an official kernel exists for another architecture. |

## Reference implementations

| Repository | Review it for | Adoption boundary |
|---|---|---|
| [MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks) | Multi-Spark SGLang launch and tuning, native-weight performance, engram handling, startup behavior, and its published benchmark methodology. | Use as the principal performance comparison, then reproduce each promising mechanism independently in an experiment. Do not infer parity from differently shaped workloads. |
| [MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks) | Memory layout, cache/offload, scheduling, and the magnitude of performance available from a highly specialized two-node implementation. | EXL3 changes numerical quality and is not eligible for the native-weight baseline. Only quality-neutral techniques should move forward. |
| [MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark) | DSpark implementation changes, draft/verification behavior, JIT and CUDA-graph choices, and Spark-specific performance commits. | Confirm model and topology applicability, then prefer the equivalent upstream vLLM change or a small upstreamable patch. |
| [rhys101/DeepSeek-V4.1-Flash-vLLM-DGX-Spark-8](https://github.com/rhys101/DeepSeek-V4.1-Flash-vLLM-DGX-Spark-8) | Eight-Spark scaling, native resident Engram, native-prefill integration, TP8/EP4 organization, switched multi-rail RoCE operations, long-context capacity trials, acceptance tooling, and its documented earlier vLLM path. | Its current headline SG18 results use SGLang on eight Sparks and a switched dual-interface MTU-9000 fabric, not this vLLM TP3 switchless ring. Reuse methodology or independently isolated mechanisms only; do not compare its published throughput directly with this deployment. |
| [knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4](https://github.com/knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4) | Native-weight TP4 measurements for bounded and rank-split indexer prefill, Engram row caching and prefetch, DSpark depth, shared-expert alignment, fast loading, small-collective RoCEnante routing, long-context memory, and switched versus switchless operation. | It is an AGPL SGLang TP4/EP2 profile derived from the Mia implementation, not a vLLM TP3 source. Treat its measurements and quality gates as hypotheses to reproduce; port only isolated quality-neutral mechanisms through their canonical upstreams. |
| [jakejharris/jspark3-deepseek](https://github.com/jakejharris/jspark3-deepseek/tree/v2.0.1) | Three-node orchestration, network topology, benchmark scripts, context/concurrency choices, and operational ergonomics. | Compare against tag `v2.0.1`; do not import mutable scripts or configuration without recording an exact source revision and experiment. |
| [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) | Container construction and newer combinations of CUDA, NCCL, PyTorch, FlashInfer, and vLLM on DGX Spark. | Version novelty alone is not a promotion reason. Rebuild from pinned canonical sources and qualify correctness, memory, TTFT, and TPS together. |
| [jnardiello/GLM-5.3-Flash-FP8-4-DGX-Spark-Switchless](https://github.com/jnardiello/GLM-5.3-Flash-FP8-4-DGX-Spark-Switchless) | Switchless multi-rail topology, rank/interface mapping, launch discipline, collective profiling, and four-Spark operational lessons. | Separate generally useful topology ideas from GLM-, TP4-, and four-node-specific assumptions before testing on DS4.1 TP3. |
| [tonyd2wild/GLM-5.3-Flash-NVFP4-1M-KV-4x-DGX-Spark](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-1M-KV-4x-DGX-Spark) | GB10 memory-ceiling experiments, explicit KV-memory ladders, page-cache effects during startup, worker-first launch discipline, batched-token and sequence-concurrency sweeps, CUDA-graph A/B tests, DFlash2 acceptance measurements, long-context failure gates, and reproducible benchmark receipts. | Its GLM TP4 checkpoints, NVFP4 attention/KV paths, DFlash2 drafter, four-node fabric, container patches, and cache-dropping policy are not DS4.1 TP3 defaults. Reproduce each quality-neutral operational or scheduler hypothesis independently; do not transfer its quantized arithmetic, memory values, or destructive host-cache behavior without a controlled experiment. |
| [osolmaz/oomwrap](https://github.com/osolmaz/oomwrap) | Process-level memory-pressure supervision and failure handling. | Compare its semantics with `scripts/memguard.sh`; adopt it only if it demonstrably improves protection without interfering with JIT startup or adding opaque policy. |

## End-to-end and quality methodology

| Source | Review it for | Adoption boundary |
|---|---|---|
| [deepseek-ai/deepseek-harness](https://github.com/deepseek-ai/deepseek-harness) | Official agent-harness behavior, DeepSeek provider integration, DSML tool calls, reasoning controls, multimodal request handling, and realistic long-horizon compatibility smoke tests. | It is an application harness, not an inference benchmark or serving implementation. Add pinned scenarios as a separate end-to-end workload; do not use its elapsed time to attribute vLLM kernel performance. |
| [Unsloth DeepSeek V4 guide](https://unsloth.ai/docs/models/deepseek-v4) | Tensor identity checks, KL-divergence, top-token agreement, and quantization quality reporting methodology. | The page targets the earlier 284B V4-Flash-0731 GGUF/llama.cpp stack, not the 552B V4.1 native checkpoint. Do not import its weights, memory figures, launch flags, chat template, or DSpark results. Use the official V4.1 paper, model artifact, and vLLM recipe above for this deployment. |

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
9. Does measured global KV storage approach the report's 890-byte-per-token
   architectural figure once fixed SWA, alignment, allocator groups, and runtime
   metadata are accounted for separately?
10. Do reasoning, DSML tool calls, and multimodal requests remain compatible in
    a pinned DeepSeek Harness scenario, independently of the server microbenchmarks?
11. Do candidate output distributions preserve the native baseline under
    deterministic token agreement and KL-divergence checks, not just a handful
    of exact prompts?

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
| 2026-09-20 | Official V4.1 sources, vLLM recipe, DeepSeek Harness, FlashMLA, and Unsloth V4 guide | Added the release note and technical report as model authorities; added the vLLM recipe and FlashMLA as implementation references; classified DeepSeek Harness as an end-to-end workload and Unsloth's earlier-model page as quality methodology only. |
| 2026-09-20 | rhys101 eight-Spark DS4.1 implementation | Added its native-prefill, resident-Engram, RoCE scaling, long-context capacity, acceptance, and earlier-vLLM work as inspiration while recording that its headline SG18 measurements are not a like-for-like TP3 vLLM comparison. |
| 2026-09-20 | knapcio four-Spark TP4 DS4.1 profile | Added its measured bounded/rank-split prefill, Engram, DSpark, shared-expert, loader, RoCEnante, long-context, quality, and switchless findings as hypotheses to reproduce without treating its SGLang TP4 stack as a deployment input. |
| 2026-09-21 | tonyd2wild four-Spark GLM 5.3 profile | Added its measured GB10 memory ladder, startup page-cache investigation, concurrency and batched-token sweeps, CUDA-graph comparison, speculative-acceptance checks, long-context gates, and run receipts as operational hypotheses; its GLM-specific NVFP4 and TP4 implementation remains outside the DS4.1 native-weight deployment. |
