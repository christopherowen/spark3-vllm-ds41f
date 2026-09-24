# Improvement leads for the promoted karmic-kraken stack

The leads below came from desk research; the results section records what
then ran on the idle cluster. Sources: a code audit
of B12X's dense MXFP8 GEMM path and vLLM's DS4.1 layers, and a survey of the
repositories in `docs/inspiration.md`. Profile shares refer to
`../2026-09-24-kk-speed-tuning/runs/profile-c1/` (rank-0 decode GPU time:
routed MoE 33%, dense FP8 GEMM 27%, RoCE collectives 20%, LM head 6%).

## Where we stand

Published DeepSeek V4.1 Flash numbers on three Sparks do not beat ours on
prose, concurrency, or prefill (MiaAI SGLang TP3: 37.9 tok/s prose, 78.6 at
four streams, about 1.6k tok/s prefill; jspark3 and tonyd2wild vLLM TP3 with
lower-bit EXL3 experts: prose 33-34, six streams 146-153). Single-stream code
is where others lead (62-76 tok/s against our 52), using DSpark depth 5 with a
verification cap or lossy EXL3 experts. Four- and eight-Spark setups are
faster with 33-167% more hardware. Several published numbers use sparkDash
prompts that inflate prose and prefill.

## Ranked leads

1. **Shard the Engram WKV projection.** Every rank reads the full 25600x6144
   FP8 projection twice per forward (about 315 MB per rank per step). vLLM's
   `engram_config.projection_tp` shards it with a `ColumnParallelLinear`, which
   requires 25600 to divide by the TP size; at TP3 it needs a small padding
   patch (pad the output to 25632 rows, slice after the gather). Same math, so no
   quality change expected. Saves about 210 MB per step (roughly 1 ms, ~2%).
   A four-Spark deployment saved 1.4 ms/step sharding this and the fused
   WQA/WKV projection with bit-identical output.
2. **Dense GEMM plans for under-filled grids.** DS4.1's 32x32-scaled weights
   run as plain MXFP8 with split-K off. The fused attention input projection
   (1792x5120) launches 14 CTAs and the shared-expert gate/up (1536x5120) 24 CTAs
   on 48 SMs; together about 45% of dense-GEMM bytes. The targeted in-engine
   autotune measured neutral end to end, so first measure per-shape default
   versus best plan with B12X's offline sweep (`window1.sh`). If the gains are
   real, pin plans through `block_fp8_linear.plan(override=...)` (one-line
   vLLM change) or extend B12X's split rule upstream.
3. **Fuse activation quantization into the GEMM.** Each dense GEMM runs a
   separate MXFP8 row-quantization kernel first, with no programmatic
   dependent launch: about 170 extra serialized kernel boundaries per step.
   B12X already has a fused BF16-to-MXFP8 path for M <= 8
   (`dense_gemm_fused_quant_a`, used by the WO projection). Kernel/integration
   work in B12X.
4. **Block rejection for sampled traffic.** `rejection_sample_method: "block"`
   (the upstream vLLM recipe's choice) raised accepted tokens 1.8-2.5% per step
   on sampled traffic elsewhere. Lossless; no effect at temperature 0, so it
   needs a sampled benchmark.
5. **LM head.** Unmerged Local Inference Lab work (vLLM #873 with B12X #421)
   adds a tensor-core GEMV for BF16 linears at M <= 8; an FP8 draft head halved
   its time elsewhere. Wait for merge and SM121 validation.
6. **Prefill.** vLLM #880 (prefill-only steps use the full token budget) and
   rank-split indexer prefill (21-28% on long prompts in an SGLang port) are
   candidates; our eager prefill all-reduces spend most of short-prompt TTFT on
   host launch skew.
7. **Engram host time.** B12X `6e2090bc` reads disk Engram rows concurrently
   (0.55 ms host time per decode cycle elsewhere); needs a newer B12X pin and
   the matching vLLM call site.

Already in place (no action): tensor-parallel MoE (no expert parallelism),
trimmed NCCL buffers, `disk_resident_scales: false`, 4/2/2 compile workers,
two HCAs per peer.

Not pursued: lower-bit EXL3 experts (lossy; a published TP3 run passed 2 of 4
code tasks), FlashMLA (its V4.1 kernels target SM90/SM100 only).

## Maintenance window 1 (`window1.sh`)

Stops the promoted service, sweeps the dense MXFP8 plans for every DS4.1 TP3
shape at M=4 and M=32 in a 24 GiB-capped container under a host memory
watchdog, then restarts the promoted service and runs `doctor --live`.
Expected downtime 20-40 minutes (the sweep compiles every candidate on first
use). Output: `runs/window1/*.tsv`, default plan against the fastest per shape.

## Results (2026-09-24, cluster idle)

- **Window 1, dense GEMM plans** (`runs/window1/`): at every DS4.1 TP3 shape
  the heuristic plan is within 0-4% of the best offline plan at M=4 and M=32
  (weights stream at 195-216 GB/s, the same practical ceiling as the MoE);
  only the tiny DSpark context-KV GEMM gains (1.25x at M=4, about 4 us).
  Pinning plans is not worth a patch. Rejected.
- **Engram projection TP** (`overlay-vllm/engram.py`, then vLLM patch
  `0001-engram-projection-tp-padding`): pads the 25600-row output to 25632
  (8544 rows and 267 scale blocks per rank) and loads the last rank's missing
  rows through `allow_tp_padding`; the first attempt, which concatenated
  padding, was refused by the B12X checkpoint loader. Profile
  (`runs/profile-engramtp-c1/`, `compare_profiles.py`): the projection falls
  from ~690 to ~235 us per call, about 0.4 ms per output token at one stream
  (~2%). The serving matrix cannot resolve 2% (all points within noise); LRU
  5/5 twice; dgx1 minimum MemAvailable +0.25 GiB. Accepted.
- **Block rejection** (`bench_sampled.py`, temperature 1.0, 16 requests per
  cell): mean accepted drafts per step prose c1 0.857 vs 0.899, prose c4 0.890
  vs 0.885, code c1 2.274 vs 2.112, code c4 2.253 vs 2.242 (block vs
  standard); about +1% on average, within sampling noise. Lossless and the
  upstream recipe's choice; adopted as a zero-risk setting with inconclusive
  measured gain. No effect at temperature 0.
- **vLLM #880** (prefill-only steps use the full token budget): not applicable.
  This vLLM sets `max_num_scheduled_tokens` equal to `max_num_batched_tokens`
  under DSpark (the startup warning only flags values below 8192), so there is
  no cap to lift.
- **Full candidate** (capture32, no FlashInfer autotune, Engram TP, block
  rejection; `cluster-blockrej.json`): LRU 5/5 twice, every serving point
  within noise of capture32, dgx1 minimum MemAvailable 6.50 GiB (from 6.07).
- **Deferred:** fused activation quantization (B12X kernel work, ~1-2%),
  LM head (bandwidth-bound BF16 GEMV; only an FP8 head would help, with a
  numerics risk), rank-split indexer prefill (SGLang port), concurrent disk
  Engram reads (newer B12X plus vLLM call-site change; host time mostly
  hidden behind the GPU).

The Engram patch is carried in the image recipe
(`../2026-09-23-karmic-kraken-reference/patches/vllm/`), producing image
`vllm-ds41f-kkref:01f1b874c774-r2` (vLLM tree `90fdd043`); `cluster-r2.json`
is the candidate on that image with no overlays.
