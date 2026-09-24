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

## External review: MiaAI overnight and sfxnz (2026-09-24 evening)

MiaAI-Lab `6b40a5e` reports a TP3 SGLang configuration 17-40% faster at one
stream and 19% faster at four than its previous one. `mia_workload.py`
reimplements their `benchmarks/overnight_bench.py` workload without running
their code: same prompts, thinking off, 512 tokens, decode tok/s from the first
to the last content delta, and C4 as 2 prose plus 2 code. It ran against our
promoted r2 service with 8 repetitions
(`runs/mia-workload-r2.json`; their numbers are 5 repetitions from
`docs/overnight-results.md`):

| Their workload | MiaAI final | Ours (r2) |
|---|---|---|
| C1 prose | 34.2 | 39.6 (+16%) |
| C1 prose2 | 32.6 | 37.2 (+14%) |
| C1 code | 62.2 | 61.4 (tie) |
| C1 sampled chat, T=0.7 | 37.2 | 42.1 (+13%) |
| C4 aggregate | 75.8 | 93.8 (+24%) |
| C4 per-stream | 24.3 | 29.4 (+21%) |

Most of their gain is changes our stack already has:

- The checkpoint's FP8 `wo_a`: our V4.1 attention consumes the e4m3 weights
  and UE8M0 scales directly (`_WOProjectionWeightMethod`).
- Tensor-parallel MoE, which is their EP1.
- Block verification.
- Deterministic kernel tactics: no FlashInfer autotune.
- One image on every rank, checked by `doctor --live`. Their rank 2 had been
  running a different build.

Their single-stream code lead comes from DSpark k=5 with a draft-confidence
cap (4.0 accepted tokens per step on code).

sfxnz `45d3303` (vLLM TP2, EXL3 2-bit experts) is not comparable on quality
or topology. Its output-neutral findings agree with MiaAI's on one point:
the disk-Engram host path stalls the GPU.

Our own profile shows the same stall (`runs/profile-engramtp-c1/`, one stream).
The GPU is busy 78% of the window. In a representative 40.1 ms step, the only
gaps over 50 us are 0.72 + 0.27 + 0.88 ms around the two Engram `_lookup`
kernels at step start: hashes copied to the host, host disk reads, then the
row copy. That is about 1.9 ms (4.7%) of GPU idle per step.

Leads, in order:

1. **Overlap the disk Engram lookup with compute.** MiaAI forks the lookup
   onto a side stream once the hash ids exist, and joins it at each Engram
   layer. Layer 14's rows are then ready about 15 ms before they are needed.
   The change is bit-exact, with a check mode. sfxnz stages Engram tables in
   parallel and prefetches next-step rows with `fadvise(WILLNEED)` after
   propose. Expected: about 4% at one stream. This needs a vLLM patch around
   `prepare_disk_engram`, and possibly B12X's lookup binding.
2. **Draft-only FP8 LM head.** The DSpark draft reuses the target's bf16
   `lm_head` for full-vocab draft logits. An e4m3 copy used only by the draft
   leaves every accepted token unchanged: MiaAI saw 0.1-1.3 ms per step, and
   acceptance 3.0882 to 3.0875 offline. It costs about 200 MiB per rank; dgx1
   headroom is 5.8 GiB above the 3 GiB guard.
3. **Depth 5 with a confidence cap, for code.** Our depth-5 run raised
   acceptance but not throughput, because verification streams each row's
   experts. MiaAI's cap maps dead rows to the anchor row's experts, so capped
   positions cost no MoE bandwidth. vLLM's adaptive verification is our
   analogue. Retest after 1 and 2, measured on code.
4. **Memory hygiene from sfxnz:** page-cache drop after load, indexer
   workspace factor, logits cap. It is worth measuring only as dgx1 headroom
   for KV or sequences.

Already true for us: capture sizes cover every verify width (k+1=4:
4, 8, ..., 32), and there is no autotune. Both repos' negative results (k
sweeps, autotune) match ours.

A safety note from MiaAI: a single prompt of about 200k tokens ran their head
node out of memory during prefill, with the KV cache only 31% full (per-chunk
indexer transient). The node hung for 40 minutes with no memory guard. Our
context limit is 160K; prefill is measured to 128K by `bin/spark3 bench` and
to 254K in `kkt-len512k`, both under the 3 GiB steady guard.
