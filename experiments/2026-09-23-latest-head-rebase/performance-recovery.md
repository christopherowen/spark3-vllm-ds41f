# DS4.1 performance attribution and recovery plan

2026-09-23. This source and configuration audit is paired with the consolidated
runtime benchmark in `performance-bridge.md`. All new measurements use the
consolidated 10-vLLM/5-B12X series pinned in `series.json`.

## What the existing comparison establishes

The September 20 weekend service reached 31.4/44.3 output tokens/s at
concurrency 1 for prose/code; the patch-0029 target-only service reached
10.1/10.1. The consolidated current-head service reached 9.1/8.9 on the first
pass and 9.3/9.3 on the single-request repeats. At concurrency 2–8 it was
within roughly 3% of patch0029. All eight matched cases were slower than the
weekend setup in `performance-bridge.md`. This is an as-operated regression,
not a patch attribution. The weekend used three-token DSpark and
`FULL_AND_PIECEWISE` CUDA graphs. Patch-0029 has no DSpark and uses
`cudagraph_mode=NONE`. The V4.1 model does not support `torch.compile`, so
turning graphs off leaves the target decode path eager. The source, block size,
KV reservation, and container memory cap also differ.

We own the decision to leave both accelerators off after the new path's
correctness and startup failures. Those failures justified a safe target-only
launch; they do not establish that the consolidated source must remain slow.
No local patch has been measured in isolation against the same source, config,
workload, and hardware state. The consolidated result shows that patch cleanup
did not create the large loss, but it does not separate upstream code from the
disabled accelerators. Do not attribute the 69–82% low/moderate concurrency
loss to an upstream kernel or to B12X by inspection.

The TP3 virtual geometry (64/8 real heads/groups to 72/9 padded) is present
in the shared `state/config.json` for both services. It costs extra work, but
it is not a new variable in the weekend-to-patch-0029 comparison. The patch
consolidation itself preserved the old candidate's exact source trees; the
September 23 rebase changes upstream source, now measured as a complete arm.

## Local hot-path candidates

The consolidated series has a few patches worth profiling after acceleration
is restored. The B12X V4.1 adapter selects attention/cache writers and native
24-head partitions; disk Engram stages rows outside graph capture; RoCEnante
routes three-rank collectives. These are correctness requirements today, but
their decode costs are unmeasured. ModelOpt scales, the file-backed loader,
selected-JIT registration, and plan preparation mostly act at load or startup,
so they are weak explanations for a steady 256-token decode slowdown. The
vocabulary patch affects runtime padding, but both services use the same TP3
geometry. B12X's 3,072-token mHC tuning is a prefill hypothesis, not a
single-token decode fix. None should be removed solely to improve a benchmark.

## Upstream work and hardware limits

The pinned vLLM head already contains sparse metadata cleanup
([#57885](https://github.com/vllm-project/vllm/pull/57885)), DSpark profile
batch limiting ([#56448](https://github.com/vllm-project/vllm/pull/56448)),
and the speculative vocabulary GPU-sync removal
([#57396](https://github.com/vllm-project/vllm/pull/57396)). The pinned B12X
head already contains Spark decode routing and indexer scheduling
([1dc77276](https://github.com/local-inference-lab/b12x/commit/1dc77276e9d0ba297ad753eb3af04327759c1a3b)).
They need target-hardware qualification, but no missing cherry-pick is required
to get these changes.

B12X also replaced its managed weight pool with ordinary CUDA allocations in
[1ec67ef](https://github.com/local-inference-lab/b12x/commit/1ec67ef61104b33c309166dc39254fb5279882a6).
That commit reports a V4.1 target/draft smoke on **four** Sparks. It does not
establish the memory fit of our larger per-rank TP3 shard. Our first
current-head target-only load reached the dgx3 5 GiB startup floor just after
weight loading, with driver allocation failures. The loader-lifecycle fix alone
failed at the same memory floor. The separate scoped managed-weight patch then
passed bounded checkpoint copy, guarded TP3 startup, API readiness, and the
serving matrix. It recovered memory fit; it did not recover weekend decode TPS.

Two prominent upstream vLLM speedups cannot run on this topology:

| Path in pinned source | Why this Spark TP3 setup cannot select it |
| --- | --- |
| mHC coefficient overlap [#57603](https://github.com/vllm-project/vllm/pull/57603) and fused mHC all-reduce [#57643](https://github.com/vllm-project/vllm/pull/57643) | Overlap requires SM100/DeepGEMM; the fused collective additionally requires TP4 and NVLink multicast. GB10 is SM121, and our ranks communicate over RoCE. |
| Fused GEMM/reduce-scatter [#57428](https://github.com/vllm-project/vllm/pull/57428) | Its code requires SM100, one NVLink domain, sequence parallelism, and TP dividing 128. TP3 fails the divisibility check even on otherwise compatible hardware. |

The V4.1 indexer reports device-decided query-length support only for its
SM100 or SM90 paths. On SM121, upstream's adaptive-verification guard therefore
rejects that DSpark mode. Fixed-block DSpark remains a separate candidate;
the [official V4.1 recipe](https://github.com/vllm-project/recipes/blob/main/models/deepseek-ai/DeepSeek-V4.1-Flash.yaml)
documents fixed-block verification where adaptive mode is unsupported. Our
B12X adapter advertises CUDA-graph support and prepares speculative decode
row capacity, but its actual SM121 graph behavior has not been qualified.

Upstream FlashInfer sparse MLA accepts 8/16/32/64/128 Q heads and rounds
our 24 local TP3 heads up to 32. Our B12X adapter keeps a native 24-head
partition. This is a potential B12X advantage, not a measured one. The
128-token page and 64-state indexer geometry are required by the selected
SM12x kernels; the old 256-token manager block failed under this source.

At the 2026-09-23 08:06 UTC vLLM head, only three commits followed our pinned
vLLM base: per-token NVFP4 MoE selection for a one-sided EP path, then two ROCm
CI changes. B12X `master` still matched our pin. None is an identified fix for
this Spark TP3 decode loss. Chasing those commits would change the candidate
without addressing the measured configuration difference.

## Controlled recovery sequence

Build one immutable image from the consolidated series, distribute that digest
to all three ranks, and change one runtime variable per guarded launch:

1. **Target-only, eager:** reproduce semantics, long-context admission, memory,
   and the serving matrix on the consolidated image. API, memory, long-context
   admission, and throughput passed; deterministic code output remains a
   quality failure needing a known-good same-prompt control. This is the new
   source baseline, not the speed target.
2. **Target-only, `FULL_DECODE_ONLY`:** isolate graph acceleration while keeping
   mixed prefill eager. A [GB10/TP2 V4 field report](https://github.com/vllm-project/vllm/issues/40969#issuecomment-5190414871)
   found this mode stable and fast where `FULL_AND_PIECEWISE` hung; it is a test
   lead, not proof for V4.1/B12X/TP3.
3. **Fixed-block DSpark, eager:** start with the weekend's three draft tokens,
   adaptive verification off. Isolate draft benefit and record acceptance per
   position, output integrity, draft residency, and host memory.
4. **Fixed-block DSpark plus `FULL_DECODE_ONLY`:** require short and 128K
   request correctness, graph stability, and the full throughput matrix.
   Only then compare five draft tokens, the checkpoint's trained block size,
   as a separate experiment.
5. Evaluate `FULL_AND_PIECEWISE` only if it has a distinct prefill benefit.
   [#56771](https://github.com/vllm-project/vllm/issues/56771) reports a
   V4.1 DSpark/graph long-prefill illegal access on SM120, so short prompts
   alone cannot qualify this mode.

Use the same prompts, model, client location, warmed state, and low foreign
load for each arm; preserve all request-level receipts. Record graph mode
actually selected, DSpark drafts/accepts, TTFT, per-stream and aggregate TPS,
four 128K admissions, normal Strix traffic, minimum MemAvailable, and errors.
Repeat near-tie comparisons across complete launches (A-B-A); do not infer a
few-percent win from a single noisy boot. The weekend speed and output-quality
gates remain the acceptance targets. If the combined acceleration still misses
them, profile the B12X adapter, disk Engram, and RoCEnante separately on this
same consolidated source before changing kernels.
