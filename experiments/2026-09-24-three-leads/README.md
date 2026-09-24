# Three decode leads on the promoted r2 stack

Base: `main` with the promoted r2 configuration (`vllm-ds41f-kkref:01f1b874c774-r2`).
Each arm changes one thing, starts fresh under the launcher's memory guards,
and is measured with `bin/spark3 bench --suites quality,decode,sampled`.
`cluster-base.json` is the promoted configuration under its own container name,
booted and measured the same way, so boot-to-boot effects cancel.

Primary metrics:

- **Single-stream step time** (ms per decode step at concurrency 1). This is
  steady to about ±0.3% and independent of acceptance, so it measures kernel
  and host changes.
- **Aggregate tok/s** at concurrency 1-8. It includes acceptance, so it
  measures speculative-decoding changes.
- **Accepted drafts per step** on temperature-1.0 traffic.

## Leads

1. **Engram disk lookup off the critical path** (`cluster-engram.json`, in
   progress). At one stream the host reads both Engram layers' rows from disk
   before launching the forward graph. In `../2026-09-24-improvement-leads/runs/profile-engramtp-c1/`
   the GPU is idle for 0.72 ms (layer 1 read), 0.27 ms (layer 14 read), and
   0.86 ms (the `cudaGraphLaunch` call, which normally overlaps the previous
   step but here follows the Engram waits). That is about 1.85 ms of a
   40 ms step.
2. **Draft-only FP8 LM head** (`cluster-fp8head.json`,
   `overlay-vllm/dspark_utils.py`, `SPARK3_DRAFT_HEAD_FP8=1`). The DSpark draft
   computes full-vocabulary logits every step through the target's bf16 LM head
   (about 440 MB per rank read). The overlay quantizes a copy once at load
   (32x32 blocks, UE8M0 scales, the checkpoint's FP8 layout) and runs it on the
   B12X block-32 GEMM. The target keeps the bf16 head, so verification and
   every committed token are unchanged; only draft proposals can differ.
   Expected: about 1 ms less per step, +220 MB per rank.
3. **Depth 5 under adaptive verification** (`cluster-depth5.json`,
   `cluster-depth5-cost4.json`). Adaptive verification already trims each
   step's draft rows out of the batch, by expected acceptance per unit of
   measured cost, using the previous step's confidences. The earlier depth-5
   test raised acceptance without a throughput gain, but on single passes with
   ±9% spread. This retest uses the precise bench, at the promoted cost scale
   2.0 and at 4.0 (cost 1.0 was replaced by 4.0 after the 2.0 arm verified
   3.3 drafts per step on prose for 1.34 accepted).

## Running

On dgx1 from a clean deployment checkout at the published commit:

```sh
experiments/2026-09-24-three-leads/run_arm.sh base
experiments/2026-09-24-three-leads/run_arm.sh depth5
experiments/2026-09-24-three-leads/restore.sh
```

Reports land in the ignored `results/private/bench/tl-<arm>/bench.json`;
summaries are recorded below.

## Results

### Engram overlay, first attempt (hung)

The first version launched the B12X decode kernel from the reader thread and
waited on CUDA events there. The delay arm captured its graphs (the in-graph
flag wait captures fine) and then hung in the first adaptive-verification
profiling step, with every rank's main thread asleep. It was stopped by hand
after 30 minutes; nothing was measured. A healthy arm also runs one extra
thread at 100% CPU, so the busy thread seen during the hang was not the
fault. The likely cause is a kernel launch through CuTe's Python runtime from
a second thread. The overlay now keeps every kernel launch and stream
operation on the main thread: the decode is queued at stage time behind a
host-released gate (`cuStreamWaitValue32` on a mapped host word), and the
reader thread only polls for the row IDs, reads the rows, and releases the
gate, also on failure. The reader's wait and the main thread's join have
deadlines, so a repeat fails the start instead of hanging.

Other arms: pending summary.
