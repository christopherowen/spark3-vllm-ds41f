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

### Measurements (2026-09-24, cluster otherwise idle)

Every arm was a fresh boot measured with `bin/spark3 bench --suites
quality,decode,sampled` (the delay check ran quality plus single-stream decode
only). Reports are in `runs/<arm>.json`. `base` opened the session and `base2`
closed it. Decode is aggregate tok/s at temperature 0 with reasoning on, each
point sampled until its 95% CI is within 2%:

| Arm | prose c1 | c2 | c4 | c8 | code c1 | c2 | c4 | c8 |
|---|---|---|---|---|---|---|---|---|
| base | 42.4 | 68.1 | 99.8 | 146.3 | 52.2 | 80.2 | 118.2 | 165.6 |
| base2 | 41.0 | 65.8 | 100.9 | 141.7 | 51.4 | 80.8 | 115.2 | 161.9 |
| depth5 | 41.3 | 65.6 | 99.1 | 139.0 | 53.3 | 81.8 | 121.4 | 163.2 |
| depth5-cost4 | 41.4 | 61.0 | 93.7 | 143.5 | 52.5 | 80.7 | 118.3 | 161.6 |
| fp8head | 42.5 | 66.6 | 101.7 | 141.4 | 52.0 | 79.8 | 118.0 | 166.7 |
| engram | 42.5 | 68.4 | 102.7 | 155.0 | 54.0 | 83.2 | 122.0 | 172.3 |

Percent change; `*` marks a Welch 95% interval that excludes zero:

| Comparison | prose c1 | c2 | c4 | c8 | code c1 | c2 | c4 | c8 |
|---|---|---|---|---|---|---|---|---|
| base2 vs base | -3.2* | -3.4* | +1.2 | -3.2* | -1.4 | +0.6 | -2.6* | -2.2 |
| depth5 vs base | -2.4 | -3.7* | -0.7 | -4.9* | +2.1 | +1.9 | +2.7* | -1.5 |
| depth5 vs base2 | +0.8 | -0.3 | -1.8 | -1.9 | +3.6* | +1.3 | +5.4* | +0.8 |
| depth5-cost4 vs base | -2.2 | -10.4* | -6.1* | -1.9 | +0.6 | +0.5 | +0.1 | -2.4 |
| depth5-cost4 vs base2 | +1.0 | -7.3* | -7.2* | +1.3 | +2.0 | -0.1 | +2.8* | -0.2 |
| fp8head vs base | +0.4 | -2.2 | +1.9 | -3.3* | -0.3 | -0.6 | -0.1 | +0.7 |
| fp8head vs base2 | +3.7* | +1.2 | +0.7 | -0.2 | +1.2 | -1.2 | +2.5* | +2.9* |
| engram vs base | +0.3 | +0.3 | +3.0* | +6.0* | +3.5* | +3.7* | +3.2* | +4.0* |
| engram vs base2 | +3.6* | +3.9* | +1.8 | +9.4* | +5.0* | +3.1* | +6.0* | +6.4* |

Single-stream step time (ms) and verified drafts per step (not recorded for
`base`, which ran before the field existed):

| Arm | prose step | prose verified | code step | code verified |
|---|---|---|---|---|
| base | 49.94 | - | 52.72 | - |
| base2 | 52.37 | 3.00 | 53.22 | 3.00 |
| depth5 | 54.48 | 3.30 | 57.32 | 3.99 |
| depth5-cost4 | 52.53 | 2.85 | 56.66 | 3.57 |
| fp8head | 48.74 | 2.53 | 51.97 | 2.94 |
| engram | 49.92 | 2.88 | 51.26 | 3.00 |

Findings:

- **Boot-to-boot variation is about 3%.** The same configuration differs
  between boots by up to 3.4% at several points. Adaptive verification
  profiles its costs at startup, and each boot settles on a different number
  of verified drafts: `base2` verified all three on single-stream prose,
  `fp8head` 2.5. The within-run intervals do not cover this, so a single boot
  per arm resolves only effects larger than about 3%. Step time follows the
  verified rows, about 7-8 ms per verified row at one stream, so step times
  compare cleanly only between boots that verified the same rows.
- **Engram overlap: accept for productization.** It is faster than both
  baselines at 13 of 16 comparisons and slower at none. The cleanest
  mechanism check is code at one stream, where `engram` and `base2` both
  verified 3.00 drafts per step. There the step fell from 53.22 to 51.26 ms,
  1.96 ms, matching the 1.85 ms of idle GPU the profile attributed to the
  Engram reads and the graph launch. The gain grows with concurrency, as more
  rows are read per step: +6.0 to +9.4% at prose c8, +4.0 to +6.4% at code c8.
  Sampled traffic at c4 rose 5-7%. Quality passed 5/5, and the delay check
  proved the gate holds: a 20 ms read delay lengthened single-stream steps by
  27 ms, with acceptance and quality unchanged.
- **FP8 draft LM head: reject.** On code, with matching verified rows, the
  step saved about 0.8 ms, but accepted drafts per step fell about 2.5%. The
  net is within boot noise against both baselines.
- **Depth 5: reject at both cost scales.** At cost 2.0 it raised acceptance,
  from 1.88 to 2.21 on single-stream code, but lengthened steps 8-9% because
  the draft always proposes five positions. The net was code +2 to +5% and
  prose -5 to +1% across the two baselines, within boot noise. Cost 4.0
  trimmed more rows but kept the draft cost; prose fell 6-10% at c2 and c4.
  Sampled code with reasoning off did gain (+8% at c1 on cost 4.0), but not
  enough to change the default.
- **Thermals:** no thermal slowdown on any node in any arm. dgx3 ran warmest,
  GPU up to 70 C (24 C below its limit) and board zones up to 91.3 C (critical
  trip 104 C).
