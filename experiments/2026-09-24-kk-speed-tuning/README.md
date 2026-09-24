# Karmic-kraken speed tuning

Base: the 2026-09-24 promoted baseline (`base.json`). Each arm changes one
variable of `config/cluster.json` and is measured on the three Sparks with the
same client as the baseline: `measure.sh` runs `run_arm.sh` twice (prose and
code, concurrency 1-8, 256 output tokens, then the five-repeat LRU gate) and
records the minimum MemAvailable per node (`sample_mem.sh`). Two passes per arm
because single passes vary by about ±7%. The unchanged 5 GiB startup and 3 GiB
steady memory guards stay in force; a guard kill fails the arm.

## Arms

1. `capture32`: CUDA-graph capture sizes extended with 28 and 32. With DSpark
   verifying three draft tokens, seven or eight streams issue 28- or 32-token
   decode batches, which the baseline's largest graph (24) cannot replay; those
   steps run eagerly.
2. B12X autotune with a bounded candidate-race budget. vLLM creates B12X's
   `PreparationSession` without a budget, so racing may hold half of free
   memory; on unified memory that has tripped the startup guard before. One
   budgeted tuning run fills the persistent selection cache
   (`/cache/kkref/jit/b12x/cute/preparation`), then the arm is measured.
3. `batched8192`: 8,192 batched tokens (the 2026-09-20 value) for long-prompt
   prefill; needs a prefill workload in addition to the decode matrix.
4. DSpark depth 5 or 7 (Local Inference Lab's recipe uses 7), with capture
   sizes covering `(1 + depth) x 8`.

Acceptance: faster than the baseline beyond pass-to-pass variation, LRU 5/5 in
both passes, no guard kill, and at least 3 GiB above the steady guard on every
node.

## Results

Two passes per arm; mean aggregate tok/s with the pass range (c1 includes the
repeat runs, four samples). `summarize.py runs kkt-baseline ...` regenerates it.

| Point | Baseline | capture32 | depth5 (on capture32) |
|---|---|---|---|
| prose c1 | 42.4 [38.9-45.6] | 39.8 [37.6-42.0] | 40.1 [37.6-43.2] |
| prose c2 | 61.6 | 65.9 | 66.2 |
| prose c4 | 99.1 | 101.4 | 99.0 |
| prose c8 | 137.8 | 138.4 | 142.9 |
| code c1 | 49.1 [45.8-55.1] | 52.7 [50.7-53.9] | 52.5 [49.8-56.7] |
| code c2 | 79.3 | 81.0 | 81.6 |
| code c4 | 118.0 | 117.0 | 122.3 |
| code c8 | 155.7 [154.5-156.9] | 166.9 [160.7-173.1] | 164.4 |
| LRU | 5/5, 5/5 | 5/5, 5/5 | 5/5, 5/5 |
| min MemAvailable (GiB, dgx1/2/3) | 6.98/7.68/7.72 | 6.07/7.71/6.89 | 6.77/6.82/7.50 |

- `capture32`: code c8 +7% with non-overlapping ranges; every other point is
  within pass-to-pass variation. Keep.
- `depth5`: accepted draft tokens per step rose from 1.53 to 1.74 over the
  matrix (4.9-5.0 at one stream), but throughput did not. On GB10 the routed
  MoE streams each verified token's experts from memory
  (`../2026-09-23-karmic-kraken-reference/runs/moe-decode-bench/`), so
  verification cost grows almost linearly with depth. Reject; keep depth 3.
- `batched8192`: three launches, each stopped by the 5 GiB startup guard on
  dgx1 (the API-server rank, always the tightest) at 4.98-5.00 GiB, 30-40 s
  after KV allocation, during the warmup forward at the full 8,192-token batch
  (`runs/failed-starts/*batched8192*`, `*b8192-nofia*`, `*b8192-kv15*` on dgx1).
  Removing `--enable-flashinfer-autotune` changes nothing: the pass is on by
  default, and it saves 0 configs in this stack either way. Freeing 0.5 GiB of
  KV (1.5 GiB) did not make room, and KV capacity is not linear in bytes:
  1.5 GiB holds 439,068 tokens against 933,168 at 2 GiB, because part of the
  reservation is fixed per-sequence cache. Reject; keep 4,096.
- Autotune r1 (`runs/autotune-r1-guard-kill/`): stopped by the startup guard on
  dgx1 at 5.0 GiB after 28 of 861 tuning obligations (3,005 compilations,
  2:43). The race budget bounds only candidate scratch; the four spawned,
  device-pinned compile workers and resident compiled programs grow with
  tuning. B12X saves selections only when a stage installs, so a killed run
  keeps nothing. r2 (`cluster-autotune2.json`) uses one compile worker per
  stage, a 256-program compile cache, a 1 GiB race budget, and `memlog.sh`.
- Autotune r2 (one compile worker, 256-program cache, 1 GiB race budget;
  `runs/kkt-autotune2-memlog.final.txt`): MemAvailable fell about 0.95 GiB per
  minute, tracking compilations (about 1.6 MB per compiled candidate), with 20
  of 861 obligations done after four minutes; stopped by hand before the guard.
  Candidate programs stay resident, so a full in-engine tune cannot fit on GB10
  at this model size. `norm.mhc` produced 2,139 of r1's ~3,000 compilations.
- Draft TP 1: not run. The DSpark drafter is 7.39 GiB; at TP 1 all of it lands
  on dgx1, which has 6-7 GiB of steady headroom.

## Decode profile (`runs/profile-c1/`)

Torch profiler on rank 0, capture32 configuration, one stream, 48 output
tokens after warmup. The GPU is busy 90% of the 1.31 s window. Kernel time:
routed MoE 33%, B12X dense FP8 GEMMs 27%, RoCE collectives 20%, LM-head BF16
GEMM 6%, mHC/norms 3%, the rest under 3% each. 83 of the 1,840 all-reduce
calls are eager ones in the prefill step, each spinning about 2.2 ms for peer
ranks' host launches (host skew, most of the 0.25 s TTFT); the graph-captured
decode all-reduces average about 40 µs. The routed MoE runs near memory
bandwidth (`moe-decode-bench`). The dense GEMMs take about 15 ms per step
against roughly 8-9 ms of FP8 weight traffic at the measured 215 GB/s, so
they are the tunable share. Autotune r3 restricts tuning to
`gemm.block_fp8_linear`, `gemm.bf16_gemv`, and `moe.decode` through the
overlay's `SPARK3_B12X_AUTOTUNE_ONLY`.
- Autotune r3 (`cluster-autotune3.json`, tuning limited to
  `gemm.block_fp8_linear`, `gemm.bf16_gemv`, `moe.decode`): reached readiness
  (dgx1 startup minimum about 6.2 GiB, `runs/kkt-autotune3-startup-memlog.txt`)
  and saved 283 KB of selections to `/cache/kkref/jit/b12x/compile/preparation/`.
  Matrix against capture32: prose c1 44.5 vs 39.8, code c4 112.8 vs 117.0,
  code c8 160.0 vs 166.9, the rest within ±5%; LRU 5/5 twice. Neutral: tuned
  selections buy nothing measurable over B12X's heuristics here. Keep
  `B12X_AUTOTUNE=0`; with it on, the untuned components (mHC, attention) would
  race again at startup.

FlashInfer: only the top-k/top-p sampler uses it ("Using FlashInfer for top-p
& top-k sampling"); at temperature 0 the profile contains no FlashInfer kernel.
The `--enable-flashinfer-autotune` warmup saves 0 configs.
