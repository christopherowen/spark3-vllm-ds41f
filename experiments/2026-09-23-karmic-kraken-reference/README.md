# Karmic Kraken reference build

## Purpose

This is a reference, not the target base. The target is canonical
`vllm-project/vllm` main plus a minimal patch series. Where our local patches
differ from Local Inference Lab's `integration/karmic-kraken-beta`, LIL's
version is preferred. Before extracting that series, this experiment measures
LIL's current DS4.1 line as a whole on our switchless three-Spark fabric:

- Does it pass the fixed LRU quality gate that both the 2026-09-20 weekend
  image and the canonical-main consolidated candidate fail?
- Does it match or beat the weekend serving matrix?

A pass gives the minimal canonical-main series a known-good behavioral target.
A fail means extraction alone cannot meet the goal.

## Sources (`sources.json`)

- vLLM: LIL `integration/karmic-kraken-beta` `01f1b874c` on canonical
  `0f8fa53ac` (2026-09-16), unmodified.
- B12X: LIL `integration/karmic-kraken-beta` `0f846212` plus one local patch,
  `patches/b12x/0001-switchless-rocenante.patch`. It adds per-peer HCA routing
  for direct-cabled rings; LIL's own launcher targets a switched fabric. The
  beta's `b12x/comm/roce` is identical to master, so the patch applies unchanged.
- Runtime base: canonical ARM64 nightly `af1c0149`, the weekend image's base.
  It is seven commits before LIL's base, with no native-source changes in
  between. Its FlashInfer 0.6.18.post1, humming 0.1.12 and Torch 2.13 match
  the beta's pins. Only `_C_stable_libtorch` and `_moe_C_stable_libtorch` are
  rebuilt, for SM121.
- CuTe DSL: B12X beta declares 4.6.2 exactly, as the weekend image used, so the
  hash-locked 4.6.2 wheels replace the nightly's 4.7.1. This drops the previous
  candidate's local DSL-alignment patch.

LIL's DS4.1 package pads TP3 heads itself (`update_model_config_for_parallelism`,
`allow_tp_padding`). The mounted 72/9 `state/config.json` overlay and the R37
`indexer.py` override are therefore not used. The 24-head fused-Q dispatch
patch is also unnecessary: LIL's DS4.1 path does not call that operator.

## Configuration (`cluster-kkref.json`)

Derived from the promoted weekend config, with LIL's DS4.1 TP3 recipe
(`scripts/serve-ds41-flash-dspark-tp3-rdma.sh`) where the two differ:

- DSpark 7 tokens, draft TP 3, adaptive verification with cost scale 2;
- `--async-scheduling`, `--no-scheduler-reserve-full-isl`,
  `--enable-flashinfer-autotune`, `custom_ops: ["all"]`;
- disk Engram with `disk_resident_scales: false`;
- `VLLM_PLUGINS=b12x_loader`, `B12X_DENSE_SPLITK_TURBO=1`, and the TP3 compile
  worker counts.

LIL's launcher also exports `VLLM_USE_B12X_*`, `B12X_MOE_FORCE_A8`,
`B12X_W4A16_TC_DECODE` and `B12X_MLA_SM120_UNIFIED`. Neither the beta vLLM,
the beta B12X, nor the weekend source reads them, so they are omitted.
`--kv-cache-dtype fp8` is omitted as well: both stacks hard-code the B12X
DS4.1 attention cache format.

The first launch stays conservative on memory: 2 GiB KV (as in the other
arms), CUDA graphs captured up to 32 tokens, the weekend's 8 sequences and
8192 batched tokens, and the unchanged 5/3 GiB guards. JIT caches are isolated
under `/cache/kkref`.

## Gates

Same as `../2026-09-23-canonical-minimal/`: five-repeat LRU probe, the
prose/code c1-c8 serving matrix from dgx1 against localhost, and the
startup/steady memory guards.

## Results so far (r4: autotune off, weekend memory envelope, DSpark 3)

Serving matrix (`runs/kkref-r4/`, same client as the weekend control):
prose c1 40.9 tok/s (weekend today 34.6), code c1 53.5 (44.0), prose c8 133.0
(83.1), code c8 161.3 (71.9); TTFT 0.26 s vs 0.32 s. LIL's line meets the
speed target with B12X autotuning disabled.

Quality: the LRU gate fails 0/5 twice (`valueerk`, `valueazon`, syntax errors).
The defect is shared by weekend R37, LIL beta, and the canonical-main
candidate. Token-by-token generation where every step is a fresh full prefill
(`stepwise-salt.json`) produces correct code. Chunked generation (fresh
prefill, then K ordinary decode steps) passes at K=4 and K=16 and fails at
K=48 and K=256 (`stepwise-chunk*.json`). The error accumulates across
consecutive decode steps and a prefill clears it: decode-written state
differs from what prefill writes for the same tokens.

`cluster-kkref-trace.json` removes DSpark and mounts `trace_model_state.py`,
a diagnostic copy of LIL's `DeepseekV41ModelState` that logs the Engram
lookback window and input ids per step on TP rank 0. It tests whether decode
feeds Engram the wrong n-grams.

## Trace results (no DSpark, per-step device synchronize)

- Engram lookback and input ids matched the true token sequence at all 181
  steps of a traced request (`runs/kktrace/kktrace-lookback.jsonl`), so decode
  feeds Engram the correct n-grams.
- That first request after startup passed. Five repeats then failed, even
  with a full device synchronize before every step. An unsynchronized
  per-step race is not the cause.
- Five requests with a unique `cache_salt` each (`probe_quality_salted.py`)
  also failed, so cross-request prefix reuse is not the cause.
- The compressor state cache is FP32 (`CircularBufferSpec`), so a
  reduced-precision state cache is not the explanation either.

Ruled out so far: RoCEnante all-reduce, B12X FP8/8-head split decode (checked
on the canonical-main stack), Engram lookback, prefix-cache reuse, DSpark, and
a per-step race. Established: decode-written state drifts from what prefill
writes for the same tokens. At most 16 consecutive decode steps stay correct;
48 or more do not. The defect is shared by weekend R37, LIL beta and the
canonical-main candidate. Every stack also shares TP3 head padding, B12X
cache writers and kernels, and disk Engram.

## Result: the drift requires CUDA-graph replay

On LIL's image in eager mode (`cudagraph_mode: NONE`, no DSpark), the LRU gate
passed 30/30 across all synchronization arms, including no synchronization
(`runs/kktrace/sync-ab-eager.json`). With FULL_AND_PIECEWISE graphs it passed
about 1 time in 24, with or without DSpark and with or without a per-step
device synchronize. Kernel numerics, Engram, collectives and host races are
not the cause. The next arms separate PIECEWISE graphs (attention eager
between pieces) from FULL_AND_PIECEWISE with breakable CUDA graphs enabled,
which runs the `eager_break_during_capture` regions eagerly.

The "PIECEWISE" arm (`cluster-kkref-g-piecewise.json`) failed 7/8
(`runs/kktrace/g-piecewise-nonbreakable.json`). DS4.1 is not torch-compiled,
and vLLM provides piecewise graphs for it only with
`VLLM_USE_BREAKABLE_CUDAGRAPH=1`. With that off, this arm still replayed full
decode graphs. `cluster-kkref-g-breakable.json` is PIECEWISE with breakable
graphs on: the `eager_break_during_capture` regions (SWA write, compressor and
compressed-cache write, indexer and attention) run eagerly on every replay.
