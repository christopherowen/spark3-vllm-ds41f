# Canonical-main minimal series, with weekend control

## Why this experiment exists

The consolidated current-head candidate (canonical vLLM `0f2a15c9` plus 10
patches, B12X `0332cc50` plus 5 patches) serves on all three ranks but fails
the fixed LRU code request. It failed 5/5 with RoCEnante all-reduce
(`../2026-09-23-consolidated-quality/runs/roce-current.json`) and 5/5 with
NCCL (`runs/consolidated-nccl-lru.json`). RoCEnante is therefore not the cause.
The same 114 tokens submitted as one prefill select the correct next token,
while incremental decode does not. The fault is in decode-time state on the
canonical-main path.

Direction: vLLM tracks canonical `vllm-project/vllm` main and B12X tracks
`local-inference-lab/b12x` master. Local Inference Lab's vLLM release branches
are behavioral references only, not a base.

## What the weekend setup actually was

The 2026-09-20 image `vllm-dsv41:r37-b12x-e4m3fix1-arm64-sm121` is Local
Inference Lab `release/ds41-optimized-r37` on both repositories (vLLM
`c687594b`, B12X `f1c4e9dd`). It is overlaid on the canonical ARM64 nightly
`af1c0149`. Diffing the image's `/opt/ds41-r37/vllm` against `c687594b`
shows the complete local vLLM delta (`runs/weekend-vs-r37-models.diff`):

- new `vllm/models/deepseek_v4_1/virtual_heads.py`, the TP3 64/8 to 72/9
  padding, hooked from `attention.py`;
- `allocate_weights` / `copy_weight` wrappers so custom DS4.1 parameters use
  B12X's managed weight pool;
- the mounted `state/indexer.py` (`DSV41_INDEXER_PREFILL_FACTOR`), plus the
  TP3 `state/config.json`.

The B12X delta is switchless per-peer RoCEnante routing, the 3072-token mHC
policy, and two fixes already upstream (`3ab21b3d`, `02407f65`).

Canonical vLLM has its own `vllm/models/deepseek_v41` model with no B12X
attention for it. It pins `b12x==1.3.0` (2026-08-25), which predates DS4.1.
Canonical main plus B12X master therefore needs an adapter series. This
experiment's job is to make that series coherent and minimal.

## Sequence (one variable per launch, all through `bin/spark3` guards)

1. `cluster-weekend-control.json`: the promoted weekend config, unchanged
   except launch-enabled, a distinct container name, and a new rendezvous
   port. It runs the LRU probe and the serving matrix with the identical
   client on dgx1 against localhost. This is the missing known-good control
   for the LRU gate.
2. `cluster-consolidated-fi-attn.json`: the exact consolidated image and NCCL
   configuration, with only `--attention-backend B12X` changed to canonical's
   `FLASHINFER_MLA_SPARSE_DSV41` (SM12x FlashInfer sparse MLA). `MAX_JOBS=1`
   bounds any post-load FlashInfer JIT compile under the startup guard. If
   this is coherent, the fault is confined to the local B12X attention
   adapter (vLLM patches 0009/0010).
3. Rebuild a layered series on the then-current canonical main and B12X
   master. First the three-Spark requirements (TP3 vocabulary, TP3 virtual
   heads, disk Engram), then each B12X performance layer as a separately
   selectable runtime option.

## Gates

- Quality: `../2026-09-23-consolidated-quality/probe_quality.py`, five
  repeats; every complete output is retained. A single pass does not qualify.
- Memory: 5 GiB startup and 3 GiB steady memguards, as configured. A guard
  kill is a failed run, never a reason to lower the guard.
- Performance: `../2026-09-20-upstream-main-rebase/benchmark_serving.py`,
  prose and code at concurrency 1, 2, 4 and 8, 256 output tokens, on dgx1
  against localhost.

## Startup finding: FlashInfer JIT after weight residency

The first two weekend-control launches (`cluster-weekend-control.json`, then
`-kv2.json` with 2 GiB KV) were both stopped by the 5 GiB startup memguard.
The kill came 1-5 s after "Graph capturing finished", when FlashInfer began
"Building JIT module sampling" with about 97 GiB of weights resident. The
2026-09-20 containers had built `sampling.so` once into their own writable
layers (`docker diff` on the retained `dsv41-tuned-8192-s3-kv30-g4`), so the
weekend service never paid this cost again. Every fresh container repeats it.
The weekend image's `/root/.cache/flashinfer` holds only its baked modules.

`prebuild_flashinfer.sh` builds the module on an idle host in a 24 GiB-capped
container from the same image, seeding `cache/flashinfer-r37` with the image's
baked FlashInfer modules. `cluster-weekend-control-kv2-fi.json` sets
`FLASHINFER_WORKSPACE_BASE=/cache/flashinfer-r37`. The guard thresholds are
unchanged. Any image that uses `VLLM_USE_FLASHINFER_SAMPLER=1` needs the same
pre-weight build.

## Control result: the weekend setup fails the same LRU gate

`runs/weekend-control-kv2/` holds the unchanged weekend image with 2 GiB KV
and a prebuilt FlashInfer sampler. It reproduces the 2026-09-20 serving matrix
within single-run noise: prose c1 34.6 tok/s (Sep 20: 31.4), code c1 44.0
(44.3), code c4 96.0 (94.4). It fails the LRU gate 4/5 with the same malformed
tokens as the consolidated candidate (`valueerk`, ` value splash`). The
defect is therefore shared by both stacks and predates the canonical-main
rebase.

## FlashInfer-attention bisect: not runnable at 128-token blocks

`cluster-consolidated-fi-attn.json` failed cleanly in warmup:
FlashInfer 0.6.18 SM120 sparse-MLA decode supports only a compressed page size
of 64, and 128-token blocks give 32 for ratio-4 layers. It would need 256-token
blocks, which the B12X path cannot use. No memory event occurred (minimum
about 9.8 GiB available).

## Shared-component hypothesis: B12X split-KV V4.1 decode

Both stacks call B12X `run_unified_decode` for decode. For the V4.1 cache,
decode is never single-pass: it runs split-KV with FP8-internal compute. With
24 heads per rank (TP3), the default selects 8-head blocks. Prefill
("extend") is single-pass. That matches "correct as one prefill, wrong during
incremental decode", and 24 heads per rank arises only at TP3.
`b12x_tuning_bf16_decode.py` changes only the split-KV decode selection to
BF16 compute with 16-head blocks (one full block plus an 8-head remainder).
Single-pass extend keeps the upstream FP8 default.
`cluster-consolidated-bf16-decode.json` is the failing consolidated NCCL
configuration plus that one read-only mount.

Result: `runs/consolidated-bf16dec/lru.json` still fails 5/5 (`valueerk`,
syntax errors) with split-KV decode forced to BF16/H16. The decode plan is
built with `mode="decode"` and no persisted selection cache exists, so the
override was in effect. B12X's FP8-internal / 8-head-block decode is not the
cause.

## FlashInfer attention at 256-token blocks

`cluster-consolidated-fi-attn-b256.json` repeats the FlashInfer-attention
bisect with `--block-size 256`, which gives FlashInfer's required 64-entry
compressed page. It isolates the B12X attention/cache path from the other
shared components: TP3 virtual heads, disk Engram, and B12X MoE/linear.
