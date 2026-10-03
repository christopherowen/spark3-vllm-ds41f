# TileLang kernel backend

**Status:** sources complete and verified for TP3 and TP4; no image built.
TileLang, TileKernels and the DS4.1 TileLang vLLM patch (0027) are pinned,
prepared and recorded, and doctor passes. Launch is disabled until the images
are built and qualified. No cluster operation was performed.

Deployment base: `e15547a`. The promoted configuration stays TP3 with B12X
kernels.

## Purpose

Run the DS4.1 model compute kernels (attention, including the DSpark drafter's,
linear layers and MoE) on TileLang instead of B12X, with everything else
matched to the B12X configuration of the same topology. The policy and its
required settings are in [kernel-backends.md](../../docs/kernel-backends.md).

## Profiles

One profile per topology, each the B12X configuration of that topology with
the kernel policy changed:

| Profile | Mirrors | Image | vLLM tree |
| --- | --- | --- | --- |
| [tp3/cluster.json](tp3/cluster.json) | `config/cluster-64k.json` (promoted TP3) | `vllm-ds41f-kkref:04c30fa98e79-r5o-tilelang-v2` | `e9e04990` |
| [tp4/cluster.json](tp4/cluster.json) | [the TP4 candidate](../2026-10-03-collective-contract/candidate.json) | `vllm-ds41f-kkref:04c30fa98e79-r5o-roce-contract-tilelang-v1` | `15160070` |

The delta in both, with the settings doctor requires:

- `kernel_backend: tilelang`;
- `--attention-backend TILELANG`, `--linear-backend tilelang`,
  `--moe-backend tilelang`, and `"attention_backend":"TILELANG"` in
  `--speculative-config`;
- `VLLM_DS41_KERNEL_BACKEND=tilelang`; `TILELANG_CACHE_DIR` stays
  `/cache/kkref/jit/tilelang`;
- the image, its source lock and the expected source-tree labels.

TP4 also keeps its per-boot artifacts apart from the B12X arm's: pinned DSpark
cost curves under `/cache/kkref/dspark-costs/ring4-tilelang-20261003` (the
curves are keyed by shapes only, so sharing the B12X directory would reuse
B12X timings) and profiler traces under
`/cache/kkref/profiles/ring4-tilelang-20261003`.

The node maps, transport, collective limits and NCCL settings (RoCEnante plus
NCCL), the B12X checkpoint loader, model, context, KV budget, memory guards,
deployment path and every other setting equal the mirrored configuration.
`tests/test_kernel_backend.py` checks this field by field.

[profiles.json](profiles.json) is the [transport tuning catalog](../2026-10-03-transport-profiles/README.md)
with these profiles as its bases; its `rocenante` and `nccl` groups are the
B12X catalog's own. It materializes a site configuration the same way:

```sh
bin/spark3 tuning --profiles-config experiments/2026-10-03-tilelang-kernels/profiles.json show tp4
bin/spark3 tuning --profiles-config experiments/2026-10-03-tilelang-kernels/profiles.json create tp4 \
  --nodes-config config/nodes-ring4.local.json --output experiments/<new>/tp4.json
```

The derived layout matches the B12X profile's: patch 0027 changes kernels, not
padding or sharding (heads, output groups, vocabulary, Engram and drafter
widths are unchanged), so the [layout audit](../2026-10-03-transport-profiles/model-layout.json)
lists both TileLang trees. Its `kernel_scratch` reports no compact-MoE scratch:
TileLang's expert GEMMs pad nothing at either width.

## Tensor-parallel widths

| Per rank | TP3 | TP4 |
| --- | --- | --- |
| Attention heads | 24 (72 padded) | 16 |
| Routed and shared expert rows | 768 | 576 |
| Expert GEMM K blocks (gate/up, down) | 128, 128 | 128, 192 |
| Shared expert down K blocks (≤64 rows, more) | 128, 128 | 192, 64 |

The block-scaled GEMMs take any K block that is a multiple of 64. A block that
starts off a 128-element boundary reads the two packed UE8M0 words it straddles
and passes the MMA its offset, so no operand or scale is padded. TileKernels'
packed scales pad each row to whole words (576 columns: 18 exponents in 20
bytes); the workspace and weight packing use that layout. Each output
accumulates over K in the same order whatever the block, so rows stay identical
across batch sizes and tile configurations.

All eight GEMM variants of both widths compile for `sm_121a`. The TP4 down
projection's 192-wide blocks use 96.6 KiB of shared memory at three stages,
within the 99 KiB limit.

## Sources

Each profile's lock is the mirrored B12X lock plus:

| Source | Revision | Local changes | Tree |
| --- | --- | --- | --- |
| `tilelang` (`tile-ai/tilelang`, main) | `b95ee4ff` (0.1.15) | `patches/tilelang/series` 0001-0003; patch head `30d854b6` | `1697ad52` |
| `tile_kernels` (`deepseek-ai/TileKernels`, main) | `66258df6` (2.0.0) | none | `64770881` |

TileLang's submodules (TVM and CUTLASS, with TVM's own, recursively) are fetched
at the commits its tree records; each `source.json` lists them and the
patch-set fingerprint `34504514…`. TileLang's contribution fork is
`christopherowen/tilelang`, branch `deepseek-v41-sm120`.

Both vLLM series end with the same
[0027-deepseek-v41-tilelang-kernels.patch](vllm/0027-deepseek-v41-tilelang-kernels.patch):

| Series | Before the TileLang patch | Patch head | Tree | Fingerprint |
| --- | --- | --- | --- | --- |
| [tp3/vllm/series](tp3/vllm/series) | the 26 promoted patches | `65a60156` | `e9e04990` | `cab61e96…` |
| [tp4/vllm/series](tp4/vllm/series) | the 26 promoted patches and the [explicit collective policy](../2026-10-03-collective-contract/README.md) | `bd188b66` | `15160070` | `02f49962…` |

B12X and NCCL are the mirrored profile's: the promoted trees for TP3, the
balanced-policy trees for TP4. Each vLLM record carries capability
`tilelang-kernels`, and the image label `local.spark3.vllm.tree` expects its
tree.

## vLLM patch 0027

`VLLM_DS41_KERNEL_BACKEND=tilelang` selects the TileLang family for the whole
model, drafter included, and the model config requires the attention, linear
and MoE backends to agree. The patch adds:

- `vllm/models/deepseek_v4_1/tilelang/`: sparse MLA, the MXFP4 indexer
  (query quantization, index-key writes, scores, exact top-k, candidates),
  MXFP8 and BF16 GEMMs and the routed experts, plus the attention layer,
  linear methods and RMSNorm built on them. Plans hold no buffers; scratch
  comes from vLLM's workspace, reserved before memory profiling. Cache pages
  are read at the allocator's row stride (vLLM's BLHNC layout interleaves
  layers).
- `AttentionBackendEnum.TILELANG`, the `tilelang` linear and MoE backends,
  and `TileLangExperts` in the MXFP4 oracle.
- `vllm/models/deepseek_v4_1/kernels.py`, the one place that picks a
  component's implementation, and `Block32FP8LinearMethod`, the checkpoint
  weight layout both families share.
- `requirements/cuda.txt`: `tilelang==0.1.15`.

Still on the shared B12X kernels under both families: mHC (with its fused
norms), rotary, the SWA and indexed cache writers, the index-weight scale, the
WO projection, the compressor, Engram, the vocabulary head, the DSpark
context-KV projection and NVFP4 heads, and vision.

## Validation

```sh
bin/spark3 --cluster-config experiments/2026-10-03-tilelang-kernels/tp3/cluster.json build prepare --only vllm,tilelang,tile_kernels
bin/spark3 --cluster-config experiments/2026-10-03-tilelang-kernels/tp4/cluster.json build prepare --only vllm
python3 -m unittest tests.test_kernel_backend tests.test_transport_profiles
```

Completed on 2026-10-03:

- Preparation reproduced every recorded patch head and tree for both series,
  including all eleven TileLang submodule commits. Doctor passes for both
  profiles and their B12X twins with the example node maps; both catalog
  profiles resolve.
- Before the TP4 widths were added, the GPU kernel tests ran on dgx4 in
  one-off containers from the r5o image, against torch references:
  - sparse MLA, for SWA-only and indexed layers at 16 and 64 heads;
  - MXFP4 quantization and index-key writes, matching B12X's rounding
    bit for bit;
  - dense, candidate and candidate-publishing indexer scores and top-k, and
    multi-level top-k merges;
  - MXFP8 and BF16 GEMMs, RMSNorm, and the routed experts.

  Cache pages were padded and strided like BLHNC. Every kernel of one
  contract is bit-identical, and outputs do not depend on batch composition.
- Reduced-model runs of vLLM on dgx4 (TP1): a 6-layer DS4.1 checkpoint built
  from checkpoint layers 0, 1, 2, 3, 20 and 24, which covers SWA-only,
  ratio-2, ratio-1, candidate-source and reindex layers.
  - Both backends start, profile, capture CUDA graphs and serve teacher-forced
    prompts of up to 2,072 tokens.
  - TileLang was bitwise identical across three runs. B12X differed from
    itself by a mean 0.45-0.63 nats per position.
  - TileLang differed from B12X by the same 0.47-0.63, and swapping any single
    component (attention, linear, MoE, norm) moved B12X by a similar amount,
    which is the truncated model's sensitivity, not a component error.

## After the images exist

The kernel tests, now with TP3's 24 heads and TP4's 576-wide GEMMs, run from
each image's own vLLM tree as kernel-lab bundles
([kernel-tests-tp3](bundles/kernel-tests-tp3/candidate.json),
[kernel-tests-tp4](bundles/kernel-tests-tp4/candidate.json); see
[lab.md](../../docs/lab.md)):

```sh
scripts/lab.py kernel-local experiments/2026-10-03-tilelang-kernels/bundles/kernel-tests-tp4
```

Then the GPU import smoke and, in a reserved window, each profile against its
B12X twin: startup and steady memory, the LRU quality gate, the determinism
checks, one- and eight-stream decode, 32K/64K/256K source-text prefill, prefix
replay and admission. Kernel output may differ from B12X, so qualification
compares quality before speed. The TP4 comparison needs the B12X TP4 image as
well; lab windows run only on the promoted topology, so TP4 runs use explicit
cluster commands.
