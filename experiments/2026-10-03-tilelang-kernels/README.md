# TileLang kernel backend

**Status:** both images built on dgx4 and their kernel tests pass; not
qualified. TileLang, TileKernels and the DS4.1 TileLang vLLM patch (0027) are
pinned, prepared and recorded, and doctor passes. Launch is disabled until the
images are qualified against their B12X twins. No cluster operation was
performed.

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
| [tp3/cluster.json](tp3/cluster.json) | `config/cluster-64k.json` (promoted TP3) | `vllm-ds41f-kkref:04c30fa98e79-r5o-tilelang-v6` | `b7e7a6f5` |
| [tp4/cluster.json](tp4/cluster.json) | [the TP4 candidate](../2026-10-03-collective-contract/candidate.json) | `vllm-ds41f-kkref:04c30fa98e79-r5o-roce-contract-tilelang-v5` | `b13fdda5` |

The delta in both, with the settings doctor requires:

- `kernel_backend: tilelang`;
- `--attention-backend TILELANG`, `--linear-backend tilelang`,
  `--moe-backend tilelang`, and `"attention_backend":"TILELANG"` in
  `--speculative-config`;
- `VLLM_DS41_KERNEL_BACKEND=tilelang`; `TILELANG_CACHE_DIR` stays
  `/cache/kkref/jit/tilelang`;
- the image, its source lock and the expected source-tree labels.

TP4 also keeps its per-boot artifacts apart from the B12X arm's: pinned DSpark
cost curves under `/cache/kkref/dspark-costs/ring4-tilelang-v5-20261003` (the
curves are keyed by shapes only, so sharing the B12X directory would reuse
B12X timings; each TileLang image version pins its own for the same reason) and profiler traces under
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
| Expert GEMM K blocks (gate/up, down) | 128, 128 | 128, 4 x 128 + a 64 tail |
| Shared expert down K blocks (≤64 rows, more) | 128, 128 | 192, 64 |

The expert GEMMs load E2M1 weights by TMA, which unpacks FP4 for the MMA
(`CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B`) only in boxes of exactly 128 K
elements. A K that ends 64 past a multiple of 128 (TP4's down projection)
therefore keeps its last 64 columns as compact tile tails beside the 128-wide
tiles. The tail is copied with `cp.async`, eight packed bytes into each
sixteen-byte group of the same unpacked shared layout, and multiplied over its
64 columns only, so neither weights, activations nor scales are padded. That
matters because TileKernels leaves the two padding bytes of each 576-wide row's
packed activation scales unwritten. The dense MXFP8 GEMMs have no such load
constraint: they take any K block that is a multiple of 64 and read the two
packed UE8M0 words a block straddles. Each output accumulates over K in the
same order whatever the block, so rows stay identical across batch sizes and
tile configurations.

All GEMM variants of both widths compile for `sm_121a`.

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
| [tp3/vllm/series](tp3/vllm/series) | the 26 promoted patches | `3c63268e` | `b7e7a6f5` | `01be16cc…` |
| [tp4/vllm/series](tp4/vllm/series) | the 26 promoted patches and the [explicit collective policy](../2026-10-03-collective-contract/README.md) | `fc4a5b12` | `b13fdda5` | `095e39ea…` |

B12X and NCCL are the mirrored profile's: the promoted trees for TP3, the
balanced-policy trees for TP4. Each vLLM record carries capability
`tilelang-kernels`, and the image label `local.spark3.vllm.tree` expects its
tree.

## Decode projections and vocabulary heads

The first TP4 profile ([TP4 performance](../2026-10-03-tilelang-tp4-performance/README.md))
put the whole decode gap to B12X in the dense projections, and found the DSpark
draft and transition heads running BF16 cuBLAS under the TileLang family:

- Decode rows (up to 64) now read one whole 64-row MXFP8 activation tile from
  the padded workspace and store only their live rows. A partial tile cost
  about 0.4 us per missing row on SM121 (q_b: 43 us at one row, 16 us at 64);
  with whole tiles every decode row count runs at the 64-row speed.
- BF16 projections with few output tiles (the 384-expert router: six) split K
  until tiles x shards covers the 48 SMs (eight shards), with FP32 partials
  from the reserved workspace reduced in shard order. The large-row kernel
  folds the same shards, so decode and prefill rows stay bit-identical.
- `_supports_default_lm_head_quantization` accepted only the `auto` and `b12x`
  linear backends, so the online NVFP4 draft head and transition head fell
  back to BF16 cuBLAS (about six vocabulary GEMMs per step on the critical
  tail). The `tilelang` backend now keeps B12X's head kernels, as the policy
  states.

Kernel-lab microbenchmarks on dgx4 under CUDA graphs, warm L2 (per call):

| Projection | Before | After | B12X in serving |
| --- | ---: | ---: | ---: |
| Router gate, BF16, 384 x 5120 | 27 us | 9.4-9.9 us (6.4-6.7 at 16-48 rows) | ~11 us |
| Q-B, 8192 x 1280 | 43 us | 15 us | ~22 us |
| Fused Q-A/KV, 1792 x 5120 | 33-36 us | 15 us | ~19 us |
| Indexer Q-B, 4096 x 1280 | 19-22 us | 9.4 us | |
| Shared expert gate/up, 1152 x 5120 | 23-25 us | 15 us | |
| Shared expert down, 5120 x 576 | 14-15 us | 6.6 us | |

The one projection whose weights exceed L2 (6400 x 5120, 33 MB) stays
DRAM-bound at about 145 us either way. Every result matches a torch reference,
and every row is bit-identical across 1-200 rows and both kernel paths.

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

## Images and kernel tests

Built on dgx4 with `bin/spark3 build prepare`, `build check` and
`build image --apply` from deployment commit `681cc2e`:

| Profile | Image | ID | Build |
| --- | --- | --- | --- |
| tp3 | `vllm-ds41f-kkref:04c30fa98e79-r5o-tilelang-v3` | `sha256:17ab6c4c…` | 459 s |
| tp4 | `vllm-ds41f-kkref:04c30fa98e79-r5o-roce-contract-tilelang-v2` | `sha256:a0a5abc0…` | 453 s |

Both passed the GPU and TileLang import smokes. The kernel tests, with TP3's 24
heads and TP4's 576-wide GEMMs, ran from each image's own vLLM tree as
kernel-lab bundles ([kernel-tests-tp3](bundles/kernel-tests-tp3/candidate.json),
[kernel-tests-tp4](bundles/kernel-tests-tp4/candidate.json); see
[lab.md](../../docs/lab.md)):

```sh
scripts/lab.py kernel-local experiments/2026-10-03-tilelang-kernels/bundles/kernel-tests-tp4
```

Each image passed all 49: sparse MLA at 16, 24 and 64 heads, the MXFP4 indexer
paths and top-k, the MXFP8 GEMMs with K blocks of 128, 192 and 64, the BF16
GEMM, RMSNorm, and the routed experts at 256-, 576- and 320-wide rows against
FP32, with rows unchanged by batch composition. The bundles pass
`--noconftest`: vLLM's root conftest imports test-only packages the serving
image does not carry, and these tests use no fixtures.

The first run, on an earlier TP3 image, failed the 12 routed-expert cases at
576 and 320 because TMA cannot unpack FP4 in 192- or 64-wide boxes; the compact
K tails above fixed it. Images `-r5o-tilelang-v2` and
`-r5o-roce-contract-tilelang-v1` carry that defect and are superseded.

## Qualification

Next, in a reserved window, each profile against its B12X twin: startup and
steady memory, the LRU quality gate, the determinism checks, one- and
eight-stream decode, 32K/64K/256K source-text prefill, prefix replay and
admission. Kernel output may differ from B12X, so qualification compares
quality before speed. The TP4 comparison needs the B12X TP4 image as well;
lab windows run only on the promoted topology, so TP4 runs use explicit
cluster commands.
