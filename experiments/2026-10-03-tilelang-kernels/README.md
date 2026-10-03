# TileLang kernel backend

**Status:** sources complete and verified; no image built. TileLang,
TileKernels and the DS4.1 TileLang vLLM patch (0027) are pinned, prepared and
recorded, and doctor passes. Launch is disabled until an image is built and
qualified. No cluster operation was performed.

Deployment base: `e15547a`. The promoted configuration stays TP3 with B12X
kernels.

## Purpose

Run the DS4.1 model compute kernels (attention, including the DSpark drafter's,
linear layers and MoE) on TileLang instead of B12X, with everything else
matched to the promoted configuration. The policy and its required settings are
in [kernel-backends.md](../../docs/kernel-backends.md).

## Intended delta

One variable: `kernel_backend: tilelang` in [cluster.json](cluster.json), with
the settings doctor requires:

- `--attention-backend TILELANG`, `--linear-backend tilelang`,
  `--moe-backend tilelang`, and `"attention_backend":"TILELANG"` in
  `--speculative-config`;
- `VLLM_DS41_KERNEL_BACKEND=tilelang`; `TILELANG_CACHE_DIR` stays
  `/cache/kkref/jit/tilelang`.

The node map, transport, collectives (RoCEnante plus NCCL), the B12X checkpoint
loader, model, context, KV budget, memory guards and every other setting equal
`config/cluster.json`. The image tag is
`vllm-ds41f-kkref:04c30fa98e79-r5o-tilelang-v1`; the B12X and NCCL trees are the
promoted ones.

## Sources

[upstreams.lock.json](upstreams.lock.json) is the promoted lock plus:

| Source | Revision | Local changes | Tree |
| --- | --- | --- | --- |
| `tilelang` (`tile-ai/tilelang`, main) | `b95ee4ff` (0.1.15) | `patches/tilelang/series` 0001-0003; patch head `30d854b6` | `1697ad52` |
| `tile_kernels` (`deepseek-ai/TileKernels`, main) | `66258df6` (2.0.0) | none | `64770881` |

TileLang's submodules (TVM and CUTLASS, with TVM's own, recursively) are fetched
at the commits its tree records; [source.json](source.json) lists them and the
patch-set fingerprint `34504514…`. TileLang's contribution fork is
`christopherowen/tilelang`, branch `deepseek-v41-sm120`.

The vLLM series [vllm/series](vllm/series) is the 26 promoted patches plus
[0027-deepseek-v41-tilelang-kernels.patch](vllm/0027-deepseek-v41-tilelang-kernels.patch),
stored in this directory: patch head `6b352163`, tree `edbb8598`, patch-set
fingerprint `d02c2080…`. The vLLM record in `source.json` carries capability
`tilelang-kernels`, and the image label `local.spark3.vllm.tree` expects that
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
bin/spark3 --cluster-config experiments/2026-10-03-tilelang-kernels/cluster.json build prepare --only vllm,tilelang,tile_kernels
bin/spark3 --cluster-config experiments/2026-10-03-tilelang-kernels/cluster.json doctor
python3 -m unittest tests.test_kernel_backend
```

Completed on 2026-10-03:

- Preparation reproduced every recorded patch head and tree, including all
  eleven TileLang submodule commits. Doctor passes with the example site
  configuration.
- GPU kernel tests on dgx4 in one-off containers from the r5o image (the
  patch's `tests/kernels/{attention,quantization,moe}/test_deepseek_v41_tilelang*.py`),
  against torch references:
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

## Qualification after the image exists

The GPU import smoke, then in a reserved window: startup and steady memory, the
LRU quality gate, the determinism checks, one- and eight-stream decode,
32K/64K/256K source-text prefill, prefix replay and admission, each against the
promoted B12X baseline. Kernel output may differ from B12X, so qualification
compares quality before speed.
