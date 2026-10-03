# Packed BF16 head under the TileLang kernel family

The same variable as the [packed BF16 head](../README.md), measured under the
[TileLang kernel family](../../2026-10-03-tilelang-kernels/README.md), which
keeps B12X's vocabulary projections: the TileLang TP4 profile with native
drafter heads, plus `VLLM_DS41_PACKED_BF16_LM_HEAD=1`, its own DSpark cost and
profiler directories, and sources that add B12X 0012 and vLLM 0028.

- vLLM: the TileLang TP4 series with [0028](../vllm/0028-deepseek-v41-packed-bf16-lm-head.patch)
  before the TileLang kernels, which [0029](vllm/0029-deepseek-v41-tilelang-kernels.patch)
  replays on top; besides the replay, it lets the packed head's backend check
  accept `tilelang`, as it already does for B12X's BF16 projection. Tree
  `6415d579`, head `54013620`.
- B12X: the packed head's series ([../b12x/series](../b12x/series)), tree `bd2d95c4`.
- Candidate: [candidate.json](candidate.json). Window arms:
  [control.json](control.json), identical to the TileLang TP4 round 3 arm
  (image `-r5o-roce-contract-tilelang-v5`), and [packed.json](packed.json)
  (image `-r5o-roce-contract-tilelang-packedhead-v1`).
- Bundles: [tilelang-tests](bundles/tilelang-tests/candidate.json),
  [vllm-tests](bundles/vllm-tests/candidate.json) and
  [b12x-tests](bundles/b12x-tests/candidate.json) against the new image.

**Status:** sources reproduced; image, tests and window follow.
