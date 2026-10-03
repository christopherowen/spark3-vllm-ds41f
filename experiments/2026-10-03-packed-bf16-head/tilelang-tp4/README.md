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

**Status:** image `vllm-ds41f-kkref:04c30fa98e79-r5o-roce-contract-tilelang-packedhead-v1`
(`sha256:5bcbc860…`) built on dgx4 from `0c0149f` with the regular
`bin/spark3 build` commands and loaded on dgx1–dgx3. Its bundles pass on dgx4
(2026-10-03 22:52–22:53 UTC): TileLang 51 passed, vLLM 2 passed, B12X 15
passed. TP4 window 2026-10-03 22:55–23:10 UTC, summary in
[results.json](results.json).

## TP4 window

The same hardened procedure as the B12X window, control first:

| Case | TileLang control | TileLang packed | Change |
| --- | ---: | ---: | ---: |
| prose, 1 stream | 62.4 ± 0.3% | 62.3 ± 0.3% | −0.2% |
| prose, 8 streams | 214.1 ± 3.4% | 213.6 ± 0.2% | −0.2% |
| code, 1 stream | 79.9 ± 1.3% | 80.8 ± 1.0% | +1.1% |
| code, 8 streams | 246.3 ± 0.4% | 247.5 ± 2.0% | +0.5% |

Single-stream step: prose 33.45 → 33.58 ms, code 36.54 → 36.73 ms (+0.4% and
+0.5%, intervals −0.3% to +1.4%). Prefill: 1K −1.2%, 32K −0.9%, 64K −0.3%,
256K −0.5%. Quality 5/5 in both arms. The control reproduces the TileLang
round 3 arm (62.7 / 215.9 / 80.0 / 247.9 tok/s).

The head does under TileLang what it does under B12X: 1.088 / 1.119 ms per
read (rank 0 / rank 3) against 1.305 / 1.328 ms for cuBLAS BF16, 0.43 ms less
per decode step. The step did not shorten because TileLang's routed-expert
kernels ran 0.6 ms slower in the packed boot (18.86 against 18.26 ms per
six-row step on rank 0). Those kernels do not touch the head, and their time
varies across TileLang boots far more than this: 16.15 ms in round 2,
18.23 ms in round 3, 18.26 and 18.86 ms here. B12X's routed experts stayed
within 21.82–22.25 ms over the same four boots. The 2.7 ms TileLang spread is
its own finding: it hides any change smaller than itself in one-boot A/Bs,
and closing it would be worth more than this head.

Host MemAvailable minima were lower in the packed arm on dgx1 (26.33 against
29.21 GiB) and dgx4 (29.94 against 30.86), level on dgx2 and dgx3. As in the
B12X window, only the packed arm profiled and pinned fresh DSpark cost curves
at boot, while vLLM's free memory at KV allocation was the same (114.04
against 114.12 GiB).
