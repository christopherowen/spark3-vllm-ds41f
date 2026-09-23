# Serving performance bridge, 2026-09-23

## Scope and identity

This is a measured comparison of the successful 2026-09-20 promoted service
with the **currently running patch-0029 service**. It is not a performance
measurement of the consolidated latest-head series in this experiment: that
series has not been built or launched. The two operating configurations differ
substantially, so the numbers measure the complete setups, not the causal
effect of patch consolidation.

- Original: `config/cluster.json`, image
  `vllm-dsv41:r37-b12x-e4m3fix1-arm64-sm121`, raw receipts in
  `../2026-09-20-upstream-main-rebase/runs/baseline-live/`.
- Interim: `../2026-09-20-upstream-main-rebase/cluster-v2-patch0029.json`,
  image `vllm-ds41f-upstream-main:d9ddab37ab97-b69feee3022c-d4243ad067ab8`,
  raw receipts in `runs/patch0029-live/`.
- Client: the identical `../2026-09-20-upstream-main-rebase/benchmark_serving.py`,
  SHA-256
  `888fcfebc223dd971c362474f2f543c48443437484021adcfc0cfad5797528ef`.
  Both sets use the same model snapshot, prompts, temperature 0, seed 42,
  256 requested completion tokens, and HTTP streaming. The interim client ran
  on dgx1 against localhost. The weekend receipts identify the script but do
  not record the client host or URL, so its network path is unknown.

The interim sequence used one 64-token prose warmup, then prose and code at
concurrency 1, 2, 4, and 8. Each case was run once, sequentially; both
single-request cases were then repeated. The command shape was:

```sh
python3 experiments/2026-09-20-upstream-main-rebase/benchmark_serving.py \
  --base-url http://127.0.0.1:8000 --case prose \
  --concurrency 1 --output-tokens 256
```

The old receipts also contain a 64-token prose warmup. All 30 matched interim
requests succeeded and each returned 256 completion tokens. There were no
benchmark HTTP failures. Foreign request traffic was not measured for the
interim runs; the old run recorded zero foreign requests. Thus these are
single-run directional results, not a statistical confidence estimate.

## Matched serving results

Aggregate end-to-end tokens per second includes first-token and request wall
time. Negative change means the interim setup is slower. TTFT is mean seconds.

| Prompt | Concurrency | Weekend TPS | Patch-0029 TPS | Change | Weekend TTFT | Patch-0029 TTFT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Prose | 1 | 31.41 | 10.13 | -67.7% | 0.327 | 0.289 |
| Prose | 2 | 53.18 | 12.09 | -77.3% | 0.485 | 0.486 |
| Prose | 4 | 77.66 | 23.97 | -69.1% | 0.585 | 0.605 |
| Prose | 8 | 57.78 | 47.72 | -17.4% | 0.609 | 0.774 |
| Code | 1 | 44.28 | 10.06 | -77.3% | 0.372 | 0.303 |
| Code | 2 | 64.92 | 11.96 | -81.6% | 0.477 | 0.494 |
| Code | 4 | 94.35 | 23.72 | -74.9% | 0.592 | 0.625 |
| Code | 8 | 76.08 | 46.87 | -38.4% | 0.690 | 0.789 |

The two later single-request repeats reached 10.00 prose and 10.10 code TPS,
consistent with the first interim measurements. Completion hashes differed
both from the weekend receipts and between identical concurrent requests in
*each* setup; the interim single-request repeats also produced different
hashes. These hashes alone cannot establish output-quality regression. The
patch-0029 target-only semantic smoke passed earlier; full quality and
long-context gates remain open.

## Interpretation and next comparison

The interim setup has a clear as-operated speed regression. Its first-token
latency is roughly comparable, while decode dominates the slowdown. It has
speculative decoding disabled and CUDA graphs set to `NONE`; the successful
weekend setup used three-token DSpark speculation and
`FULL_AND_PIECEWISE` CUDA graphs. It also changed block size from 256 to 128,
KV reservation from 3 GiB to 2 GiB, and the vLLM/B12X source. These changes
prevent attribution to any one patch. The 1-to-4-way loss is especially large.

After the consolidated image is built and safely launched on all three ranks,
repeat the same matrix with raw receipts, the semantic and long-context gates,
and the normal Strix workload. Then test performance features separately where
the new upstream supports them, preserving output correctness and host memory
headroom. The consolidated image has no measured performance result yet.
