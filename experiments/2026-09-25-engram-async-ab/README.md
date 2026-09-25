# Asynchronous disk Engram rows: A/B on one image

`experiments/2026-09-24-three-leads/` measured the asynchronous Engram
overlay at +3-9% against two baselines, but single boots of one
configuration vary by about 3% (adaptive verification's startup cost
profile). This experiment confirms the effect on the productized patch
before promotion:

- Image `vllm-ds41f-kkref:01f1b874c774-r3`, built by `bin/spark3 build` from
  the vLLM series with `0002-engram-async-disk-rows.patch` (tree `033fd0cc`).
- Two arms that differ only in `SPARK3_ENGRAM_ASYNC`: `cluster-off.json`
  (unset) and `cluster-on.json` (`1`). The patch traces the same graphs
  either way, so both share one compile cache.
- Four fresh boots in the order off, on, off, on, each measured with
  `bin/spark3 bench --suites quality,decode`. Arms are compared by pooling
  each arm's two boots.

## Results

The image built in 484 s (lowest MemAvailable 79.8 GiB), passed the GPU import
smoke, and reached all three nodes with one ID (`sha256:9c9ca541…`). The owner
stopped the four-boot A/B during the first boot: the three-leads arms had
already measured the change, and a routine promotion should not need
hours of benchmarking. r3 was promoted with the asynchronous path on, and a
short confirmation run of the promoted service became the r3 reference
(`manifests/benchmarks/2026-09-25-karmic-kraken-r3.json`).
