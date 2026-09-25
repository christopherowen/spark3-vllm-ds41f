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

Pending.
