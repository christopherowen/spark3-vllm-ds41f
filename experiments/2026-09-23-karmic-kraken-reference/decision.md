# Decision: promote (2026-09-24)

Promote `cluster-kkref-notiny.json` as `config/cluster.json`, with baseline
`manifests/baselines/2026-09-24-karmic-kraken.json`. The owner accepted the
promotion and parked canonical vLLM main for now.

What is promoted:

- vLLM: Local Inference Lab `integration/karmic-kraken-beta` at `01f1b874c`,
  unchanged.
- B12X: Local Inference Lab `integration/karmic-kraken-beta` at `0f846212` plus
  `patches/b12x/0001-switchless-rocenante.patch` (tree `d661de31`).
- Image `vllm-ds41f-kkref:01f1b874c774-r1`, content digest
  `sha256:9f507864…` on all three nodes, built by this directory's
  `prepare-sources` and `build-candidate` at deployment commit `9ffea8f3`.
- `B12X_W4A8_TINY_DECODE=0`: tiny decode omits `swiglu_limit`, which caused the
  LRU incoherence shared by every earlier stack, and gives no speed on GB10.
  Patch 0002 (the clamp fix) is kept here as an upstream contribution only.

Evidence (`results.json`, `runs/kkref-notiny/`): LRU 5/5; faster than the
2026-09-20 weekend image at every serving point (prose c1 44.9 vs 34.6 tok/s,
code c8 160.1 vs 71.9); 933,168 KV tokens in 2 GiB per rank (2.8x the
weekend's density).

Not claimed: tuned speed (autotune off, DSpark 3, capture sizes up to 24,
4096 batched tokens) or broad quality. Those are follow-up experiments against
this baseline.
