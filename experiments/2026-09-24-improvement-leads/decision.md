# Decision: promote r2 (2026-09-24)

Promote image `vllm-ds41f-kkref:01f1b874c774-r2` (vLLM tree `90fdd043` =
pinned karmic-kraken-beta plus `patches/vllm/0001-engram-projection-tp-padding`)
with `engram_config.projection_tp: true` and DSpark `rejection_sample_method:
block`, as baseline `manifests/baselines/2026-09-24-karmic-kraken-r2.json`.

- Engram projection sharding: the projection falls from ~690 to ~235 us per
  call in the rank-0 decode profile (~2% of decode GPU time) and frees about
  210 MB per rank. Same math: padded rows carry zero weights.
- Block rejection: lossless; measured +1% mean accepted drafts on sampled
  traffic, within noise. Kept as the upstream recipe's choice.
- Gates: LRU 5/5 in four passes; serving matrix within noise of the
  promoted baseline at every point. A two-stream dip in the first r2 matrix
  did not reproduce: five focused repetitions give r2 66.2 +/- 1.6 (prose) and
  83.7 +/- 3.3 (code) against 64.9 +/- 2.5 and 76.2 +/- 2.0 for the same logic
  on r1 with the patch mounted. r2's `engram.py` is byte-identical to the
  overlay; its rebuilt native extensions differ only by build reproducibility.

Rejected: dense GEMM plan pinning (0-4% per shape), vLLM #880 (no cap to
lift). Deferred: fused activation quantization, LM head, rank-split indexer
prefill, concurrent disk Engram reads (see README).
