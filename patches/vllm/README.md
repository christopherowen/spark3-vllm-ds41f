# vLLM patch stack

Base: `local-inference-lab/vllm@01f1b874c774b4fade087d5f311970ee53745e01`
(`integration/karmic-kraken-beta`).

- `0001-engram-projection-tp-padding.patch` lets `engram_config.projection_tp`
  shard DeepSeek V4.1's Engram WKV projection when its 25600-row output does
  not divide by the TP size (TP3): the output is padded to whole 32-row
  block-FP8 scale blocks per rank, the last rank's missing checkpoint rows are
  zero-filled through `allow_tp_padding`, and the gathered output is sliced
  back. Applying it to the base yields tree `90fdd043`, the promoted image's
  `local.spark3.vllm.tree` label. Evidence:
  `experiments/2026-09-24-improvement-leads/`. Upstream status: candidate for
  Local Inference Lab, not submitted.
