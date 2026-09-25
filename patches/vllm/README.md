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

- `0002-engram-async-disk-rows.patch` (`SPARK3_ENGRAM_ASYNC=1`) reads disk
  Engram rows on a reader thread while the forward graph launches, instead of
  finishing both layers' reads before the launch. The main thread queues the
  row decode on a side stream behind a host gate and resets a per-layer ready
  flag; the reader thread only waits for the row IDs, reads the rows, and
  releases the gate (also on failure); each Engram layer waits on its flag,
  captured into the CUDA graph. Graphs are identical with the path on or off.
  Quality: the rows and their decode are unchanged, only their timing; a
  20 ms injected read delay lengthened steps by 27 ms with quality and
  acceptance unchanged, showing the gate holds. Applying 0001-0002 to the base
  yields tree `033fd0cc`. Evidence: `experiments/2026-09-24-three-leads/`
  and `experiments/2026-09-25-engram-async-ab/`. Upstream status: candidate
  for Local Inference Lab, not submitted.

