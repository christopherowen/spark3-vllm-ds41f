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


- `0003-dsml-optional-string-attribute.patch` keeps DSML tool parameters
  that omit `string="true|false"`; the V4 and V4.1 parsers dropped them
  silently. A missing attribute is treated like `string="false"` (JSON,
  falling back to the literal). Port of vllm-project/vllm#56271 (merged as
  `39e33db7`), Python parts only. Quality: tool-call arguments only. Tests:
  the PR's parser tests (128 pass in the r3 image).

- `0004-mm-block-hash-window.patch` hashes every multimodal item that
  overlaps a block completed by generated tokens; the incremental hasher
  started at the last item only, so a block holding two images hashed as if
  it held one. That lost the next turn's prefix-cache hit and let two
  conversations that differ only in an earlier image of that block share
  its KV. Port of vllm-project/vllm#51694 (merged as `e6c07ea5`). Tests: the
  PR's test fails on r3 and passes with the patch.

- `0005-dspark-pinned-cost-curves.patch` (`SPARK3_DSPARK_COST_DIR`) saves TP
  rank 0's adaptive-verification cost curves for a shape set and reuses them
  on later boots with the same shapes, so verified draft counts no longer
  follow one boot's startup timing. `SPARK3_DSPARK_PROFILE_REPLAYS` raises
  the replay count of the pinned profile. Every rank still profiles, so
  collectives stay matched. Output unchanged. Upstream status: candidate
  for Local Inference Lab, not submitted.

- `0006-dspark-marginal-verification-rule.patch`
  (`SPARK3_DSPARK_VERIFY_RULE=marginal`) chooses the draft budget that
  maximizes expected tokens minus the achieved rate times modeled cost,
  instead of each step's expected tokens per millisecond. The rate is an
  exponential average per request count, floored at the no-draft rate, and
  derived identically on every rank. Output unchanged: the target verifies
  every kept draft. Upstream status: candidate, not submitted.

- `0007-dspark-v41-gathered-markov-bias.patch` admits the V4.1 drafter to
  the existing gathered Markov-bias path (`dspark_draft_topk` in the
  speculative config): the bias is added to the top-k base candidates only,
  instead of projecting onto the whole vocabulary at every draft position.
  Only drafts can change, so acceptance can move but output cannot. Off
  unless `dspark_draft_topk` is set. Upstream status: candidate, not
  submitted.

- `0008-dspark-draft-trace.patch` (`SPARK3_DSPARK_TRACE=<path>`) makes TP
  rank 0 append each step's sampled tokens, verified draft counts, drafts
  and raw confidences as JSON lines, for pricing verification policies
  offline at temperature 0. It copies tensors to the host every step: replay
  runs only. Output unchanged.

Applying 0001-0008 to the base yields patch head `0ccd0f62` and tree
`0debe853`.
