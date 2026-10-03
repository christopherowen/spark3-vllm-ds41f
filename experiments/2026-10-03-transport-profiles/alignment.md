# TP3 and TP4 alignment inventory

This completes the layout report for the two named profiles. It describes
adapter-owned model partitions, serialized weight/scale buffers, graph row
capacities and collective layout. These are derived expectations for the
audited checkpoint/source combinations, not new performance measurements or
an inventory of every tile-dependent GEMM workspace.

The [checkpoint config receipt](checkpoint-config.json) was read from dgx1's
pinned snapshot without changing the host or service. Its hash is recorded in
[model-layout.json](model-layout.json). Dimensions come directly from that
receipt; they are not duplicated as tunable constants. Source tree identities,
file hashes and executable helper checks are in [source-checks.json](source-checks.json).
The helper checks execute extracted pure functions with allocation stubs; they
do not test GPU memory accesses, quantization values or serving correctness.

## Model partitions

| Object | Logical dimension | TP3 allocation | TP4 allocation |
| --- | ---: | ---: | ---: |
| Attention heads | 64 global | 72 global; 24/rank | 64 global; 16/rank |
| Attention output groups | 8 global | 9 global; 3/rank | 8 global; 2/rank |
| Engram WKV output | 25,600 global | 25,632 global; 8,544/rank | 25,600 global; 6,400/rank |
| Draft auxiliary projection output | 5,120 global | 5,184 global; 1,728/rank | 5,120 global; 1,280/rank |
| Target/draft/Markov output vocabulary | 129,280 global | 129,408 global; 43,136/rank | 129,280 global; 32,320/rank |
| Expert intermediate channels | 2,304 global | 768/rank, exact | 576/rank, exact |

The auxiliary projection input concatenates three target hidden states:
3 × 5,120 = **15,360** columns. The configured projection shards its **output**
columns, using the same `TP × 32` alignment helper as Engram. TP3 therefore
has 64 trailing zero-weight output rows and trims the gathered result; TP4
needs neither extra output rows nor that trim.

TP3 vocabulary ranks own 43,136 / 43,136 / 43,008 real token rows, with
128 padding rows on the final rank. TP4 owns 32,320 real rows on every rank.
The target and draft heads mask padded token logits; the Markov output
partition is checked against the draft head's partition.

The Markov input embedding remains replicated at **129,280 × 256**, with no
vocabulary padding. The indexer's 32 heads and weights remain replicated;
indexer sequence-parallel **rows** are a separate matter. The vision tower
also remains replicated. A head count divisible by TP does not automatically
make its implementation tensor-parallel.

## NVFP4 storage

This layout applies only to a head a profile re-quantizes to NVFP4
(`VLLM_DS41_DRAFT_NVFP4_HEAD=1`, `VLLM_DS41_MARKOV_NVFP4=1`). The TP4 candidate
keeps both heads at the checkpoint's BF16
([native drafter heads](../2026-10-03-native-drafter-heads/README.md)); the
report then lists their native shapes under `drafter.head_formats`, and the
BF16 draft head is the target head's own tensor.

The draft LM head has K=5,120; the Markov output projection has K=256.
Both use packed values `[N, K/2]` (two values per byte), one E4M3 scale per
16 values, and swizzled scales with rows rounded to 128 and columns to four.
Neither K dimension needs padding. The scale plane can contain padded rows
even when the value matrix and vocabulary do not.

| Per-rank tensor | TP3 shape | TP4 shape |
| --- | --- | --- |
| Draft head packed values, bytes | 43,136 × 2,560 | 32,320 × 2,560 |
| Draft head scale storage, bytes | 43,136 × 320 | 32,384 × 320 |
| Markov output packed values, bytes | 43,136 × 128 | 32,320 × 128 |
| Markov output scale storage, bytes | 43,136 × 16 | 32,384 × 16 |

TP4's extra 64 scale rows occupy **20,480 bytes** for the draft head and
**1,024 bytes** for the Markov output per rank. Those are not extra logits,
checkpoint weights or expert channels. TP3 has no additional scale-row
rounding beyond its already padded vocabulary shard. These figures exclude
scalar scales, allocator rounding, temporary quantization storage and
plan-dependent GEMM workspaces; they are not total model-memory estimates.

Quantized activations follow a related rule: M actual input rows produce
`[M, K/2]` packed values and `[ceil(M/128) × 128, ceil((K/16)/4) × 4]`
scale storage. For M=1, 8 or 48, the draft-head scale buffer is 128 × 320
bytes, and the Markov scale buffer is 128 × 16 bytes. These are **head input
rows**, which need not equal a scheduler's token count.

The TP4 compact W4A8 MoE path separately rounds 576 intermediate channels to
640 scratch channels. Its intermediate region has `max_tokens × top_k` rows
and `ceil(N/128)` scale tiles. This does not pad the expert weights to 640.
Actual compact-path selection and workspace capacity depend on the prepared
plan and routed row count; the report states the layout when selected.

## Graph and sequence rows

Both profiles configure the target ladder
`1,2,3,4,6,8,12,16,20,24,28,32,40,48`. Adaptive verification also has exact
low-concurrency graphs: one request at 1–6 rows, and two requests at
2/4/6/8/10/12 total rows. Consequently the ladder alone cannot predict the
graph used. Five rows can use the exact five-row graph rather than padding
to six. Request count, query lengths, mixed/decode mode and actual captured
descriptors participate in dispatch.

With five drafts and at most eight requests, the drafter has two distinct
capacity lists when full graph capture is supported:

- **Query forward:** six rows per request, with capacities
  `6,12,18,24,30,36,42,48` after rounding the configured ladder to six.
- **Auxiliary context preparation:** `1,2,4,8,16,32,48`, bounded by the
  decode limit, hidden-state buffer and maximum graph capture size. Its
  reusable BF16 auxiliary input buffer is **48 × 15,360** in both profiles.
  Nineteen eligible context rows use capacity 32; larger prefills use the eager
  context path. Padded KV slots are invalidated and replay tails are cleared.

Prefill SP rounds live rows to a multiple of TP, independently of those graph
ladders. The current 2 MiB registered capacity gives the same 205-live-row
SP boundary in both profiles; lowering transport dispatch to 1 MiB does not
lower that boundary.

| Live rows in an eligible prefill forward | TP3 collective rows | TP4 collective rows |
| --- | ---: | ---: |
| 205 | 207 | 208 |
| 4,096 | 4,098 | 4,096 |

These counts apply to eligible encoder work before CED compaction. A long
prompt is scheduled into chunks; its total token count is not one padded
GEMM or graph capacity. The report does not predict chunk boundaries from
prompt length alone because other requests, cache reuse and scheduling matter.

## Transport scratch

RoCEnante uses 16-byte packs. All-reduce eligibility requires a positive payload
whose size is a multiple of 16. Misaligned pointers use alignment scratch;
unsupported payload sizes follow the declared backend policy.

Direct all-gather requires aligned input/output pointers and a total byte
count divisible by 16. A last-dimension gather also requires each row's byte
width to be divisible by 16. Otherwise the supported input uses a staged
**whole shard** rounded to 16 bytes, followed by gather and reshape. It does
not pad every row independently.

For example, a contiguous 3 × 5 BF16 shard contains 30 bytes and stages into
32 bytes. Gathered scratch contains 96 bytes at TP3 or 128 at TP4. An 8 × 5
BF16 shard contains 80 bytes: a dimension-zero gather with aligned pointers
can be direct, but a last-dimension gather needs staging/reshape because each
row contains ten bytes—even though that staging adds no payload bytes.

## Source checks and reproduction

The implementation locations, relative to each source repository, are:

- vLLM `models/deepseek_v4_1/common/engram.py`: padded column projection.
- vLLM `models/deepseek_v4_1/nvidia/dspark.py`: auxiliary projection, context
  graphs, draft/Markov selection and partition agreement.
- vLLM `model_executor/models/qwen3_dspark.py`: Markov input/output ownership.
- vLLM `model_executor/layers/vocab_parallel_embedding.py`: vocabulary layout.
- vLLM `model_executor/layers/quantization/online/nvfp4.py` and
  `model_executor/kernels/linear/nvfp4/b12x.py`: NVFP4 quantization/packing.
- vLLM `_custom_ops.py`: actual FP4 value/scale allocations.
- vLLM `v1/worker/gpu/cudagraph_utils.py` and
  `v1/worker/gpu/spec_decode/dflash/speculator.py`: target/draft descriptors.
- vLLM `models/deepseek_v4_1/sp_prefill.py`: SP boundary and row layout.
- B12X `_lib/intrinsics.py`, `gemm/blockscaled/_a16.py`: swizzled weight
  scales and borrowed packed storage.
- B12X `moe/_shared/kernels/w4a8_compact_micro.py`: compact intermediate scratch.
- B12X `comm/roce/roce_oneshot.py`: direct/padded collective layouts.

```sh
python3 experiments/2026-10-03-transport-profiles/check-layout-source.py \
  --vllm-repo <prepared-vllm-repository> \
  --b12x-repo <prepared-b12x-repository>
```

For each audited source pair this checks eight NVFP4 storage cases against
vLLM's actual allocation helpers, the SP boundary, exact target graph shapes,
and five direct-gather alignment cases against B12X's actual predicate.
Other listed layouts were inspected in source. Unit tests also cover expected
partitions, scratch sizes, graph lists and rejection of incompatible source,
checkpoint or execution flags. No inference kernel or production setting is
changed by this reporting work.

Omitting the repository arguments locates the prepared explicit-policy source
trees automatically. CI runs this checker after fresh source preparation,
alongside the collective-contract tests.
