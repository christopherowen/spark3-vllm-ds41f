# Native BF16 drafter heads

Base deployment commit: `aafb3cf` (main).

**Variable:** the DSpark draft head and Markov transition head keep the
checkpoint's BF16 tensors instead of re-quantizing them to NVFP4 at load
(`VLLM_DS41_DRAFT_NVFP4_HEAD=0`, `VLLM_DS41_MARKOV_NVFP4=0`). With the flag off
the draft head is the target head's own tensor, so no second copy is loaded.
The pinned DSpark cost curves are keyed by shapes only, so the profile also
gets its own cost directory (`ring4-collective-nativeheads-20261003`).

**Status:** the [TP4 collective-contract candidate](../2026-10-03-collective-contract/README.md)
now runs native heads. The promoted TP3 configuration still sets both flags;
switching it needs a new immutable baseline, and the promoted triangle cannot
serve on the current four-node ring cabling.

## Why

The owner's rule is that served weights stay native unless a transform is
purely lossless. An audit of every load-time weight transform against the
checkpoint headers (`dba1be0a40aa45a94ad051997016db3960a90277`) found exactly
two lossy ones:

| Tensor | Checkpoint | Served before | Verdict |
| --- | --- | --- | --- |
| `head.weight` as the draft head | BF16 [129280, 5120] | NVFP4 copy | lossy |
| `mtp.2.markov_head.head.weight` | BF16 [129280, 256] | NVFP4 | lossy |
| Dense projections, including WO | FP8, block-32 scales | unchanged | native |
| Routed experts | FP4 | repacked | lossless |
| Engram tables | F8_E4M3 with F8_E8M0 scales | unchanged | native |
| Target LM head | BF16 | unchanged | native |
| Display carve-out | — | placement only | lossless |

## Measurement

Four-node ring, TP4, B12X, image `vllm-ds41f-kkref:04c30fa98e79-r5o-roce-contract-v1`
(`sha256:e72ba6a4…`), the regular `bin/spark3 bench` decode and prefill matrix
with three decode samples per point. The NVFP4 arm is the B12X control of the
TileLang TP4 screen's round 1 (2026-10-03 18:09–18:25 UTC); the native arm is
the same profile with the three settings above (20:09–20:24 UTC). The windows
differ, so treat small differences as noise.

| Case | NVFP4 heads | Native heads | Change | Accepted per verified draft |
| --- | ---: | ---: | ---: | ---: |
| prose, 1 stream | 62.0 ± 8.4% | 62.3 ± 25.3% | +0.4% | 0.354 → 0.375 (+5.9%) |
| prose, 8 streams | 221.0 ± 1.6% | 213.3 ± 3.8% | −3.5% | 0.597 → 0.656 (+10.0%) |
| code, 1 stream | 78.4 ± 10.0% | 78.1 ± 11.6% | −0.3% | 0.491 → 0.522 (+6.3%) |
| code, 8 streams | 255.6 ± 3.9% | 246.0 ± 2.5% | −3.8% | 0.708 → 0.722 (+2.1%) |

Decode is aggregate tok/s with 95% intervals. Single-stream steps lengthen by
1.8–2.0 ms (prose 31.91 → 33.66 ms, code 36.16 → 38.18 ms): each step reads
the 331 MB BF16 head shard once more for the first draft position, and the
Markov head four times. Acceptance rises in every case, which holds one-stream
throughput level; at eight streams the extra bytes cost 3.5–3.8%. Prefill:
1K +0.5%, 32K +1.1%, 64K +1.1%, 256K +0.9%. Quality 5/5 in both arms.

## Layout audit

[`scripts/model_layout.py`](../../scripts/model_layout.py) now describes either
format instead of requiring NVFP4: `drafter.head_formats` names the format of
each head, NVFP4 heads keep their packed-value and scale layouts, and BF16
heads report their native shape (the draft head marked as shared with the
target head). The source check in
[`check-layout-source.py`](../2026-10-03-transport-profiles/check-layout-source.py)
verifies packed layouts only for heads a profile stores as NVFP4.

## Conclusion

Native heads are the owner's requirement, and they do not cost one-stream
throughput. The eight-stream cost is the extra bytes of BF16 vocabulary reads;
an exact 12-bit packing of those heads (sign and mantissa byte plus a 4-bit
exponent code, measured lossless on this checkpoint) is the follow-up that
recovers part of it without changing a weight.
