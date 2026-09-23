# Patch-stack simplification audit

2026-09-23. The running target-only image is unchanged by this audit. Its
candidate source trees are vLLM `92aa9f7d15c88b69bc6117963e2dcb52b81a55e5`
and B12X `a251f563557bcbff88b17dedd6fd6bccd859929a`.

## Finding

The 29 vLLM and 5 B12X commits are not 34 independent runtime requirements.
Many are repair commits for local features added earlier in the same series.
An old-base compact replay folded those repairs into their parent behavior:
29 vLLM patches became 10 and 5 B12X patches became 4. Fresh replay from
both recorded upstream bases reproduced the exact running-image source trees,
so that grouping changed neither output nor speed. The old-base compact files
were then removed as duplicate data; the original series remains the image's
provenance. The [September 23 current-head series](../2026-09-23-latest-head-rebase/README.md)
retains the conceptual grouping with 10 vLLM and 3 B12X patches.
Squashing alone does not reduce the 82-file vLLM source delta; code removal
requires a separate source and guarded runtime experiment.

The candidate source diff contains no temporary `print`, breakpoint, NaN
probe, or TODO diagnostic to strip. Runtime-only probes are outside the image
patch series.

| Parent change | Foldable follow-ups | Final behavior to retain |
| --- | --- | --- |
| vLLM 0001, TP3/virtual heads | 0010, 0011 | Lazy and loader-backed virtual head groups. Keep the generic TP3 vocabulary correction (0006) separate for upstream review. |
| vLLM 0002, B12X sparse attention | 0007, 0025–0029 | No child-module cycle; SWA-only routing; decode warmup capacity; 64-state indexer pages; contiguous cache writes; compressed-cache block mapping. Patch 0026 also covers the target-only synthetic startup warmup, so it cannot simply be dropped with DSpark disabled. |
| vLLM 0003, RoCEnante adapter | 0019, 0020 | Prepare and synchronize plans before memory profiling. |
| vLLM 0004, owned/file-backed loader | 0008, 0014 | Route custom weights through allocation and preserve native MXFP8 scale metadata. |
| vLLM 0005, disk Engram | 0012, 0013 | Deliver checkpoint file descriptors to the leaf and support the VL wrapper. Current upstream still offers CPU offload but no equivalent disk row staging. |
| vLLM 0015/0017, B12X preparation | 0016, 0018, 0021–0023 | Bounded pre-weight preparation and selected-kernel registration; keep MXFP8, MoE, draft and RoCE capacity requirements explicit. |
| B12X 0002, switchless RoCEnante | 0005 | Compile prepared flag layout for route stripes rather than opened HCA count. |

## Code worth challenging separately

- **vLLM 0022 is the largest scope concern.** It changes 30 files; 23 files
  are touched by no other local patch, including FlashInfer and several
  TRTLLM/MoE providers outside the selected DS4.1 B12X path. Check whether
  explicit DS4.1/B12X registration can replace those generic edits. Patch
  0023's pre-weight memory boundary remains necessary for the guarded cold
  start until a replacement is measured.
- **vLLM 0004 is the second scope concern.** It changes 16 files, with 11
  touched by no other local patch. Some generic loader edits may be localizable
  to the B12X/Engram integration, but disk-backed checkpoint routing is an
  actual runtime requirement. Do not remove these edits by inspection alone.
- **B12X 0001 is optional performance tuning.** It adds a 3,072-token MHC
  shape to a 4,096-token tuning branch. Keep it out of the correctness stack
  and retain it only if an identical workload benchmark shows a benefit.
- **The V1 runner edit in vLLM 0022 is outside the live path.** The working
  service requires `VLLM_USE_V2_MODEL_RUNNER=1`; the V1 runner caused the
  earlier unprepared disk Engram NaNs. Test removing that V1-specific edit in
  a source candidate, with the V2 path unchanged.
- `state/nan_probe.py` and the q3–q13 configurations are diagnostic evidence.
  The clean running configuration does not mount the probe. Do not carry its
  monkeypatch into the product image.

## Corrections that are not disposable

The TP3 vocabulary alignment (0006), fused 24-head native dispatch (0024),
64-state SM12x indexer pages (0027), B12X cache writer layout (0028), and
compressed-cache block mapping (0029) fix distinct demonstrated boundaries.
B12X's CUTLASS DSL pin (0003) matches this exact vLLM base image, and B12X's
managed-weight loader adaptation (0004) matches the selected load format.
These can be folded into topical patches, but dropping their behavior would
invalidate the current runtime evidence.

## Safe sequence

1. The temporary old-base compact replay matched the exact final tree IDs
   above. Preserve the original series and receipts as historical evidence.
2. In a separate candidate, remove optional or broad code one causal change
   at a time. Run focused tests, rebuild an immutable image, and repeat the
   guarded cold start and semantic smoke. Do not alter the running image while
   testing source-only simplifications.
3. Rebase that compact series onto a pinned current upstream snapshot, resolve
   the existing vLLM 0001 and B12X 0002 conflicts, and use a base image with a
   matching native ABI. Run the full long-context, graph, performance, memory,
   and failure gates before promotion.

This audit separates proven consolidation from possible source deletion. It
does not claim that a source patch can be deleted without a measured replacement.
