# Compatibility matrix

## Exact prepared revisions

| Component | Revision | Declared dependency |
|---|---|---|
| vLLM | `d05da62e9ccdf8e342b15bf6785d83224cc165af` | `torch==2.13.0`, `nvidia-cutlass-dsl[cu13]==4.7.1`, optional `b12x==1.3.0` |
| B12X | `0f3a8cbfd1c11d27f04e3ab37a802d522f4f1c68` plus carry through `8099cee92c78dd2472749ca8003053ee73be908d` | all CUTLASS DSL packages `==4.7.1` |

The dependency sets are now co-installable as declared. The official vLLM
nightly nevertheless deliberately ships NCCL 2.30.7 while Torch metadata names
2.29.7, and its pristine `pip check` also reports the ARM64 cuSPARSELt marker.
Allowing pip to resolve an unrelated B12X install therefore downgrades the
nightly's native communication stack. The experiment instead preserves the
base NCCL version, verifies every applicable direct B12X requirement, and
requires the complete pre/post `pip check` output to remain identical. This is
an explicit base-image exception, not dependency-conflict suppression.

The migration still requires target qualification because B12X compiles and
caches kernels through CUTLASS DSL, so the toolchain change can affect build
correctness, cache behavior, memory use, and performance.

The source-integration blocker is also resolved. The candidate adds a
fail-closed B12X choice to the native DS4.1 model and a RoCEnante communicator
using current prepared plans. These are consumers of current upstream model
and B12X interfaces, not copies of the older downstream model stack.

The second build probe found that current B12X loading still assumed interfaces
from its downstream vLLM fork. The candidate now supplies only the generic
owned/file-backed routing primitives in vLLM and a matching B12X adapter for
coherent managed weights on GB10. Device/GDS loading fails closed because its
coordinated transfer hook is not upstream. Disk Engram consumes file ranges
directly and never allocates the roughly 63 GiB-per-rank table as resident model
memory.

The first complete TP3 model-construction attempt also showed that upstream
`VocabParallelEmbedding`'s fixed 64-row padding is not always divisible by
non-power-of-two TP sizes. The candidate now retains 64-row alignment while
padding to its least common multiple with TP size. For DS4.1 TP3 this adds only
inert rows beyond the native vocabulary and leaves token selection unchanged.

## Valid routes

### Selected route: qualify B12X on CUTLASS DSL 4.7.1

This keeps the exact ARM64 vLLM nightly. The migration is a separate B12X
commit and must pass:

1. dependency and generated-lock updates;
2. import, compile, and cache-key tests;
3. every DS4.1 kernel compiled on an SM121 Spark;
4. deterministic output comparison against the promoted runtime;
5. the complete latency, throughput, memory, and failure benchmark matrix.

Only after those gates pass should the B12X dependency change be proposed
upstream.

### Rejected fallback: use an exact vLLM/CUTLASS DSL 4.6.2 source build

The parent of vLLM's 4.7.1 bump is
`31759ccb6d560e150849273b47e15680a0120169`. This route is valid only if its
entire native stack is built from that exact revision for ARM64; an unverified
nightly tag or a Python overlay on another image is not acceptable. It is a
fallback because it deliberately trails the selected upstream revision.

## Invalid shortcuts

- Installing B12X with `--no-deps` without checking every direct requirement,
  preserving the recorded base-native versions, and proving there are no new
  `pip check` findings.
- Downgrading CUTLASS DSL inside the published vLLM nightly.
- Overlaying current vLLM Python on native extensions from an older commit.
- Promoting the upstream-native FlashInfer result as equivalent to the B12X
  performance path without measuring both.

## Qualification order

1. Build the isolated image and pass its loader/Engram import and contract gates
   on SM121 before any service switch.
2. Prove the B12X selection, V4.1 cache ABI, native 24-head TP3 boundary, and
   RoCEnante dispatch in startup logs and focused probes.
3. During a maintenance window, compare the promoted runtime, upstream-native
   image, and B12X image with identical launch configuration and prompts.
4. Promote only the best quality-preserving candidate, then update the upstream
   locks and baseline receipt.
