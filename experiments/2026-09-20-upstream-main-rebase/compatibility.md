# Compatibility matrix

## Exact prepared revisions

| Component | Revision | Declared dependency |
|---|---|---|
| vLLM | `d05da62e9ccdf8e342b15bf6785d83224cc165af` | `torch==2.13.0`, `nvidia-cutlass-dsl[cu13]==4.7.1`, optional `b12x==1.3.0` |
| B12X | `0f3a8cbfd1c11d27f04e3ab37a802d522f4f1c68` | all CUTLASS DSL packages `==4.6.2` |

The dependency sets are not currently co-installable as declared. This is a
qualification blocker, not a resolver inconvenience: B12X compiles and caches
kernels through CUTLASS DSL, so changing its exact toolchain can change build
correctness, cache behavior, memory use, and performance.

There is also a source-integration blocker. Canonical vLLM at the prepared
revision has a generic B12X paged-attention backend, but its native DS4.1 model
selects only FlashInfer, FlashMLA, or Mega attention. It has no RoCEnante
collective implementation. The B12X source carries in this experiment cannot
affect DS4.1 serving until those two consumer adapters are ported and tested.

## Valid routes

### Route A: qualify B12X on CUTLASS DSL 4.7.1

This keeps the exact ARM64 vLLM nightly and is the preferred route if B12X can
be migrated cleanly. The migration must be a separate B12X commit with:

1. dependency and generated-lock updates;
2. import, compile, and cache-key tests;
3. every DS4.1 kernel compiled on an SM121 Spark;
4. deterministic output comparison against the promoted runtime;
5. the complete latency, throughput, memory, and failure benchmark matrix.

Only after those gates pass should the B12X dependency change be proposed
upstream.

### Route B: use an exact vLLM/CUTLASS DSL 4.6.2 source build

The parent of vLLM's 4.7.1 bump is
`31759ccb6d560e150849273b47e15680a0120169`. This route is valid only if its
entire native stack is built from that exact revision for ARM64; an unverified
nightly tag or a Python overlay on another image is not acceptable. It is a
fallback because it deliberately trails the selected upstream revision.

## Invalid shortcuts

- Installing B12X with `--no-deps` and declaring the combination supported.
- Downgrading CUTLASS DSL inside the published vLLM nightly.
- Overlaying current vLLM Python on native extensions from an older commit.
- Promoting the upstream-native FlashInfer result as equivalent to the B12X
  performance path without measuring both.

## Qualification order

1. Build and smoke-test the upstream-native vLLM revision without replacing
   the live service. This validates the model, virtual-head carry, and new KV
   allocator independently of B12X. Explicitly select the native FlashInfer
   DS4.1 backend; do not call this a B12X result.
2. Port the R38 DS4.1 B12X and RoCEnante behavior onto current vLLM interfaces
   as separate commits, with selection tests that fail if either silently falls
   back. Do not cherry-pick the old model stack.
3. Complete Route A in an isolated image and compile all kernels before any
   service switch.
4. During a maintenance window, compare the promoted runtime, upstream-native
   image, and B12X image with identical launch configuration and prompts.
5. Promote only the best quality-preserving candidate, then update the upstream
   locks and baseline receipt.
