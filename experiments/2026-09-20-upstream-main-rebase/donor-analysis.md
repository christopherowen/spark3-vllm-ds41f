# R38 donor analysis

## Boundary

The exact Local Inference Lab R38 sources are pinned in `donor.json`. They are
behavioral donors, not another base. The R38 vLLM tree and canonical vLLM have
diverged substantially since their common ancestor; replaying the R38 stack
would discard current upstream DS4.1, allocator, CED, Engram, preparation, and
kernel-selection work.

The B12X R38 revision is already an ancestor of the selected B12X `master`
base. Its kernel work should be consumed through current B12X APIs rather than
copied.

## Missing behavior to negative-port

### DS4.1 B12X sparse MLA

R38 contains a DS4.1-specific B12X attention selection, backend metadata,
workspace preparation, compressed-cache handling, and selection tests.
Canonical vLLM now has a newer DS4.1 abstraction with shared cache insertion,
compression, indexer, and backend-specific subclasses. A clean port should:

1. add a B12X subclass to the current DS4.1 abstraction rather than replace the
   model or base attention class;
2. implement only current backend hooks for padded heads, sparse-MLA execution,
   and output projection;
3. use B12X's current plan/capacity then bind/run APIs and caller-owned
   workspace;
4. prepare every plan before CUDA graph capture and reject an unprepared or
   over-capacity graph instead of allocating lazily;
5. preserve upstream cache specs and heterogeneous grouping so the allocator
   gain is not lost; and
6. add a model-selection test proving `B12X` chooses this subclass and fails
   closed when it is unavailable.

The donor's entire `deepseek_v4_1` package must not be copied. That would trade
the known release-branch fork for a newer fork and make future upstream rebases
hard again.

### RoCEnante

The prepared B12X per-peer/HCA commit only supplies transport behavior. The
separate vLLM carry `d6b40631cad4bf406f25093f56436f2b8ef10800` connects it to
current collective interfaces, bounds the message sizes for which it wins, and
retains NCCL for larger collectives. Target logs and probes must still prove
which transport handled each collective; environment variables alone are not
evidence.

## Two-stage qualification

### Stage A: upstream-native

Build the exact ARM64 vLLM revision with the virtual-head commit and explicitly
use upstream FlashInfer sparse MLA plus NCCL. This isolates the current model
and heterogeneous allocator. Static accounting predicts about 1,354,538 cached
tokens from 3 GiB, or 8.46 complete 160K windows, versus 498,145 live tokens.
That is a prediction to validate, not a benchmark result.

### Stage B: performance overlay

Use the prepared current-interface B12X DS4.1 adapter, RoCEnante adapter, three
B12X carries, and CUTLASS DSL 4.7.1. Compare this image against both Stage A and
the promoted runtime with identical weights, prompts, cache budget,
concurrency, and graph shapes.

This split makes attribution possible: Stage A measures upstream architecture;
Stage B measures only the performance overlay. It also yields small commits
that can be proposed upstream independently after target validation.

## Likely upstream units

- vLLM: quality-preserving TP3 virtual-head loading, after output-parity tests.
- vLLM/B12X: a current-interface DS4.1 B12X adapter with fail-closed selection.
- B12X: switchless per-peer/HCA RoCEnante routing, after three-node fault and
  throughput tests.
- vLLM: RoCEnante collective integration, separately from the B12X transport.
- B12X: the 3072-token mHC tuning point only after it is reprofiled on current
  kernels and CUTLASS DSL.

No upstream pull request should be opened from this experiment until the
target-hardware evidence is recorded and the commits are rebased onto the then
current project heads.
