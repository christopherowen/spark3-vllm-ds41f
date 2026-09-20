# Carry matrix

| Behavior | Main status | Candidate disposition | Upstream disposition |
|---|---|---|---|
| Native DeepSeek V4.1 model and CED path | Implemented in vLLM main | Adopt | Drop local copy |
| Heterogeneous KV-cache grouping | Implemented in vLLM main | Adopt; predicts about 1.35M tokens from 3 GiB | Drop local allocator patch |
| DS4.1 TP3 72-head/9-group geometry | Not implemented in vLLM main | Carry as isolated `277171c959846de8870732e96d23276575e88ae5` | Candidate for vLLM after target validation |
| DS4.1 B12X sparse-MLA model adapter | Generic B12X exists, but DS4.1 selection has no B12X case | Port only the current-interface adapter; do not copy the R38 model stack | Candidate for vLLM and B12X after parity and performance tests |
| 24-head-per-rank sparse MLA partition | Implemented at B12X `master` revision `0f3a8cbf` | Adopt | Drop local workaround |
| Bounded compressed-MLA prefill indices | Implemented at B12X `3ab21b3d` | Adopt | Drop local patch 0001 |
| 3072-token DS4.1 mHC winner | Old policy did not survive the plan/tuning refactor | Port as isolated `348145ea8b6adaea7c32b1b87898ff59b5d43380` | Candidate after current-main GPU requalification |
| Switchless peer/HCA RoCEnante routing | Not in B12X `master` | Carry as isolated `803d7d795d05be3902a30e82743a817a2c26d53f` | Candidate for B12X after three-node tests |
| vLLM RoCEnante collective adapter | Not in canonical vLLM main | Port against current communicator interfaces after the B12X transport is qualified | Separate vLLM candidate; large collectives remain on NCCL until then |
| CUTLASS DSL toolchain | vLLM pins 4.7.1; B12X pins 4.6.2 | Block image integration until one toolchain is explicitly qualified | Prefer a focused B12X 4.7.1 migration if target tests pass |
| Indexer workspace factor override | Similar upstream PRs exist and conflict | Do not carry yet | Re-evaluate after main-native measurements |
| Bounded parallel prefill scheduler | Only on the divergent release branch | Do not mix into the base rebase | Separate experiment after base qualification |
| Disk-backed Engram table | Not supported by vLLM main DS4.1 Engram | Do not silently replace | Separate memory/TTFT experiment; upstream design needed |
| CUDA graph and DS4.1 hot-path release commits | Main has newer, independently refactored implementations | Drop by default | Reintroduce only from profiler evidence |

## Why this is a rebase rather than another patch stack

The live release revision diverged before vLLM's native DS4.1 model, allocator,
Engram configuration, and current B12X interfaces landed. Replaying its large
commits would restore obsolete subsystems and hide which behavior is still
needed. The candidate begins at upstream main and gives each remaining delta
one commit, one test surface, and one upstream decision.

The vLLM base is the newest commit with a published ARM64 nightly at the time of
preparation. Upstream main was nine commits newer, but using it over that image
would mix Python source with native extensions built from another revision. The
candidate should advance to a newer main only with a matching nightly or a full
source build.

The source commits in this table are intentionally insufficient to produce a
B12X/RoCEnante serving image. That is a useful gate: the upstream-native image
can first measure allocator and model improvements independently, while the
performance overlay remains a separate, reviewable series.
