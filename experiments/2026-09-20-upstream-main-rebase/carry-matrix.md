# Carry matrix

| Behavior | Main status | Candidate disposition | Upstream disposition |
|---|---|---|---|
| Native DeepSeek V4.1 model and CED path | Implemented in vLLM main | Adopt | Drop local copy |
| Heterogeneous KV-cache grouping | Implemented in vLLM main | Adopt; predicts about 1.35M tokens from 3 GiB | Drop local allocator patch |
| DS4.1 TP3 72-head/9-group geometry | Not implemented in vLLM main | Carry as isolated `277171c959846de8870732e96d23276575e88ae5` | Candidate for vLLM after target validation |
| DS4.1 B12X sparse-MLA model adapter | Generic B12X exists, but DS4.1 selection has no B12X case | Carried as isolated `fa075bc82948914cb778a33822eea3a5d537202d`; upstream model semantics remain authoritative | Candidate for vLLM and B12X after parity and performance tests |
| 24-head-per-rank sparse MLA partition | Implemented at B12X `master` revision `0f3a8cbf` | Adopt | Drop local workaround |
| Bounded compressed-MLA prefill indices | Implemented at B12X `3ab21b3d` | Adopt | Drop local patch 0001 |
| 3072-token DS4.1 mHC winner | Old policy did not survive the plan/tuning refactor | Port as isolated `348145ea8b6adaea7c32b1b87898ff59b5d43380` | Candidate after current-main GPU requalification |
| Switchless peer/HCA RoCEnante routing | Not in B12X `master` | Carry as isolated `803d7d795d05be3902a30e82743a817a2c26d53f` | Candidate for B12X after three-node tests |
| vLLM RoCEnante collective adapter | Not in canonical vLLM main | Carried as isolated `d6b40631cad4bf406f25093f56436f2b8ef10800`; prepared at startup and captured with the TP graph | Candidate for vLLM; eligible small collectives use RoCE and larger ones remain on NCCL |
| CUTLASS DSL toolchain | vLLM pins 4.7.1; B12X upstream pins 4.6.2 | B12X carry `de8e7fa971eb7ae4c21e634a8931407a0b404568` aligns all declared packages to 4.7.1 | Candidate for B12X after SM121 compile and performance qualification |
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

The source commits in this table now form a complete build candidate. Each
performance overlay remains a separate, reviewable commit, and none replaces
the upstream DS4.1 model or allocator. Target compilation, parity, memory, and
performance measurements remain promotion gates.
