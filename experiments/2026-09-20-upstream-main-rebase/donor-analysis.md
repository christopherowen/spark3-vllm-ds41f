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

### B12X MoE preparation

The initial negative port covered the current sparse-MLA and collective plan
interfaces but incorrectly assumed canonical vLLM's MoE adapter was already
compatible. It was not: the selected B12X revision replaced the old keyword
planner with `PackedSource`, `ActivationSpec`, `MoEGeometry`, `PackedWeights`,
`ExecutionCapacity`, `RoutingSpec`, and preparation sessions. Candidate commit
`d7e22286ffc9299800e4e448ee525a0b5259f738` ports that boundary without copying
the donor model. It also moves retained-plan preparation ahead of KV memory
profiling, while the existing later warmup continues to cover ordinary kernels
and graph shapes. Follow-up `8051ebbaadb9dfc2bb6f60554dfdaf8eede22326`
gives that early phase a dedicated provider method instead of constructing all
warmup units and filtering them afterward; this prevents sparse attention from
being touched before KV-cache storage exists.

The next launch proved the same API review had missed the ordinary ModelOpt
MXFP8 linear adapter. Current B12X requires a prepared blockscaled plan, while
the carried adapter still requested first-use compilation through the removed
`expected_m` interface. Candidate commit
`0974d6ec70a94db3483fd19b678437403d469319` declares one bounded capacity plus
the exact CUDA-graph token regimes, prepares their retained state before KV
profiling, and executes only through the retained plan. This is a narrow
vLLM/B12X integration change rather than donor model code.

The following profile proved the plan collector itself was target-only. The
DSpark draft model is a separate vLLM ownership root, so its `main_proj` did not
appear in `worker.get_model().modules()`. Candidate commit
`fcdb8715837314ee899308154af380e064a6345d` scans the canonical target and draft
accessors and deduplicates aliases by plan key. This is lifecycle integration
for current upstream vLLM, not a donor-specific model fork.

Once the draft plans were present, profiling reached a small collective and
showed that RoCEnante's prepared plan was still scheduled only in normal
post-profile warmup. Candidate commit
`96dc80400edf18ca97ad40d99ddb7ac620af3c33` opts the communicator into the same
explicit early contract. It does not broaden early enumeration to attention or
Engram and does not alter RoCEnante routing or collective numerics.

### RoCEnante

The prepared B12X per-peer/HCA commit only supplies transport behavior. The
separate vLLM carry `d6b40631cad4bf406f25093f56436f2b8ef10800` connects it to
current collective interfaces, bounds the message sizes for which it wins, and
retains NCCL for larger collectives. Target logs and probes must still prove
which transport handled each collective; environment variables alone are not
evidence.

### Loader-owned custom parameters

The generic allocation/copy/flush boundary is insufficient unless every
checkpoint-owned parameter created directly by a model opts into it. The
working donor established this pattern for DS4 in commit `6daf00361e`, but the
newer DS4.1 model adds its own attention, mHC, Engram, vision, and DSpark
parameters. Candidate commit `e08ab6356e93f9cfc2087683b441c52997b5300e`
ports the allocation consumers without moving runtime buffers into the weight
pool. This split is intentional: the generic transport remains independently
upstreamable, while model owners can review their explicit weight factories.

The next load boundary is loader-neutral rather than DS4.1-specific. ModelOpt's
MXFP8 scale loader numerically repeats checkpoint rows when `block_rows > 1`.
Candidate commit `0ede60db3a362f7679419e40bfb3c94756e92ee7` calls the existing
`materialize_weight` interface before that transform. It preserves the exact
row expansion and keeps B12X's immutable lazy source fail-closed for undeclared
transformations.

The TP3 virtual-head loader has the same loader-neutral requirement for its
custom group selection. Candidate commit
`18088aecd7bd1173217d9db31d0924f8c4beb832` materializes only the source groups
owned by the local TP rank, retains one owned copy per distinct group, and does
not read groups whose destination is defined as zero. This preserves the native
virtual-head mapping without falling back to eager checkpoint loading.
Target loading then showed that group selection must compose with ModelOpt's
scale expansion rather than replace the registered loader. Follow-up commit
`164c5d5bde7869bff5f2004f01f9f8e432d63d59` passes each selected source group
through that original loader while temporarily exposing only its final local
destination. The original numerical transform remains authoritative and the
group is not tensor-parallel sharded a second time.

Disk-backed Engram also needs the parent-loader handoff already present in the
working donor. The initial port carried the raw checkpoint filter and leaf
`DiskTable` loader but omitted the branch in `DeepseekV4Model.load_weights`
that resolves `engram.embed_tokens` and delegates its descriptors. Candidate
commit `d878afb20dc3aa1d8f68a3a2ed94d1158520cc1a` restores that path only when
Engram is disk-backed. A composed regression enters through the parent loader,
and an inventory confirms the Engram weight and scale are the only DS4.1
checkpoint tensors represented by deliberately absent parameters.
The configured checkpoint advertises the multimodal architecture even in
language-only mode, so the default loader queries the outer VL model for this
filter. Follow-up commit `a3964182327ec2b9a06146f1f2a49f1dd942eb51`
restores the donor's delegation to the inner language model. Its regression
starts with the raw checkpoint name and passes the immutable descriptor through
the real outer mapper, inner loader, backbone, and leaf module.

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
- vLLM: explicit loader allocation for DS4.1 custom checkpoint parameters,
  independently from the generic loader transport.
- vLLM: materialize lazy MXFP8 scale tensors before ModelOpt row expansion.
- vLLM: materialize selected lazy TP3 virtual-head groups without reading the
  full unsharded projection.
- B12X: the 3072-token mHC tuning point only after it is reprofiled on current
  kernels and CUTLASS DSL.

No upstream pull request should be opened from this experiment until the
target-hardware evidence is recorded and the commits are rebased onto the then
current project heads.
