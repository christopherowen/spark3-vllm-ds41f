# Upstream-main rebase

## Hypothesis

Rebuilding the serving stack from current vLLM main and B12X's default
`master` branch, then carrying
only the still-required local behavior, will remove accumulated release-branch
debt and materially increase usable KV capacity without changing model output.

This is a source-preparation experiment. It does not authorize replacing or
restarting the running service.

## Intended delta

The candidate uses the exact vLLM source revision embedded in the current ARM64
nightly image and the current B12X default branch, rather than adding more
patches to the divergent DS4.1 release snapshots.

- vLLM base: `d05da62e9ccdf8e342b15bf6785d83224cc165af`.
- B12X base: `0f3a8cbfd1c11d27f04e3ab37a802d522f4f1c68`.
- vLLM carries: quality-preserving 64/8 to 72/9 virtual attention geometry for
  TP3, a native-weight DS4.1 B12X sparse-MLA adapter, and a current-interface
  RoCEnante collective adapter; generic loader-owned checkpoint routing; and
  an upstream-model disk Engram boundary that stages only requested native rows;
  plus TP-aware vocabulary padding for non-power-of-two tensor parallel sizes.
  The B12X warmup-provider reference is deliberately non-owning so it cannot
  create a cycle in PyTorch's child-module graph.
- B12X carries: the qualified 3072-token mHC policy, per-peer/HCA RoCEnante
  routing for the switchless ring, explicit CUTLASS DSL 4.7.1 alignment, and a
  fail-closed managed-memory loader adapter for current upstream vLLM.

The candidate deliberately adopts upstream DS4.1 model support, heterogeneous
KV allocator, Engram implementation, generic B12X 1.3 integration, bounded
index reads, and TP3 24-head sparse-MLA partition without local copies of those
changes.

The missing consumer interfaces are now ported without copying the downstream
model fork. The upstream model still owns CED, the indexer, cache allocation,
the compressor, hashing, and output projection. B12X owns the sparse attention
kernel, caller-owned scratch, ordinary-weight direct I/O, and bounded Engram row
staging. RoCEnante handles eligible small TP collectives while NCCL remains the
deliberate fallback above the configured size limits.

An earlier source-complete image was rejected before launch when its target
probe exposed missing build-time `liburing` support. The corrected candidate
includes the managed-loader interfaces and disk-backed Engram, passes focused
GB10 runtime qualification, and has been copied byte-for-byte to all ranks over
the ConnectX-7 internal links. The promoted service remains stopped and
preserved as the rollback target while the corrected image proceeds through
full model-load and serving qualification.

The first full TP3 construction attempt then exposed one additional generic
upstream boundary before weight loading: the 129,280-token vocabulary was
padded to 64-row storage alignment but not to the TP3 world size. The isolated
fix pads storage to `lcm(64, TP size)`—129,408 rows for TP3—while preserving the
original token rows and logits. The extra 128 rows are inert padding. The exact
failure and invariant are recorded in
`runs/launch-8649ec7238e3-tp3-vocab.json`.

The vocabulary-corrected image passed TP3 construction and entered checkpoint
loading. That exposed a second isolated adapter defect: assigning the attention
module to its own warmup-provider attribute through `nn.Module.__setattr__`
registered it as its own child. The resulting graph cycle made checkpoint
loader discovery recurse indefinitely. Patch 0007 keeps the identical provider
reference through `object.__setattr__`, and a regression proves it is absent
from `_modules` and that `named_modules(remove_duplicate=False)` terminates.
The exact failure is recorded in
`runs/launch-f332c52cb106-module-cycle.json`.

The module-cycle-corrected image then initialized the complete three-rank
distributed topology and reached the first real checkpoint copy. B12X correctly
rejected `layers.0.attn.attn_sink` because that checkpoint destination had not
been allocated through the loader-owned pool. The rebased path had carried the
generic allocate/copy/flush interface but missed the model-specific allocation
sites already established for the working DS4 path. Patch 0008 applies that
explicit boundary comprehensively to DS4.1 attention, mHC, Engram, MoE routing,
DSpark, and vision weights. It deliberately does not wrap runtime buffers,
scratch, caches, staging, or outputs. The exact failure and allocation inventory
are recorded in `runs/launch-cb5fe5cf0755-weight-allocation.json`.

The allocation-corrected image then reached ModelOpt's first numerical MXFP8
scale transformation. Lazy checkpoint tensors intentionally cannot execute
`repeat_interleave`, so B12X rejected the undeclared transformation. Patch 0009
uses the generic `materialize_weight` boundary immediately before the existing
row expansion; the arithmetic, dtype, expanded values, and destination loader
remain unchanged. The reproduced failure is recorded in
`runs/launch-b6791daa67f1-mxfp8-scale-transform.json`.

The scale-corrected image passed that transform and reached the TP3 virtual-head
loader. Its custom group selection still used a raw `copy_` from lazy checkpoint
metadata. Patch 0010 materializes only the TP-local source groups through the
same generic transform boundary, caches the duplicated final group, and leaves
zero-padding groups unread. Native values and the virtual-head arithmetic are
unchanged. The exact failure is recorded in
`runs/launch-3d7121b4d687-virtual-head-materialization.json`.

The first materialization-corrected image proved that the selected lazy groups
were readable, then exposed a composition error: the wrapper copied a raw
160-row scale group directly into its final 5120-row destination and therefore
bypassed ModelOpt's 32-row expansion. Patch 0011 delegates every selected group
through its original loader using a temporary final-destination view. This
retains all ModelOpt transforms, prevents a second TP shard, reuses a completed
group for virtual duplicates, and still avoids reading virtual zero groups. The
exact failure is recorded in
`runs/launch-5e87113c2318-virtual-head-loader-composition.json`.

The composition-corrected image then passed that boundary and reached the two
nonresident disk-Engram checkpoint descriptors. The rebase had ported both the
file-backed checkpoint filter and leaf `DiskTable` loader but omitted the
working branch's parent-to-leaf handoff. Patch 0012 restores that proven route
for disk-backed Engram only, so neither native table is mistaken for a resident
parameter or allocated in memory. A composed regression enters through
`DeepseekV4Model.load_weights`, and an inventory proves these are the only
nonresident DS4.1 checkpoint weights. The reproduced failure is recorded in
`runs/launch-289f83bc7e59-disk-engram-parent-routing.json`.

The parent-route-corrected image proved that handoff executes, then exposed the
preceding wrapper boundary: even with image support disabled, the checkpoint
advertises the VL architecture, so the default loader asks the outer model—not
the inner language model—which raw tensors must remain file-backed. Patch 0013
restores the working wrapper delegation. Its regression begins with the raw
checkpoint name and verifies the unchanged file descriptor through the outer
mapper, inner loader, backbone, and Engram leaf. The exact failure is recorded
in `runs/launch-ad8a2eb74d80-outer-vl-engram-filter.json`.

The outer-filter-corrected image then preserved the disk Engram selection and
entered the generic file-backed metadata reader. That reader was introduced by
this rebase series and reused safetensors' Python torch dtype table, which can
trail native PyTorch support and did not contain the checkpoint's `F8_E8M0`
MXFP8 scale dtype. Patch 0014 extends only the descriptor dtype map with
PyTorch's native `float8_e8m0fnu`; tensor payloads remain lazy and numerical
values are unchanged. A real-header regression and all 20 focused loader and
DS4.1 tests pass on GB10. The exact failure is recorded in
`runs/launch-f829b45a514d-mxfp8-metadata.json`.

The first image containing patch 0014 was rejected before distribution because
the Dockerfile still supplied the preceding vLLM and B12X identities as default
OCI labels. Its source trees were correct, but its provenance was not. The
build script now passes every verified base, patch head, and tree explicitly;
the Dockerfile defaults to `unrecorded` so a future omission cannot silently
claim an older source identity. The rejection is recorded in
`runs/build-2326847a0a02-rejected-provenance.json`.

The correctly labelled patch-0014 image passed all earlier boundaries and
completed distributed checkpoint loading. It then exposed a rebase error in
the carried MoE adapter: vLLM still called B12X's preceding `plan_weights`
interface, while the selected B12X revision requires its structured weight,
capacity, routing, and preparation APIs. Patch 0015 ports that boundary as one
unit and prepares retained MoE plans before vLLM profiles memory, so compiled
program state cannot silently consume an already-admitted KV allocation. On a
GB10 it passes real MXFP4 W4A16 and W4A8 numerical execution, repeated
preparation and execution across 1/4/16-token capacity, and CUDA-graph replay
in both modes. The full-model failure is recorded in
`runs/launch-df249c44bfd5-b12x-moe-preparation-api.json`.

The patch-0015 image completed checkpoint loading and the corrected MoE weight
boundary, then exposed an ordering defect in that patch's new pre-memory hook.
It constructed every ordinary B12X warmup unit before filtering the returned
units for MoE, which caused sparse attention to inspect placeholder cache
storage before KV-cache finalization. Patch 0016 replaces the returned-unit flag
with a dedicated provider method implemented only by B12X MoE. Attention,
Engram, linear, and collective providers are now structurally ineligible for
pre-memory enumeration. The failure is recorded in
`runs/launch-bdc1deee14da-preprofile-provider-order.json`.

The patch-0016 image then cleared both preparation failures, completed model
and draft loading at 98.1 GiB per rank, and prepared all 40 MoE plans. Memory
profiling exposed one more stale boundary: the ModelOpt MXFP8 linear adapter
still used B12X's preceding first-use interface without a prepared plan. Patch
0017 declares the bounded serving capacity plus exact graph-capture regimes,
prepares and accounts their retained memory before KV admission, and requires
that plan during execution. Its real GB10 regression executes the vLLM adapter
at 1/4/16 tokens and exactly matches the dequantized MXFP8 reference. The
failure is recorded in
`runs/launch-7c12c5a6f9a3-mxfp8-preparation-api.json`.

The immutable patch-0017 image is
`sha256:209944e6423216643cd99780e08194d6cbdf0b5189b88d462799a0688ba419cd`.
It reproduces all seventeen vLLM and four B12X patches from the exact bases and
passes the selected DS4.1 loader, ModelOpt MXFP8, disk Engram, MXFP4 MoE,
multi-capacity, CUDA-graph, Ruff, and format gates on GB10. Its receipt is
`runs/build-209944e64232-validation.json`; full TP3 startup remains a separate
required gate.

That image cleared the original MXFP8 boundary, preparing 210 target-model
linear plans and all 40 MoE plans before the real profile run. The remaining
unprepared `DSparkDeepseekV4Model.main_proj` proved the collector had omitted
the separately owned draft model. Patch 0018 makes both pre-profile and normal
warmup enumerate target plus draft roots while deduplicating any aliased layers
by their plan key. The failure is recorded in
`runs/launch-209944e64232-draft-plan-discovery.json`.

The immutable patch-0018 image is
`sha256:811f3078cc1d177b82b59fd0bcc0650e6d2a567e185efda05b84b073814c3e76`.
It reproduces all eighteen vLLM and four B12X patches from the exact bases and
passes 60 selected vLLM tests, 8 production-capability disk Engram tests, both
selected MXFP4 numerical modes, four multi-capacity/CUDA-graph cases, Ruff, and
format on GB10. Its receipt is `runs/build-811f3078cc1d-validation.json`.

That image prepared 226 MXFP8 and 43 MoE plans, then entered the real profile
run and reached an eligible DSpark embedding collective. RoCEnante rejected the
call because its plan was still scheduled only by normal post-profile warmup.
Patch 0019 opts the communicator into the explicit pre-profile provider
contract and includes eligible distributed providers in that phase. The failure
is recorded in `runs/launch-811f3078cc1d-rocenante-preparation.json`.

The resulting immutable patch-0019 image is
`sha256:553dc78794447882fac6959788124fdc2dca32d1a8a25fc959d55f6035388f6f`.
It reproduces all nineteen vLLM and four B12X patches from the exact bases and
passes 60 selected vLLM tests, all 15 RoCEnante tests, 8 production-capability
disk Engram tests, both selected MXFP4 numerical modes, four
multi-capacity/CUDA-graph cases, Ruff, and format on GB10. Its receipt is
`runs/build-553dc7879444-validation.json`; TP3 runtime remains a separate gate.

The TP3 launch then showed that distributed priming cannot run with the timing
of an ordinary local warmup unit. All ranks loaded the model within the same
second, but they reached the first RoCEnante collective seven seconds apart and
timed out at sequence 1. Patch 0020 adds a rendezvous after local plan
materialization and immediately before the priming collectives; it adds no
steady-state barrier. The failure and rejected memory-limit experiment are
recorded in `runs/launch-553dc7879444-rocenante-priming-rendezvous.json`.

The resulting immutable patch-0020 image is
`sha256:bfed12063fa09467733b58e3d131330f432f65221e7f051b9052949cab871d8f`.
It reproduces all twenty vLLM and four B12X patches from the exact bases and
passes 76 selected vLLM tests, 8 production-capability disk Engram tests, both
selected MXFP4 numerical modes, four multi-capacity/CUDA-graph cases, Ruff, and
format on GB10. Its receipt is `runs/build-bfed12063fa0-validation.json`; TP3
runtime remains a separate gate.

See [carry-matrix.md](carry-matrix.md) for the complete disposition.
[donor-analysis.md](donor-analysis.md) pins the latest known downstream R38
implementation and records how the missing behavior was negative-ported
without replacing current upstream subsystems.

## Reproduce the rebased source trees

Run the experiment-local preparer from the repository root:

```sh
experiments/2026-09-20-upstream-main-rebase/prepare-sources
```

It fetches the exact upstream commits into
`.work/experiments/2026-09-20-upstream-main-rebase/`, applies only the isolated
carry commits, and refuses the result unless each final Git tree matches the
recorded identity in `candidate.json`. It does not update promoted locks, build
an image, contact a Spark, or modify the running service.

Once a Spark is free, build the exact ARM64 candidate from those verified
trees:

```sh
experiments/2026-09-20-upstream-main-rebase/build-candidate check
experiments/2026-09-20-upstream-main-rebase/build-candidate
```

The build uses the content-addressed ARM64 manifest in `Dockerfile`, resolves
B12X dependencies normally, runs `pip check`, and refuses dirty or mismatched
source trees. Build steps use the host network so CMake's immutable dependency
fetches follow the host's network path. If GitHub throttles anonymous clones,
set `GITHUB_TOKEN`; `build-candidate` passes it through an ephemeral BuildKit
secret mount, never a build argument or image layer. It does not stop, restart,
or deploy the service. The current
patch-0018 image and checks are recorded in
`runs/build-811f3078cc1d-validation.json`. Earlier immutable receipts remain in
`runs/` as evidence for the isolated compatibility boundaries found during the
rebase.

The experiment has its own deterministic cluster configuration. It can be
inspected without changing the promoted configuration:

```sh
bin/spark3 \
  --cluster-config experiments/2026-09-20-upstream-main-rebase/cluster.json \
  cluster start
```

The configuration allowlists only the required `b12x_loader` vLLM plugin. The
separate B12X FP6 plugin depends on downstream-only interfaces and is not used
by this native FP4/FP8 model.

## Compatibility gate

Source preparation uncovered and resolved five source-level integration
boundaries. B12X now declares the same CUTLASS DSL `4.7.1` toolchain as vLLM;
DS4.1 B12X attention, MoE, and RoCEnante use current prepared-plan interfaces;
the upstream loader can route owned and file-backed tensors; and the upstream
DS4.1 model can retain Engram on immutable checkpoint storage. No candidate
image suppresses dependency conflicts, downgrades the vLLM environment, or
mixes Python with native extensions from a different vLLM revision.

[compatibility.md](compatibility.md) records the exact matrix and remaining
SM121 qualification gates. Patch 0010 and the combined DS4.1 adapter suite
passed on GB10, and full model loading then exposed its missing composition
with ModelOpt's scale transform. Patch 0011 and the exact 160-to-5120 MXFP8
scale regression pass on GB10. Patch 0012 then restores the omitted disk-Engram
parent-to-leaf handoff. Patch 0013 restores the preceding outer VL filter
delegation. Patch 0014 closes the file-descriptor dtype gap for native MXFP8
scales, and the real-header regression plus all 20 focused tests pass on GB10.
Patch 0015 closes the B12X MoE preparation-API gap and passes numerical,
multi-capacity, repeated-preparation, and CUDA-graph tests in both native model
activation modes on GB10. Patch 0016 then makes pre-memory discovery
MoE-specific rather than constructing unrelated warmup units and filtering
afterward. The candidate is not promotable until full
three-rank model-load, quality, capacity, and performance gates pass.

The patch-0020 TP3 launch synchronized RoCEnante priming but revealed a
switchless preparation mismatch: four local HCA functions are open across the
two peer links, while each peer route has two stripes. The prepared GPU plan
used four as its flag-lane count and therefore read offsets the two-lane proxy
never wrote. B12X commit `b69feee3022c` makes prepared kernel geometry use the
routed stripe count. A standalone three-rank probe proves the promoted eager
path and rebased direct path were already sound, reproduces the failure only
through `PreparationSession`, and verifies the correction against NCCL from
16 bytes through 1 MiB on every rank. The exact evidence is recorded in
`runs/launch-bfed12063fa0-rocenante-stripe-layout.json`.

The corrected immutable image then carried real model all-reduce and all-gather
tensors through RoCEnante and allocated 1,360,738 KV tokens before upstream's
adaptive-verification guard rejected the retained legacy setting. The
hard-wired DeepSeek V4.1 indexer on SM121 cannot consume device-decided query
lengths. This is a configuration carry regression, not a transport or memory
failure. The candidate now uses supported fixed-block DSpark verification;
this preserves target-model quality and leaves its concurrency-dependent
performance impact to the benchmark matrix. The failure receipt is
`runs/launch-f3de2589ce25-adaptive-verification.json`.

With adaptive verification disabled, the next upstream guard rejected the
legacy 256-token manager block: the SM121 indexer and B12X attention share a
128-token kernel page, while the required block-outer `BLHNC` layout cannot
split a padded 256-token page into dense views. `LBNHC` is not an alternative
for this model because the indexer packs its pages beside the MLA latent pages.
The candidate therefore uses the exact common 128-token block. The receipt is
`runs/launch-f3de2589ce25-kv-block-layout.json`.

That corrected launch then allocated 1,353,553 KV tokens and completed the
shared JIT registry before every rank stalled at the new upstream `ll_bf16`
router-GEMM warmup. Kernel journals show repeated NVIDIA
`NV_ERR_NO_MEMORY` allocations beginning at the same second, followed by
unclean host resets. The preceding parallel TileLang stage had already
finished, so reducing B12X compiler workers would not address the failure.

The warmup was unused work on this hardware. `GateLinear` deliberately excludes
SM121 from its specialized router kernels and dispatches the same BF16 weights
through cuBLAS with FP32 output, but the global warmup collected every BF16
gate without checking `allow_ll_bf16_gemm`. Patch 0021 makes preparation follow
the runtime eligibility decision. A clean isolated SM121 probe confirmed that
the unused five-key compile consumed 435,113,984 bytes of CUDA-visible
headroom; the regression test proves an ineligible gate is no longer
collected. This changes neither weights nor arithmetic and removes no runtime
kernel selected by DGX Spark. The crash evidence and isolation result are in
`runs/launch-f3de2589ce25-ineligible-router-warmup.json`.

Before rebuilding patch 0021, a 196-test focused replay sweep found an
unrelated one-line NVFP4 change embedded in the original loader-routing patch:
it disabled the upstream `reorder_w13=use_a16` behavior. That line is neither
needed by the DS4.1 MXFP4 path nor part of owned/file-backed checkpoint
routing, so it was removed from patch 0004 and the complete series was replayed
from the pinned upstream base. The corrected replay head is
`9de03517c6be50eef0d20fe2d6cf02c34151a01e` with tree
`c9ce910e3709cbe2967642d9a157b402ac254698`. Historical launch receipts retain
the identities of the images they actually exercised.

The corrected replay was built as immutable image
`sha256:56820972a862a084c6d0d34dd8faa5ecc8a198b1ec276cb6cfd591fa983ef993`
and copied byte-for-byte over the ConnectX-7 link to dgx2. A 32 GiB-capped
focused sweep passed 193 tests with one expected skip. The two excluded failures
are separately reproduced baselines: B12X's unselected NVFP4/BF16-SiLU cosine
threshold and a pinned-upstream Qwen ModelOpt FP8 scale-dtype test. A model-free
two-rank prepared-plan probe then matched NCCL from 16 bytes through 1 MiB,
used both dgx1-dgx2 rails, reported no transport errors, and exited without a
cgroup OOM on either node. The evidence is recorded in
`runs/build-56820972a862-two-rank-validation.json`. No DS4.1 weights were loaded;
full TP3 qualification remains gated on recovery of dgx3.

Patch 0022 replaces the remaining architecture-wide warmup guesses with
model/backend registration. Eligible router layers register their exact shape
and reachable token range; the finalized DeepSeek indexer metadata builder
registers its actual strides, verification depth, compression ratio, and PCP
alignment classes; and B12X paged attention compiles only the page size bound
to its KV cache. FlashInfer autotuning now requires an explicit capability from
a selected FlashInfer attention, linear, or MoE backend, rather than merely an
installed package on a supported GPU. The replay head is
`6964357a771af30c341a008813ca7bba8810b2e4` with tree
`f7e10aa21545cf60f41d72a2d39a8b0913efcdf2`. A 32 GiB-capped, model-free GB10
run passed all 55 focused registration and geometry tests. This source revision
is prepared for a new immutable image; it has not replaced or restarted the
stopped patch-0021 candidate.

The resulting immutable image is
`sha256:0be891cb05b710b4b4cd67a057b1fd4d4dd41967006c654da80d354d71daa3df`.
Its OCI labels exactly match deployment revision `441d0486e246`, vLLM head
`6964357a771a` and tree `f7e10aa21545`, and the retained B12X identities. The
same 55 focused tests pass from the image itself under a 32 GiB memory cap. It
has been copied uncompressed over dgx1's ConnectX-7 links, and all three nodes
report the same image ID with zero swap use and no running containers. The
validation receipt is `runs/build-0be891cb05b7-validation.json`. Full TP3 model
load and selected-key evidence remain required before promotion.

The cold TP3 qualification rejected that image. All ranks loaded the native
target plus DSpark weights at 98.1 GiB and prepared the selected B12X plans,
then the model-owned JIT registry compiled 37 DeepSeek V4.1 indexer kernels
with only 1.38--1.57 GiB host headroom. The same 301 Triton cache files and
content digest appeared on every rank at the NVIDIA `NV_ERR_NO_MEMORY`
timestamp, followed by host recovery reboots. Patch 0022 therefore identifies
the right kernels but compiles them in the wrong memory phase. It is not
promotable; the next candidate must prime those configuration-derived kernels
before weight residency and treat any resident-model cache miss as a failure.
The evidence is recorded in
`runs/launch-0be891cb05b7-indexer-jit-memory.json`.

A controlled retry retained the exact image and configuration but reused the
compiler caches written by the cold attempt. Checkpoint loading left 16--17 GiB
available, proving the 0.7 GiB model-residency increase over the promoted image
is not the dominant regression. B12X preparation/profile work nevertheless
reached 0.80--1.68 GiB and emitted driver allocation failures before reclaiming
to about 10 GiB. The eager registry then processed 61 keys, beginning with a
four-worker TileLang compile of 15 mHC variants, and drove dgx3 to zero available
memory. The retry is rejected and recorded in
`runs/launch-0be891cb05b7-warm-cache-parallel-jit-memory.json`.

The deferred-JIT recovery launch also failed and is rejected. It omitted the
61-key registry sweep, but the required profile forward pass then compiled the
same model-selected TileLang mHC kernels after 98.1 GiB of weights and all B12X
plans were resident. B12X preparation itself fell to 1.01--1.72 GiB available
and emitted driver allocation failures before reclaiming. The runtime reached a
1,353,553-token KV allocation (8.46 complete 160K windows), but all three control
paths subsequently became unavailable before API readiness. The preserved
container evidence is in
`runs/launch-0be891cb05b7-deferred-first-use-memory.json`; previous-boot kernel
journals are unavailable, so the final control loss is not attributed more
narrowly.

Patch 0023 corrects both identified startup phases without changing model
arithmetic. The active model/backend JIT registry is drained immediately after
model construction and before checkpoint payloads become resident. It also
fixes the concrete B12X preparation-lifetime regression found in the rebased
source: discovery constructed all 226 MXFP8 units before compiling the first,
and each unit retained serving-size dummy activations in its request closures.
Discovery and compilation now proceed one unit at a time, MXFP8 trial
activations are allocated only inside the active plan's preparation, and
allocator scratch is reclaimed before the next plan. Its replay head is
`c6d531aa609f83496a8471d157ef32e62895bd4b` with tree
`ab564f8dd3477f70b50b970e07250ff64256c2ac`. Ruff, formatting, Python
compilation, and patch whitespace checks pass. The fresh deterministic replay
was built as immutable image
`sha256:9076b0bd9fee09d1dda4958667ba4fa29dea3a9251f0ba682fd616e6db058dd3`.
Its OCI labels exactly match deployment revision `397d35a4e051`, the recorded
vLLM and B12X heads, and both source trees. The finished image passed all 55
focused registration, ordering, incremental-preparation, and activation-lifetime
tests on dgx1 under a 32 GiB memory/swap cap with networking disabled. No model
weights or serving process were started. The receipt is
`runs/build-9076b0bd9fee-validation.json`.

Cluster startup is now fail-closed as well: a protected startup host-memory
guard confirms its first sample before each rank is
allowed to launch, API readiness requires that guard to remain active, and only
a healthy cluster switches to the 3 GiB steady guard. A candidate that repeats
either observed trough is killed locally before it can exhaust driver and
management headroom. The repository default is 5 GiB at quarter-second
intervals; this first qualification deliberately uses 10 GiB at 100 ms. The
kill path has been exercised on all three nodes with a GPU-free 64 MiB smoke
container and left no residual units or containers. The promoted configuration
remains launch-locked. After the image was copied
byte-for-byte to every rank and the owner authorized hardware qualification,
the dedicated candidate configuration was opened for a guarded cold-cache and
warm-cache TP3 startup pair. The first cold attempt used the new
`patch0023-9076b0bd9fee` cache namespace, a 108 GiB container cap, a protected
10 GiB startup guard sampled every 100 ms, and CUDA graphs disabled. All ranks
completed the 24-key pre-weight JIT, but dgx2 and dgx3 crossed that intentionally
conservative guard immediately after weight loading. The guard stopped them
with explicit SIGKILLs, not cgroup OOM kills; coordinated rollback succeeded,
all hosts stayed reachable, and no current-boot `NV_ERR_NO_MEMORY` was present.
That incomplete cache is preserved as evidence and is not eligible for a warm
test. The receipt is `runs/launch-9076b0bd9fee-cold-r1-guarded.json`.

The second cold attempt used a fresh `patch0023-9076b0bd9fee-r2` namespace and
an 8 GiB startup guard sampled every 100 ms. All ranks completed JIT, checkpoint
loading, draft-model loading, and preparation of 226 MXFP8, 43 MoE, and one
RoCEnante plan. The head then reached live RoCEnante dispatch while compiling
the required MHC post kernel, and the guard stopped it at 8,381,748 KiB
available. It was again an explicit SIGKILL with no cgroup OOM, NVIDIA allocation
error, or host loss. That incomplete cache is also preserved only as evidence;
the receipt is `runs/launch-9076b0bd9fee-cold-r2-guarded.json`.

The third cold attempt used a fresh `patch0023-9076b0bd9fee-r3` namespace and a
6 GiB startup guard sampled every 100 ms. It passed the second attempt's limit,
made both RoCEnante collectives live, allocated the fixed 3 GiB KV cache with
1,353,553-token capacity, skipped dynamic memory profiling, and completed the
model-selected indexer JIT warmup. The head then crossed the guard at 6,283,992
KiB available before API readiness. It was again an explicit SIGKILL with no
cgroup OOM, NVIDIA allocation error, or host loss. That incomplete cache remains
evidence only; the receipt is `runs/launch-9076b0bd9fee-cold-r3-guarded.json`.

The fourth cold attempt used a fresh `patch0023-9076b0bd9fee-r4` namespace and
the repository's normal 5 GiB startup reserve, sampled at the stricter 100 ms
cadence. This still preserves approximately 3.3--4 GiB more host/driver headroom
than the 1.01--1.72 GiB trough associated with the previous control-plane loss.
It passed every memory boundary with a 5,961,904 KiB telemetry minimum on dgx1,
allocated the full 1,353,553-token KV cache, and completed the final indexer JIT
warmup. Profile execution then failed deterministically because upstream's fused
DeepSeek V4 QNorm/RoPE insertion kernel omitted a compiled instantiation for the
24 padded Q heads produced by quality-preserving TP3 virtual-head geometry. The
container exited 1 rather than being killed; no cgroup OOM, NVIDIA allocation
error, or host loss occurred. The receipt is
`runs/launch-9076b0bd9fee-cold-r4-tp3-qhead.json`.

Patch 0024 adds only that existing template instantiation and extends the native
kernel matrix with the exact 24-head shape. It changes no data type, layout,
padding, or model arithmetic and is suitable for upstream submission. The
candidate launch is locked until a fresh deterministic replay, immutable image
build, native regression, and model-free qualification complete. That image
will receive a new cold-cache namespace; the complete 925-file patch-0023 cache
is evidence, not an input to the patch-0024 cold test.

The patch-0025 image then completed fresh compilation, loaded the 98.1 GiB
target-plus-draft model, prepared all selected B12X plans, allocated the fixed
3 GiB KV cache, and passed the former SWA-only failure. vLLM's runtime warmup
exposed a separate adapter capacity error: its eight synthetic DSpark prompts
contain five rows each, but the decode plan used the eight-request count as
B12X's row-validated `max_batch` and covered only the 32-row serving-decode
shape. Patch 0026 sizes that plan for the 40-row warmup and passes the actual
row capacity to B12X. The change adds only 11.123 MiB of decode scratch per
rank (44.491 MiB to 55.614 MiB) and has model-free coverage for the warmup
bound, graph-capacity precedence, and B12X caps.

That run also recorded one non-fatal `NV_ERR_NO_MEMORY` on dgx2 while the fixed
3 GiB KV cache was being reserved. The 5 GiB guard still preserved every host,
but a first startup of a new image must not repeat that allocation boundary.
Patch 0026 qualification therefore uses a temporary 2 GiB KV cache: it still
provides roughly 902,000 KV tokens (enough for four 128K windows), while
restoring about 1 GiB of host/driver headroom. The 3 GiB production target is a
separate post-readiness capacity step, never part of the cold-compilation test.
The exact failure evidence is retained in
`runs/launch-9e1e10e5a428-cold-specdecode-scratch.json`.

The patch-0026 cold run then passed that 40-row speculative warmup and all
earlier integration boundaries, including the reduced 2 GiB KV allocation. It
failed closed at an independently reported upstream SM12x defect: ratio-1
DeepSeek V4.1 indexer pages contained 128 states, while DeepGEMM's arch-12
paged-MQA logits kernel accepts 64. This is vLLM issue #56461, not a B12X
scratch regression. Patch 0027 follows the documented heterogeneous fix rather
than globally shrinking the cache: ratio-1 indexer pages use 64 tokens and
ratio-2 pages use 128 tokens, so both contain exactly 64 states; the main MLA
cache retains its own 128-token page. The focused regression and 21 other
indexer tests pass in the pinned image (one unrelated Hugging Face lookup test
is unavailable under the deliberately offline test container). The cold-run
telemetry, failure, and rollback are recorded in
`runs/launch-4c1660dc825e-cold-indexer-page.json`.

Because the official base image provides precompiled vLLM native extensions,
the candidate image now rebuilds only `_C_stable_libtorch` from the recorded
patched source for SM121. The build stage is bounded to two compile jobs and one
NVCC thread, installs only that component into the source overlay, and rejects
the image unless the resulting binary contains the 24-head dispatcher. All
other native libraries continue to come byte-for-byte from the pinned base.

The startup pair is ordered and fail-closed:

1. prove the qualification cache path is absent on all three nodes;
2. start once from that cold state and require API readiness with every startup
   guard still active;
3. stop only the exact candidate containers, retaining the newly populated
   cache and complete logs;
4. prove the cache is non-empty on every rank, then start the identical image
   and configuration again; and
5. compare cold/warm elapsed time, compile/cache messages, minimum available
   memory, container exits, driver errors, and API readiness before changing any
   performance setting.

Each rank runs `startup-telemetry` (in this directory) as a protected transient service for
the duration of a startup. It samples host `MemAvailable`, swap use, and exact
container state twice per second into the ignored qualification directory, then
reports cache file count and bytes without placing mutable runtime artifacts in
Git. Summary parsing rejects incomplete rows so an interrupted `docker inspect`
cannot be mistaken for a zero-memory sample.

Patch 0027 completed both guarded startup tests. The empty-cache run reached API
readiness in 288.798 seconds and the populated-cache run in 199.406 seconds, a
31.0% improvement. Warm reuse reduced reported model loading from 187.542 to
135.616 seconds and engine initialization from 36.01 to 13.68 seconds, but the
pre-weight JIT phase changed only from 34.86 to 31.84 seconds because TileLang
still invoked the MHC compilation path. Every rank stayed reachable, exited
cleanly, reported no OOM kill or `NV_ERR_NO_MEMORY`, and retained complete
telemetry and container logs. The 5 GiB startup guard was material: dgx1 reached
5,919,072 KiB available during the cold run.

This candidate is not qualified for inference or promotion. Temperature-zero
chat smokes on both cold and warm starts returned garbled and repetitive tokens
despite valid HTTP responses and usage accounting. The temporary 2 GiB safety
cache exposed 520,817 tokens, or 3.26 simultaneous 160K requests. That is
nominally enough for four 128K windows with 8,817 tokens (1.7%) of aggregate
slack, but the four-way long-context execution gate remains deferred until
generation correctness passes. The launch is locked while correctness is
isolated between the target-only and DSpark paths. Full evidence is in
`runs/launch-64e2b8cf5925-cold-warm.json`.

Layerwise finite-value probes then isolated the first corruption to layer 0's
SWA attention input cache. Query projection, normalization, RoPE preparation,
and the checkpoint values remained finite; both the B12X kernel and an
independent reference decoder returned NaNs from the live cache. The two
projects used incompatible byte layouts with an identical page size: vLLM's
V4.1 writers place every token payload before the page's scale plane, while
B12X reads and writes one contiguous 528-byte MXFP8 or 288-byte NVFP4 record
per token. Shape validation therefore could not detect the mismatch. Patch
0028 routes both cache kinds through B12X's existing prepared writers and adds
backend regressions proving that RoPE is applied before publication, the
compressor's concurrent indexer input is not mutated, and the selected cache
is flattened only at the explicit B12X boundary. The B12X writer's real SM121
dynamic-row and CUDA-graph tests pass for both record types.

The first rebuilt image reached guarded API readiness on all three ranks, then
the first five-token request failed before attention: the replacement SWA
writer passed a two-dimensional latent row beside three-dimensional Q to the
DeepSeek rotary module. That module broadcasts over an explicit head axis.
Patch 0028 now exposes the latent as a single replicated head for RoPE and
removes that axis before the B12X write; its unit double rejects the previously
accepted shape. Image `b9c4e126` is recorded as failed, launch is locked, and a
replacement image (`5627de11`) has passed the focused runtime regressions and
is present identically on all three nodes. Its guarded semantic smoke remains
required before promotion.

## Quality and safety gates

The V2 runner initializes disk Engram state, resolving the first NaN observed
with the V1 runner. Subsequent target-only requests were finite but garbled.
The full live-cache reference isolated a large layer-2 indexed-attention value;
the compressed cache writer itself stored bounded records. The B12X adapter
was mapping indexed top-k positions through the independent SWA block table.

The q12 probe first failed on its own assertion for a shared-cache consumer;
it is recorded in `runs/launch-v2-blockfix-q12.json`. The corrected q13 probe
substituted the compressed-cache metadata table while preserving the patch-0028
image. SWA and compressed block IDs were 3 and 7, respectively. Three short
deterministic prompts returned coherent answers, and layer-2 native attention
agreed with the full live-cache reference within BF16 error on all three ranks.
All q13 containers stopped cleanly without OOM; the receipt is
`runs/launch-v2-blockfix-q13.json`.

Patch 0029 puts that table lookup in the B12X adapter for indexed decode and
prefill, including layers that read another layer's compressed cache. A fresh
patch replay, source identity check, syntax check, and diff check pass. Image
`470bd39a` built successfully and has the same ID on all three nodes. The new
mapping regression and both cache-writer regressions pass in that image. The
first unauthenticated build attempt failed fetching the MSA dependency; its
receipt is `runs/build-patch0029-first-attempt.json`. The authenticated build
receipt is `runs/build-470bd39a205c-validation.json`.

The clean patch-0029 image is now running on all three DGX nodes with V2 model
runner and active 3 GiB steady memory guards. The ordinary completion and chat
APIs returned coherent arithmetic and factual answers; four concurrent short
chat requests all returned the expected numbers, and a 4,827-token retrieval
request returned its passphrase. No diagnostic probe was mounted. All containers
remained running without OOM and with 8.8–10.0 GiB available host memory after
the smoke. Details are in `runs/launch-v2-patch0029-r1.json`. This is a working
target-only candidate, not a promoted baseline; the long-context, CUDA graph,
performance, and failure gates below remain open.

On 2026-09-23, upstream `vllm/main` was at `e581e14002a0` (146 commits after
this candidate's `d05da62e9ccd` base), and `b12x/master` was at `4f3028b19c1d`
(18 commits after `0f3a8cbfd1c`). A dry patch replay against those heads
conflicted in vLLM patch 0001 at `deepseek_v41/attention.py` and B12X patch 0002
in RoCEnante source. Moving this working candidate to those newer heads needs a
separate patch-by-patch rebase, removal of changes now provided upstream, a
matching ARM64 base/native ABI build, and the same guarded runtime gates. The
current running image retains the September 20 source pins.

- Keep the stopped promoted containers intact until the candidate passes its
  gates, so rollback remains a coordinated start of known state.
- Native checkpoint weights and arithmetic must remain unchanged.
- TP3 virtual groups must contribute exactly zero at the output projection.
- A candidate image must use a base image whose native extension ABI was built
  from the same vLLM revision as the overlaid Python source.
- vLLM and B12X must share one explicitly qualified CUTLASS DSL toolchain;
  dependency errors may not be bypassed.
- A DS4.1 B12X selection must fail closed until a current-interface adapter is
  installed; silently falling back to FlashInfer is not an acceptable B12X
  qualification result.
- RoCEnante routing must be invoked by a separately tested vLLM collective
  adapter; a passing B12X topology unit test alone does not prove that the
  serving process uses it.
- All three ranks must use one content-addressed image.
- A protected startup memguard must confirm its first memory sample before
  `docker run` and remain active through API readiness; an inactive guard or
  less than 5 GiB MemAvailable kills and rejects the candidate rather than
  waiting for driver allocation failure.
- Before promotion, pass model-load, deterministic-output, long-context,
  CUDA-graph, RoCEnante, minimum-memory, and request-failure gates.
- Preserve the current image and launch as the rollback target.

## Workloads

Source preparation ran no serving workload. Maintenance-window qualification
uses `benchmark_serving.py` for identical temperature-zero prose and code
requests at concurrency 1, 2, 4, and 8. Raw request-level receipts are retained
under `runs/`; failed requests are never discarded. The complete qualification
also includes the 4-by-128K target and the normal eight-way Strix workload
against both the promoted baseline and this candidate.

## Acceptance criteria

- Exact native-weight output parity on fixed prompts and tokens.
- At least four simultaneous 128K windows with the configured cache, plus the
  normal eight-request Strix workload without admission failures.
- No statistically meaningful regression in single-stream decode TPS,
  aggregate eight-stream TPS, warm/cold TTFT, or minimum available host memory.
- No request failures, OOMs, silent truncation, or RoCE/NCCL collective errors.
- Every retained local commit is independently reviewable and has an explicit
  upstream or downstream-only disposition.
