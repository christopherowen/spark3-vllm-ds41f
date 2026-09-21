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
source trees. It does not stop, restart, or deploy the service. The current
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

## Quality and safety gates

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
