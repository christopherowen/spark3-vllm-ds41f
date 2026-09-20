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
  RoCEnante collective adapter.
- B12X carries: the qualified 3072-token mHC policy, per-peer/HCA RoCEnante
  routing for the switchless ring, and explicit CUTLASS DSL 4.7.1 alignment.

The candidate deliberately adopts upstream DS4.1 model support, heterogeneous
KV allocator, Engram implementation, generic B12X 1.3 integration, bounded
index reads, and TP3 24-head sparse-MLA partition without local copies of those
changes.

The missing consumer interfaces are now ported without copying the downstream
model fork. The upstream model still owns CED, the indexer, cache allocation,
the compressor, and output projection. B12X owns the sparse attention kernel
and caller-owned scratch. RoCEnante handles eligible small TP collectives while
NCCL remains the deliberate fallback above the configured size limits.

This is a source-complete but not yet target-qualified serving candidate. Its
image has not been built while the current service is using nearly all unified
memory on each Spark. Building and runtime qualification wait for a maintenance
window; the live service has not been modified.

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
source trees. It does not stop, restart, or deploy the service.

## Compatibility gate

Source preparation uncovered and resolved two source-level integration
boundaries. B12X now declares the same CUTLASS DSL `4.7.1` toolchain as vLLM,
and the DS4.1 B12X attention and RoCEnante consumers use current vLLM and B12X
prepared-plan interfaces. No candidate image suppresses dependency conflicts,
downgrades the vLLM environment, or mixes Python with native extensions from a
different vLLM revision.

[compatibility.md](compatibility.md) records the exact matrix and remaining
SM121 qualification gates. The source is ready to build, but it is not a
deployable serving image until those target tests pass.

## Quality and safety gates

- Do not build on, restart, stop, or replace the live vLLM service while it has
  active work.
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
