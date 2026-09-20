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
- vLLM carry: quality-preserving 64/8 to 72/9 virtual attention geometry for
  TP3.
- B12X carries: the qualified 3072-token mHC policy and per-peer/HCA
  RoCEnante routing for the switchless ring.

The candidate deliberately adopts upstream DS4.1 model support, heterogeneous
KV allocator, Engram implementation, generic B12X 1.3 integration, bounded
index reads, and TP3 24-head sparse-MLA partition without local copies of those
changes.

This is not yet a performance-complete serving candidate. Canonical vLLM main
does not currently connect the DS4.1 model's sparse-MLA path to B12X, and it
does not contain the vLLM-side RoCEnante collective adapter. With no further
port, an SM121 DS4.1 launch selects the upstream FlashInfer attention path and
keeps large collectives on NCCL even if the launch still spells
`--attention-backend B12X`. The prepared B12X source commits are therefore not
reachable from this vLLM source yet.

See [carry-matrix.md](carry-matrix.md) for the complete disposition.
[donor-analysis.md](donor-analysis.md) pins the latest known downstream R38
implementation and describes how to negative-port only the missing interfaces
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

## Compatibility gate

Source preparation uncovered two real integration boundaries. First, vLLM
`d05da62e9ccdf8e342b15bf6785d83224cc165af` requires CUTLASS DSL `4.7.1`,
while B12X `0f3a8cbfd1c11d27f04e3ab37a802d522f4f1c68` pins every CUTLASS DSL
package to `4.6.2`. Second, the DS4.1 B12X attention and RoCEnante consumer
adapters still need to be ported to current vLLM interfaces. No candidate image
should suppress the dependency conflict with
`--no-deps`, downgrade the vLLM environment, or mix Python from one vLLM
revision with native extensions from another.

[compatibility.md](compatibility.md) records the exact matrix and the honest
qualification routes. Until both integration boundaries pass, the commits here
are prepared source carries, not a deployable serving image.

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

No serving workload is run during source preparation. The maintenance-window
qualification must run the stable matrix in `docs/methodology.md`, including
the 4-by-128K target and the normal eight-way Strix workload, against both the
promoted baseline and this candidate.

## Acceptance criteria

- Exact native-weight output parity on fixed prompts and tokens.
- At least eight complete 160K KV windows with the configured 3 GiB cache, or
  a documented allocator accounting explanation for any shortfall.
- No statistically meaningful regression in single-stream decode TPS,
  aggregate eight-stream TPS, warm/cold TTFT, or minimum available host memory.
- No request failures, OOMs, silent truncation, or RoCE/NCCL collective errors.
- Every retained local commit is independently reviewable and has an explicit
  upstream or downstream-only disposition.
