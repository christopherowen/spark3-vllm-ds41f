# September 23 current-head source candidate

## Hypothesis

The compact DS4.1 source series can be rebased to a pinned snapshot of both
upstreams while preserving the corrected target-model output path and serving
speed. The intended variable is the upstream source revision; the running
three-rank service and promoted configuration are unchanged.

Fetched at 2026-09-23 07:30 UTC:

| Project | Canonical branch | Pinned commit | Local concept patches |
| --- | --- | --- | ---: |
| vLLM | `main` | `0f2a15c9277f34c9afe141cf578d1f02b3bebdfe` | 10 |
| B12X | `master` | `0332cc5089137753d3af43d1c643516b4350359f` | 3 |

Run `prepare-sources` to fetch the pinned bases, replay the patches, and verify
the exact trees in `series.json`. The source candidate is isolated in `.work/`.
The old series and running image remain untouched.
`build-candidate check` verifies clean source and deployment commit identities;
`build-candidate build` uses the pinned ARM64 dependency image and rebuilds the
two vLLM CUDA stable extensions from this head. The build command refuses a
host with a running DS4.1 container.
`bin/spark3 status` checks the promoted container name from `config/cluster.json`
and does not list this unpromoted experiment. Direct read-only Docker and
systemd checks on all three nodes found the current patch-0029 containers and
their memory guards active.

## Changes from the qualified September 20 candidate

- B12X's managed-weight loader compatibility patch is omitted. Upstream now
  uses ordinary CUDA weight allocations and a bounce/GDS reader with the
  current vLLM transfer hook. Reapplying our old pool patch would replace that
  path and remove upstream's current loader behavior.
- The switchless RoCEnante patch retains upstream's traffic-class setting,
  checks it across ranks, and passes it through the C proxy alongside peer
  route stripes.
- vLLM's selected-kernel registration retains upstream's new router dispatch
  and GEMM-RS/AR location. It removes model-wide router warmups after the
  relevant kernels register their own selected shapes.
- Disk Engram retains upstream's automatic DP shared-memory resolution and
  restores loader-format validation after upstream moved config resolution.
- The remaining TP3, cache-mapping, file-backed weight, and B12X plan fixes
  are retained. They have not been proven obsolete in upstream.

## Acceptance boundary

Syntax and exact source replay are source gates only. A compatible native base
image for the exact vLLM commit is not yet published, so this candidate has no
runtime result. Before replacing the working image, build all affected native
extensions against this vLLM source, run focused tests on the target hardware,
then repeat the guarded three-rank cold start, semantic and long-context
smokes, quality comparison, and performance matrix from `docs/methodology.md`.
The service must not be stopped or replaced without explicit authorization.

`performance-bridge.md` records a same-client serving comparison between the
successful weekend baseline and the running patch-0029 service. This is an
interim measurement; the consolidated latest-head image remains unbuilt and
unmeasured.
