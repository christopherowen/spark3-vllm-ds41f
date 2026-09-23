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
| B12X | `master` | `0332cc5089137753d3af43d1c643516b4350359f` | 5 |

Run `prepare-sources` to fetch the pinned bases, replay the patches, and verify
the exact trees in `series.json`. The source candidate is isolated in `.work/`.
The old series and image remain available for rollback.
On September 23 the initial 13 local patch commits were reauthored to Christopher
Owen's GitHub noreply identity. A fresh canonical replay produced the same
vLLM and B12X source trees; `runs/source-replay-identity.json` records both
sets of commit IDs. This changes attribution only, not the candidate code.
`build-candidate check` verifies source identities and reports uncommitted
deployment changes; `build` requires a clean deployment commit.
`build-candidate build` uses the pinned ARM64 dependency image and rebuilds the
two vLLM CUDA stable extensions from this head. The build command refuses a
host with a running DS4.1 container. On an idle Spark, it chooses compile jobs
from the CPU allowance and `MemAvailable`: two NVCC threads per job, four CPUs
left for the host, an 8 GiB allowance per compile job, and at most ten jobs.
It requires 64 GiB available before starting and watches memory once a second,
canceling the build below a 24 GiB host reserve. A host lock prevents two
candidate builds from running together. A 20-CPU Spark with 120 GiB
available gets eight compile jobs instead of the previous two. These are build
limits, not serving limits; the 5 GiB startup and 3 GiB steady memory guards
still protect the model after weights load. Build on one idle node at a time,
then distribute the verified image digest to the other ranks.
`bin/spark3 status` checks the promoted container name from `config/cluster.json`
and does not list this unpromoted experiment. Direct read-only Docker and
systemd checks on all three nodes found the current patch-0029 containers and
their memory guards active.

## Changes from the qualified September 20 candidate

- Upstream B12X replaced managed final weights with ordinary CUDA allocations.
  Patch 0005 restores only scoped managed final-weight allocation on GB10; it
  retains upstream's bounded bounce/GDS reader and ordinary runtime buffers.
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

Exact source replay and native smoke tests are source gates only. The first
guarded three-rank image failed on a removed B12X/vLLM progress import. The
corrected image loaded weights on dgx2 and dgx3, then dgx3 reached the 5 GiB startup
memory floor during post-load initialization. The memory guard stopped it, and
the coordinated launcher removed all candidate containers. The known working
patch0029 image was restored on all three nodes with active guards and API
readiness. Neither of those early candidates passed semantic, quality,
long-context, or speed qualification. See
`runs/launch-eager-cb3f6320-failed.json`.

Patch 0004 groups current-vLLM loader lifecycle compatibility: removed
private progress imports, explicit transfer completion at `load_weights`, and
CUDA cache release before post-load weight preparation. The superseded
`_log_loading_time` callback is removed because current vLLM does not call it.
Image `sha256:c068aebe` passed bounded loader and GPU smoke, but its guarded
three-rank load also reached dgx3's 5 GiB startup floor immediately after
weight loading. See `runs/launch-eager-c068aebe-failed.json`. This rules out
the loader-lifecycle fix alone as sufficient. Patch 0005 restores scoped
managed final weights as the next single causal change. Image `sha256:a3152e8c`
passed a capped 1 MiB native checkpoint copy into managed weight storage with
exact values after pool scope exit. It then reached three-rank API readiness,
passed 159K context admission, and completed the matched serving matrix. A
deterministic LRU request produced malformed code, so quality qualification and
promotion remain blocked. See `runs/launch-eager-a3152e8c-observed.json`.

`performance-bridge.md` records the same-script serving comparison across the
successful weekend baseline, patch0029, and the consolidated managed-weight
candidate. The latter completed all 30 matched requests; throughput remains
far below the weekend setup at low and moderate concurrency.
`performance-recovery.md` records the source/configuration attribution audit
and the controlled qualification sequence for recovering decode speed.
`quality-diagnosis.md` records the exact-token comparison that blocks
promotion while incremental output remains intermittently malformed.

`cluster-eager.json` keeps the working patch-0029 target-only serving flags and
guarded memory limits. Its latest revision pins the managed-weight image and
reuses the JIT cache from the preceding loader-only attempt because no kernel
source changed. It is an experiment configuration, not a promotion.
`cluster-rollback.json` pins the previously working patch-0029 image on the
same deployment branch, so a failed candidate can be stopped and the prior
three-rank service relaunched with the coordinated cluster commands.

The first guarded launch of image `sha256:d25a2e05` exposed a removed B12X
progress import. Image `sha256:d4c5337e` exposed a second private output import
in a bounded GPU smoke. Image `sha256:cb3f6320` passed that smoke but reached
the startup memory floor after checkpoint loading. The guards and rollback
worked as designed. Image `sha256:c068aebe` then repeated the post-load memory
failure. The managed-weight candidate now serves on all three nodes with
active guards, pending the code-output quality gate.
