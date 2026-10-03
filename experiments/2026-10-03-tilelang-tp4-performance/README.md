# TileLang versus B12X at TP4

**Status:** measured 2026-10-03 18:09–18:25 UTC. TileLang passes the quality
gate and matches B12X on prefill, but decodes 5–10% slower: its single-stream
step is 4.5–8.8% longer. B12X stays the faster backend at TP4.

A same-window performance screen of the TileLang kernel backend against its
B12X twin on the four-node ring. Each arm is the matching tuning profile,
materialized for the site ring map with `tuning create tp4` and launch enabled
for this run only:

| Arm | Config | Catalog | Image |
| --- | --- | --- | --- |
| B12X (control) | [b12x.json](b12x.json) | [transport profiles](../2026-10-03-transport-profiles/profiles.json) | `vllm-ds41f-kkref:04c30fa98e79-r5o-roce-contract-v1` |
| TileLang | [tilelang.json](tilelang.json) | [TileLang profiles](../2026-10-03-tilelang-kernels/profiles.json) | `vllm-ds41f-kkref:04c30fa98e79-r5o-roce-contract-tilelang-v2` |

The arms differ only in the kernel policy, its image and sources, and TileLang's
own pinned DSpark cost and profiler directories (`tests/test_kernel_backend.py`
checks the bases field by field). Both images were built on dgx4 with the
regular `bin/spark3 build` commands: the control from `aafb3cf` (main), the
TileLang image from `681cc2e`.

## Procedure

One window, held through `~/spark3-hold.json` on dgx1. Lab windows run only on
the promoted topology, so the arms use the coordinated cluster commands:

```sh
bin/spark3 --cluster-config experiments/2026-10-03-tilelang-tp4-performance/b12x.json cluster sync --apply
bin/spark3 --cluster-config experiments/2026-10-03-tilelang-tp4-performance/b12x.json cluster start --apply
# bench on dgx1 (below), then
bin/spark3 --cluster-config experiments/2026-10-03-tilelang-tp4-performance/b12x.json cluster stop --remove --parallel --apply
# the same start, bench and stop with tilelang.json
```

One boot per arm, B12X first. On dgx1, from the synced
`~/projects/spark3-ring4-qualification` checkout, each arm runs the published
TP4 comparison's workloads as a lean screen, in two invocations so each section
starts cooled (a continuous run tripped dgx2's thermal guard there):

```sh
bin/spark3 --cluster-config <arm>.json bench --url http://10.0.1.71:8000 \
  --suites quality,decode --decode-cases prose,code --concurrency 1,8 \
  --min-samples 3 --max-samples 3 --compare <reference> --output <dir>
bin/spark3 --cluster-config <arm>.json bench --url http://10.0.1.71:8000 \
  --suites prefill --prefill-text source --prefill-sizes 1024,32768,65536,262144 \
  --prefill-repeats 2 --compare <reference> --output <dir>
```

The control compares against the promoted baseline manifest; TileLang compares
against the control's own reports from this window. Quality must pass before
speed counts. The cluster returns to its idle entry state (no serving
containers) and the hold is removed at the end.

## Results

One boot per arm, the same window, client on dgx1. Summary:
[results.json](results.json). Every point has zero thermal slowdown (GPU at most
66 °C, 26 °C below the limit), no swap growth and at least 29.0 GiB of host
memory available; both arms allocate the same 2,845,543-token KV cache.

| | B12X | TileLang |
| --- | ---: | ---: |
| Boot to API ready | 112 s | 214 s (cold TileLang JIT and a first DSpark cost profile) |
| Quality gate | 5/5 | 5/5 |

### Decode

Aggregate tok/s over a 256-token reasoning output, three samples per point
after one discarded warm-up; ± is the 95% interval. Accepted and verified are
DSpark drafts per step.

| Case | B12X tok/s | TileLang tok/s | Change | Accepted, B12X / TileLang | Verified, B12X / TileLang |
| --- | ---: | ---: | ---: | ---: | ---: |
| prose, 1 stream | 62.0 ± 8.4% | 58.4 ± 1.1% | −5.9% [−14.3, +2.5] | 1.06 / 1.11 | 3.00 / 2.55 |
| prose, 8 streams | 221.0 ± 1.6% | 206.3 ± 0.4% | −6.6% [−8.3, −5.0] | 1.14 / 1.12 | 1.90 / 1.81 |
| code, 1 stream | 78.4 ± 10.0% | 70.2 ± 1.0% | −10.4% [−20.4, −0.4] | 1.99 / 1.78 | 4.05 / 3.47 |
| code, 8 streams | 255.6 ± 3.9% | 242.1 ± 0.4% | −5.3% [−9.2, −1.4] | 1.80 / 2.03 | 2.54 / 3.09 |

Single-stream step time, which does not depend on acceptance: prose
**31.91 → 34.73 ms (+8.8%, [+6.0, +11.7])**, code **36.16 → 37.79 ms (+4.5%,
[+2.0, +7.0])**. TileLang's adaptive verification, priced from its own cost
curves, verifies fewer drafts per single-stream step and still takes longer per
step, so the decode gap is kernel time, not extra speculative work. Draft
acceptance moves both ways because the two backends' outputs differ.

The control reproduces the published TP4 result: against the promoted TP3
baseline it decodes 17.5–30.5% faster (prose 62.0 / 221.0, code 78.4 / 255.6
tok/s at 1 / 8 streams).

### Prefill

Source-text prompts, two repeats per size, one output token.

| Nominal size | Input tokens | B12X tok/s | TileLang tok/s | Change | Mean TTFT, B12X → TileLang |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1K | 947 | 2,661 | 2,577 | −3.2% [−41.4, +35.1] | 0.343 → 0.354 s |
| 32K | 28,907 | 4,690 | 4,925 | +5.0% [−10.0, +20.1] | 6.311 → 6.010 s |
| 64K | 64,943 | 4,695 | 4,910 | +4.6% [−11.2, +20.3] | 13.104 → 12.529 s |
| 256K | 240,637 | 4,535 | 4,660 | +2.8% [−0.3, +5.9] | 52.973 → 51.549 s |

Every interval crosses zero; TileLang prefill is level with B12X, nominally
3–5% faster from 32K up.

### Leads for the decode gap

- TileLang's compiler reported, 16 times while building the serving kernels,
  that an 8-wide `T.vectorized` loop was lowered as a serial loop. The kernel
  is not named in the warning; finding it is the first follow-up.
- The decode-sized GEMMs (MXFP8 linear at 1–48 rows and the expert GEMMs at
  one tile per expert) and the per-step indexer are the candidates a torch
  profile of one decode step should rank against B12X's.

## Window

Hold `tilelang-tp4-performance` on dgx1, 18:04–18:25 UTC, with a heartbeat.
Entry and exit inventories (`docker ps -a` on every node, and each shared
checkout's commit) match: no serving container before or after, the unrelated
stopped containers untouched, and the `spark3-ring4-qualification` checkouts
returned to their entry commits (`d87e91e`; `9571973` on dgx4).

Two earlier attempts in the same hour did not reach a GPU. The first stopped at
`cluster sync` (the Mac's temporary path exceeded SSH's control-socket limit).
The second's start was refused because stopped production containers
(`-r5o`, stopped at 14:02 UTC) still held the serving name, and its automatic
cleanup, `cluster stop --remove`, then deleted those stopped containers on dgx1,
dgx2 and dgx3 with their logs. That was an error in the run script: it now
inventories with `docker ps -a` and refuses to start over any existing serving
container. The containers' definitions are reproducible from the promoted
configuration.
