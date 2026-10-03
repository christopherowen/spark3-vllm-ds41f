# TileLang versus B12X at TP4

**Status:** planned; results follow the run.

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
