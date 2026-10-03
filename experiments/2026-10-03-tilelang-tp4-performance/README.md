# TileLang versus B12X at TP4

**Status:** round 2 (2026-10-03 20:09–20:24 UTC): with the decode projections
fixed, TileLang is level with or ahead of B12X at TP4: decode +1.4% to +4.2%
except code at eight streams (−1.7%), single-stream steps 1.5–4.5% shorter,
prefill +2.4% to +4.3% (all within noise), and 5.3 ms less main-stream kernel
time per profiled decode step. Both arms ran with the DSpark heads at native
BF16. Round 1 (18:09–18:25 UTC, NVFP4 drafter heads) found TileLang 5–10%
slower and attributed the gap to the dense projections.

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

### Where the decode time goes

A second window (18:32–18:38 UTC, hold `tilelang-tp4-profile`, the same
restoration) booted each arm once more and wrapped the repository's
`profile_decode.py` (one stream) and `profile_c8.py` (eight streams) in vLLM's
torch profiler. [step_median.py](step_median.py) takes rank 0's main stream,
splits it into target steps at the shared SWA cache writer (40 per step) and
reports the median of every steady six-row decode step (23 TileLang, 25 B12X).
TileLang kernels all appear as `main_kernel`; each was identified from its
cached launch geometry and parameter names.

| Median ms per six-row decode step | B12X | TileLang | Change |
| --- | ---: | ---: | ---: |
| Dense projections | 2.29 | 5.68 | **+3.39** |
| Routed experts | 19.43 | 19.19 | −0.24 |
| Sparse MLA and indexer | 1.49 | 1.28 | −0.21 |
| Activation quantization and norms | 0.48 | 0.31 | −0.17 |
| RoCE collectives, waits included | 4.37 | 4.03 | −0.34 |
| Shared B12X kernels (WO, mHC, rotary, cache writers, head) | 5.79 | 5.33 | −0.46 |
| Main stream total | 33.84 | 35.82 | +1.97 |

The whole decode gap is in the dense projections at decode row counts:

| Projection, per call | B12X | TileLang |
| --- | --- | --- |
| Router gate, BF16, N=384, K=5120 | GEMV, 768 CTAs: ~11 µs | 6 CTAs: ~41 µs |
| Q-B, N=8192, K=1280 | split over 48 CTAs: ~22 µs | 128 CTAs: ~46 µs |
| Fused Q-A/KV, N=1792, K=5120 | split over 14 CTAs: ~19 µs | 28 CTAs: ~34 µs |

TileLang's dense GEMMs give each CTA a 64×64 output tile and walk the whole K
in one pass, the batch-invariant design used for every row count. At six to
48 rows that leaves the GPU mostly idle (6 CTAs for the router on 48 SMs) and
each CTA latency-bound on a long serial K loop; B12X uses a GEMV for the router
and split-K for the projections. The L2 prefetch serves both arms equally:
its windows fire at the same points, and TileLang's Q-B is slow with its
weights already in L2.

The components optimized and benchmarked during development hold up in
serving: sparse MLA and the indexer are faster than B12X, and the routed
experts (including TP4's compact K tails) are at parity. The dense GEMMs were
only checked for correctness and batch invariance, never timed against B12X.

TileLang's compiler also reports that the FP8-to-BF16 dequantization of staged
KV records in all twelve sparse-MLA variants is lowered as an eight-element
serial loop; attention is still ahead of B12X, so this is a smaller follow-up.

The next change is a decode path for the dense projections: a fixed split-K
(same split for every row count, so outputs stay batch-invariant) for the MXFP8
projections and a GEMV-shaped kernel for the BF16 router. Removing the 3.4 ms
would put TileLang about 1.4 ms per step ahead of B12X on this profile.

## Window

Hold `tilelang-tp4-performance` on dgx1, 18:09–18:25 UTC, with a heartbeat.
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

## Round 2: fixed projections, native drafter heads

The decode projections were fixed in the [TileLang candidate](../2026-10-03-tilelang-kernels/README.md)
(image `-r5o-roce-contract-tilelang-v4`). The audit of load-time weight
quantization found that the DSpark draft head and Markov transition head
re-quantize the checkpoint's BF16 tensors to NVFP4
(`VLLM_DS41_DRAFT_NVFP4_HEAD`, `VLLM_DS41_MARKOV_NVFP4`), which the owner did
not intend; both arms of round 2 run them at native BF16
([round2-b12x.json](round2-b12x.json), [round2-tilelang.json](round2-tilelang.json)),
each with fresh pinned cost curves. Every other weight is native or repacked
losslessly. Summary: [round2-results.json](round2-results.json).

| Case | B12X | TileLang | Change |
| --- | ---: | ---: | ---: |
| prose, 1 stream | 62.3 ± 25.3% | 63.1 ± 0.5% | +1.4% [−23.9, +26.6] |
| prose, 8 streams | 213.3 ± 3.8% | 219.7 ± 0.2% | +3.0% [−0.8, +6.8] |
| code, 1 stream | 78.1 ± 11.6% | 81.4 ± 0.3% | +4.2% [−7.4, +15.8] |
| code, 8 streams | 246.0 ± 2.5% | 241.7 ± 0.6% | −1.7% [−4.3, +0.9] |

Single-stream step: prose 33.66 → 33.15 ms, code 38.17 → 36.44 ms. Prefill:
32K +4.3%, 64K +4.0%, 256K +2.4%. Quality 5/5 in both arms; no thermal
slowdown; at least 30.0 GiB host memory available. TileLang's intervals are
tight because its output is bit-reproducible run to run.

Median kernel time per six-row decode step (rank 0 main stream):

| Category | B12X | TileLang | Change |
| --- | ---: | ---: | ---: |
| Routed experts | 21.99 | 17.38 | −4.61 |
| Dense projections | 2.27 | 2.44 | +0.17 (round 1: +3.39) |
| Sparse MLA and indexer | 1.48 | 1.71 | +0.23 |
| Collectives, shared kernels, quantization and norms | 10.91 | 9.83 | −1.08 |
| Main stream total | 36.65 | 31.35 | −5.30 |

Native drafter heads on B12X (round 2 against round 1, different windows):
decode tok/s −3.5% to −3.8% at eight streams and level at one; accepted drafts
per verified draft rise 2–10% in every case, within the three-sample noise.

The step-end tail also showed the DSpark transition head on BF16 cuBLAS under
TileLang (about 1.4 ms per step, against 0.42 ms with B12X's vocabulary row
kernel): the logits processor enabled B12X's vocabulary projection only for
the `b12x` linear backend. That is fixed in image v5; round 3 measures it.

## Round 3: B12X's vocabulary projection under TileLang

Image `-r5o-roce-contract-tilelang-v5` ([round3-tilelang.json](round3-tilelang.json))
against the unchanged round-2 control, 2026-10-03 20:40–20:55 UTC, both with
native drafter heads.

| Case | B12X | TileLang | Change |
| --- | ---: | ---: | ---: |
| prose, 1 stream | 62.6 ± 3.7% | 62.7 ± 0.4% | +0.3% |
| prose, 8 streams | 213.7 ± 6.4% | 215.9 ± 0.2% | +1.0% |
| code, 1 stream | 79.8 ± 28.3% | 80.0 ± 0.3% | +0.2% |
| code, 8 streams | 249.0 ± 7.3% | 247.9 ± 0.2% | −0.4% |

Every difference is within noise; quality 5/5 in both arms. The transition
head now runs B12X's vocabulary row kernel under TileLang (0.42 ms per step,
as in the B12X arm), and the step-end tail is level: median 6.37 ms (B12X)
against 6.18 ms (TileLang). On the profiled workload TileLang spends 3.1 ms
less main-stream kernel time per decode step (routed experts −2.7 ms, dense
projections +0.17, sparse MLA and indexer +0.19).

Across rounds 2 and 3 TileLang is at parity with B12X at TP4 (decode −1.7% to
+4.2%, prefill +2% to +4%). What remains in both arms' step-end tail is 2.6 ms
of BF16 vocabulary GEMMs (the target head and the first native draft head),
which already stream at about 250 GB/s; reading fewer bytes losslessly (an
exact 12-bit BF16 form) is the remaining lever there that keeps native weights.
