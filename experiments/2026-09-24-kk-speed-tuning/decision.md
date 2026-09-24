# Decision: promote capture32 (2026-09-24)

Promote CUDA-graph capture sizes `[1,2,3,4,6,8,12,16,20,24,28,32]` with
`--max-cudagraph-capture-size 32` into `config/cluster.json`, as baseline
`manifests/baselines/2026-09-24-karmic-kraken-capture32.json`. With DSpark
verifying three draft tokens and eight admitted sequences, decode batches
reach 32 tokens; the previous largest graph (24) left 28- and 32-token
batches eager. Code at eight streams rose 155.7 to 166.9 tok/s with
non-overlapping pass ranges; every other matrix point is within pass-to-pass
variation. LRU passed 5/5 in both passes. dgx1's minimum MemAvailable fell
from 6.98 to 6.07 GiB, still 3 GiB above the steady guard.

Everything else was measured and not promoted (`results.json`, `README.md`):

| Candidate | Outcome |
|---|---|
| DSpark depth 5 | neutral: +14% accepted drafts, but verification cost is almost linear in tokens on GB10 |
| 8,192 batched tokens | infeasible: warmup crosses the 5 GiB startup guard on dgx1, even with 1.5 GiB KV |
| draft TP 1 | infeasible: the 7.39 GiB drafter would all land on dgx1 |
| full B12X autotune | infeasible in-engine: candidate programs stay resident (~1.6 MB per compilation) |
| targeted autotune (GEMM, MoE) | runs, neutral |
| adaptive cost scale 1.0 | neutral |
| adaptive cost scale 4.0 | trade: prose c8 +7%, code c8 -5% |
| max model length 512K | works (254K prefill at 3,490 tok/s, 1.63M KV tokens), but dgx1 startup margin 0.55 GiB and c4/c8 about -6%; kept as an opt-in, not the default |

Where decode time goes (`runs/profile-c1/`): routed MoE 33% (near memory
bandwidth), dense FP8 GEMMs 27%, RoCE collectives 20% (decode all-reduces
about 40 µs; eager prefill all-reduces wait on host skew), LM head 6%. The
remaining headroom is in kernels (dense GEMM efficiency), not configuration.
