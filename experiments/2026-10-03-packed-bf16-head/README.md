# Exact 12-bit packed BF16 vocabulary head

Base deployment commit: `3fb888b` (the [native drafter heads](../2026-10-03-native-drafter-heads/README.md)
change on `aafb3cf`).

**Variable:** the DS4.1 target vocabulary head, which the native DSpark draft
head shares, is stored in an exact 12-bit packed form instead of BF16
(`VLLM_DS41_PACKED_BF16_LM_HEAD=1`), with its own DSpark cost directory
(`ring4-collective-packedhead-20261003`). Every weight bit is kept; only the
storage and the kernel that reads it change.

Candidate: [candidate.json](candidate.json), the TP4 collective-contract
candidate with native heads plus the variable above. Sources:
[source.json](source.json) and [upstreams.lock.json](upstreams.lock.json),
extending the candidate's series with
[B12X 0012](b12x/0012-packed-bf16-vocab-projection.patch) and
[vLLM 0028](vllm/0028-deepseek-v41-packed-bf16-lm-head.patch).

**Status:** image `vllm-ds41f-kkref:04c30fa98e79-r5o-roce-contract-packedhead-v1`
(`sha256:5e112503…`) built on dgx4 from `2ed0fd8` with the regular
`bin/spark3 build` commands and loaded on dgx1–dgx3. Its kernel bundles pass on
dgx4 (2026-10-03 22:16–22:18 UTC): B12X 15 passed, vLLM 2 passed, head bench
26 checks passed. The TP4 window (22:19–22:36 UTC) measured 0.44 ms less head
time per decode step and a 0.60 ms shorter step-end tail, with throughput
level within the benchmark's noise; see [TP4 window](#tp4-window).

## Why

With native drafter heads every decode step reads the BF16 head shard twice:
once for target verification and once for the first draft position. At TP4
that is 2 × 331 MB per rank, about 2.6 ms of each step's tail, and these GEMMs
already stream near the GB10's bandwidth. Reading fewer bytes is the only
lever left, and the owner's rule allows it only if it is purely lossless.

## Format

A BF16 value is a sign bit, an 8-bit exponent and a 7-bit mantissa. The packed
form keeps the sign and mantissa in one byte and the exponent in a 4-bit code:
code `c` in 1–15 means exponent `base + c − 1` for a 15-exponent window chosen
per tensor shard; code 0 means exponent 0, so zeros and subnormals stay exact.
Values whose exponent falls outside the window are exceptions: their slot holds
+0 and their exact BF16 bits sit in a per-row list that the projection adds
after the main dot product. A row is `K` sign-and-mantissa bytes followed by
`K/2` code bytes, 7,680 bytes for K = 5,120.

Packing runs once after loading and refuses to start unless unpacking
reproduces every input bit. The checkpoint (`dba1be0a…`) has no zero or
subnormal values in `head.weight` and uses 31 exponents (97–127); the best
window, 112–126, leaves about 1.6 values in 10,000 as exceptions:

| Shard | Rows | Exceptions | Most in one row | BF16 bytes | Packed bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| TP4, each rank | 32,320 | 25,718–27,034 | 23 | 330,956,800 | 248,554,820 |
| TP3, each rank | 43,092–43,094 | 34,416–35,683 | 23 | 441,262,080 | 331,404,396 |

The packed bytes replace the BF16 tensor, so the head also frees 82 MB per
TP4 rank, and the display carve-out holds the packed form.

## Kernel

B12X's prepared `packed_bf16_vocab_projection` (Triton) decodes in registers
and accumulates in FP32 on tensor cores. Each logit is its row's 128-column
groups summed in order, then its exceptions, independent of token count and
tile configuration, so a token's logits are the same bits at any batch size.
They differ from cuBLAS BF16 logits only by FP32 accumulation order (maximum
absolute difference equal to cuBLAS's own against an FP32 reference).

Development sweep on dgx4 (real head shards, CUDA graphs, best of 432
configurations per row count; the defaults use these winners):

| Shard | Rows | cuBLAS BF16 | BF16 row kernel | Packed | Packed GB/s | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TP4 rank 0 | 1 | 1,362 µs | 1,306 µs | 1,028 µs | 242 | 1.33× |
| TP4 rank 0 | 6 | 1,389 µs | — | 1,049 µs | 237 | 1.32× |
| TP4 rank 0 | 8 | 1,396 µs | — | 1,044 µs | 238 | 1.34× |
| TP4 rank 0 | 16 | 1,393 µs | — | 1,072 µs | 232 | 1.30× |
| TP4 rank 0 | 32 | 1,485 µs | — | 1,062 µs | 234 | 1.40× |
| TP4 rank 0 | 48 | 1,546 µs | — | 1,107 µs | 224 | 1.40× |
| TP3 rank 2 | 1 | 2,664 µs | 1,741 µs | 1,401 µs | 237 | 1.24× vs row kernel |
| TP3 rank 2 | 6 | 1,847 µs | — | 1,414 µs | 234 | 1.31× |
| TP3 rank 2 | 48 | 1,912 µs | — | 1,537 µs | 216 | 1.24× |

At TP4 that is 0.33–0.44 ms less per head read, two reads per decode step.

The built image, default prepared configuration ([head-bench](bundles/head-bench/candidate.json)):

| Shard | Rows | cuBLAS BF16 | Packed | Packed GB/s | Speedup | Max error, packed / cuBLAS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TP4 rank 0 | 1 | 1,368 µs | 1,057 µs | 235 | 1.29× | 0.123 / 0.123 |
| TP4 rank 0 | 6 | 1,392 µs | 1,051 µs | 237 | 1.33× | 0.125 / 0.125 |
| TP4 rank 0 | 8 | 1,390 µs | 1,056 µs | 235 | 1.32× | 0.125 / 0.125 |
| TP4 rank 0 | 16 | 1,394 µs | 1,063 µs | 234 | 1.31× | 0.125 / 0.125 |
| TP4 rank 0 | 32 | 1,509 µs | 1,043 µs | 238 | 1.45× | 0.125 / 0.125 |
| TP4 rank 0 | 48 | 1,525 µs | 1,104 µs | 225 | 1.38× | 0.125 / 0.125 |
| TP3 rank 2 | 1 | 2,670 µs | 1,411 µs | 235 | 1.89× | 0.125 / 0.125 |
| TP3 rank 2 | 6 | 1,851 µs | 1,431 µs | 232 | 1.29× | 0.125 / 0.125 |
| TP3 rank 2 | 48 | 1,895 µs | 1,504 µs | 220 | 1.26× | 0.204 / 0.204 |

Errors are against an FP32 product of the same BF16 weights; every row count
kept row 0's logits bit-identical to a single-token run.

## TP4 window

Same-window A/B on the four-node ring, 2026-10-03 22:19–22:36 UTC, with the
hardened window procedure (hold file with heartbeat, idle-entry check,
`cluster sync`, per arm start, `bin/spark3 bench` decode and prefill, decode
profile, stop, entry checkouts restored, hold released; all four nodes idle
afterwards). Arms: [control.json](control.json) (the TP4 profile with native
drafter heads, image `-r5o-roce-contract-v1`) and [packed.json](packed.json)
(image `-r5o-roce-contract-packedhead-v1`). Summary:
[results.json](results.json).

| Case | Control | Packed | Change |
| --- | ---: | ---: | ---: |
| prose, 1 stream | 61.7 ± 11.0% | 61.8 ± 11.1% | +0.2% |
| prose, 8 streams | 216.0 ± 1.0% | 216.9 ± 3.0% | +0.4% |
| code, 1 stream | 76.9 ± 29.8% | 75.9 ± 4.5% | −1.3% |
| code, 8 streams | 250.7 ± 8.1% | 248.7 ± 4.3% | −0.8% |

Single-stream step: prose 33.60 → 33.00 ms, code 37.82 → 37.10 ms (−0.6 and
−0.7 ms, inside the three-sample intervals). Prefill: 1K −0.2%, 32K −0.4%,
64K −0.2%, 256K −0.2%. Quality 5/5 in both arms; no thermal slowdown. The
control reproduces the round-2 native-heads control of the TileLang screen
(prose 62.3 / 213.3, code 78.1 / 246.0 tok/s).

Each rank logged the exact packing (exponents 112–126; 25,992 exceptions on
rank 0, 27,034 on rank 3, as measured offline), and the display carve-out
held 552.3 MiB instead of 631.2 MiB. The decode profiles (median over
six-row steps, main stream) show the effect directly:

| | Control | Packed |
| --- | ---: | ---: |
| Vocabulary head kernel, per read (rank 0 / rank 3) | 1.311 / 1.336 ms (cuBLAS BF16) | 1.075 / 1.095 ms (packed) |
| Head reads per step | 2 | 2 |
| Step-end tail after the last layer | 6.09 ms | 5.49 ms |

Under serving the packed read saves 18% rather than the 24% of the offline
bench, because the L2 prefetch and collectives share the bandwidth.

Open item: host MemAvailable minima during the benches were 0.5–0.8 GiB lower
in the packed arm on every node (packed 30.09 / 31.14 / 31.09 / 30.64 GiB,
control 30.63 / 31.93 / 31.91 / 31.39). vLLM's own accounting does not show
it: model loading took 73.34 GiB against 73.33, the KV cache is fixed at
3.5 GiB, and the packed arm had 0.24 GiB more free memory when the KV cache
was allocated. The arms differ in one more way: the control loaded DSpark
cost curves pinned earlier, while the packed arm, with its own new cost
directory, profiled and pinned them at this boot. A TileLang window of the
same change shows the same pattern. Rebooting the packed arm with its now-pinned curves,
and comparing per-process memory after boot, separates the two before any
promotion.

## Conclusion

The packed head is lossless and does what it was built for: 0.44 ms less
vocabulary-head time per decode step at TP4, with logits that are the
checkpoint's weights accumulated in a fixed order. At 33–38 ms per step that
is about 1.5%, below what three-sample decode benches resolve, so it shows in
the profiles and step times rather than in tok/s.

Per TP4 rank and decode step, the vocabulary reads were 443 MB with the old
NVFP4 drafter heads (BF16 target head 331 MB, NVFP4 draft copy 93 MB, NVFP4
Markov head 4 × 4.7 MB) and 728 MB with native heads (the shared BF16 head
twice, BF16 Markov head 4 × 16.5 MB). Packing brings it to 563 MB, recovering
58% of the extra bytes without changing a weight. The rest is the packed head
still being larger than the NVFP4 copy, and the BF16 Markov head, which this
change does not pack because it gathers individual rows.

## Changes

- **B12X 0012**: the `gemm.packed_bf16_vocab_projection` component: `pack` /
  `unpack`, the Triton kernel, its tuning contract (with the measured
  shared-memory model that rejects configurations SM12x cannot hold) and
  tests.
- **vLLM 0028**: `PackedBf16LmHeadMethod` packs the loaded BF16 head; the
  DS4.1 target head selects it with `VLLM_DS41_PACKED_BF16_LM_HEAD=1`; the
  logits processor declares, prepares and runs B12X's packed projection per
  token capacity, for the target and for the drafter that shares the head.
  The DSpark Markov head is unchanged: it gathers individual rows.

## Bundles

- [b12x-tests](bundles/b12x-tests/candidate.json): B12X 0012's tests from the
  image's own tree.
- [vllm-tests](bundles/vllm-tests/candidate.json): vLLM 0028's logits-head
  test and the existing shared/Markov preparation test.
- [head-bench](bundles/head-bench/candidate.json): the real head shards,
  packed projection against cuBLAS, with the default configuration.
