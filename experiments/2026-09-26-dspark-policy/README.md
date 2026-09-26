# DSpark verification policy and draft cost

Base: the promoted `2026-09-25-karmic-kraken-r3-vision-kernel70`
configuration, on candidate image `vllm-ds41f-kkref:01f1b874c774-r4a`
(vLLM series 0001-0008, tree `0debe853`). Patches 0005-0008 are off unless
their variable is set, so one image serves every arm.

## Hypotheses

1. Pinning adaptive verification's cost curves
   (`SPARK3_DSPARK_COST_DIR`, patch 0005) removes most of the ~3%
   boot-to-boot throughput variation, which comes from the startup profile.
2. The marginal-rate verification rule (patch 0006) raises accepted tokens
   per step at unchanged step cost, most on code, where confidence varies
   most between steps.
3. The gathered Markov bias (patch 0007, `dspark_draft_topk` 1024) shortens
   the drafter by reading 1,024 rows of `markov_w2` per draft position
   instead of all 129,280, without lowering acceptance.
4. A five-draft trace at temperature 0 (patch 0008) prices draft counts,
   rules and confidence calibration offline.

## Arms

`make_arms.py` derives every arm from `config/cluster.json`. All arms use
the r4a image and its own compile cache, drop
`--default-chat-template-kwargs {"thinking":true}` (thinking is already the
default when a request names neither key, and the explicit default
overrode a client's `enable_thinking: false`), and profile 15 replays into
a pinned cost directory.

| Arm | Change from `base` |
|---|---|
| `base` | none (all new paths off; pinned curves in `dspark-costs/k3`) |
| `marginal` | `SPARK3_DSPARK_VERIFY_RULE=marginal`, cost scale 1.0 (the marginal rule prices drafts directly; scale 2.0 would double-count) |
| `topk` | `dspark_draft_topk` 1024; own pinned curves and compile cache |
| `k5trace` | 5 drafts, capture sizes to 48, trace to `dspark-trace/k5.jsonl`; replay only, never timed |

## Workload

`run_arm.sh ARM LABEL --suites quality,decode --decode-cases
prose,code,prose-nothink,code-nothink,json-nothink`: the LRU gate and the
temperature-0 decode matrix at 1/2/4/8 streams. `prose` and `code` are the
reference cases (reasoning on, the server default); the `-nothink` cases
turn reasoning off to measure answers, and `json-nothink` is structured
output. Arms alternate (base, marginal, base, marginal) so each pair spans
two boots.

## Gates

LRU 5/5 on every arm; no failed requests; memory guards unchanged (5 GiB
startup, 3 GiB steady), and dgx1's lowest MemAvailable recorded per arm.
`--allow-mismatch` because `doctor --live` reports only the nvidia-drm
modeset host setting (display), which does not touch inference.

## Results

Pending.
