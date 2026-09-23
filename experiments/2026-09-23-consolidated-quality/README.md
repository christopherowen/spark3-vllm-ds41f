# Consolidated source quality qualification

The current-head 10-vLLM/5-B12X candidate serves with guarded TP3 but can
produce malformed code from an identical temperature-zero LRU request. This
experiment holds the image, model, prompt, and serving settings fixed while
isolating the faulty generation path. No setting is eligible for promotion on
the strength of a single valid response.

The first controlled variable is RoCEnante collectives versus NCCL. Both
cluster configurations use the exact existing image and guarded limits;
`cluster-roce.json` reproduces the present setup and `cluster-nccl.json` changes
only `VLLM_ENABLE_ROCE_ALLREDUCE`. Use `probe_quality.py` against each profile
and retain every complete output. The test's Python syntax and `put` signature
checks are a focused gate for the observed failure, not a general model-quality
evaluation.

The prior B12X numerical attention test used 64-token pages while the service
uses 128-token pages. On dgx3, a capped 24-head FP8 decode test with 128-token
pages and a matching two-page oracle passed in 4.74 seconds. An initial local
test edit changed page size without changing the two-page fixture width and
failed in the reference gather; it did not reach a valid kernel comparison.
The permanent source test must include both page geometries before promotion.
