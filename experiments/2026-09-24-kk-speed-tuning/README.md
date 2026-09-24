# Karmic-kraken speed tuning

Base: the 2026-09-24 promoted baseline (`base.json`). Each arm changes one
variable of `config/cluster.json` and is measured on the three Sparks with the
same client as the baseline: `measure.sh` runs `run_arm.sh` twice (prose and
code, concurrency 1-8, 256 output tokens, then the five-repeat LRU gate) and
records the minimum MemAvailable per node (`sample_mem.sh`). Two passes per arm
because single passes vary by about ±7%. The unchanged 5 GiB startup and 3 GiB
steady memory guards stay in force; a guard kill fails the arm.

## Arms

1. `capture32`: CUDA-graph capture sizes extended with 28 and 32. With DSpark
   verifying three draft tokens, seven or eight streams issue 28- or 32-token
   decode batches, which the baseline's largest graph (24) cannot replay; those
   steps run eagerly.
2. B12X autotune with a bounded candidate-race budget. vLLM creates B12X's
   `PreparationSession` without a budget, so racing may hold half of free
   memory; on unified memory that has tripped the startup guard before. One
   budgeted tuning run fills the persistent selection cache
   (`/cache/kkref/jit/b12x/cute/preparation`), then the arm is measured.
3. `batched8192`: 8,192 batched tokens (the 2026-09-20 value) for long-prompt
   prefill; needs a prefill workload in addition to the decode matrix.
4. DSpark depth 5 or 7 (Local Inference Lab's recipe uses 7), with capture
   sizes covering `(1 + depth) x 8`.

Acceptance: faster than the baseline beyond pass-to-pass variation, LRU 5/5 in
both passes, no guard kill, and at least 3 GiB above the steady guard on every
node.
