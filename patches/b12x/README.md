# B12X patch stack

Base: `local-inference-lab/b12x@0f84621264a1eef19e5bcb195fd3fa4f168c2976`
(`integration/karmic-kraken-beta`), which already contains upstream correctness
commit `02407f65`.

- `0001-switchless-rocenante.patch` adds per-peer HCA routing for the
  direct-cabled three-node ring; Local Inference Lab's launcher targets a
  switched fabric. Applying it to the base yields tree `d661de31`, the tree
  recorded in the promoted image's `local.spark3.b12x.tree` label. Qualified on
  the three Sparks by the 2026-09-23 karmic-kraken experiment (serving matrix and
  LRU gate).

Not in the series: the W4A8 tiny-decode `swiglu_limit` fix
(`experiments/2026-09-23-karmic-kraken-reference/patches/b12x/0002-tiny-decode-swiglu-limit.patch`)
is an upstream contribution. The promoted runtime disables tiny decode instead
(`B12X_W4A8_TINY_DECODE=0`).
