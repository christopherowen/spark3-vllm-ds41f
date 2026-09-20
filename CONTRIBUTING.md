# Contributing

This repository owns deployment composition, configuration, experiments, and
evidence. It does not replace the vLLM, B12X, FlashInfer, or CUTLASS projects.

For a deployment change, open a branch here and follow `docs/methodology.md`. For
source work, first identify the owner in `upstreams.lock.json`, prepare that tree
with `bin/spark3 upstream prepare NAME`, and create one narrowly scoped branch in
the appropriate source fork. Do not combine unrelated vLLM and B12X changes into
one upstream contribution.

Remote names have fixed meanings:

- `upstream` is the canonical external project;
- `origin` is Christopher Owen's fork of that project, when recorded;
- this repository's `origin` is only the deployment repository.

After an upstream merge, advance the pinned commit, remove any now-redundant
local patch, rebuild one immutable image, and qualify it as an experiment before
promotion. Preserve the upstream pull-request link and benchmark receipts in the
promotion commit.
