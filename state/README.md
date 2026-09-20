# Runtime deployment artifacts

These files are mounted read-only into the serving container and are part of the
promoted deployment, not node-local configuration:

- `config.json` is the native DeepSeek V4.1 Flash model configuration with the
  recorded TP3 virtual-head overlay.
- `indexer.py` is the materialized vLLM source produced by the pinned upstream
  revision plus `patches/vllm/series`.

Their SHA-256 values are recorded under `deployment_artifacts` in
`manifests/sources/2026-09-20-active-source.json` and checked by
`bin/spark3 doctor`. Regenerate them from the recorded inputs; do not edit a
copy on an individual Spark.

No credentials, tokens, private keys, or machine-local environment files belong
in this directory or anywhere else in Git.
