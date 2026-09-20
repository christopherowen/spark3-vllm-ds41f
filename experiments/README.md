# Experiments

Copy `_template/` to `YYYY-MM-DD-short-name/` before changing a candidate. Keep
raw results, including failures. An experiment does not change promoted state
until a later commit updates `config/` and adds an immutable baseline manifest.

Prefer one intended variable per experiment. If an interaction must be tested,
name every changed variable and include the factorial comparison needed to
separate their effects.
