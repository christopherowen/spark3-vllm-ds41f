# B12X patch stack

Base: `local-inference-lab/b12x@f1c4e9dd5b1d841c10cda70ee49c3190cf3db0be`.

First apply upstream correctness commit
`02407f653609e2c4c49db2efb5f77835d2e04e32`, then the local series.

The current image differs from that base in exactly six B12X files:

- `b12x/_lib/intrinsics.py` — the already-upstream E4M3 correctness commit;
- `b12x/attention/_shared/mla/prefill_mg.py` — prefill index bounds;
- `b12x/norm/mhc/_policy.py` — DeepSeek prefill policy;
- `b12x/comm/roce/_proxy.py` — peer-aware RoCE proxy;
- `b12x/comm/roce/_roce_proxy.c` — peer-aware RoCE transport;
- `b12x/comm/roce/roce_oneshot.py` — peer routing and dual-HCA selection.

The three local deltas are exported as independent patches in `series`. A prepared
upstream tree must reproduce the active B12X source hash before the new Docker build
is promoted.
