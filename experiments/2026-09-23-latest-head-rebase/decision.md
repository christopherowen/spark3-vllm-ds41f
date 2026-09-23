# Decision

Keep the consolidated current-head source as an unpromoted experiment. The
smaller B12X loader-lifecycle fix did not recover startup memory headroom: its
guarded load crossed the dgx3 5 GiB floor after weights. Patch 0005 restores
scoped managed final weights on GB10 while retaining upstream's bounded reader.
The new image passed a capped native copy, guarded TP3 startup, API readiness,
159K context admission, and the 30-request serving matrix with active guards.

Do not promote yet. The same temperature-zero LRU request produced malformed
identifiers twice, and a thinking-disabled request also produced an undefined
assignment. Logprobs favor stray suffixes immediately after confidently
generated identifiers, pointing to model-state/logit behavior rather than
response formatting. The 256-token code throughput cases may contain only
reasoning tokens and cannot establish code quality. Keep the candidate running
for controlled diagnosis; compare the exact quality prompt against a known-good
arm before changing CUDA graphs or DSpark. The large weekend speed gap remains.
