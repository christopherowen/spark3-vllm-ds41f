# Decision

Keep the consolidated current-head source as an unpromoted experiment. The
smaller B12X loader-lifecycle fix did not recover startup memory headroom: its
guarded load crossed the dgx3 5 GiB floor after weights. Patch 0005 restores
scoped managed final weights on GB10 while retaining upstream's bounded reader.
The new image passed a capped native copy, guarded TP3 startup, API readiness,
159K context admission, and the 30-request serving matrix with active guards.

Do not promote yet. The same temperature-zero LRU request intermittently
produces malformed identifiers and undefined assignments. The exact generated
token-ID prefix gives a correct next token when supplied as one prompt, while
an incremental request from that prefix can produce a stray suffix. The
defect also occurs through completions, and a successful repeat proves it is
not deterministic. `quality-diagnosis.md` records the bounded comparison and
remaining state-path candidates. The 256-token code throughput cases may
contain only reasoning tokens and cannot establish code quality. Keep the
candidate running for controlled diagnosis before changing CUDA graphs or
DSpark. The large weekend speed gap remains.
