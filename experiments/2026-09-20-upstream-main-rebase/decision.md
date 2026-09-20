# Decision

Status: **retain for more evidence**

The source rebase is prepared and its local changes are separated into
reviewable commits. Static checks pass, and the CPU-testable 3072-token policy
gate passes. No image has been built or deployed, and the live vLLM service has
not been restarted or modified.

Promotion first requires two independently reviewable integrations:

1. port the DS4.1 B12X sparse-MLA and RoCEnante consumer adapters to current
   vLLM interfaces without reviving the downstream model fork; and
2. resolve the declared CUTLASS DSL incompatibility: the exact vLLM nightly
   uses 4.7.1 while B12X `master` pins 4.6.2.

The preferred dependency path is a separately reviewable B12X 4.7.1 migration,
qualified on target hardware. Dependency suppression or an in-place nightly
downgrade is not an acceptable substitute. See `compatibility.md` and
`donor-analysis.md`.

After those gates, the matching ARM64 image must pass the full target-hardware
quality/performance matrix. In particular, DS4.1 B12X attention integration,
Engram storage, TP3 virtual-head loading, RoCEnante transport, and allocator
capacity must be validated together before changing `upstreams.lock.json`,
`config/`, or the promoted baseline.
