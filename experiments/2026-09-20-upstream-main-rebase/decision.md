# Decision

Status: **source-complete; await target qualification**

The source rebase is prepared and every local change is separated into a
reviewable commit. Static checks pass. The native DS4.1 model now reaches B12X
sparse MLA and RoCEnante through current prepared-plan interfaces, and B12X
declares the same CUTLASS DSL 4.7.1 toolchain as vLLM. No image has been built
or deployed, and the live vLLM service has not been restarted or modified.

The remaining work is target evidence, not another source port:

1. build the content-addressed ARM64 candidate when a Spark is free;
2. compile and run the selected B12X kernels and RoCEnante plan on all ranks;
3. establish deterministic output parity, four-by-128K capacity, memory, TTFT,
   single-stream TPS, and eight-stream aggregate TPS against the live baseline.

The current nodes have only about 5--6 GiB available while serving, so an image
build or pull was deliberately not started. Dependency suppression or an
in-place nightly downgrade remains unacceptable. See `compatibility.md` and
`donor-analysis.md`.

The matching ARM64 image must pass the full target-hardware quality/performance
matrix. In particular, DS4.1 B12X attention, Engram storage, TP3 virtual-head
loading, RoCEnante transport, and allocator capacity must be validated together
before changing `upstreams.lock.json`, `config/`, or the promoted baseline.
