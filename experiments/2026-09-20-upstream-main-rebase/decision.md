# Decision

Status: **MoE compatibility correction qualified; immutable rebuild required**

The source rebase is prepared and every local change is separated into a
reviewable commit. Static checks pass. The native DS4.1 model now reaches B12X
sparse MLA, MoE, and RoCEnante through current prepared-plan interfaces,
ordinary weights load through the coherent GB10 path, and native Engram rows
are staged from disk instead of consuming about 63 GiB per rank. B12X declares
the same CUTLASS DSL 4.7.1 toolchain as vLLM.

The first candidate image was built and distributed but deliberately not
launched because its probe revealed missing downstream loader contracts. It is
superseded by this source revision. The promoted containers are stopped but
retained for rollback.

The patch-0014 image completed checkpoint loading but proved that the MoE carry
still targeted B12X's preceding API. Patch 0015 replaces that boundary with the
current structured preparation API and passes real W4A16/W4A8, multi-capacity,
repeated-preparation, and CUDA-graph tests on one Spark. This is the final
identified compatibility correction, not yet proof of a full TP3 startup.

The remaining work is target evidence:

1. rebuild the content-addressed ARM64 candidate on dgx3;
2. compile and run the selected B12X kernels and RoCEnante plan on all ranks;
3. establish deterministic output parity, four-by-128K capacity, memory, TTFT,
   single-stream TPS, and eight-stream aggregate TPS against the live baseline.

The nodes are now free for qualification. Dependency suppression or an in-place
nightly downgrade remains unacceptable. See `compatibility.md` and
`donor-analysis.md`.

The matching ARM64 image must pass the full target-hardware quality/performance
matrix. In particular, DS4.1 B12X attention, Engram storage, TP3 virtual-head
loading, RoCEnante transport, and allocator capacity must be validated together
before changing `upstreams.lock.json`, `config/`, or the promoted baseline.
