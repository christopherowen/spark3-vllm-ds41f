# Decision

Status: **Patch-0019 image qualified; startup-headroom policy under TP3 qualification**

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
repeated-preparation, and CUDA-graph tests on one Spark. Its TP3 launch then
showed that the pre-memory collector constructed unrelated warmup units before
filtering them. Patch 0016 gives that phase a dedicated provider method exposed
only by MoE, and its regression fails if an ordinary provider is even invoked.
That image cleared both failures and exposed the same preparation-API drift in
the ModelOpt MXFP8 linear adapter during memory profiling. Patch 0017 prepares
bounded capacity and exact graph regimes before KV admission and passes a real
numerical adapter test on GB10. Its immutable image passes the selected source,
numerical, disk Engram, MoE, CUDA-graph, and style gates. This is not yet proof
of a full TP3 startup.

That image then loaded the complete target and DSpark models, prepared all 210
target-model MXFP8 plans plus 40 MoE plans, and entered the real profile run.
The profile exposed one omitted ownership root: the collector scanned
`worker.get_model()` but not the separately owned `worker.get_draft_model()`.
Patch 0018 enumerates both roots and deduplicates shared layers by their existing
plan key. The failure was not an OOM and did not require a configuration change.
Its immutable image passes the complete selected-path prelaunch suite on GB10.

The next launch proved that fix by preparing 226 MXFP8 and 43 MoE plans, then
reached the first eligible small collective during the real profile run.
RoCEnante rejected it because its plan was still deferred to the normal
post-profile warmup. Patch 0019 exposes the same explicit pre-profile contract
on RoCEnante and includes eligible distributed providers in that phase.
Ordinary post-cache providers remain structurally excluded. This was not an OOM.

The immutable patch-0019 image now passes the selected source, numerical, disk
Engram, MoE, CUDA-graph, RoCEnante, and style gates on GB10. The remaining work
is target evidence:

Its first TP3 launch loaded the target and DSpark model but exhausted physical
host headroom during pre-profile preparation. All three kernels recorded NVIDIA
`NV_ERR_NO_MEMORY`; Docker's cgroup OOM flag remained false. The candidate now
caps resident container memory at 110 GiB and permits up to 6 GiB of swap only
inside a 116 GiB memory-plus-swap ceiling. This preserves the 3 GiB KV cache and
all quality/performance settings while giving the driver deterministic physical
headroom. Promotion requires bounded, non-growing swap and unchanged steady
performance.

1. distribute the qualified content-addressed ARM64 image to all ranks;
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
