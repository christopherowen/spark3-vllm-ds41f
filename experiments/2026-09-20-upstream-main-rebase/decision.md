# Decision

Status: **RoCEnante stripe-layout fix proven; immutable image qualification required**

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

The immutable patch-0019 image passes the selected source, numerical, disk
Engram, MoE, CUDA-graph, RoCEnante, and style gates on GB10. Its TP3 launch
loaded target plus DSpark at 98.1 GiB on every rank, then proved the distributed
priming unit lacked a rendezvous: ranks entered its first real collective seven
seconds apart and timed out at sequence 1. Patch 0020 synchronizes ranks after
local plan materialization and immediately before priming. The temporary Docker
memory-limit experiment did not address the cause and has been reverted. The
remaining work is target evidence:

The immutable patch-0020 image now passes the complete selected source,
numerical, disk Engram, MoE, CUDA-graph, RoCEnante, and style gates on GB10.

Its synchronized TP3 launch exposed one more carry boundary in our switchless
extension. Each rank opens four local HCA functions but routes two stripes to
each peer. The new B12X preparation refactor compiled the GPU flag layout with
the number of opened functions (four), while the unchanged proxy correctly
wrote the routed-lane layout (two). A model-free three-rank A/B proved the old
eager image and the rebased direct kernel both pass, while only the prepared
path failed with every RDMA write already complete. B12X commit `b69feee3022c`
uses `runtime.stripe_count` for prepared all-reduce and all-gather geometry.
The fixed prepared path now primes reductions and gather and matches NCCL from
16 bytes through 1 MiB on all ranks, using both functions on both peer links.

The corrected immutable image subsequently reached KV initialization with real
model traffic over RoCEnante. It allocated 1,360,738 KV tokens (8.50 times the
160K hard limit) before current upstream rejected the carried
`enable_adaptive_verification=true` setting. DeepSeek V4.1 hard-wires an indexer
backend that does not support device-decided query lengths on SM121. The
candidate therefore uses upstream's supported fixed-block DSpark verification.
This changes speculative scheduling rather than target-token quality and must
be quantified across the concurrency matrix.

The following launch cleared that guard and exposed the second stale runtime
setting: a 256-token manager block cannot be split into the 128-token SM121
kernel pages under the required block-outer `BLHNC` layout. The candidate now
uses 128, the exact common page size for the DeepSeek V4.1 indexer and B12X
attention. This is expected to reduce tail fragmentation at the cost of more
block-table entries; effective capacity and throughput remain measured gates.

1. complete TP3 startup with the supported fixed-block DSpark configuration;
2. establish deterministic output parity, four-by-128K capacity, memory, TTFT,
   single-stream TPS, and eight-stream aggregate TPS against the live baseline.

The nodes are now free for qualification. Dependency suppression or an in-place
nightly downgrade remains unacceptable. See `compatibility.md` and
`donor-analysis.md`.

The matching ARM64 image must pass the full target-hardware quality/performance
matrix. In particular, DS4.1 B12X attention, Engram storage, TP3 virtual-head
loading, RoCEnante transport, and allocator capacity must be validated together
before changing `upstreams.lock.json`, `config/`, or the promoted baseline.
