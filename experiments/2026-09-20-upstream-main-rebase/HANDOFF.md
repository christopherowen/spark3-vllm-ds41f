# Garbled-output isolation handoff

Last updated: 2026-09-23 (Europe/Belgrade)

## Latest live finding

The candidate's `VLLM_USE_V2_MODEL_RUNNER=0` selected
`vllm/v1/worker/gpu_model_runner.py`, which does not initialize
`DeepseekV41ModelState` or call the disk Engram preparation path. A guarded
live request exposed 94 NaNs in layer-1 `staged_rows` **before** TP reduction
on every rank; an instrumented `Engram.prepare_disk` wrapper was never called.
These were stale/unwritten staged rows, not a RoCEnante or WKV result. See the
q3/q4 receipts in `runs/`.

An otherwise identical V2-runner launch initialized `DeepseekV41ModelState`
on all three ranks with one disk Engram model and a three-token lookback.
`prepare_disk` ran for every request step, staged rows, WKV, and gate were
finite, and the deterministic request returned HTTP 200. Its text was still
garbled (`" selectsbedaelanibit"` for `"1 + 1 ="` at temperature zero), so
correctness is unresolved. See `runs/launch-v2-live-trace-q6.json`. The next
diagnostic compares live layer-0 B12X attention output against its compressed
reference and records attention/MoE/decoder activation ranges by layer.

That comparison passed closely on layer 0 (maximum absolute error 0.0625 on
native/reference values of magnitude at most 3). The layer-2 decoder residual
then jumped from about ±1 to about ±800, while its MoE output remained small.
The next live trace compares `mhc_shifted_post_pre`'s residual update against
the direct tensor formula and records the carried MHC mix ranges. See
`runs/launch-v2-numeric-q7.json`.

The MHC residual update matches its direct tensor reference; the large value
arrives as the layer-2 attention output (about ±4,672) while layers 0 and 1
are about ±4 and ±6. The next trace extends the live-cache native/reference
B12X attention comparison to layers 1 and 2. See
`runs/launch-v2-mhc-q8.json`.

## Review correction: disk Engram collective

Source review found that the proposed dim-0 all-gather probe does **not**
exercise this candidate's disk-Engram path. `cluster.json` selects
`table_memory="disk"`; the NVIDIA Engram implementation rejects sequence
parallelism in that mode and its `embed()` calls
`tensor_model_parallel_all_reduce` on `[5, 24, 256]` BF16 staged rows. The
dim-0 gather and `_engram_select_rows` belong to a different Engram mode. The
`probe_engram_collective.py` now tests the actual all-reduce shape against NCCL.
The corrected probe passed on all three nodes under an 8 GiB container cap;
see `runs/probe-engram-reduce-d189f87.json`. The earlier two setup failures
were for the irrelevant gather probe and are not evidence about this failure.

The WKV replay below used synthetic finite activations, not the all-reduced
checkpoint rows. It validates weight loading and an isolated numerical path,
but it does not rule out WKV output on the real payload. The first observed
non-finite tensor remains the following MHC input; attribution to Engram is a
strong source-order inference until intermediate Engram values are measured.

## Objective

Restore correct native-weight DeepSeek V4.1 Flash inference on the three-node
switchless DGX Spark cluster while retaining the upstream-main vLLM rebase and
the smallest defensible B12X integration.  Do not trade model quality for the
fix.  Once correctness is restored, resume TTFT, TPS, and memory tuning.

## Safety and current cluster state

- Never reboot or deliberately wedge a DGX.  The machines cannot be physically
  power-cycled for about one week.
- Use bounded, fail-closed tests.  Do not start the full model casually.
- Do not start DSpark, speculative, auxiliary, reranking, or application
  workers.  The current qualification target is target-only inference.
- `deployment.launch_enabled` remains `false` in the candidate cluster file.
- `dgx1`, `dgx2`, and `dgx3` are reachable and currently idle.  At handoff,
  each reported about 118 GiB available memory and no running containers.
- Preserve the stopped candidate containers and their logs as evidence:
  `dsv41-upstream-67e0-b12x-b69f-cachefix-q2` is exited with code 0 and
  `OOMKilled=false` on every node.
- The disposable collective-probe containers have exited and were removed.

The candidate configuration intentionally remains conservative while
correctness is unresolved: V1 runner, CUDA graphs `NONE`, 2 GiB KV cache,
160K advertised context, text-only, target-only, 5 GiB startup memory guard,
and 3 GiB steady-state guard.

## Candidate identity

- Image tag:
  `vllm-ds41f-upstream-main:67e0b2ada271-b69feee3022c-de7d44fd41ad6`
- Image ID:
  `sha256:5627de1112656aaab227d26cb3bc0edfa46e5f54934000e4ec6f83f9a70e545a`
- vLLM patch head:
  `67e0b2ada271f75b2f9f0de473889a0f99b2062c`
- B12X patch head:
  `b69feee3022c05fa840833a80dd4010156cf7fe1`
- Prepared source tree: `/tmp/spark3-prepare-ded2f`
- Repository and nodes were last synchronized at repository commit:
  `eb2476df2576717758601eef050f948e9185d91d`
- Model snapshot:
  `dba1be0a40aa45a94ad051997016db3960a90277`

## Proven failure boundary

Patch 0028 corrected a real B12X cache-record ABI mismatch and the associated
RoPE latent shape.  Its focused source tests and GPU graph replay passed.  A
fresh-cache, guarded three-rank startup then completed safely and reached API
readiness.

The deterministic request `"1 + 1 ="` (`max_tokens=4`, temperature zero)
returned HTTP 500.  The first observed all-NaN tensor was shape
`[5, 4, 5120]`, BF16, immediately after the checkpoint's first Engram
injection (layer 1) and before the following MHC pre-normalization.  All
102,400 values were NaN and none were infinity.  The layer input, layer-0
attention reference, and preceding MHC post boundary were finite.  This
corrects the earlier, inaccurate attribution to layer-0 MHC input.

See `runs/launch-5627de111265-mhc-input-nan.json`.

## Boundaries conclusively ruled out

### Disk-backed Engram lookup

`probe_engram_checkpoint.py` replayed the exact prompt through B12X's real
io_uring disk-Engram path on all three nodes and both checkpoint Engram owners
(layers 1 and 14), one TP rank at a time.

- Token IDs were `[19, 940, 223, 19, 438]` on every rank.
- Layer-1 hash digest was identical on all ranks:
  `86ccd773ea3e0f57f6b7584382710b376ebad960738b683ec8f29616ccd56bfe`.
- Layer-14 hash digest was identical on all ranks:
  `26166708844c97a98c13c887489295ba8b04e7d002f4e923b03a5debf9934c5e`.
- Each rank/layer produced 30,720/30,720 finite BF16 values.
- All 40 local rows used by the prompt matched direct checkpoint reads exactly;
  the other 80 rank-owned slots were correctly zero.

The immutable checkpoint ranges, hash computation, direct reader,
dequantization, and local staging are therefore not the source.

### Resident q/k weights and ModelOpt MXFP8 WKV on synthetic input

Static checkpoint inspection found finite q/k weights, valid WKV value bytes,
and valid E8M0 scale bytes.  The new `probe_engram_wkv.py` then reproduced the
production managed-allocation, deferred-read flush, 32-row scale expansion,
B12X pack, prepared plan, and real MXFP8 GEMM on dgx1 under an 8 GiB container
cap.

Layer 1 results:

- q SHA-256:
  `0aca22a679f1a2479e8065a7b3c64e035b13fe1fa63189644e3556b0a333da56`
- k SHA-256:
  `61152efdabab20ca8c29fe2e087d7de9c5c1684bfef930fa00541efe58fff422`
- q and k: 20,480/20,480 finite each.
- WKV scale bytes: range 109 through 117, zero `0xff` bytes.
- Sampled WKV values: 65,536/65,536 finite.
- Real projection: 128,000/128,000 finite, zero NaNs, zero infinities.
- Maximum error against the independent dequantized BF16 reference over 32
  output rows: `0.0009765625`.

This disproves the working theory that B12X deferred reads were flushed only
after vLLM packed WKV. The loader flushes before generic quantization
post-processing. The replay does not establish WKV behavior for the real Engram
rows after their TP reduction.

### Existing fused-kernel coverage

The upstream-tree Engram fused post-WKV test compares the Triton gate/residual
kernel against a Torch reference for the real `hc_mult=4`, `dim=5120`
geometry.  It passes synthetic finite inputs.  That does not yet prove the
complete real distributed Engram path, but there is no evidence of a general
gate-kernel arithmetic defect.

## Current open boundary

The first unvalidated operation between finite local staged rows and the
synthetic-input WKV replay is the disk path's distributed TP all-reduce:

1. each rank stages `[5, 24, 256]` BF16 rows, zero for rows it does not own;
2. TP all-reduce sums the three staging buffers;
3. WKV projects the reduced rows;
4. the fused q/k gate adds the shared value to the hidden state.

The candidate routes eligible all-reduces through RoCEnante. The actual failing
Engram path is not covered by the single-rank disk or synthetic-input WKV
probes.

## Collective probe status

`probe_engram_collective.py` now compares the `[5, 24, 256]` BF16 RoCEnante
all-reduce byte-for-byte with NCCL. It allocates only small control buffers and
tens of KiB of payload. The test containers must remain capped at 8 GiB.

Two setup attempts did not reach a payload:

1. The first omitted production's `NCCL_CROSS_NIC=1` and
   `NCCL_IB_SUBNET_AWARE_ROUTING=1`.  NCCL consequently attempted invalid
   non-adjacent ring-port pairs and timed out during endpoint exchange.  This
   result is invalid as a model/network verdict.
2. The second used the complete production NCCL environment.  NCCL endpoint
   exchange succeeded, but B12X's native RoCEnante queue pairs failed to enter
   RTR on all ranks.  The probe was still using NCCL itself as B12X's endpoint
   exchange group, unlike vLLM, which gives B12X a Gloo CPU group and retains a
   separate NCCL device group.

All four ConnectX ports currently report `PORT_ACTIVE`, MTU 4096, Ethernet
link-layer.  Explicit per-interface pings pass in both directions over all six
ring subnets (`192.168.0.0/24` through `192.168.5.0/24`).  The preceding full
candidate launch logged RoCEnante ready with all four HCAs and later logged its
all-gather and all-reduce paths live.  Do not conclude that production
RoCEnante is broken from the two probe setup failures.

The corrected probe mirrors the production group split: Gloo for B12X endpoint
exchange and a separate NCCL group for the numerical oracle. It returned the
same 30,720 finite values and SHA-256 hash on all three ranks, exactly matching
NCCL, with no reported RoCEnante transport errors or OOM kills. This clears
the synthetic collective geometry; it does not yet validate the real rows after
reduction.

The real-row TP3 probe also passed checkpoint row verification (40 slices per
rank) and RoCEnante/NCCL bitwise equality. The reduced rows are identical on
all ranks and finite. The WKV output and fused gate are finite, but WKV hashes
differ by rank despite identical reduced input. See
`runs/probe-engram-real-pipeline-454c8e0.json`. The probe now records loaded
and packed WKV hashes and compares the first 32 projected columns against an
independent dequantized MXFP8 reference to identify the source of that
difference.

## Suggested next steps

1. Run the bounded V2 configuration `cluster-v2-trace.json` with the updated
   attention and per-layer activation probe. Preserve the old stopped service
   containers; use a distinct container name and rendezvous port. Keep target
   only, CUDA graphs `NONE`, 2 GiB KV, the 5/3 GiB memory guards, and one
   tiny deterministic request.
2. Compare native layer-0 B12X attention against the live-cache compressed
   reference. Locate the first layer with a large activation jump. Isolate
   that layer's attention, MHC, or MoE output and fix the proven numerical
   cause; repeat the guarded request.
3. After the deterministic request produces correct text, run several prompt
   and decode checks and refresh the run receipts. Then decide which minimal
   fixes belong upstream in B12X or vLLM before promotion.

## Relevant source paths

- Candidate configuration: `candidate.json`, `cluster.json`
- Failure receipt: `runs/launch-5627de111265-mhc-input-nan.json`
- Exact disk lookup probe: `probe_engram_checkpoint.py`
- Exact managed WKV probe: `probe_engram_wkv.py`
- Completed synthetic TP3 collective probe: `probe_engram_collective.py`
- Real-row TP3 pipeline probe: `probe_engram_pipeline.py`
- Common Engram implementation:
  `/tmp/spark3-prepare-ded2f/vllm/vllm/models/deepseek_v41/common/engram.py`
- B12X disk Engram implementation:
  `/tmp/spark3-prepare-ded2f/vllm/vllm/models/deepseek_v41/nvidia/engram.py`
- vLLM B12X loader:
  `/tmp/spark3-prepare-ded2f/b12x/b12x/integration/vllm/loader.py`
- Direct checkpoint loader:
  `/tmp/spark3-prepare-ded2f/b12x/b12x/loader/_checkpoint.py`
- B12X MXFP8 vLLM kernel:
  `/tmp/spark3-prepare-ded2f/vllm/vllm/model_executor/kernels/linear/mxfp8/b12x.py`
- RoCEnante vLLM adapter:
  `/tmp/spark3-prepare-ded2f/vllm/vllm/distributed/device_communicators/b12x_roce_all_reduce.py`
- RoCEnante runtime:
  `/tmp/spark3-prepare-ded2f/b12x/b12x/comm/roce/roce_oneshot.py`

## Repository state

Commit `d189f876534fa88315c8844ef0dcb8f425fe5fda` records the corrected
failure boundary, the first three probes, and this handoff. All three nodes
were synchronized to that commit before the passing collective probe. The
real-row pipeline probe and its result remain separate work in progress.

Do not discard unrelated user changes.  Do not promote or push a candidate
until correctness is demonstrated by a deterministic live request.
