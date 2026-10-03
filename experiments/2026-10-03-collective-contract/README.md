# Explicit collective policy

The measured balanced candidate correctly dispatches all-reduces above 1 MiB
to NCCL, but its vLLM adapter reports 2 MiB because it reads registered capacity.
Its `all_reduce_max_bytes` property also describes eligibility while returning
capacity. This successor fixes the contract and removes backend-error fallback
from explicitly selected multi-node RoCEnante groups.

Base deployment commit: `ccbc5db`. Source manifest: [source.json](source.json).
Candidate: [candidate.json](candidate.json). Source patch:
[vllm/0027-explicit-collective-policy.patch](vllm/0027-explicit-collective-policy.patch).
It follows production vLLM patches 0001–0026 and uses the exact B12X and NCCL
sources from [the measured candidate](../2026-10-03-balanced-policy/selected.json).
The authored vLLM branch is `explicit-collective-policy`.

**Status:** source-qualified, unbuilt and not deployed. Launch is disabled.
The measured image and profiles remain frozen; their measurements do not qualify
this successor image. No cluster operations were performed for this change.

**Drafter heads:** since 2026-10-03 the candidate keeps the DSpark draft and
Markov heads at the checkpoint's BF16 (`VLLM_DS41_DRAFT_NVFP4_HEAD=0`,
`VLLM_DS41_MARKOV_NVFP4=0`) with its own DSpark cost directory
(`ring4-collective-nativeheads-20261003`). The
[native drafter heads record](../2026-10-03-native-drafter-heads/README.md)
has the audit and the TP4 measurement.

## One policy in execution and reporting

| Quantity | Selected value | Meaning |
| --- | ---: | --- |
| All-reduce dispatch maximum | 1,048,576 bytes (1 MiB) | Largest eligible input sent through RoCEnante |
| All-reduce registered capacity | 2,097,152 bytes (2 MiB) | Preparation/priming capacity; not the serving dispatch cutoff |
| All-gather dispatch maximum | 2,097,152 bytes (2 MiB) | Input shard per rank; a four-rank result can be 8 MiB |
| Sequence-parallel prefill boundary | 205 tokens | Existing conservative floor derived from registered capacity; transport tuning does not also retune model scheduling |
| NCCL buffer setting | 4,194,304 bytes (4 MiB) | Independent NCCL protocol buffer; not any of the preceding limits |

The adapter exposes `all_reduce_max_bytes`, `all_reduce_capacity_bytes`, and
`all_gather_max_bytes`. Startup and first-use messages read these runtime
properties. The SP helper explicitly reads capacity and logs that basis.
B12X continues to validate the dispatch cutoff across peers; no duplicate vLLM
eligibility predicate or independently parsed dispatch limit was introduced.
The capacity/SP coupling is now documented, not removed. Retuning SP independently
of capacity is a separate experiment.

For the selected multi-node TP group:

| Operation/condition | Deliberate execution path |
| --- | --- |
| All-reduce: contiguous local CUDA FP16/BF16/FP32, positive size divisible by 16, up to and including 1 MiB | RoCEnante |
| Other NCCL-supported all-reduce inputs | Direct PyNCCL; no accelerator search |
| All-gather: contiguous local CUDA shard, positive size up to and including 2 MiB, dimension 0 or last, non-scalar/non-bool/non-complex/non-sparse | RoCEnante |
| Other NCCL-supported all-gather inputs | PyNCCL with layout conversion |
| Reduce-scatter | PyNCCL |
| Variable all-gather/reduce-scatter | PyNCCL; the existing batch-invariant root-reduce/scatter branch is retained when that mode is enabled |
| Missing capability, differing peer configuration or failed RoCE construction | Initialization error |
| Required communicator lost/disabled, transport exception, or all-reduce unexpectedly returning no result | Error; no switch to another backend |

RoCEnante gather still has two deliberate layouts: direct output for aligned
rows, or padded scratch plus reshape otherwise. Both use the same transport.
A prepared low-level all-reduce can use the full 2 MiB capacity during priming;
serving dispatch cannot. FP16 is supported by the source predicate but was not
included in the prior BF16/FP32 hardware timing screen.

Non-RoCE groups keep upstream dispatch. In the recorded selected boot, TP logged
`B12X_ROCENANTE, PYNCCL`, while the separate EP group logged `PYNCCL`. CPU/Gloo
setup and control traffic are also outside this TP policy. Single-node RoCE
construction still skips the transport. Contradictory explicit enable/disable
or PCIe/RoCE selections now produce an error.

## Fallback audit

These findings refer to the pinned source, not every possible upstream version.
Paths below are relative to the prepared vLLM tree.

| Path | Finding and disposition |
| --- | --- |
| `distributed/device_communicators/b12x_roce_all_reduce.py` | Previously warned and disabled RoCE after capability votes or constructor errors. The candidate raises instead and includes PyNCCL availability in the capability vote. |
| `distributed/device_communicators/cuda_communicator.py` | Previously allowed the generic accelerator chain after RoCE declined, and a PyTorch process-group all-reduce if PyNCCL was disabled or returned `None`. The selected RoCE policy now bypasses that chain and rejects lost backends. Gather/scatter paths similarly bypass optional symmetric-memory/AITER paths. |
| B12X `comm/roce/roce_oneshot.py` | Already rejects closed runtimes and checks health on execution. Exceptions are not retried through NCCL. Aligned/padded gather is a layout decision, not failure recovery. |
| `model_executor/kernels/linear/__init__.py` | An explicit linear backend with no kernel for a layer type can warn and return the automatic kernel list. This remains upstream behavior; this audit has not established which serving layers hit that branch. |
| `model_executor/kernels/linear/b12x_unquantized.py` | Generic unquantized linear routing intentionally uses `F.linear` when no eligible native plan exists, inputs are unsuitable, or preparation selected the torch backend. Eligibility also depends on autotuning being enabled. It is not proof that every DS4.1 projection takes that generic path. |
| `models/deepseek_v4_1/b12x_layers.py`, `compressor.py` | DS4.1 BF16 projections use an exact row-count plan when present, otherwise their prepared capacity plan. mHC similarly uses a capacity plan within bounds and raises above capacity. These are actual model plan substitutions and remain unchanged. |
| `models/deepseek_v4_1/nvidia/b12x_vision.py` | Vision linears also substitute the prepared capacity plan for an unlisted row count. |
| `model_executor/layers/fused_moe/b12x.py` | Unsupported B12X MoE quantization schemes raise; this is not a generic retry into another MoE backend. |

The broader model is not yet a fully enumerated, closed execution plan. The next
useful boundary is a per-layer manifest of prepared plan, permitted capacities,
backend and substitution reason, checked at startup and capture. Do not remove
all PyTorch or capacity paths indiscriminately: mixed layer types and unlisted
row counts need supported implementations. Reject an unexpected selection once
its intended alternatives are specified and qualified.

NCCL's internal protocol/channel choices remain intentional implementation
choices under the existing Ring/network profile. This source change does not
claim balanced payload for every variable collective, point-to-point operation,
or sub-alignment-sized payload; the earlier hardware evidence covers its stated
all-reduce/all-gather/reduce-scatter cases only.

## Validation

Run from the deployment repository:

```sh
TMPDIR=/tmp bin/spark3 --cluster-config experiments/2026-10-03-collective-contract/candidate.json doctor
TMPDIR=/tmp bin/spark3 --cluster-config experiments/2026-10-03-collective-contract/candidate.json build prepare
python3 experiments/2026-10-03-collective-contract/test-contract.py
TMPDIR=/tmp bin/spark3 --cluster-config experiments/2026-10-03-collective-contract/candidate.json build check
python3 -m unittest discover -s tests
git diff --check
```

Completed locally: both default and candidate doctor, fresh preparation of all
sources and verified wheels, build-input verification, **14 contract tests**,
**176 repository tests**, and lint/format checks of changed source and test files.
Contract tests execute the actual source methods with mocked dependency
boundaries on a CPU host. They cover cutoff boundaries, all three reduction
dtypes, input-shard accounting, distinct capacities, the 205-token SP boundary,
startup votes, failed backends, explicit NCCL selection, and log contents.
They do not validate CUDA execution or a distributed failure-injection run.
CI now prepares this exact vLLM/B12X series and runs the contract tests too.

Before enabling deployment: build a new immutable image, run a coordinated
four-rank startup/collective/serving smoke, verify the new messages, verify graph
capture/replay and health propagation, and check that the prepared policy is the
same on all ranks. The previous performance screen need not be repeated in full
for logging; any change to active backend selection or the SP boundary needs
new matched measurements. Full upstream vLLM GPU tests have not run here.
