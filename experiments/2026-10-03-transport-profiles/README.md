# Named TP3 and TP4 transport profiles

Base deployment commit: `5a14239e40369557662212d3ba424a2dda287a13`.
This change organizes configuration; it changes no serving kernel, selected
transport setting or production default. No GPU measurement is needed to
establish configuration equivalence. Hardware qualification of the TP4 image
remains a separate promotion task.

[`profiles.json`](profiles.json) puts the transport tuning controls side by
side. `tp3` reproduces the promoted 64 KiB triangle configuration. `tp4`
reproduces the [explicit-policy candidate](../2026-10-03-collective-contract/README.md):
the measured balanced transport plus its source-tested vLLM policy/reporting
fix. That final image is **not built or hardware-qualified**. The registry
does not make either topology appropriate for different physical cabling.

| Setting | TP3 | TP4 |
| --- | --- | --- |
| Physical fabric | Direct triangle | Four-node neighbour ring |
| Far-rank transfer | Every peer is direct | Whole fragments, half each direction |
| RoCEnante all-reduce dispatch | 2 MiB, same as capacity | 1 MiB |
| RoCEnante all-reduce capacity | 2 MiB | 2 MiB |
| RoCEnante all-gather dispatch | 4 MiB input per rank | 2 MiB input per rank |
| Reduce-scatter | NCCL | NCCL |
| NCCL initialized channels | Upstream minimum; maximum 8 | Minimum and maximum 4 |
| NCCL buffer | 1 MiB | 4 MiB |
| NCCL LL128 buffer setting | 256 KiB; LL128 excluded | 256 KiB; LL128 excluded |
| NCCL algorithm | Existing upstream selection | Ring |
| NCCL direction/thread policy | Existing upstream selection | Mode 2, both directions and roots |
| NCCL traffic allocation floor | Upstream default | 512 bytes |
| NCCL thread thresholds | Upstream default | `-2 -2 -2 1 1 1` |

The table describes policy for eligible inputs. Tensor shape, dtype, alignment
and tiny indivisible payloads retain the documented eligibility boundaries.
Four initialized NCCL channels do not mean every call uses all four.

## Derived model layout

`tuning show` also reports `derived_layout`, calculated from the pinned
checkpoint dimensions in the hashed [checkpoint receipt](checkpoint-config.json), TP size and
the resolved graph/transport settings. Generated configs retain this report
inside `tuning_origin`. It is documentation/provenance; serving does not read
it as a second set of padding controls. The source audit is checked against
the selected vLLM/B12X tree pair and mounted model snapshot before reporting.
It also lists the [TileLang candidate's](../2026-10-03-tilelang-kernels/README.md)
trees: its vLLM patch changes kernels, not padding or sharding, and the report's
kernel scratch follows the profile's `kernel_backend`.

| Model dimension | TP3 | TP4 |
| --- | --- | --- |
| Attention heads, global | 64 → 72 | 64 |
| Attention output groups, global | 8 → 9 | 8 |
| Engram WKV width, global | 25,600 → 25,632 | 25,600 |
| Draft auxiliary projection output, global | 5,120 → 5,184 | 5,120 |
| Target vocabulary rows, global | 129,280 → 129,408 | 129,280 |
| Expert intermediate width, per rank | 768, exact | 576, exact |
| Compact MoE intermediate scratch when selected | This tail path is inactive | 576 → 640 |

The report separates model dimensions, kernel scratch, scheduled rows and
transport layout. TP4 removes the listed model-dimension padding while still
using graph-capacity, sequence-parallel and tile alignment. For example, a
4,096-row SP prefill uses 4,098 collective rows at TP3 and 4,096 at TP4;
205 rows become 207 and 208 respectively. Nineteen rows have a next configured
target ladder entry of 20 and draft-context capacity of 32; actual target
dispatch also depends on request count and captured descriptors. These are rows in one forward,
not total prompt length. Whole prompts are scheduled into prefill chunks,
and CED compaction changes which layers process which rows.

Vision weights remain replicated in both served implementations. The report
also enumerates draft/Markov vocabulary partitions, packed values and scale
storage, auxiliary/query graph capacities and transport alignment. The
[complete alignment inventory](alignment.md) explains the rules and exact
source checks. Experimental batch-invariant
LM-head row duplication is not enabled. See the
[source-level padding audit](../2026-10-03-tp3-tp4-comparison/padding.md) for
implementation locations. Changing source trees or the checkpoint requires
rechecking this audit; the numbers are not live kernel measurements.

## Inspect and generate

```sh
bin/spark3 tuning show tp3
bin/spark3 tuning show tp4
bin/spark3 tuning create tp4 \
  --nodes-config config/nodes-ring4.local.json \
  --output experiments/2026-10-03-transport-profiles/candidate.local.json
bin/spark3 --cluster-config experiments/2026-10-03-transport-profiles/candidate.local.json doctor
```

Use the actual site's node map. The generation command requires the correct
node count and validates reciprocal links, ring order, parallel sizes and
transport constraints. It contacts no node and writes a **launch-disabled**
config. Existing files are never overwritten. The `.local.json` example is
for local inspection; use a tracked filename and publish it before deployment.

Each recipe pins a base config by SHA-256. It retains that base's source lock,
image, memory/page-size policy, KV allocation, context limit, graph capacities,
speculative-decoding settings and cost-table location. A changed base requires
an explicit review and hash update. TP3 and TP4 use their respective source
and image identities; this is not a claim that the new TP4 image has been
qualified on TP3. Model scheduling and memory settings belong in a separate
experiment rather than changing implicitly with a transport tuning knob.

The knobs use the actual native environment names. `null` means **omit the
variable and use the pinned implementation's default**, not zero. Unknown
controls are rejected. In particular, the old TP3 image cannot accept the
custom TP4 dispatch/direction controls. Streaming and GPUNetIO are excluded
from these recipes: the streaming implementations were slower, and GPUNetIO
has no equivalent collective performance result.

To tune a variable, copy the small catalog to a new experiment, change that
variable, then generate a fresh full config:

```sh
bin/spark3 tuning --profiles-config experiments/<new-experiment>/profiles.json \
  create tp4 --nodes-config config/nodes-ring4.local.json \
  --output experiments/<new-experiment>/candidate.json
```

The resolved JSON is the input to all existing tools. There is no inheritance
or node-count autodetection at launch. Its provenance records the catalog hash,
profile name and base hash; benchmark identity includes the resolved config.
Changing a recipe never alters an already generated candidate. Regenerate to
incorporate changes. `--cluster-config` is intentionally rejected for `tuning`:
the recipe already identifies its base.

## Path to main and production

Merging the branch's tooling and experiment records is distinct from selecting
a production image. The current PR leaves `config/cluster.json` on TP3; merging
alone neither starts serving nor promotes the experimental TP4 settings.

The production candidate is the balanced **whole-fragment** TP4 transport with
the explicit policy adapter. Complete these finite steps:

1. Build its pinned vLLM/B12X/NCCL sources into one immutable image. Distribute
   that exact image digest to every rank. Keep streaming, trace instrumentation
   and GPUNetIO out of the production patch series.
2. In a coordinated window, check four-rank startup, correct dispatch/capacity
   messages, matching peer policy, CUDA graph replay, boundary-size collectives,
   and transport-error propagation. The adapter's CPU tests do not establish
   those properties on hardware.
3. Run the final image's quality and serving matrix: decode at 1/2/4/8 streams,
   cold short/32K/64K/long prefill, prefix reuse, the promised four-long-request
   admission and near-limit retrieval, multimodal input, and the fixed agent
   workflow. Include sustained mixed prefill/decode with normal fan control;
   separately cooled bursts have not qualified sustained load. Record errors,
   memory, throttling, actual work and latency, preserving unsuccessful runs.
4. Use the measured balanced image as the matched control for a lean comparison
   of the new adapter. Keep prompts and the TP4 verification-cost table fixed.
   Expand only if a difference is unresolved. There is no need to repeat the
   discarded channel/buffer/streaming sweeps or recable to TP3 merely to qualify
   this TP4-only image. The final candidate still needs its own complete
   benchmark reference under the repository promotion procedure.
5. After owner acceptance, publish one promotion commit updating configuration,
   source locks/production patch series, documentation, and new immutable
   baseline and benchmark manifests. Deploy through the coordinated commands
   and verify `doctor --live` against that exact commit on all four nodes.

The explicit-policy adapter is unbuilt; full configured context, multimodal and
sustained-load checks remain open for this final policy. These are release
gates, not invitations to keep tuning indefinitely. If a gate fails, isolate
that failure and rerun the affected check. TP3 remains a separate supported
configuration for a physical triangle; three nodes selected from the current
four-node ring do not reproduce that topology.

## Local validation

The tests verify byte-for-byte execution-setting equivalence with each pinned
base, topology mismatch rejection, invalid/unsupported tuning controls, base
hash drift, launch disabling, provenance and overwrite protection. Run:

```sh
TMPDIR=/tmp python3 -m unittest discover -s tests
TMPDIR=/tmp bin/spark3 doctor
git diff --check
```

These checks qualify the generator. They add no new performance or hardware
claim to the linked experiment results.

Completed locally: all **186 repository tests**, default doctor, generated TP4
doctor using the four-node example map, verified prepared build inputs, and
`git diff --check`. Source preparation reused the already verified
vLLM/B12X/NCCL trees; no upstream patch changed in this task. No cluster window,
node mutation, image build or new benchmark was performed. The alignment
follow-up read the pinned checkpoint config from dgx1 and ran the recorded
CPU-only checks against both source pairs. Its initial build check found only
source preparation recorded; full local preparation restored the build contexts
and the subsequent build check passed. Both generated topology profiles pass
doctor, and CI now checks the layout against freshly prepared source helpers.
