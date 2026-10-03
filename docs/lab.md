# Experiment windows and the kernel lab

`scripts/lab.py` runs experiments on the three Sparks without restoring the promoted
service after every run. It replaces the copy-and-edit `runNN.sh` pattern.

## Windows

```sh
scripts/lab.py window open --minutes 90 --note "mHC screen"
scripts/lab.py window status
scripts/lab.py window close
```

`open` refuses unless:

- the deployment commit is on `origin/main` and every node's checkout matches;
- the cluster has served no requests for 30 s;
- no one else holds `~/spark3-hold.json`.

It then writes the hold file and starts a watchdog. The hold file records the time cap
and a heartbeat that the runner refreshes every minute. The watchdog restores the
promoted service and removes the hold if the heartbeat goes stale for 15 minutes.

`close` boots `config/cluster.json` (unless it is already live), checks
`doctor --live`, and removes the hold.

A window closes itself:

- when its run completes, unless the run uses `--keep-open`;
- when a job fails;
- at its time cap;
- when someone writes `~/spark3-request.json`. The runner finishes the current job first.

## Runs

```sh
scripts/lab.py run scripts/lab_specs/lab1-first-window.json --dry-run
scripts/lab.py run scripts/lab_specs/lab1-first-window.json
```

A spec names the experiment directory, a run name and a list of jobs. A run opens a
window if none of ours is open.

- **`measure`:** boots each arm with `cluster start --replace`, with no stop in
  between, and measures it.
  - Profile `lean`, about 2.5 min per arm:
    - bench prose and JSON with 3 samples each;
    - 1K and 16K prefill, 2 repeats each;
    - 12 distinct single-stream prompts;
    - 8 streams with 2 samples;
    - short prompts with 3 reps;
    - 3 rounds of mixed traffic.
  - Profile `full`: the round-16 matrix.
  - `"bracket": true` measures the first arm again at the end, labelled `<label>-end`, to show drift.
  - Results land where `tables_arms.py` reads them: `results/private/bench/<run>-<label>/` and
    `results/private/determinism/<run>/<kind>-<label>.jsonl`.
- **`validate`:** a trace arm.
  - Sequence: boot A, `c8_trace.py`, `scenario_trace.py --chunk <threshold>`, then
    `trace_mixes.py` (tier `full` only), then a restart (boot B) with the scenarios
    repeated under `-bootB`.
  - Then a stop and `analyze_trace4.py` for each node; the three nodes run concurrently, capped at 16 GB each.
  - Tier `quick`:
    - boot A runs the mixed, chunked, chunked_end, cache_long and distinct scenarios;
    - the restart runs cache_long and distinct only.
  - `analysis-summary.json` reports rows, differences, cross-restart pairs and a pass flag.
- **`kernel`:** runs kernel-lab bundles on the named nodes concurrently, with the cluster
  stopped. It writes `results/private/lab/<run>/verdicts.json` and checks that the
  nodes agree.

Before an arm's first boot, every mounted B12X file must carry at least as many
`fence_proxy` calls as the image's own copy. Overlays built from an older B12X tree
fail this check. The overlay hashes of each run go to
`results/private/lab/overlays-<run>.txt`.

## Kernel-lab bundles

A bundle is a directory holding `candidate.json` and `overlay/{vllm,b12x}/<package path>`.
Overlay files are mounted over `/opt/spark3/candidate/<package>/<package>/...`.

`candidate.json` holds:
- `image`, `env`, `argv` (the harness command inside the container);
- `mounts`: `[source, destination]` pairs. A source is resolved from the bundle first, then
  from `~/spark3-lab/inputs`, then from the node's checkout;
- `sync`: inputs that are not in Git, copied to `~/spark3-lab/inputs` on the other nodes;
- `variant_configs`: configurations allowed more than one row group, such as production;
- `workdir`: the container working directory (default `/opt/spark3/candidate/b12x`);
- `verdict`: `"exit"` for a test-suite bundle, such as pytest over the image's own
  `/opt/spark3/candidate/vllm/tests`. It passes on exit status 0 with tests passed and none
  failed, and records the failed test ids, which the nodes must agree on.

Before replaying, a kernel job brings its nodes below the bench's cooling threshold
(55 °C hottest zone). If any is hotter, all of them cool together at the maximum fan
floor until the last is below it; then their usual fan control returns. This is the
same thermal baseline every bench starts from ([methodology](methodology.md)).

Bundles, synced inputs and outputs stay under `~/spark3-lab/` on each node, outside
the deployment checkout.

```sh
scripts/lab.py kernel-local experiments/2026-09-29-determinism/bundles/mhc-smoke --dry-run
```

`kernel-local` runs on the node itself. It:
- refuses while the serving container runs there;
- takes `/tmp/spark3-lab-gpu.lock`;
- writes `output.txt` and `verdict.json`: row groups, bit-equality sets, timings, errors,
  input hashes and pass/fail.

A verdict passes when:
- every configuration not listed in `variant_configs` has one row group;
- every compared pair is bit-equal;
- nothing crashed.

A test-suite verdict passes when the suite exits 0 and reports passed tests and no
failures or errors.

Compile caches persist per node in `~/.cache/spark3-lab/compile`.
