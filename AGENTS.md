# Agent guide

Read `README.md`, `docs/methodology.md`, `docs/upstreams.md`, `docs/inspiration.md`,
`docker/README.md`, `config/cluster.json`, and the current baseline manifest
before changing runtime or build state. For performance work, `TODO.md` lists
the open levers, measured dead ends, and the lean screening routine.

## Sources of truth

- `config/cluster.json` is the promoted runtime configuration.
- `config/cluster.json` selects the promoted site topology through `nodes_config`
  (default: `config/nodes.json`). The default is git-ignored and initialized
  from `config/nodes.example.json`. Candidate profiles may select a separate map;
  `cluster sync` preserves and copies the selected site map.
- `kernel_backend` in a cluster profile selects the model compute kernels:
  `b12x` (the default and the promoted backend) or `tilelang`.
  `docs/kernel-backends.md` lists the settings, sources and image labels
  doctor requires for each.
- `upstreams.lock.json` owns the promoted external source URLs and revisions.
  Experimental cluster profiles may select a repository-relative
  `upstreams_config` lock with their own source manifest and patch series.
- `requirements/` owns hashed non-Git build inputs.
- `patches/*/series` owns the ordered local patch stacks.
- `manifests/baselines/` is immutable evidence. Never rewrite a published baseline.
- `manifests/benchmarks/<baseline>.json` is that baseline's reference
  `bin/spark3 bench` run. It is immutable too; a new promotion adds its own.
- `experiments/` is the only place for unpromoted tuning.
- `docs/inspiration.md` is a research watchlist, never a source or deployment
  authority. It names each source, what to review it for, and the adoption
  boundary; nothing more.
- Reviews of inspiration sources are for our eyes only: their results,
  comparisons with ours, and analyses of their code or methods never go into
  this repository (docs, experiments, commits, or scripts). An idea taken from
  a source enters as an ordinary experiment of ours, measured on our stack.

Do not treat a running container, shell history, an uncommitted file on a node, or
an image tag as configuration authority. Inspect them as evidence and reconcile
differences into this repository.

## Change discipline

Use neutral task names for branches, commits, pull requests and documentation.
Do not add agent or tool attribution, including branch prefixes or commit trailers.
Preserve required third-party license notices.

Create `experiments/YYYY-MM-DD-short-name/` for performance work. Record the base
commit, one intended variable, exact commands, workload identity, all runs, errors,
memory observations, and a conclusion. Do not discard an unfavorable run.

Change one causal variable at a time unless the experiment explicitly tests an
interaction. A candidate is promoted only when the owner accepts it and the same
commit updates configuration, documentation, and a new immutable baseline.

Never start, stop, replace, or restart the serving cluster without explicit
authorization for that operation. A coordinated runtime change must cover all
configured nodes; mixed rank state is invalid. See `docs/switchless-topology.md`
for three-node direct-peer and four-node neighbour-ring profiles. A new topology
needs its own hardware qualification before promotion.

Use `bin/spark3 cluster sync`, `start`, and `stop` for node operations. They are
plans unless `--apply` is supplied. Never deploy with rsync or copy a dirty
working tree: publish one commit, require clean node checkouts, and detach every
node at that exact commit. Do not bypass the coordinated start with a one-rank
launch script.

## Shared cluster windows

The three Sparks serve production and host experiments, and several agents use them.
Coordinate through `~/spark3-hold.json` on dgx1:

- Before any GPU experiment, benchmark, cluster start, stop, restart or sync, read the
  hold file. If it exists and you are not its holder, do not act. To ask for the
  cluster, write `~/spark3-request.json` (who, why, how long); a runner holding a
  window closes it after its current job.
- To take the cluster, write the hold file with `holder`, `since`, `expected_end`,
  `heartbeat` and the rule it imposes, and remove it once production is restored.
  `scripts/lab.py window open` does this behind the publish and idle guards.
- A holder refreshes `heartbeat` while it works. The watchdog that
  `scripts/lab.py` starts with a window restores the promoted service and removes
  the hold when the heartbeat is more than 15 minutes old.

Inside a window, run jobs back to back without restoring the promoted service in
between; restore it once, when the window closes. See `docs/lab.md`.

## Upstream work

Never make durable changes in exported vLLM, B12X or TileLang trees. Start from the commit in
`upstreams.lock.json`, apply the ordered patch series, and work on a named branch.
Prepared trees under `.work/upstreams/` and `.work/build/` are disposable, not sources
of truth.

Use remote names consistently:

- `upstream`: canonical project repository;
- `origin`: Christopher Owen's fork when it exists;
- `deployment`: this repository, never a source-code fork.

Each local patch must state its upstream base, rationale, quality implications,
tests, and upstream status. Prefer a narrowly scoped patch that can become one
upstream commit. When upstream accepts a change, replace the local patch with the
new upstream commit pin rather than carrying both.

## Safety and secrets

No Hugging Face tokens, GitHub tokens, Tailscale credentials, SSH private keys, or
other secrets belong here. Site IPs and interface mappings are configuration;
credentials remain outside Git.

Ignored caches, `.env*`, `*.local.*`, private keys, and credentials must remain
node-local. Synchronization is Git-based specifically so ignored files are never
an accidental deployment input.

Use deterministic paths and pinned revisions. Build once and distribute the same
OCI digest to every rank. Never overwrite a tag in place and assume ranks match.

Before proposing a change, run `bin/spark3 doctor`, apply the affected upstream
series in a fresh prepared tree, and run `git diff --check`. When complete build
inputs are present, also run `bin/spark3 build check`.
