# Agent guide

Read `README.md`, `docs/methodology.md`, `docs/upstreams.md`, `docs/inspiration.md`,
`docker/README.md`, `config/cluster.json`, and the current baseline manifest
before changing runtime or build state.

## Sources of truth

- `config/cluster.json` is the promoted runtime configuration.
- `config/nodes.json` is the promoted site topology.
- `upstreams.lock.json` owns every external source URL and revision.
- `requirements/` owns hashed non-Git build inputs.
- `patches/*/series` owns the ordered local patch stacks.
- `manifests/baselines/` is immutable evidence. Never rewrite a published baseline.
- `experiments/` is the only place for unpromoted tuning.
- `docs/inspiration.md` is a research watchlist, never a source or deployment
  authority.

Do not treat a running container, shell history, an uncommitted file on a node, or
an image tag as configuration authority. Inspect them as evidence and reconcile
differences into this repository.

## Change discipline

Create `experiments/YYYY-MM-DD-short-name/` for performance work. Record the base
commit, one intended variable, exact commands, workload identity, all runs, errors,
memory observations, and a conclusion. Do not discard an unfavorable run.

Change one causal variable at a time unless the experiment explicitly tests an
interaction. A candidate is promoted only when the owner accepts it and the same
commit updates configuration, documentation, and a new immutable baseline.

Never start, stop, replace, or restart the three-rank service without explicit
authorization for that operation. A coordinated runtime change must cover all
three nodes; mixed rank state is invalid.

Use `bin/spark3 cluster sync`, `start`, and `stop` for node operations. They are
plans unless `--apply` is supplied. Never deploy with rsync or copy a dirty
working tree: publish one commit, require clean node checkouts, and detach every
node at that exact commit. Do not bypass the coordinated start with a one-rank
launch script.

## Upstream work

Never make durable changes in exported vLLM or B12X trees. Start from the commit in
`upstreams.lock.json`, apply the ordered patch series, and work on a named branch.
Prepared trees under `.work/upstreams/` are disposable build inputs, not sources
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
