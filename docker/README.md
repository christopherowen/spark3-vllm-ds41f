# Image build

`docker/Dockerfile` builds the promoted image: Local Inference Lab's
karmic-kraken-beta vLLM and B12X with the local patch series, on the canonical
vLLM ARM64 nightly `af1c0149`. Only vLLM's `_C_stable_libtorch` and
`_moe_C_stable_libtorch` are rebuilt, for SM121. FlashInfer 0.6.18.post1 comes
from the base image. The r1 and r2 images came from the same recipe in
`experiments/2026-09-23-karmic-kraken-reference/`; the running r3 image
(`vllm-ds41f-kkref:01f1b874c774-r3`) is the first built by `bin/spark3 build`.

```sh
bin/spark3 build prepare        # create or repair the build directory
bin/spark3 build check          # verify it
bin/spark3 build image          # print the build command
bin/spark3 build image --apply  # build and smoke-test on an idle host
bin/spark3 build smoke          # rerun the GPU import smoke
```

## Build directory

`upstreams.lock.json` and the source manifest it names determine every build
input:

- the vLLM and B12X revisions;
- the patch-series fingerprints, recorded patch heads, and trees;
- the CUTLASS revision;
- the CuTe DSL wheel lock;
- the base image digest.

A hash of those inputs names the directory:

```
.work/build/vllm-<12>-b12x-<12>-<input hash>/
  inputs.json            # the inputs and a content digest of each context
  src/vllm, src/b12x     # pinned revision + git am of patches/*/series
  src/cutlass            # pinned CUTLASS revision (headers only)
  context/vllm-source    # clean exports: no .git, no bytecode
  context/b12x-source
  context/cutlass-source
  context/cutlass-dsl-wheels   # hash-verified wheels
  context/vllm-deletions/deleted.txt
  context/empty          # the main context; every COPY uses a named context
  images/<image id>.json # receipt for each image built from it
```

The same lock always yields the same directory with the same contents.

**Patch heads:** `git am` runs with a fixed committer and
`--committer-date-is-author-date`, so the patched heads reproduce the source
manifest's `patch_head` commits exactly. Prepare refuses a head or tree that
differs from the manifest. The patch-set fingerprint must also match the
manifest, so a changed patch has to be recorded before it can be built.

**Reuse and repair:** re-running `prepare` reuses every verified part and
rebuilds only what is missing or wrong. It writes `inputs.json` last, so an
interrupted run is never mistaken for a complete one. `check` recomputes each
context's digest.

**Contexts:** these are exports without Git metadata. File modes are
normalized to Git's 644 and 755, so the contexts don't depend on the host
umask and are identical on every host. The earlier recipe copied whole
checkouts, which put about 100 MB of `.git` history into the image and
changed the build cache key on every fresh clone.

**CI:** CI runs `bin/spark3 build prepare --only vllm` and `--only b12x` to
check that the series still apply and reproduce the recorded trees.

## Resource policy on DGX Spark

`build image --apply` runs only on an idle Spark:

- It refuses while a DS4.1 container is running and holds a host lock against
  overlapping builds.
- It needs 64 GiB MemAvailable to start.
- It picks compile jobs from memory and CPUs: 8 GiB per job, 24 GiB kept in
  reserve, four CPUs reserved, two NVCC threads per job, at most ten jobs.
- It samples MemAvailable every second and cancels the build below 24 GiB.

After the build it checks the tree labels, then imports the DS4.1 serving
modules on the GPU in a 16 GiB container. It writes a receipt with the image
ID, job count, elapsed time, and lowest MemAvailable.

The default tag is `container.image` from the cluster configuration. The
command refuses to overwrite an existing tag, so a node never silently holds a
different image under the promoted name. Build once, then load the same image
on every node and confirm all three report the same ID (`docs/replicate.md`).
Start the service only through `bin/spark3 cluster start`, with its startup and
steady memory guards.
