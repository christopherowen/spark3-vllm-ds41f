# Image build

`docker/Dockerfile` builds the promoted image: Local Inference Lab's
karmic-kraken-beta vLLM and B12X with the local patch series, on the canonical
vLLM ARM64 nightly `af1c0149`. Only vLLM's `_C_stable_libtorch` and
`_moe_C_stable_libtorch` are rebuilt, for SM121, and NCCL 2.30.7 (the base
image's version) is rebuilt from its release tag with `patches/nccl`, replacing
the wheel's `libnccl.so.2`; the image build checks the installed library's
version and checksum. FlashInfer 0.6.18.post1 comes from the base image. The r1
and r2 images came from the same recipe in
`experiments/2026-09-23-karmic-kraken-reference/`; r3 was the first built by
`bin/spark3 build`, and the running image is `vllm-ds41f-kkref:04c30fa98e79-r5o`.

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

- the vLLM, B12X and NCCL revisions;
- the patch-series fingerprints, recorded patch heads, and trees;
- the CUTLASS revision;
- the CuTe DSL wheel lock;
- the base image digest.

A hash of those inputs names the directory:

```
.work/build/vllm-<12>-b12x-<12>-<input hash>/
  inputs.json            # the inputs and a content digest of each context
  src/vllm, src/b12x, src/nccl  # pinned revision + git am of patches/*/series
  src/cutlass            # pinned CUTLASS revision (headers only)
  context/vllm-source    # clean exports: no .git, no bytecode
  context/b12x-source
  context/nccl-source
  context/cutlass-source
  context/cutlass-dsl-wheels   # hash-verified wheels
  context/vllm-deletions/deleted.txt
  context/empty          # the main context; every COPY uses a named context
  images/<image id>.json # receipt for each image built from it
```

A lock that lists the optional `tilelang` and `tile_kernels` sources adds
`src/tilelang` (with its submodules), `src/tile_kernels` and their
`*-source` contexts; see [TileLang kernel backend](#tilelang-kernel-backend).

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
check that the series still apply and reproduce the recorded trees, and
`--only vllm,tilelang,tile_kernels` for the TileLang candidate.

**Missing patches:** a series entry whose patch file does not exist yet stops
`prepare`, `check` and `image` with the missing path. `prepare --only` still
prepares the other parts.

## TileLang kernel backend

For a lock that lists `tilelang` and `tile_kernels`
([kernel-backends.md](../docs/kernel-backends.md)), `prepare` checks out
TileLang's submodules recursively at the commits the patched tree records and
refuses commits that differ from the source manifest. `build image` passes
their contexts, `TILELANG_*` and `TILE_KERNELS_*` arguments (revision, patch
head, tree and version), the compile jobs as `TILELANG_COMPILE_JOBS`, and
`RUNTIME_STAGE=runtime-tilelang`. That stage starts from the B12X runtime
stage, builds wheels of both, replaces the base image's `tilelang` 0.1.12,
installs TileKernels without dependencies, and checks the installed files,
versions and every shared `apache-tvm-ffi` and `tilelang` requirement. The
image gains `local.spark3.tilelang.*` and `local.spark3.tile_kernels.*`
labels, and the GPU smoke imports both packages.

Without those sources the command has no new argument, BuildKit skips the
TileLang stages and their contexts, and the image is the unchanged `runtime`
stage.

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
