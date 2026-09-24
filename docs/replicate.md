# Replicating the promoted baseline

This reproduces `manifests/baselines/2026-09-24-karmic-kraken-r2.json`:
DeepSeek V4.1 Flash on three DGX Spark (GB10) nodes, tensor parallelism 3,
direct-cabled dual ConnectX-7 ring, Local Inference Lab's
`integration/karmic-kraken-beta` vLLM and B12X.

## Hardware and hosts

- Three DGX Spark nodes, each pair joined by direct ConnectX-7 cables (no
  switch), plus a management network for SSH, Gloo, and the API.
- Docker with the NVIDIA container runtime and `buildx`, `/dev/infiniband`
  RDMA devices, and passwordless SSH from the head node to the other two as the
  `ssh_user` in `config/nodes.json`.
- About 480 GiB of free disk per node for the model (the Engram tables are read
  from disk) plus caches, and 64 GiB of free memory on the build host.

## Site configuration

Edit these for your site, then run `bin/spark3 doctor`:

- `config/nodes.json`: node names, ranks, management IPs, and
  `roce_peer_hcas`, which maps each peer rank to the local RoCE devices cabled
  to it (`ibv_devices` and `rdma link` show the names).
- `config/cluster.json`: `distributed.master_addr` (the head node's management
  IP), `host.home`, `deployment.repository` if you use a fork, and the interface
  names in `GLOO_SOCKET_IFNAME`, `NCCL_SOCKET_IFNAME`, `TP_SOCKET_IFNAME`, and
  `NCCL_IB_HCA`.

## Model

On every node, download the pinned revision into the Hugging Face cache that
`config/cluster.json` mounts:

```sh
huggingface-cli download deepseek-ai/DeepSeek-V4.1-Flash \
  --revision dba1be0a40aa45a94ad051997016db3960a90277
```

## Build the image once

On the build host, from a clean checkout of `main`:

```sh
bin/spark3 build prepare
bin/spark3 build image --apply
```

`build prepare` fills `.work/build/<vllm>-<b12x>-<input hash>/`, a directory
named by the lock's inputs. It fetches the pinned vLLM and B12X revisions,
applies the patch series, and checks both against the source manifest's
patch heads and trees. It also fetches CUTLASS and the hash-locked CuTe DSL
wheels, then exports clean build contexts. Re-running it reuses or repairs the
directory. `build image --apply`:

- refuses to run next to a live service;
- sizes compile jobs to available memory, and cancels if MemAvailable falls
  below 24 GiB;
- tags the image `container.image` from `config/cluster.json`, and never
  overwrites an existing tag;
- checks the tree labels, and runs a GPU import smoke test in a
  memory-capped container.

The launcher checks the image's source-tree labels
(`local.spark3.vllm.tree`, `local.spark3.b12x.tree`) against the
configuration. Your digest will differ from ours because the image also
records the deployment commit; the tree labels must match. See
[docker/README.md](../docker/README.md).

## Distribute one digest

Copy the image to the other two nodes and confirm all three report the same ID:

```sh
for host in dgx2 dgx3; do
  docker save vllm-ds41f-kkref:01f1b874c774-r2 | ssh "$host" docker load
done
for host in dgx1 dgx2 dgx3; do
  ssh "$host" docker image inspect vllm-ds41f-kkref:01f1b874c774-r2 --format '{{.Id}}'
done
```

## First start

The first start compiles B12X kernels into `cache/kkref/jit/` and takes longer;
later starts reuse the cache. To keep FlashInfer's sampling-module compile out
of the memory-guarded startup, prebuild it on each node first:

```sh
experiments/2026-09-23-canonical-minimal/prebuild_flashinfer.sh \
  vllm-ds41f-kkref:01f1b874c774-r2 kkref/flashinfer
```

Then, from the head node with a clean checkout of the published `main` commit:

```sh
bin/spark3 cluster sync --apply
bin/spark3 cluster start --apply
bin/spark3 doctor --live
```

`cluster start` arms a 5 GiB startup memory guard on every node before any
container starts, waits for API readiness, then switches to the 3 GiB steady
guard. A guard stop rolls the launch back and keeps each rank's log under
`results/private/failed-starts/` (git-ignored; experiment configurations use
their own `runs/failed-starts/`).

The API is OpenAI-compatible on the head node's port 8000, model
`deepseek-v4.1-flash`. Reasoning is on by default; pass
`"chat_template_kwargs": {"thinking": false}` to turn it off per request.

## Verify

From the head node against the running, otherwise idle service:

```sh
bin/spark3 bench
```

It checks that the live cluster matches `config/cluster.json`, runs the quality
gate and every benchmark suite (about 35 minutes), and compares the result with
the promoted reference run in `manifests/benchmarks/`. It exits non-zero if the
quality gate fails, any request fails, or a point is significantly slower than
the reference by more than 3%. See the README's Benchmarking section.
