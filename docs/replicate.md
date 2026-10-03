# Replicating the promoted baseline

This reproduces `manifests/baselines/2026-10-02-karmic-kraken-r5o-64k.json`:
DeepSeek V4.1 Flash on three DGX Spark (GB10) nodes, tensor parallelism 3,
direct-cabled dual ConnectX-7 ring, Local Inference Lab's
`integration/karmic-kraken-beta` vLLM and B12X.

## Hardware and hosts

- Three DGX Spark nodes, each pair joined by direct ConnectX-7 cables (no
  switch), plus a management network for SSH, Gloo, and the API.
- Docker with the NVIDIA container runtime and `buildx`, `/dev/infiniband`
  RDMA devices, and passwordless SSH from the head node to the other two as the
  `ssh_user` in `config/nodes.json` (see [Site configuration](#site-configuration)).
- About 480 GiB of free disk per node for the model (the Engram tables are read
  from disk) plus caches, and 64 GiB of free memory on the build host.
- DGX Spark 26.09.2 or later on every node, booted to `multi-user.target`
  (no desktop; `bin/spark3 doctor --live` reports a node that is not), with
  the installed OS first in the UEFI boot order. A network (PXE) entry first
  adds about a minute of DHCP timeouts to every boot; `doctor --live` reports
  it, and `sudo efibootmgr -o <ubuntu>,<others>` fixes it.
- NVIDIA kernel mode setting on (`nvidia-drm modeset=Y`), so a monitor or KVM
  attached after boot gets a console. DGX OS ships
  `nvidia-drm-options-modeset0`, which turns it off; purge it (`sudo apt-get
  purge nvidia-drm-options-modeset0 && sudo update-initramfs -u -k all`) and
  reboot. `doctor --live` checks it.
- The DGX Spark additive fan-floor control on every node: the signed
  `dgx-spark-fan-control` DKMS module with its `dgx_ec_fan_floor` cooling
  device, and the `dgx-fan-control` daemon. Its source and signing key live
  outside this repository; `doctor --live` reports the first missing layer.
  `bin/spark3 bench` and kernel-lab jobs use it to pre-cool before measuring.
  If any node is at or above 55 °C, all of them cool together until every node
  is below it. Without it they wait for the node to cool under
  NVIDIA's curve.
- NVMe interrupt coalescing off on every node. DGX OS's
  `nvidia-nvme-interrupt-coalescing.service` sets feature 0x08 to 0x107 at
  every boot (one interrupt per 8 completions or 100 µs). The drives default to
  0 and cannot save the feature, so masking the service keeps it off:

  ```sh
  sudo systemctl mask --now nvidia-nvme-interrupt-coalescing.service
  sudo /usr/bin/nvidia-nvme-interrupt-coalescing.sh disable
  ```

  `doctor --live` warns when coalescing is on or the service can turn it back
  on. The measured trade-off is in
  [experiments/2026-10-02-nvme-coalescing](../experiments/2026-10-02-nvme-coalescing/README.md).
- Kernel `7.0.0-1019-nvidia-64k` with `kho=off`, NVIDIA 580.178.04 and
  signed memory-saver DKMS 0.2.0 on every node. Follow
  [memory-profiles.md](memory-profiles.md) to install, validate and select the
  matching profile. The page-aware memory service selects `/swap-64k.img`,
  disables THP and sets `vm.min_free_kbytes=45166`. The 4 KiB kernel/profile
  remains available for rollback at its original context and KV allocation.
- `vm.watermark_boost_factor=0` on every node. With the kernel default (15000)
  a watermark boost hides up to 0.87 GiB from MemAvailable, which the memory
  guards read:

  ```sh
  echo "vm.watermark_boost_factor = 0" | sudo tee /etc/sysctl.d/90-watermark-boost.conf
  sudo sysctl -w vm.watermark_boost_factor=0
  ```

## Site configuration

Edit these for your site, then run `bin/spark3 doctor`:

- `config/nodes.json`: node names, ranks, management IPs, `ssh_user`, and
  `roce_peer_hcas`, which maps each peer rank to the local RoCE devices cabled
  to it (`ibv_devices` and `rdma link` show the names). The file is
  git-ignored because it describes your site. Create it on the head node
  with `cp config/nodes.example.json config/nodes.json` and edit it there.
  `bin/spark3 cluster sync --apply` copies the head node's file to every node
  after it moves their checkouts.
- `config/cluster.json` and both named `config/cluster-4k.json` /
  `config/cluster-64k.json` profiles: `distributed.master_addr` (the head node's management
  IP), `host.home`, `deployment.repository` if you use a fork, and the interface
  names in `GLOO_SOCKET_IFNAME`, `NCCL_SOCKET_IFNAME`, `TP_SOCKET_IFNAME`, and
  `NCCL_IB_HCA`. Keep `distributed.master_port` below the head node's
  ephemeral port range (`sysctl net.ipv4.ip_local_port_range`); inside it, an
  outgoing connection can hold the port and the head fails to start. The site
  range starts at 10000, so the port is 9999.

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
directory. A profile that selects the TileLang kernel backend also fetches
TileLang and TileKernels ([kernel-backends.md](kernel-backends.md)).
`build image --apply`:

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
  docker save vllm-ds41f-kkref:04c30fa98e79-r5o | ssh "$host" docker load
done
for host in dgx1 dgx2 dgx3; do
  ssh "$host" docker image inspect vllm-ds41f-kkref:04c30fa98e79-r5o --format '{{.Id}}'
done
```

## First start

The first start compiles B12X kernels into `cache/kkref/jit/` and takes longer;
later starts reuse the cache. To keep FlashInfer's sampling-module compile out
of the memory-guarded startup, prebuild it on each node first:

```sh
experiments/2026-09-23-canonical-minimal/prebuild_flashinfer.sh \
  vllm-ds41f-kkref:04c30fa98e79-r5o kkref/flashinfer
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
gate and the decode matrix (about six minutes), and compares the result with
the promoted reference run in `manifests/benchmarks/`. `--full` runs every
suite to tighter intervals (about 35 minutes). It exits non-zero if the
quality gate fails, any request fails, or a point is significantly slower than
the reference by more than 3%. See the README's Benchmarking section.
