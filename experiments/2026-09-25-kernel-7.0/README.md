# Kernel 7.0.0-1019-nvidia

Question: can the cluster move from `6.17.0-1032-nvidia` to `7.0.0-1019-nvidia`,
the kernel the DGX OS `linux-nvidia-hwe-24.04` metapackage now installs?

## Background

`7.0.0-1019-nvidia` was on the recovery denylist. Another three-Spark
deployment found RDMA memory registration failing on it (`ibv_reg_mr` returns
`ENOMEM`, even for 1 KB regions, and NCCL cannot build communicators), with
`CmaTotal` reading 0. The kernel configurations show the cause:

| | 6.17.0-1032 | 7.0.0-1019 |
|---|---|---|
| `CONFIG_KEXEC_HANDOVER` | y | y |
| `CONFIG_KEXEC_HANDOVER_ENABLE_DEFAULT` | not set | y |
| `CONFIG_CMA_SIZE_MBYTES` | 128 | 0 |

With Kexec HandOver on by default, its scratch pages stay migrate-CMA while
uncounted in `CmaTotal`, so long-term pins must migrate off them and fail under
memory pressure. DGX Spark 26.09.2 (`dgx-spark-ota-update-meta`) adds
`nvidia-spark-grub-kho` 1.0-1, which puts `kho=off` on the kernel command line.

## Hosts

All three nodes on DGX Spark 26.09.2 (2026-09-25), booted to
`multi-user.target`, `vm.watermark_boost_factor=0`, NVIDIA driver 580.178.04
(the module is built for both kernels), `kho=off` in every GRUB entry.

## One-shot boot on dgx3

`grub-reboot` into 7.0 with the denylist unchanged, so the recovery policy
re-armed 6.17 for the following boot.

- `kho=off` on the command line, `CmaTotal` 0 kB, no Kexec HandOver, mlx5,
  NVRM or Xid messages; no failed units; SSH, Tailscale and Docker up; all four
  RoCE links active.
- Idle MemAvailable 117.03 GiB (116.45 GiB on 6.17 after the same update).
- `ib_write_bw` dgx1 -> dgx3 over one direct link (GID index 3, 8 MiB
  messages): 109.07 Gb/s, no errors.

## Service run with dgx3 on 7.0

dgx1 and dgx2 on 6.17, dgx3 on 7.0; promoted configuration
(`2026-09-25-karmic-kraken-r3-vision`), `bin/spark3 cluster start`, `doctor
--live`, `vision_check.py`, quick bench against the promoted reference.

- Start within the 5 GiB startup guard; 0 `ibv_reg_mr`, `ENOMEM` or
  `ncclSystemError` lines in any container log; 0 new kernel errors on dgx3.
- `doctor --live` passed; both image checks passed; quality 5/5.
- Decode against the reference: faster at prose c4 (+4.9%), prose c8 (+4.6%)
  and code c2 (+7.7%), equal elsewhere; single-stream steps 47.52/49.18 ms
  against 49.68/51.50 ms.
- Lowest MemAvailable: dgx1 6.49, dgx2 7.50, dgx3 7.62 GiB (5.65, 6.66,
  6.80 GiB in the reference run).

The memory and speed gains also include the desktop being off and the OS
update on every node, so they are not attributed to the kernel.

## Decision

Run `7.0.0-1019-nvidia` on all three nodes. The next-boot policy and its
denylist were removed from the repository and the hosts; GRUB's default
(entry 0, the newest installed kernel) now decides. To roll a node back,
`grub-reboot` into the 6.17 entry for one boot, or remove 7.0 in GRUB by hand.
