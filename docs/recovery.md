# Host availability and recovery

## 2026-09-21 dgx3 incident

The preserved dgx3 journal identifies the failure sequence:

- At 06:07:24 the NVIDIA kernel driver logged 172 `NV_ERR_NO_MEMORY`
  allocation failures while the rejected candidate performed an unused SM121
  `ll_bf16` router warmup.
- The host did not reboot. New SSH pre-authentication children accumulated while
  the machine was impaired. At 07:51:44 the listener entered `MaxStartups`
  throttling and remained there for 6 hours, 9 minutes.
- Tailscale's internal watchdog eventually reported a daemon deadlock and
  systemd restarted it at 10:22:28. The first successful remote SSH session was
  not accepted until 14:00:55, when the SSH listener exited throttling.
- The management NIC and both dgx3 peer links were healthy after recovery. From
  dgx1 and dgx2, `10.0.1.72` returned ICMP and an OpenSSH banner normally.

This was not a DHCP, netplan, RoCE, or physical-link failure. It was a GPU-driver
memory-exhaustion event followed by management-daemon degradation, made durable
because dgx3 lacked the watchdog policy already present on dgx1 and dgx2.

## Prevention

The original unused-router cause is removed by upstream-main rebase patch 0021.
Runtime warmup now follows the same SM121 eligibility decision as execution.
The three-rank container remains capped at 112 GiB per node and TP2 full-model
loading is prohibited by the hardware envelope.

The later patch-0022 qualification proved that a steady-state-only memguard was
not sufficient: selected model kernels and B12X plans can exhaust driver and
management headroom before API readiness. Every coordinated start now pre-arms
a protected 5 GiB, quarter-second startup guard before `docker run`, waits for its
first successful memory sample, and only then permits the rank to launch. The
guard remains active throughout readiness and switches to the 3 GiB steady
guard only after the API is healthy. Rollback containers receive the same
pre-launch protection. A candidate that crosses the threshold is killed
immediately rather than receiving a five-second graceful-stop window;
protecting the host takes precedence over preserving an unqualified startup.
The orchestrator no longer waits for a driver allocation failure or
management-path loss.

The host-side recovery policy under `host/recovery/` adds bounded recovery
without changing inference arithmetic or kernel selection:

- systemd feeds the existing SBSA watchdog and reboots after a real PID-1/system
  stall;
- ordinary OOM remains non-panicking so a cgroup can kill the workload instead;
- SSH and Tailscale receive small `MemoryMin` protection and strongly negative
  OOM scores;
- SSH retains bounded but larger pre-authentication headroom; and
- a minute timer restarts only an unresponsive management service or an SSH
  listener that enters `MaxStartups` throttling.

The policy deliberately does not reboot for gateway loss, Internet loss, a
single failed health check, or a generic hung task.

The 2026-09-22 recovery also exposed an independent boot-completion race. All
three hosts had the same vendor `plymouth-quit-wait.service` with an infinite
timeout. dgx1 received the graphical quit event after 5.4 seconds; dgx2 and
dgx3 waited about 101 and 98 minutes respectively until an operator issued
`plymouth quit`. A repository-owned drop-in now preserves the normal handoff
for 30 seconds, then explicitly quits Plymouth, with a 40-second outer service
bound. The same drop-in is installed and checked on every node.

## 2026-09-22 three-node qualification incident

A patch-0022 recovery launch disabled the eager JIT registry sweep but left the
startup memguard off under the old policy. The required profile forward pass
then compiled model-selected TileLang mHC kernels after 98.1 GiB of weights and
270 B12X plans were resident. Observed MemAvailable fell to 1.01--1.72 GiB and
the NVIDIA driver reported allocation failures on every node. The runtime
reached the intended 1,353,553-token KV allocation but never became ready; all
three management paths subsequently became unavailable.

The source correction is patch 0023. It drains the active model/backend JIT
registry after construction and before checkpoint loading. It also fixes a
separate, concrete preparation-lifetime bug: the collector constructed all 226
MXFP8 units before compiling the first, and every unit retained its serving-size
dummy activations. Discovery now streams one unit at a time, the dummy
activations are created only inside that plan's preparation, and allocator
scratch is reclaimed before moving to the next plan. The operational correction
is the startup guard above. After physical recovery, all three hosts passed the
recovery-policy check and the guard's real SIGKILL path using a GPU-free, 64 MiB
smoke container. Patch 0023 then built on dgx1 and passed all 55 focused tests
from the immutable image under a 32 GiB cgroup, without loading weights. Exact
evidence is recorded in the upstream-main rebase experiment. Both promoted and
candidate configurations still set `deployment.launch_enabled` to false, so an
applied start is rejected locally until the guarded TP3 qualification is
deliberately opened.

## Kernel next-boot policy

The controlled dgx3 recovery reboot exposed a separate boot-consistency issue:
`GRUB_DEFAULT=0` selected installed kernel `7.0.0-1019-nvidia`, while this
cluster's RoCE path is qualified on `6.17.0-1032-nvidia`. The earlier one-shot
selection of 6.17 had been consumed as designed.

Removing 7.0 through apt would also remove `linux-nvidia-hwe-24.04`, preventing
the normal metapackage from delivering a future corrected kernel. Permanently
saving 6.17 in GRUB would have the same practical update lock. The recovery
policy therefore denies only exact release `7.0.0-1019-nvidia`: while that is
the generated default, a timer maintains a one-shot entry for the running
eligible kernel. A future default not on the denylist remains eligible and the
policy removes only its own override. There is no apt hold, package pin, kernel
removal, or persistent `GRUB_DEFAULT` change.

## Recovery validation

Recovery implementation commit `7f40db1` was installed on all three nodes on
2026-09-21. Each node passed `scripts/host-recovery check`, reported no failed
systemd units, and retained working SSH and Tailscale listeners through their
scheduled restarts. dgx3 then consumed the policy-generated GRUB entry, booted
`6.17.0-1032-nvidia`, and automatically re-armed the same eligible next boot.
It had zero swap use, no current-boot `NV_ERR_NO_MEMORY` events, and healthy
management and peer links after the reboot.

The qualified candidate image is byte-identical on all three nodes as image ID
`sha256:56820972a862a084c6d0d34dd8faa5ecc8a198b1ec276cb6cfd591fa983ef993`.
No inference container was started as part of host recovery.
