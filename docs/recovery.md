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

The inference-side cause is removed by upstream-main rebase patch 0021. Runtime
warmup now follows the same SM121 eligibility decision as execution, avoiding
the unused router compilation that consumed the final CUDA-visible headroom.
The three-rank container remains capped at 112 GiB per node, the 3 GiB memguard
is armed only after JIT startup, and TP2 full-model loading is prohibited by the
hardware envelope.

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
