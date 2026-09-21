# Host recovery policy

These files are the source of truth for the small management-plane recovery
policy installed on every Spark. They do not start, stop, or otherwise manage
the vLLM container.

The policy has five deliberately narrow responsibilities:

1. arm the Spark's existing SBSA hardware watchdog through systemd;
2. preserve a small amount of reclaim protection for SSH and Tailscale;
3. prevent a handful of stuck pre-authentication SSH children from exhausting
   the listener; and
4. check the local SSH listener and Tailscale control socket once per minute,
   restarting only the affected management service when it is unresponsive;
   and
5. keep an exact known-bad kernel out of the next boot without holding packages
   or preventing a future kernel from becoming the normal default.

It does not panic on an ordinary host or cgroup OOM, panic on a generic hung
task, reset networking, restart Docker, or reboot merely because the gateway or
Internet is unavailable. The hardware watchdog is the last resort for a real
PID-1/system stall.

The kernel policy is intentionally a release-specific denylist, not a pin. If
the current GRUB default is denied, it places a one-shot boot entry for the
currently running eligible kernel. When a newer eligible kernel becomes the
default, the timer removes only the one-shot entry that it owns. It never edits
`GRUB_DEFAULT`, holds an apt package, or removes the kernel metapackage.

Install or verify the policy from a clean, published checkout on each node:

```sh
scripts/host-recovery apply
scripts/host-recovery check
```

`apply` validates `sshd` before changing either listener. Fresh SSH and
Tailscale processes are scheduled after the command exits so the OOM policy is
active without stranding an installation performed through Tailscale.
