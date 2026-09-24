#!/usr/bin/env python3
"""Find shared-expert outputs that differ for bit-identical MoE inputs.

usage: gtrace_shared_race.py TRACE_ROOT LABEL[,LABEL...]
Streams one dump file at a time; keeps only per-layer hashes and layer-0 rows.
"""
import collections, glob, hashlib, os, sys
import torch

root, labels = sys.argv[1], sys.argv[2].split(",")


def digest(t):
    return hashlib.sha1(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def rel(a, b):
    a, b = a.float(), b.float()
    return float((a - b).norm() / (b.norm() + 1e-6))


# (layer, moe_x hash) -> {shared_out hash: [(label, step), ...]}
groups = collections.defaultdict(lambda: collections.defaultdict(list))
rows0 = {}
for label in labels:
    files = sorted(glob.glob(os.path.join(root, f"gtrace-{label}", "0*.pt")))[1:]
    for step, path in enumerate(files, start=1):
        d = torch.load(path)
        for layer in range(40):
            kx, ks = (2000 + layer, "moe_x"), (2000 + layer, "shared_out")
            if kx not in d or ks not in d:
                continue
            x, s = d[kx][0].cpu(), d[ks][0].cpu()
            groups[(layer, digest(x))][digest(s)].append((label, step))
            if layer == 0:
                rows0[(label, step, "s")] = s.clone()
        del d
bad = [(k, v) for k, v in groups.items() if len(v) > 1]
shared_inputs = sum(1 for v in groups.values() if sum(len(x) for x in v.values()) > 1)
print(f"runs={labels}")
print(f"input groups seen by more than one (run, step): {shared_inputs}; "
      f"groups with diverging shared_out: {len(bad)}")
per_run = collections.Counter()
per_layer = collections.Counter()
for (layer, _), outs in bad:
    members = sorted(outs.values(), key=len, reverse=True)
    for ms in members[1:]:
        for l, _s in ms:
            per_run[l] += 1
            per_layer[layer] += 1
print("deviations per run:", dict(sorted(per_run.items())))
print("deviations per layer:", dict(sorted(per_layer.items())))
for (layer, _), outs in sorted(bad, key=lambda kv: kv[0][0])[:20]:
    members = sorted(outs.values(), key=len, reverse=True)
    majority, minority = members[0], [m for ms in members[1:] for m in ms]
    detail = ""
    if layer == 0:
        ref = rows0[(majority[0][0], majority[0][1], "s")]
        detail = " rel=" + ",".join(
            f"{rel(rows0[(l, s, 's')], ref):.4f}" for l, s in minority)
    print(f"  L{layer:2d} majority={majority[:4]}{'...' if len(majority) > 4 else ''} "
          f"deviating={minority}{detail}")
