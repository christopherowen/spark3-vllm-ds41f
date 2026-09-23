#!/usr/bin/env python3
"""Compare graph-trace dumps of two runs. usage: gtrace_analyze.py DIR_A DIR_B"""
import glob, json, os, sys
import torch

def load(d):
    traj = json.load(open(os.path.join(d, "trajectory.json")))
    files = sorted(glob.glob(os.path.join(d, "0*.pt")))
    steps = [torch.load(f) for f in files[1:1 + len(traj["generated"])]]
    return traj, steps

(ta, sa), (tb, sb) = load(sys.argv[1]), load(sys.argv[2])
ga, gb = ta["generated"], tb["generated"]
same = next((i for i, (x, y) in enumerate(zip(ga, gb)) if x != y), min(len(ga), len(gb)))
print(f"A {os.path.basename(sys.argv[1])}: {len(ga)} tokens; B {os.path.basename(sys.argv[2])}: "
      f"{len(gb)} tokens; identical first {same} tokens; steps A={len(sa)} B={len(sb)}")
names = ["engram_residual", "attn_in", "attn_out", "ffn_in", "ffn_out"]
layers = sorted({k[0] for k in sa[0] if k[0] >= 0})

def rel(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return float((a - b).norm() / (b.norm() + 1e-6))

limit = min(same + 1, len(sa), len(sb))
for step in [0, 1, 2, 3, 5, 10, 20, 40, 60, 80]:
    if step >= limit:
        break
    a, b = sa[step], sb[step]
    pa, pb = a.get((-1, "positions")), b.get((-1, "positions"))
    worst = []
    first = None
    for layer in layers:
        for name in names:
            key = (layer, name)
            if key in a and key in b and a[key].shape == b[key].shape and a[key].numel():
                r = rel(a[key], b[key])
                worst.append((r, layer, name))
                if first is None and r > 0.02:
                    first = (layer, name, r)
    worst.sort(reverse=True)
    print(f"step {step:3d} pos A={pa.flatten()[:3].tolist() if pa is not None else None} "
          f"B={pb.flatten()[:3].tolist() if pb is not None else None} "
          f"first>0.02={first} worst={[(round(r,3), l, n) for r, l, n in worst[:3]]}")
# per-layer detail at the first decode step
if limit > 1:
    a, b = sa[1], sb[1]
    print("\nstep 1 per-layer relative difference (engram_residual attn_in attn_out ffn_in ffn_out):")
    for layer in layers:
        vals = []
        for name in names:
            key = (layer, name)
            vals.append(f"{rel(a[key], b[key]):8.4f}" if key in a and key in b and a[key].numel() else "       -")
        print(f"  layer {layer:2d} " + " ".join(vals))
