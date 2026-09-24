#!/usr/bin/env python3
"""Characterize layer-0 shared-expert deviations: rounding noise or corruption.

usage: gtrace_shared_diff.py TRACE_ROOT REF_LABEL LABEL STEP[,STEP...]
"""
import os, sys
import torch

root, ref_label, label = sys.argv[1:4]
steps = [int(s) for s in sys.argv[4].split(",")]
key = (2000, "shared_out")


def load(lab, step):
    import glob
    files = sorted(glob.glob(os.path.join(root, f"gtrace-{lab}", "0*.pt")))
    d = torch.load(files[step])
    assert torch.equal(d[(2000, "moe_x")][0], load.x[step]) if step in load.x else True
    return d[(2000, "moe_x")][0].clone(), d[key][0].float().clone()
load.x = {}

for step in steps:
    xr, r = load(ref_label, step)
    xg, g = load(label, step)
    same_input = torch.equal(xr, xg)
    diff = (g - r).abs()
    changed = diff > 0
    n = int(changed.sum())
    ulp = torch.where(r != 0, diff / (r.abs() * 2.0 ** -8), torch.zeros_like(diff))
    idx = changed.nonzero().flatten()
    runs = int(((idx[1:] - idx[:-1]) == 1).sum()) if n > 1 else 0
    big = int((ulp > 4).sum())
    print(f"step {step:3d} same_input={same_input} rel={float(diff.norm() / r.norm()):.4f} "
          f"changed={n}/{r.numel()} adjacent_pairs={runs} max_ulp={float(ulp.max()):.1f} "
          f">4ulp={big} max_abs={float(diff.max()):.4g} |ref|max={float(r.abs().max()):.4g}")
    if big:
        worst = torch.topk(ulp, 5).indices.tolist()
        print("   worst idx", worst, "ref", [round(float(r[i]), 5) for i in worst],
              "got", [round(float(g[i]), 5) for i in worst])
