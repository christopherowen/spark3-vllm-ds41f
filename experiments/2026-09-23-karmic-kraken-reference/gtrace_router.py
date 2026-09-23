#!/usr/bin/env python3
"""Compare router and MoE-branch traces, graph vs eager, against eager noise.

usage: gtrace_router.py TRACE_ROOT EAGER_LABELS GRAPH_LABELS  (comma lists)
"""
import glob, os, sys
import torch


def load(d, n=6):
    files = sorted(glob.glob(os.path.join(d, "0*.pt")))
    return [torch.load(f) for f in files[1:1 + n]]


root = sys.argv[1]
E = [load(os.path.join(root, f"gtrace-{l}")) for l in sys.argv[2].split(",")]
G = [load(os.path.join(root, f"gtrace-{l}")) for l in sys.argv[3].split(",")]


def rel(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return float((a - b).norm() / (b.norm() + 1e-6))


def ids(t):
    return sorted(int(x) for x in t[0].tolist())


for step in (1, 2, 3):
    print(f"== decode step {step} (row 0)")
    for layer in (0, 1, 2, 3, 10, 20, 39):
        r, m = 1000 + layer, 2000 + layer
        e0 = E[0][step]
        cells = []
        for key in ((m, "moe_x"), (r, "router_logits"), (r, "topk_weights"),
                    (m, "shared_out"), (m, "routed_out"), (layer, "ffn_out")):
            if key not in e0 or any(key not in x[step] for x in G):
                cells.append(f"{key[1]}=missing")
                continue
            noise = max([rel(e[step][key][0], e0[key][0]) for e in E[1:]] + [1e-7])
            g = max(rel(x[step][key][0], e0[key][0]) for x in G)
            cells.append(f"{key[1]} g{g:.4f}/n{noise:.4f}")
        same = [ids(x[step][(r, "topk_ids")]) == ids(e0[(r, "topk_ids")]) for x in G + E[1:]]
        cells.append(f"ids-same={same}")
        print(f" L{layer:2d} " + " | ".join(cells))
    e0, g0 = E[0][step], G[0][step]
    print(f"  L0 ids eager {ids(e0[(1000, 'topk_ids')])} graph {ids(g0[(1000, 'topk_ids')])}")
    print(f"  L0 weights eager {[round(v, 4) for v in e0[(1000, 'topk_weights')][0].tolist()]}")
    print(f"  L0 weights graph {[round(v, 4) for v in g0[(1000, 'topk_weights')][0].tolist()]}")
