import glob, os, sys
import torch
def load(d, n=6):
    files = sorted(glob.glob(os.path.join(d, "0*.pt")))
    return [torch.load(f) for f in files[1:1 + n]]
root = sys.argv[1]
E1, E2, E3 = (load(os.path.join(root, f"gtrace-eager{i}")) for i in (1, 2, 3))
G, G2 = load(os.path.join(root, "gtrace-graph")), load(os.path.join(root, "gtrace-graph2"))
names = ["engram_residual", "attn_in", "attn_out", "ffn_in", "ffn_out"]
def rel(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return float((a - b).norm() / (b.norm() + 1e-6))
for step in (1, 2, 3):
    print(f"== decode step {step} (row 0): graph-vs-eager  [eager-vs-eager noise]  ratio")
    for layer in range(0, 43):
        cells = []
        for name in names:
            k = (layer, name)
            if k not in E1[step] or k not in G[step]:
                cells.append(" " * 26); continue
            g = rel(G[step][k][0], E1[step][k][0])
            n = max(rel(E2[step][k][0], E1[step][k][0]), rel(E3[step][k][0], E1[step][k][0]), 1e-6)
            cells.append(f"{g:7.4f} [{n:7.4f}] x{g/n:6.1f}")
        if layer < 6 or layer % 6 == 0 or layer in (13, 14, 15):
            print(f" L{layer:2d} " + " | ".join(cells))
