"""Effective rank (participation ratio of the Gram spectrum) of the window's hidden state x and of each readout stage, per layer group."""
import sys, os, json
sys.path.insert(0, ".")
import numpy as np, torch
import gdn_extract as g
from gdn_flat import layer_streams, unit
NS = [32, 64]; ENDS = list(range(255, 1024, 128))
reff = lambda X: (lambda s2: s2.sum() ** 2 / (s2 ** 2).sum())(np.linalg.svd(unit(X), compute_uv=False) ** 2)
tok = g.AutoTokenizer.from_pretrained(g.MD); data = g.load_sequences(tok, n_per_domain=2)
cls = json.load(open(f"classes_{os.path.basename(g.MD)}.json")); long_heads = sorted(tuple(map(int, k.split("|"))) for k, v in cls.items() if v == "long")
acc = {n: {k: [] for k in ("x", "kpre", "K", "V", "Vrot", "Vgauss")} for n in NS}
with torch.no_grad():
    for dom, seqs in data.items():
        for seq in seqs:
            S = layer_streams(torch.tensor(seq))
            for (li, h) in long_heads:
                st = S[li]
                for n in NS:
                    for end in ENDS:
                        sl = slice(end - n + 1, end + 1)
                        for k, X in (("x", st["x"][sl]), ("kpre", st["kpre"][sl, h]), ("K", st["K"][sl, h]), ("V", st["V"]["own"][sl, h]),
                                     ("Vrot", st["V"]["rot"][sl, h]), ("Vgauss", st["V"]["gauss"][sl, h])):
                            acc[n][k].append(reff(X))
print(os.path.basename(g.MD), "- effective rank of the window Gram (rows normalised), long-head layers, medians")
for n in NS:
    print(f"n={n}: " + "  ".join(f"{k}={np.median(v):.1f}" for k, v in acc[n].items()) + f"   (max possible {n})")
