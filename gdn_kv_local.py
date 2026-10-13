"""Causal-local comparator: fit Z by ridge on the (up to) 256 tokens BEFORE the window (same sequence), apply to the window.
Computable at inference time; the bound is valid for any Z, so this is an out-of-window certificate."""
import sys, json
sys.path.insert(0, ".")
import numpy as np
import torch
import gdn_extract as g
from gdn_kv import keys_values

NS = [8, 16, 32, 64]
RIDGES = [1e-3, 1e-2, 1e-1]
ENDS = (511, 767, 1023)          # >= 448 prior tokens available
CTX = 256

if __name__ == "__main__":
    tok = g.AutoTokenizer.from_pretrained(g.MD)
    data = g.load_sequences(tok, n_per_domain=2)
    per = json.load(open("results_gdn.json"))["per_head"]
    long_heads = {d: {tuple(map(int, k.split("|")[1:])) for k, v in per.items() if k.startswith(d + "|") and v["p256_A"] > 0.5} for d in data}
    unitv = lambda V: V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)
    res = {(n, l): [] for n in NS for l in RIDGES}; base = {n: [] for n in NS}
    with torch.no_grad():
        for dom, seqs in data.items():
            for seq in seqs:
                KV = keys_values(torch.tensor(seq))
                for (li, h) in sorted(long_heads[dom]):
                    K, V = KV[li][0][:, h], unitv(KV[li][1][:, h])
                    for n in NS:
                        for end in ENDS:
                            s0 = end - n + 1
                            Ktr, Vtr = K[s0 - CTX:s0], V[s0 - CTX:s0]
                            C = Ktr.T @ Ktr / len(Ktr); B = Ktr.T @ Vtr / len(Ktr)
                            Kw, Vw = K[s0:end + 1], V[s0:end + 1]
                            G = Kw @ Kw.T; L = np.tril(G, -1)
                            T = L.T @ np.linalg.inv(np.eye(n) + L)
                            Fi = np.linalg.norm(T @ Vw, axis=1); psi_i = np.linalg.norm(T @ Kw, axis=1)
                            trow = np.linalg.norm(T, axis=1); Top, Tf = np.linalg.norm(T, 2), np.linalg.norm(T)
                            base[n].append((np.linalg.norm(T @ Vw) / np.sqrt(n), Fi.max()))
                            for lam in RIDGES:
                                Z = np.linalg.solve(C + lam * np.eye(g.DK), B).T
                                zr = np.linalg.norm(Z, 2); R = Kw @ Z.T - Vw
                                rr, rf = np.linalg.norm(R, 2), np.linalg.norm(R)
                                item = zr * psi_i + trow * rr
                                rmsb = zr * np.linalg.norm(psi_i) / np.sqrt(n) + min(Top * rf, Tf * rr) / np.sqrt(n)
                                res[(n, lam)].append(dict(zr=zr, rr=rr, rrel=rf / np.sqrt(n), rms_b=rmsb, item_b=item.max(),
                                    ok_r=float(base[n][-1][0] <= rmsb * (1 + 1e-9)), ok_i=float((Fi <= item + 1e-9).all())))
                print(dom, "seq done", flush=True)
    med = lambda n, l, k: float(np.median([r[k] for r in res[(n, l)]]))
    print(f"\nCausal-local comparator (ridge fit on preceding {CTX} tokens), long-memory heads, windows per n = {len(base[NS[0]])}")
    print(f"{'n':>4} {'ridge':>6} | {'RMS real':>8} {'RMS bound':>9} | {'max real':>8} {'item bound':>10} | {'|Z|op':>6} {'|R|op':>6} {'|R|F/sqn':>8} | holds")
    for n in NS:
        for l in RIDGES:
            print(f"{n:4d} {l:6.0e} | {np.median([b[0] for b in base[n]]):8.3f} {med(n,l,'rms_b'):9.3f} | {np.median([b[1] for b in base[n]]):8.3f} "
                  f"{med(n,l,'item_b'):10.3f} | {med(n,l,'zr'):6.2f} {med(n,l,'rr'):6.2f} {med(n,l,'rrel'):8.3f} | "
                  f"{np.mean([r['ok_r'] for r in res[(n,l)]]):.2f},{np.mean([r['ok_i'] for r in res[(n,l)]]):.2f}")
        print()
