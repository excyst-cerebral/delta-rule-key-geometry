"""Is key-value consistency a stable property of a head?  Leave-one-sequence-out global comparator.

For each long-memory head: fit ONE ridge map Z (dv x dk) on the (key, unit-value) pairs of the other 3 sequences,
then use that fixed Z in the transfer-form bound (Thm 6') on windows of the held-out sequence.
The bound is valid for any Z, so this is a genuine out-of-sample certificate.
"""
import sys, json
sys.path.insert(0, ".")
import numpy as np
import torch
import gdn_extract as g
from gdn_kv import keys_values

NS = [8, 16, 32, 64]
RIDGES = [1e-3, 1e-2, 1e-1, 1.0]
ENDS = (255, 511, 767, 1023)

if __name__ == "__main__":
    tok = g.AutoTokenizer.from_pretrained(g.MD)
    data = g.load_sequences(tok, n_per_domain=2)
    per = json.load(open("results_gdn.json"))["per_head"]
    long_heads = {d: {tuple(map(int, k.split("|")[1:])) for k, v in per.items() if k.startswith(d + "|") and v["p256_A"] > 0.5}
                  for d in data}
    seqs = [(d, s) for d in data for s in data[d]]
    KV = []
    with torch.no_grad():
        for d, s in seqs:
            KV.append(keys_values(torch.tensor(s)))
            print(d, "extracted", flush=True)
    unitv = lambda V: V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)
    res = {(n, lam): [] for n in NS for lam in RIDGES}
    base = {n: [] for n in NS}
    for ti, (dom, _) in enumerate(seqs):
        for (li, h) in sorted(long_heads[dom]):
            Ktr = np.vstack([KV[j][li][0][:, h] for j in range(len(seqs)) if j != ti])
            Vtr = np.vstack([unitv(KV[j][li][1][:, h]) for j in range(len(seqs)) if j != ti])
            C = Ktr.T @ Ktr / len(Ktr); B = Ktr.T @ Vtr / len(Ktr)
            Kte, Vte = KV[ti][li][0][:, h], unitv(KV[ti][li][1][:, h])
            Zs = {lam: np.linalg.solve(C + lam * np.eye(g.DK), B).T for lam in RIDGES}
            zops = {lam: np.linalg.norm(Zs[lam], 2) for lam in RIDGES}
            for n in NS:
                for end in ENDS:
                    Kw, Vw = Kte[end - n + 1:end + 1], Vte[end - n + 1:end + 1]
                    G = Kw @ Kw.T
                    L = np.tril(G, -1)
                    T = L.T @ np.linalg.inv(np.eye(n) + L)
                    Fi = np.linalg.norm(T @ Vw, axis=1)
                    psi_i = np.linalg.norm(T @ Kw, axis=1)
                    trow = np.linalg.norm(T, axis=1)
                    Top, Tf = np.linalg.norm(T, 2), np.linalg.norm(T)
                    base[n].append((np.linalg.norm(T @ Vw) / np.sqrt(n), Fi.max()))
                    for lam in RIDGES:
                        Z = Zs[lam]
                        R = Kw @ Z.T - Vw
                        rr, rf = np.linalg.norm(R, 2), np.linalg.norm(R)
                        item = zops[lam] * psi_i + trow * rr
                        rmsb = zops[lam] * np.linalg.norm(psi_i) / np.sqrt(n) + min(Top * rf, Tf * rr) / np.sqrt(n)
                        res[(n, lam)].append(dict(zr=zops[lam], rr=rr, rrel=rf / np.sqrt(n), rms_b=rmsb, item_b=item.max(),
                                                  ok_r=float(np.linalg.norm(T @ Vw) / np.sqrt(n) <= rmsb * (1 + 1e-9)),
                                                  ok_i=float((Fi <= item + 1e-9).all())))
        print("held-out seq", ti, "done", flush=True)
    med = lambda n, lam, k: float(np.median([r[k] for r in res[(n, lam)]]))
    print(f"\nHeld-out (leave-one-sequence-out) global comparator, long-memory heads, windows per n = {len(base[NS[0]])}")
    print(f"{'n':>4} {'ridge':>6} | {'RMS real':>8} {'RMS bound':>9} | {'max real':>8} {'item bound':>10} | {'|Z|op':>6} {'|R|op':>6} {'|R|F/sqn':>8} | holds")
    for n in NS:
        for lam in RIDGES:
            print(f"{n:4d} {lam:6.0e} | {np.median([b[0] for b in base[n]]):8.3f} {med(n,lam,'rms_b'):9.3f} | "
                  f"{np.median([b[1] for b in base[n]]):8.3f} {med(n,lam,'item_b'):10.3f} | {med(n,lam,'zr'):6.2f} {med(n,lam,'rr'):6.2f} "
                  f"{med(n,lam,'rrel'):8.3f} | {np.mean([r['ok_r'] for r in res[(n,lam)]]):.2f},{np.mean([r['ok_i'] for r in res[(n,lam)]]):.2f}")
        print()
