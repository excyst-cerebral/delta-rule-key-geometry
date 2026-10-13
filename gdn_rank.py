"""Rank-based quantities on trained Gated-DeltaNet keys vs random keys, as a function of window size n.

Per window of n consecutive keys (long-memory heads only), with alpha = beta = 1 'ideal' single pass on
random unit-norm values (dv = 256), we report:
  r_eff, r_0.05 (=#lam>=0.05), lam_min
  RMS    : measured RMS_i recall error                      (trained vs random keys)
  T5x    : Thm 5' with exact ||W||_op                       sqrt((n/r_eff-1)/2) * ||W||_op
  T5a    : Thm 5' fully a priori (Lemma 6, best tau)
  floor  : Thm 3' lower bound (relative RMS) with B = ||V||_F, best m;  adv = delta rule measured on that V
"""
import sys, json
sys.path.insert(0, ".")
import numpy as np
import torch
import gdn_extract as g
from gdn_keydiag import forward_keys

NS = [4, 8, 16, 32, 64, 128]
TAUS = np.logspace(-3, 1, 25)
rng = np.random.default_rng(5)


def delta_pass(K, V):
    S = np.zeros((V.shape[1], K.shape[1]))
    for t in range(len(K)):
        S += np.outer(V[t] - S @ K[t], K[t])
    return S


def one_window(Kw):
    n = len(Kw)
    G = Kw @ Kw.T
    lam, U = np.linalg.eigh(G)
    lam = np.clip(lam, 0, None)
    V = rng.normal(size=(n, g.DV)); V /= np.linalg.norm(V, axis=1, keepdims=True)
    S = delta_pass(Kw, V)
    rms = np.linalg.norm(Kw @ S.T - V) / np.sqrt(n)
    r_eff = n ** 2 / np.linalg.norm(G) ** 2
    L = np.tril(G, -1)
    W = np.linalg.solve(np.eye(n) + L, V)
    fac = np.sqrt(max(n / r_eff - 1, 0) / 2)
    t5x = fac * np.linalg.norm(W, 2)
    best = np.inf
    for tau in TAUS:
        keep = lam >= tau
        if not keep.any():
            continue
        Pi = U[:, keep] @ U[:, keep].T
        N = V - Pi @ V
        nop = np.linalg.norm(N, 2)
        best = min(best, np.sqrt(np.linalg.norm(Pi @ V, 2) ** 2 / tau + nop ** 2) + nop)
    t5a = fac * best
    off = G - np.eye(n)
    rho_max = max(np.linalg.norm(off[i, i + 1:]) for i in range(n))
    old5 = rho_max * np.linalg.norm(V, 2) / np.sqrt(max(lam[0], 1e-12))          # previous Theorem 5 (max over items)
    # Thm 3' floor: V = sum_{j<m} u_j w_j^T over the m smallest eigen-directions, B = sqrt(m)
    fl, mbest = 0.0, 1
    for m in range(1, min(n, g.DV) + 1):
        f = (np.clip(1 - np.sqrt(m * lam[:m]), 0, None) ** 2).mean()
        if f > fl:
            fl, mbest = f, m
    Wo = np.linalg.qr(rng.normal(size=(g.DV, mbest)))[0]
    Va = U[:, :mbest] @ Wo.T
    Sa = delta_pass(Kw, Va)
    adv = np.linalg.norm(Kw @ Sa.T - Va) / np.sqrt(mbest)
    return dict(ok5x=float(rms <= t5x*(1+1e-9)), ok5a=float(rms <= t5a*(1+1e-9)), okfl=float(fl**0.5 <= adv*(1+1e-9)), old5=old5, r_eff=r_eff, r05=float((lam >= 0.05).sum()), lam_min=lam[0], rms=rms, t5x=t5x, t5a=t5a,
                floor=np.sqrt(fl), adv=adv, m=mbest)


def agg(rows):
    return {k: float(np.median([r[k] for r in rows])) for k in rows[0]}


if __name__ == "__main__":
    tok = g.AutoTokenizer.from_pretrained(g.MD)
    data = g.load_sequences(tok, n_per_domain=2)
    per = json.load(open("results_gdn.json"))["per_head"]
    long_heads = {d: {tuple(map(int, k.split("|")[1:])) for k, v in per.items() if k.startswith(d + "|") and v["p256_A"] > 0.5}
                  for d in data}
    trained = {n: [] for n in NS}
    with torch.no_grad():
        for dom, seqs in data.items():
            for seq in seqs:
                Ks = forward_keys(torch.tensor(seq))
                for (li, h) in sorted(long_heads[dom]):
                    Kh = Ks[li][:, h]
                    for n in NS:
                        for end in (255, 511, 767, 1023):
                            trained[n].append(one_window(Kh[end - n + 1:end + 1]))
                print(dom, "seq done", flush=True)
    rand = {n: [one_window(g_ / np.linalg.norm(g_, axis=1, keepdims=True))
                for g_ in [rng.normal(size=(n, g.DK)) for _ in range(40)]] for n in NS}
    for name, src in (("trained", trained), ("random", rand)):
        for n in NS:
            print(f"per-window checks [{name} n={n:3d}]  T5x holds: {np.mean([r['ok5x'] for r in src[n]]):.3f}  "
                  f"T5a holds: {np.mean([r['ok5a'] for r in src[n]]):.3f}  floor<=measured: {np.mean([r['okfl'] for r in src[n]]):.3f}")
    out = {"trained": {n: agg(v) for n, v in trained.items()}, "random": {n: agg(v) for n, v in rand.items()}}
    json.dump({str(k): v for k, v in out.items()}, open("results_rank.json", "w"), indent=1, default=str)
    for name in ("trained", "random"):
        print(f"\n=== {name} keys ({'long-memory heads, ' + str(len(trained[NS[0]])) + ' windows' if name=='trained' else '40 draws'}) ===")
        print(f"{'n':>4} {'r_eff':>6} {'r_.05':>6} {'lam_min':>8} | {'RMS err':>8} {'T5x':>8} {'T5a':>9} {'old T5':>9} | {'floor':>6} {'adv(meas)':>9} {'m*':>4}")
        for n in NS:
            a = out[name][n]
            print(f"{n:4d} {a['r_eff']:6.1f} {a['r05']:6.0f} {a['lam_min']:8.1e} | {a['rms']:8.3f} {a['t5x']:8.2f} {a['t5a']:9.1f} {a['old5']:9.1f} | "
                  f"{a['floor']:6.3f} {a['adv']:9.3f} {a['m']:4.0f}")
