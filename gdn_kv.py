"""Key-value-aware bounds on the trained Gated-DeltaNet (long-memory heads), ideal alpha = beta = 1 pass.

Per window of n consecutive (key, value) pairs, values L2-normalised:
  RMS_real  : recall error with the model's real key-value pairing
  RMS_shuf  : same keys, same values, pairing shuffled (control: isolates key-value correlation)
  RMS_gauss : random unit values
  psi       : RMS error of the self-association run (values = keys)        [key-only quantity]
  fac       : sqrt((n/r_eff - 1)/2)
  zr, rr    : ||Z||_op and ||R||_op of the bound-minimising ridge comparator Z
  KV        : Thm 6 RMS bound  fac * (sqrt(zr^2+rr^2)+rr)
  rank-only : Thm 5' a priori bound (Lemma 8 truncation) with the same real V
  KV_max    : item-wise Thm 6, max_i rho_i * bracket   vs   measured max_i F_i
  old5      : previous Thm 5, max_i rho_i ||V||_op / sqrt(lam_min)
"""
import sys, json
sys.path.insert(0, ".")
import numpy as np
import torch
import torch.nn.functional as F
import gdn_extract as g

NS = [8, 16, 32, 64, 128]
LAMS = np.logspace(-5, 1, 25)
TAUS = np.logspace(-3, 1, 25)
rng = np.random.default_rng(7)


def keys_values(ids):
    h = g.W["model.embeddings.weight"][ids]
    out = {}
    for i in range(g.NL):
        x = g.rms(h, g.W[f"model.layers.{i}.attn_norm.weight"])
        n = x.shape[0]
        k = g.sconv(x @ g.W[f"model.layers.{i}.attn.k_proj.weight"].T, g.W[f"model.layers.{i}.attn.k_conv1d.weight"])
        v = g.sconv(x @ g.W[f"model.layers.{i}.attn.v_proj.weight"].T, g.W[f"model.layers.{i}.attn.v_conv1d.weight"])
        out[i] = (F.normalize(k.view(n, g.H, g.DK), dim=-1, eps=1e-6).numpy().astype(np.float64),
                  v.view(n, g.H, g.DV).numpy().astype(np.float64))
        a, _ = g.gdn(x, i, False)
        h = h + a
        m = g.rms(h, g.W[f"model.layers.{i}.mlp_norm.weight"])
        h = h + g.mlp(m, i)
    return out


def window(Kw, Vraw):
    n = len(Kw)
    V = Vraw / np.maximum(np.linalg.norm(Vraw, axis=1, keepdims=True), 1e-12)
    G = Kw @ Kw.T
    lam, U = np.linalg.eigh(G); lam = np.clip(lam, 1e-12, None)
    L = np.tril(G, -1)
    A = np.eye(n) + L
    T = L.T @ np.linalg.inv(A)
    rho = np.array([np.linalg.norm(G[i, i + 1:]) for i in range(n)])
    r_eff = n ** 2 / np.linalg.norm(G) ** 2
    fac = np.sqrt(max(n / r_eff - 1, 0) / 2)
    rms = lambda X: np.linalg.norm(T @ X) / np.sqrt(n)
    Fi = np.linalg.norm(T @ V, axis=1)
    shuf = np.mean([rms(V[rng.permutation(n)]) for _ in range(3)])
    Vg = rng.normal(size=V.shape); Vg /= np.linalg.norm(Vg, axis=1, keepdims=True)
    psi = np.linalg.norm(T @ Kw) / np.sqrt(n)
    # comparator profile via eigen-decomposition (Z = V^T U diag(sqrt(lam)/(lam+l)) Y^T, R = -l U diag(1/(lam+l)) U^T V)
    UtV = U.T @ V
    best = (np.inf, None, None)
    for l in LAMS:
        zr = np.linalg.norm((np.sqrt(lam) / (lam + l))[:, None] * UtV, 2)
        rr = np.linalg.norm((l / (lam + l))[:, None] * UtV, 2)
        b = np.sqrt(zr ** 2 + rr ** 2) + rr
        if b < best[0]:
            best = (b, zr, rr)
    b, zr, rr = best
    # rank-only a priori bound with the same real V (Lemma 8)
    rb = np.inf
    for tau in TAUS:
        keep = lam >= tau
        if not keep.any(): continue
        Pi = U[:, keep] @ U[:, keep].T
        N = V - Pi @ V
        nop = np.linalg.norm(N, 2)
        rb = min(rb, np.sqrt(np.linalg.norm(Pi @ V, 2) ** 2 / tau + nop ** 2) + nop)
    vop = np.linalg.norm(V, 2)
    # --- transfer form (Thm 6'):  F_i <= ||Z||_op psi_i + ||T_i|| ||R||_op ;  RMS <= ||Z|| psi_rms + min(||T||_op ||R||_F, ||T||_F ||R||_op)/sqrt(n)
    P = T @ Kw
    psi_i = np.linalg.norm(P, axis=1)
    trow = np.linalg.norm(T, axis=1)
    Top, Tf = np.linalg.norm(T, 2), np.linalg.norm(T)
    best_i, best_r = (np.inf, None), (np.inf, None)
    for l in LAMS:
        zr_ = np.linalg.norm((np.sqrt(lam) / (lam + l))[:, None] * UtV, 2)
        Rm = (l / (lam + l))[:, None] * UtV
        rr_, rf_ = np.linalg.norm(Rm, 2), np.linalg.norm(Rm)
        item = zr_ * psi_i + trow * rr_
        rmsb = zr_ * psi + min(Top * rf_, Tf * rr_) / np.sqrt(n)
        if item.max() < best_i[0]: best_i = (item.max(), item)
        if rmsb < best_r[0]: best_r = (rmsb, None)
    kv_rms = fac * b
    kv_item = rho * b
    return dict(rms_real=rms(V), rms_shuf=shuf, rms_gauss=rms(Vg), psi=psi, fac=fac, zr=zr, rr=rr, vop=vop,
                kv=kv_rms, rank_only=fac * rb, old5=rho.max() * vop / np.sqrt(lam[0]),
                max_real=Fi.max(), kv_max=kv_item.max(),
                ok_rms=float(rms(V) <= kv_rms * (1 + 1e-9)), ok_item=float((Fi <= kv_item + 1e-9).all()),
                transfer=zr * psi, tf_rms=best_r[0], tf_item=best_i[0],
                ok_tf_rms=float(rms(V) <= best_r[0] * (1 + 1e-9)), ok_tf_item=float((Fi <= best_i[1] + 1e-9).all()),
                Top=Top, Tf_rms=Tf / np.sqrt(n))


if __name__ == "__main__":
    tok = g.AutoTokenizer.from_pretrained(g.MD)
    data = g.load_sequences(tok, n_per_domain=2)
    per = json.load(open("results_gdn.json"))["per_head"]
    long_heads = {d: {tuple(map(int, k.split("|")[1:])) for k, v in per.items() if k.startswith(d + "|") and v["p256_A"] > 0.5}
                  for d in data}
    rows = {n: [] for n in NS}
    with torch.no_grad():
        for dom, seqs in data.items():
            for seq in seqs:
                KV = keys_values(torch.tensor(seq))
                for (li, h) in sorted(long_heads[dom]):
                    Kh, Vh = KV[li][0][:, h], KV[li][1][:, h]
                    for n in NS:
                        for end in (255, 511, 767, 1023):
                            rows[n].append(window(Kh[end - n + 1:end + 1], Vh[end - n + 1:end + 1]))
                print(dom, "seq done", flush=True)
    med = lambda n, k: float(np.median([r[k] for r in rows[n]]))
    json.dump({str(n): {k: med(n, k) for k in rows[n][0]} for n in NS}, open("results_kv.json", "w"), indent=1)
    print(f"\nlong-memory heads, {len(rows[NS[0]])} windows per n, medians; values L2-normalised")
    print(f"{'n':>4} | {'RMS real':>8} {'shuffled':>8} {'gauss':>6} | {'psi':>5} {'fac':>5} {'|Z|op':>6} {'|R|op':>6} {'|V|op':>6} | "
          f"{'KV bnd':>7} {'rank-only':>9} | {'max real':>8} {'KV item':>8} {'old T5':>8} | holds(rms,item)")
    for n in NS:
        pass
        print(f"{n:4d} | {med(n,'rms_real'):8.3f} {med(n,'rms_shuf'):8.3f} {med(n,'rms_gauss'):6.3f} | {med(n,'psi'):5.2f} "
              f"{med(n,'fac'):5.2f} {med(n,'zr'):6.2f} {med(n,'rr'):6.2f} {med(n,'vop'):6.2f} | "
              f"{med(n,'kv'):7.2f} {med(n,'rank_only'):9.2f} | {med(n,'max_real'):8.3f} {med(n,'kv_max'):8.2f} {med(n,'old5'):8.1f} | "
              f"{np.mean([r['ok_rms'] for r in rows[n]]):.2f},{np.mean([r['ok_item'] for r in rows[n]]):.2f}")

    print("\nTransfer form (Thm 6'):  RMS <= |Z| psi + min(|T|op |R|F, |T|F |R|op)/sqrt(n);   item: F_i <= |Z| psi_i + |T_i| |R|op")
    print(f"{'n':>4} | {'RMS real':>8} {'TF rms bnd':>10} {'(fac form)':>10} | {'max real':>8} {'TF item bnd':>11} {'(fac item)':>10} {'old T5':>8} | {'|T|op':>6} {'|T|F/sqn':>8} | holds(rms,item)")
    for n in NS:
        print(f"{n:4d} | {med(n,'rms_real'):8.3f} {med(n,'tf_rms'):10.3f} {med(n,'kv'):10.2f} | {med(n,'max_real'):8.3f} {med(n,'tf_item'):11.2f} "
              f"{med(n,'kv_max'):10.2f} {med(n,'old5'):8.1f} | {med(n,'Top'):6.2f} {med(n,'Tf_rms'):8.2f} | "
              f"{np.mean([r['ok_tf_rms'] for r in rows[n]]):.2f},{np.mean([r['ok_tf_item'] for r in rows[n]]):.2f}")
