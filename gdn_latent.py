"""Shared-latent hypothesis  K = X A,  V = X B  (X: n x m, m small), long-memory heads, unit-normalised values.

T1  joint low rank        r_eff([K V]) vs r_eff(K) + r_eff(V); real vs shuffled / other-head / other-layer pairing
T2  alignment             fraction of ||V||_F^2 inside the top-m eigenspace of G = K K^T  (key-only subspace)
T3  mechanism             q = tr(V^T G^-1 V)/n  (= ||S*||_F^2 / n, the comparator norm of Thm 2);  Spearman with measured error
T4  latent comparators    (a) joint-SVD latent X from [K V];  (b) key-only latent X = top-m left singular vectors of K
                          Z^T = pinv(X^T K) X^T V ;  certificate = Thm 6' with the true residual
T5  predictive certificate  key-only comparator, with ||Z||_op, ||R||_op, ||R||_F/sqrt(n) replaced by the 90th percentile of the same
                          quantity on windows from OTHER sequences (leave-one-sequence-out); m chosen on the training windows only
"""
import sys, json
sys.path.insert(0, ".")
import numpy as np
import torch
import gdn_extract as g
from gdn_kv import keys_values
from gdn_kv_tok import err_op, rms

NS = [16, 32, 64]
ENDS = list(range(255, 1024, 64))
MS = [1, 2, 4, 8, 16]
MJ = [4, 8, 16]
MK = [2, 4, 8, 12, 16, 24, 32]
rng = np.random.default_rng(31)
unitv = lambda V: V / np.maximum(np.linalg.norm(V, axis=-1, keepdims=True), 1e-12)


def reff(M):
    s2 = np.linalg.svd(M, compute_uv=False) ** 2
    return s2.sum() ** 2 / (s2 ** 2).sum()


def tf_bound(zr, rr, rf, psi_rms, Top, Tf, n):
    """Theorem 6' RMS bound; rf = ||R||_F / sqrt(n)."""
    return zr * psi_rms + min(Top * rf * np.sqrt(n), Tf * rr) / np.sqrt(n)


def spearman(a, b):
    rk = lambda x: np.argsort(np.argsort(x)).astype(float)
    return float(np.corrcoef(rk(np.asarray(a)), rk(np.asarray(b)))[0, 1])


def window_all(Kw, Vs, n):
    """Vs: dict name -> unit values (n x dv). Returns metrics for the window."""
    G = Kw @ Kw.T
    lam, U = np.linalg.eigh(G)
    lam, U = lam[::-1], U[:, ::-1]                       # descending
    lam = np.clip(lam, 1e-12, None)
    T = err_op(Kw)
    Top, Tf = np.linalg.norm(T, 2), np.linalg.norm(T)
    psi_rms = np.linalg.norm(T @ Kw) / np.sqrt(n)
    out = dict(rK=reff(Kw))
    for name, V in Vs.items():
        UtV = U.T @ V
        o = {}
        o["rV"] = reff(V)
        o["rJ"] = reff(np.hstack([Kw, V]))
        o["align"] = {m: float((UtV[:m] ** 2).sum() / (V ** 2).sum()) for m in MS if m < n}
        o["q"] = float((UtV ** 2 / lam[:, None]).sum() / n)
        o["rms"] = rms(T, V)
        if name == "real":
            # (a) joint-SVD latent
            Ucat = np.linalg.svd(np.hstack([Kw, V]), full_matrices=False)[0]
            o["joint"] = {}
            for m in MJ:
                if m >= n: continue
                X = Ucat[:, :m]
                A, B = X.T @ Kw, X.T @ V
                Zt = np.linalg.pinv(A) @ B
                R = Kw @ Zt - V
                zr, rr, rf = np.linalg.norm(Zt, 2), np.linalg.norm(R, 2), np.linalg.norm(R) / np.sqrt(n)
                o["joint"][m] = dict(zr=zr, rr=rr, rf=rf, b=tf_bound(zr, rr, rf, psi_rms, Top, Tf, n))
            # (b) key-only latent
            o["key"] = {}
            for m in MK:
                if m >= n: continue
                zr = np.linalg.norm(V.T @ U[:, :m] / np.sqrt(lam[:m]), 2)
                N = V - U[:, :m] @ UtV[:m]
                rr, rf = np.linalg.norm(N, 2), np.linalg.norm(N) / np.sqrt(n)
                o["key"][m] = dict(zr=zr, rr=rr, rf=rf, b=tf_bound(zr, rr, rf, psi_rms, Top, Tf, n))
            o["Top"], o["Tf"], o["psi"] = Top, Tf, psi_rms
        out[name] = o
    return out


if __name__ == "__main__":
    tok = g.AutoTokenizer.from_pretrained(g.MD)
    data = g.load_sequences(tok, n_per_domain=3)
    per = json.load(open("results_gdn.json"))["per_head"]
    long_heads = {d: {tuple(map(int, k.split("|")[1:])) for k, v in per.items() if k.startswith(d + "|") and v["p256_A"] > 0.5} for d in data}
    W = {n: [] for n in NS}                                     # list of (seq_id, result)
    sid = 0
    with torch.no_grad():
        for dom, seqs in data.items():
            for seq in seqs:
                KV = keys_values(torch.tensor(seq))
                for (li, h) in sorted(long_heads[dom]):
                    K = KV[li][0][:, h]; V = unitv(KV[li][1][:, h])
                    h2 = (h + 1 + rng.integers(g.H - 1)) % g.H
                    lj = (li + 1 + rng.integers(g.NL - 1)) % g.NL
                    Voh = unitv(KV[li][1][:, h2]); Vol = unitv(KV[lj][1][:, rng.integers(g.H)])
                    for n in NS:
                        for end in ENDS:
                            s0 = end - n + 1
                            p = rng.permutation(n)
                            Vs = {"real": V[s0:end + 1], "shuf": V[s0:end + 1][p], "oh": Voh[s0:end + 1], "ol": Vol[s0:end + 1]}
                            W[n].append((sid, window_all(K[s0:end + 1], Vs, n)))
                sid += 1
                print(dom, "seq done", flush=True)
    srcs = ["real", "shuf", "oh", "ol"]
    med = lambda n, f: float(np.median([f(r) for _, r in W[n]]))
    print(f"\nlong-memory heads, {len(W[NS[0]][0:])} windows per n, 6 sequences")
    print("\n=== T1. joint low rank (participation-ratio rank; median over windows) ===")
    print(f"{'n':>3} | {'r(K)':>5} | " + " | ".join(f"{s:>5}: r(V) r([K V]) ratio" for s in srcs))
    for n in NS:
        row = f"{n:3d} | {med(n,lambda r:r['rK']):5.1f} | "
        row += " | ".join(f"{'':6}{med(n,lambda r,s=s:r[s]['rV']):4.1f} {med(n,lambda r,s=s:r[s]['rJ']):9.1f} {med(n,lambda r,s=s:r[s]['rJ']/(r['rK']+r[s]['rV'])):5.2f}" for s in srcs)
        print(row)
    print("   (ratio = r([K V]) / (r(K)+r(V)):  0.5 = identical row spaces, 1.0 = nothing shared)")
    print("\n=== T2. fraction of ||V||_F^2 in the top-m eigenspace of K K^T ===")
    print(f"{'n':>3} {'src':>5} | " + " ".join(f"m={m:<2}" .rjust(7) for m in MS) + " | uniform m/n at m=8")
    for n in NS:
        for s in srcs:
            print(f"{n:3d} {s:>5} | " + " ".join(f"{med(n,lambda r,s=s,m=m:r[s]['align'].get(m,np.nan)):7.2f}" for m in MS) + (f" | {8/n:.2f}" if s == 'real' else ""))
    print("\n=== T3. comparator norm q = ||S*||_F^2/n = tr(V^T G^-1 V)/n  and its link to the measured error ===")
    print(f"{'n':>3} | " + " ".join(f"{s:>9}" for s in srcs) + " | RMS err: " + " ".join(f"{s:>6}" for s in srcs) + " | Spearman(log q, err) real | Spearman(align8, err) real")
    for n in NS:
        q = [med(n, lambda r, s=s: r[s]['q']) for s in srcs]
        e = [med(n, lambda r, s=s: r[s]['rms']) for s in srcs]
        sp1 = spearman([np.log(r['real']['q']) for _, r in W[n]], [r['real']['rms'] for _, r in W[n]])
        sp2 = spearman([r['real']['align'][8] for _, r in W[n]], [r['real']['rms'] for _, r in W[n]])
        print(f"{n:3d} | " + " ".join(f"{x:9.2f}" for x in q) + "           " + " ".join(f"{x:6.3f}" for x in e) + f" | {sp1:25.2f} | {sp2:25.2f}")
    print("\n=== T4. latent comparators with the true residual (Thm 6' RMS bound; medians) ===")
    print(f"{'n':>3} | {'measured':>8} | joint-SVD latent: " + " ".join(f"m={m}: |Z| |R|F b".rjust(20) for m in MJ))
    for n in NS:
        row = f"{n:3d} | {med(n,lambda r:r['real']['rms']):8.3f} | {'':17}"
        for m in MJ:
            if m < n:
                row += f"{med(n,lambda r,m=m:r['real']['joint'][m]['zr']):5.2f} {med(n,lambda r,m=m:r['real']['joint'][m]['rf']):5.2f} {med(n,lambda r,m=m:r['real']['joint'][m]['b']):6.2f}   "
        print(row)
    print(f"{'n':>3} | key-only latent (top-m left sing. vectors of K):  m: |Z|op  |R|F/sqn  bound")
    for n in NS:
        print(f"{n:3d} | " + "  ".join(f"m={m}: {med(n,lambda r,m=m:r['real']['key'][m]['zr']):.2f} {med(n,lambda r,m=m:r['real']['key'][m]['rf']):.2f} {med(n,lambda r,m=m:r['real']['key'][m]['b']):.2f}"
                                       for m in MK if m < n))
    print("   (post-hoc ridge comparator, Section 3d: 0.90 at n=32, 1.23 at n=64; measured RMS 0.47, 0.52)")
    print("\n=== T5. predictive certificate: key-only comparator, constants calibrated on OTHER sequences (q90), m chosen on training windows ===")
    print(f"{'n':>3} | {'m*':>3} | {'q90 |Z|':>7} {'q90 |R|op':>9} {'q90 |R|F/sqn':>12} | {'measured':>8} {'pred. bound':>11} {'% bound<1':>9} | {'coverage of constants':>21} {'bound valid':>11}")
    nseq = sid
    for n in NS:
        recs = W[n]
        best_overall = None
        res = {m: dict(b=[], cov=[], valid=[], consts=[]) for m in MK if m < n}
        for held in range(nseq):
            tr = [r['real'] for s_, r in recs if s_ != held]; te = [(r['real']) for s_, r in recs if s_ == held]
            for m in res:
                zq = np.quantile([x['key'][m]['zr'] for x in tr], .9)
                oq = np.quantile([x['key'][m]['rr'] for x in tr], .9)
                fq = np.quantile([x['key'][m]['rf'] for x in tr], .9)
                for x in te:
                    b = tf_bound(zq, oq, fq, x['psi'], x['Top'], x['Tf'], n)
                    k = x['key'][m]
                    res[m]['b'].append(b); res[m]['valid'].append(x['rms'] <= b * (1 + 1e-9))
                    res[m]['cov'].append(k['zr'] <= zq and k['rr'] <= oq and k['rf'] <= fq)
                    res[m]['consts'].append((zq, oq, fq))
        # choose m by median bound over all held-out windows (a mild selection; the grid has 5-7 values)
        mstar = min(res, key=lambda m: np.median(res[m]['b']))
        r_ = res[mstar]
        zq, oq, fq = np.median(np.array(r_['consts']), axis=0)
        meas = np.median([x['rms'] for _, x in [(a, b['real']) for a, b in recs]])
        print(f"{n:3d} | {mstar:3d} | {zq:7.2f} {oq:9.2f} {fq:12.2f} | {meas:8.3f} {np.median(r_['b']):11.2f} {100*np.mean(np.array(r_['b'])<1):8.0f}% | {np.mean(r_['cov']):21.2f} {np.mean(r_['valid']):11.2f}")
        print("      all m:", {m: round(float(np.median(res[m]['b'])), 2) for m in res})
