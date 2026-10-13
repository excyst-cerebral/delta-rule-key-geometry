"""Bounding q = tr(V^T G^-1 V)/n  and the error from key-side information (long-memory heads, unit-normalised values, alpha=beta=1).

Q1  how loose are the key-only worst-case bounds on q  (1/lam_min, tr(G^-1)/n)?
Q2  random-pairing null:  E_pi[q] = [tr(G^-1)(1-hbar) + hbar 1^T G^-1 1]/n   vs measured shuffled q
Q3  gain profile kappa_j = ||u_j^T V||^2 / lam_j  across key eigen-directions (q = mean_j kappa_j)
Q4  predicting q from key-side features, leave-one-sequence-out (LOSO): R^2 and a calibrated q upper bound
Q5  error:  n RMS^2 = tr(M B);  RMS ~ psi_rms * sqrt(q);  rigorous  RMS <= ||TK||_op sqrt(q);  calibrated  RMS <= r90 * psi_rms
"""
import sys, json
sys.path.insert(0, ".")
import numpy as np
import torch
import gdn_extract as g
from gdn_kv import keys_values
from gdn_kv_tok import err_op

NS = [16, 32, 64]
ENDS = list(range(255, 1024, 64))
rng = np.random.default_rng(17)
unitv = lambda V: V / np.maximum(np.linalg.norm(V, axis=-1, keepdims=True), 1e-12)


def spearman(a, b):
    rk = lambda x: np.argsort(np.argsort(x)).astype(float)
    return float(np.corrcoef(rk(np.asarray(a)), rk(np.asarray(b)))[0, 1])


def qval(UtV, lam, n):
    return float((UtV ** 2 / lam[:, None]).sum() / n)


def window(Kw, Vw):
    n = len(Kw)
    G = Kw @ Kw.T
    lam, U = np.linalg.eigh(G); lam, U = np.clip(lam[::-1], 1e-12, None), U[:, ::-1]
    T = err_op(Kw)
    TK = T @ Kw
    psi_i = np.linalg.norm(TK, axis=1); psi = np.linalg.norm(TK) / np.sqrt(n)
    Kop = np.linalg.norm(TK, 2)
    UtV = U.T @ Vw
    kap = (UtV ** 2).sum(1) / lam
    q = kap.mean()
    Mt = (UtV / np.sqrt(lam)[:, None]) @ (UtV / np.sqrt(lam)[:, None]).T            # M in the key eigenbasis
    gmax = np.linalg.eigvalsh(Mt).max()
    E = T @ Vw; Fi = np.linalg.norm(E, axis=1); rms = np.linalg.norm(E) / np.sqrt(n)
    qs = np.mean([qval(U.T @ Vw[rng.permutation(n)], lam, n) for _ in range(3)])
    hbar = (np.linalg.norm(Vw.sum(0)) ** 2 - n) / (n * (n - 1))
    ones_t = U.T @ np.ones(n)
    qnull = ((1 / lam).sum() * (1 - hbar) + hbar * (ones_t ** 2 / lam).sum()) / n
    third = n // 3
    ll = np.log(lam)
    slope = float(np.polyfit(ll, np.log(np.clip(kap, 1e-12, None)), 1)[0])
    return dict(q=q, qshuf=qs, qnull=qnull, hbar=hbar, lam_min=lam[-1], lam1=lam[0], trGinv=(1 / lam).sum() / n,
                reff=n ** 2 / np.linalg.norm(G) ** 2, psi=psi, psi_max=psi_i.max(), Kop=Kop, rms=rms, Fmax=Fi.max(), gmax=gmax,
                k_top=kap[:third].mean(), k_mid=kap[third:2 * third].mean(), k_bot=kap[2 * third:].mean(), slope=slope)


if __name__ == "__main__":
    tok = g.AutoTokenizer.from_pretrained(g.MD)
    data = g.load_sequences(tok, n_per_domain=3)
    per = json.load(open("results_gdn.json"))["per_head"]
    long_heads = {d: {tuple(map(int, k.split("|")[1:])) for k, v in per.items() if k.startswith(d + "|") and v["p256_A"] > 0.5} for d in data}
    W = {n: [] for n in NS}; sid = 0
    with torch.no_grad():
        for dom, seqs in data.items():
            for seq in seqs:
                KV = keys_values(torch.tensor(seq))
                for (li, h) in sorted(long_heads[dom]):
                    K = KV[li][0][:, h]; V = unitv(KV[li][1][:, h])
                    for n in NS:
                        for end in ENDS:
                            W[n].append((sid, window(K[end - n + 1:end + 1], V[end - n + 1:end + 1])))
                sid += 1
                print(dom, "seq done", flush=True)
    nseq = sid
    arr = lambda n, k: np.array([r[k] for _, r in W[n]])
    med = lambda n, k: float(np.median(arr(n, k)))
    print(f"\nlong-memory heads, {len(W[NS[0]])} windows per n, {nseq} sequences")

    print("\n=== Q1. key-only worst-case bounds on q (unit-norm value rows) ===")
    print(f"{'n':>3} | {'q':>6} {'q p90':>6} | {'1/lam_min (worst case)':>22} {'ratio':>8} | {'tr(G^-1)/n':>10} {'ratio':>7} | {'lam_min':>8}")
    for n in NS:
        print(f"{n:3d} | {med(n,'q'):6.2f} {np.quantile(arr(n,'q'),.9):6.2f} | {np.median(1/arr(n,'lam_min')):22.1f} {np.median(1/arr(n,'lam_min')/arr(n,'q')):8.0f} | "
              f"{med(n,'trGinv'):10.1f} {np.median(arr(n,'trGinv')/arr(n,'q')):7.1f} | {med(n,'lam_min'):8.4f}")

    print("\n=== Q2. random-pairing null: closed form vs measured shuffled q ===")
    print(f"{'n':>3} | {'hbar (value redundancy)':>23} | {'q shuffled (meas.)':>18} {'null formula':>13} {'median ratio':>13} {'Spearman':>9} | {'q real':>7} {'real/null':>10}")
    for n in NS:
        print(f"{n:3d} | {med(n,'hbar'):23.3f} | {med(n,'qshuf'):18.2f} {med(n,'qnull'):13.2f} {np.median(arr(n,'qshuf')/arr(n,'qnull')):13.2f} "
              f"{spearman(arr(n,'qshuf'), arr(n,'qnull')):9.2f} | {med(n,'q'):7.2f} {np.median(arr(n,'q')/arr(n,'qnull')):10.2f}")

    print("\n=== Q3. gain profile kappa_j = ||u_j^T V||^2/lam_j by key eigen-direction thirds (median over windows) ===")
    print(f"{'n':>3} | {'top third (large lam)':>21} {'middle':>8} {'bottom (small lam)':>19} | {'slope of log kappa on log lam':>29} | {'lam_max(M) = |Z|^2':>19}")
    for n in NS:
        print(f"{n:3d} | {med(n,'k_top'):21.2f} {med(n,'k_mid'):8.2f} {med(n,'k_bot'):19.2f} | {med(n,'slope'):29.2f} | {med(n,'gmax'):19.2f}")

    print("\n=== Q4. predicting q from key-side features (leave-one-sequence-out) ===")
    feats = ["lam_min", "trGinv", "reff", "lam1", "psi"]
    print(f"{'n':>3} | {'R2 (const)':>10} {'R2 (key feats)':>14} | single-feature Spearman with log q: " + " ".join(f"{f:>8}" for f in feats) +
          f" | {'q90 const: cover':>16} {'bound':>6} | {'q90 regr.: cover':>16} {'bound(med)':>10}")
    for n in NS:
        S = np.array([s for s, _ in W[n]])
        y = np.log(arr(n, "q"))
        X = np.column_stack([np.ones(len(y))] + [np.log(arr(n, f)) for f in feats])
        sse = sst = 0; cov_c = []; cov_r = []; bnd_r = []; qc = []
        for held in range(nseq):
            tr, te = S != held, S == held
            beta = np.linalg.lstsq(X[tr], y[tr], rcond=None)[0]
            res = y[tr] - X[tr] @ beta
            yhat = X[te] @ beta
            sse += ((y[te] - yhat) ** 2).sum(); sst += ((y[te] - y[tr].mean()) ** 2).sum()
            c90 = np.quantile(y[tr], .9); cov_c.append(y[te] <= c90); qc.append(np.exp(c90))
            r90 = np.quantile(res, .9); cov_r.append(y[te] <= yhat + r90); bnd_r.append(np.exp(yhat + r90))
        sp = [spearman(np.log(arr(n, f)), y) for f in feats]
        print(f"{n:3d} | {0.0:10.2f} {1 - sse / sst:14.2f} | {'':36}" + " ".join(f"{x:8.2f}" for x in sp) +
              f" | {np.mean(np.concatenate(cov_c)):16.2f} {np.mean(qc):6.2f} | {np.mean(np.concatenate(cov_r)):16.2f} {np.median(np.concatenate(bnd_r)):10.2f}")

    print("\n=== Q5. error from q ===")
    print(f"{'n':>3} | {'measured':>8} {'psi_rms':>8} {'psi*sqrt(q)':>11} {'ratio':>6} {'rel.err':>8} | {'|TK|op sqrt(q)':>14} {'sqrt(gmax) psi':>14} | {'(rms/psi)^2':>11} {'q':>6} {'median |g_eff^2/q-1|':>20}")
    for n in NS:
        pred = arr(n, "psi") * np.sqrt(arr(n, "q"))
        print(f"{n:3d} | {med(n,'rms'):8.3f} {med(n,'psi'):8.3f} {np.median(pred):11.3f} {np.median(pred/arr(n,'rms')):6.2f} {np.median(np.abs(pred/arr(n,'rms')-1)):8.2f} | "
              f"{np.median(arr(n,'Kop')*np.sqrt(arr(n,'q'))):14.2f} {np.median(np.sqrt(arr(n,'gmax'))*arr(n,'psi')):14.2f} | "
              f"{np.median((arr(n,'rms')/arr(n,'psi'))**2):11.2f} {med(n,'q'):6.2f} {np.median(np.abs((arr(n,'rms')/arr(n,'psi'))**2/arr(n,'q')-1)):20.2f}")
    print("\nPredictive (V-free) error bounds, LOSO:  RMS <= r90 * psi_rms  and  max_i F_i <= rmax90 * max_i psi_i  (r = 90th percentile of the ratio on other sequences)")
    print(f"{'n':>3} | {'measured RMS':>12} {'psi':>6} {'r90':>5} {'bound':>6} {'%<1':>5} {'coverage':>9} | {'meas max':>8} {'max psi':>8} {'rmax90':>7} {'bound':>6} {'%<1':>5} {'coverage':>9} | {'rms_hat=psi*sqrt(qmed): rel.err':>31} {'Spearman(psi, rms)':>19} | {'earlier predictive (3g)':>23}")
    earlier = {16: 1.46, 32: 1.88, 64: 2.32}
    for n in NS:
        S = np.array([s for s, _ in W[n]])
        rms, psi, Fm, pm, q = arr(n, "rms"), arr(n, "psi"), arr(n, "Fmax"), arr(n, "psi_max"), arr(n, "q")
        ratio, rmax = rms / psi, Fm / pm
        cov, cov2, b1, b2, relerr, r90s, rm90s = [], [], [], [], [], [], []
        for held in range(nseq):
            tr, te = S != held, S == held
            r90 = np.quantile(ratio[tr], .9); rm90 = np.quantile(rmax[tr], .9)
            r90s.append(r90); rm90s.append(rm90)
            cov.append(rms[te] <= r90 * psi[te]); b1.append(r90 * psi[te])
            cov2.append(Fm[te] <= rm90 * pm[te]); b2.append(rm90 * pm[te])
            relerr.append(np.abs(psi[te] * np.sqrt(np.median(q[tr])) / rms[te] - 1))
        b1, b2 = np.concatenate(b1), np.concatenate(b2)
        print(f"{n:3d} | {np.median(rms):12.3f} {np.median(psi):6.2f} {np.mean(r90s):5.2f} {np.median(b1):6.2f} {100*np.mean(b1<1):4.0f}% {np.mean(np.concatenate(cov)):9.2f} | "
              f"{np.median(Fm):8.3f} {np.median(pm):8.2f} {np.mean(rm90s):7.2f} {np.median(b2):6.2f} {100*np.mean(b2<1):4.0f}% {np.mean(np.concatenate(cov2)):9.2f} | "
              f"{np.median(np.concatenate(relerr)):31.2f} {spearman(psi, rms):19.2f} | {earlier[n]:23.2f}")
