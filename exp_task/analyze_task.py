"""Pre-registered analysis (exp_task/PREREGISTRATION.md).  numpy only.  usage: python analyze_task.py <model_dir_name>"""
import sys, glob, json, numpy as np
name = sys.argv[1]; HERE = "./exp_task"
Z = [np.load(f) for f in sorted(glob.glob(f"{HERE}/out/{name}_shard*.npz"))]
meta = np.concatenate([z["meta"] for z in Z]); psi_span = np.concatenate([z["psi_span"] for z in Z]); mlog = np.concatenate([z["mlog"] for z in Z])
n, j, age, fc, cl = meta[:, 0], meta[:, 1], meta[:, 2], meta[:, 5], meta[:, 7]
N = len(meta); cal = (j % 2 == 0); tst = ~cal; rng = np.random.default_rng(0)
print(f"{name}: {N} prompts; calibration {cal.sum()}, test {tst.sum()}; overall forced-choice acc {fc.mean():.3f}")
for v in sorted(set(n)): m = n == v; print(f"  n={int(v):2d}: acc {fc[m].mean():.3f} (chance {1/v:.3f})  mean age {age[m].mean():.0f}")
# head groups: median per-token half-life of the decay gate on calibration prompts only (exp_task/halflife_cal.py; deviation D1)
half = np.load(f"{HERE}/out/{name}_halflife_cal.npy"); long_h = half >= 256
print(f"  long-memory heads (median half-life >= 256): {long_h.sum()} of {long_h.size} ({long_h.mean():.0%})")
ps = psi_span.mean(1)                                      # mean over the 3 span tokens -> (N, 24, H)
P1 = ps[:, long_h].mean(1); P1b = ps.reshape(N, -1).mean(1)
# shuffled control: P1 of another prompt with the same n in the same split
P1s = P1.copy()
for v in set(n):
    for msk in (cal, tst):
        idx = np.where((n == v) & msk)[0]; P1s[idx] = P1[rng.permutation(idx)]
onehot = np.stack([(n == v).astype(float) for v in sorted(set(n))[1:]], 1)
def design(cols): return np.column_stack([np.ones(N)] + cols)
FEATS = {"B1 age": [age], "B2 n": [n], "B3 age+onehot(n)": [age, onehot], "P1": [P1], "P1b": [P1b], "P1 shuffled": [P1s], "B3+P1": [age, onehot, P1], "B3+P1b": [age, onehot, P1b], "B3+P1 shuffled": [age, onehot, P1s]}
def fit(X, y, lam=1e-3, it=60):
    mu, sd = X[cal][:, 1:].mean(0), X[cal][:, 1:].std(0) + 1e-12; Xs = X.copy(); Xs[:, 1:] = (X[:, 1:] - mu) / sd; b = np.zeros(X.shape[1]); Xc, yc = Xs[cal], y[cal]
    for _ in range(it):
        p = 1 / (1 + np.exp(-Xc @ b)); W = p * (1 - p) + 1e-9; R = lam * np.eye(len(b)); R[0, 0] = 0
        b = b + np.linalg.solve(Xc.T @ (Xc * W[:, None]) + R, Xc.T @ (yc - p) - R @ b)
    return 1 / (1 + np.exp(-Xs @ b))
def auc(y, s):
    o = np.argsort(s, kind="mergesort"); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    for u in np.unique(s):                                    # average ties
        m = s == u
        if m.sum() > 1: r[m] = r[m].mean()
    n1 = y.sum(); n0 = len(y) - n1; return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else np.nan
pred = {k: fit(design(c), fc) for k, c in FEATS.items()}
ti = np.where(tst)[0]
def boot(fn, B=2000):
    out = []
    for _ in range(B):
        samp = np.concatenate([rng.choice(np.where(tst & (n == v))[0], size=(tst & (n == v)).sum()) for v in sorted(set(n))]); out.append(fn(samp))
    return np.nanpercentile(out, [2.5, 97.5])
print("\nTEST-set AUC (pooled over loads):")
for k in pred: print(f"  {k:22s} {auc(fc[ti], pred[k][ti]):.3f}")
for a, b in (("B3+P1", "B3+P1 shuffled"),):
    pass
d1 = lambda s: auc(fc[s], pred["B3+P1"][s]) - auc(fc[s], pred["B1 age"][s]); d2 = lambda s: auc(fc[s], pred["B3+P1"][s]) - auc(fc[s], pred["B3+P1 shuffled"][s]); d3 = lambda s: auc(fc[s], pred["B3+P1"][s]) - auc(fc[s], pred["B3+age+onehot(n)".replace("B3+age+onehot(n)", "B3 age+onehot(n)")][s])
res = {}
for lab, fn in (("H1: AUC(B3+P1) - AUC(B3)", d3), ("AUC(B3+P1) - AUC(B3+shuffled P1)", d2)):
    v = fn(ti); ci = boot(fn); res[lab] = (v, ci.tolist()); print(f"  {lab}: {v:+.3f}  95% bootstrap CI [{ci[0]:+.3f}, {ci[1]:+.3f}]")
H1 = res["H1: AUC(B3+P1) - AUC(B3)"]; ok = H1[0] >= 0.03 and H1[1][0] > 0
print("  H1 criterion (delta >= 0.03 and CI excludes 0):", "MET" if ok else "NOT MET")
print("\nWithin each load (test prompts): AUC of P1 alone / shuffled P1 / age")
for v in sorted(set(n)):
    s = np.where(tst & (n == v))[0]; print(f"  n={int(v):2d}: P1 {auc(fc[s], P1[s]):.3f}  shuffled {auc(fc[s], P1s[s]):.3f}  age {auc(fc[s], age[s]):.3f}  (n_test={len(s)}, successes={int(fc[s].sum())})")
# decision-style: lowest vs highest P1 tercile within load, test set
diffs = []; 
for v in sorted(set(n)):
    s = np.where(tst & (n == v))[0]; q1, q2 = np.quantile(P1[s], [1 / 3, 2 / 3]); lo, hi = s[P1[s] <= q1], s[P1[s] >= q2]; diffs.append((len(s), fc[lo].mean() - fc[hi].mean()))
w = np.array([d[0] for d in diffs]); dec = float((w * np.array([d[1] for d in diffs])).sum() / w.sum()); print(f"\nDecision test: recall(lowest-psi tercile) - recall(highest-psi tercile), weighted over loads: {dec:+.3f}  (per load: {[round(d[1],3) for d in diffs]})")
# secondary: continuous outcome, rank correlation of P1 with candidate log-prob after removing n and age (rank residuals), test set
def rk(x): o = np.argsort(np.argsort(x)); return o / len(x)
def resid(y, X): b = np.linalg.lstsq(X, y, rcond=None)[0]; return y - X @ b
Xc = np.column_stack([np.ones(N), age, onehot]); ry, rp = resid(rk(cl[ti]), Xc[ti]), resid(rk(P1[ti]), Xc[ti]); pr = float(np.corrcoef(ry, rp)[0, 1])
bs = [float(np.corrcoef(*(lambda s: (ry[s], rp[s]))(rng.integers(0, len(ti), len(ti))))[0, 1]) for _ in range(2000)]; print(f"Secondary: partial rank correlation of P1 with candidate log-prob (controls: n, age): {pr:+.3f}  95% CI [{np.percentile(bs,2.5):+.3f}, {np.percentile(bs,97.5):+.3f}]")
print("\nExploratory: P1 by layer block (mean over long-memory heads in the block), test-set AUC alone:")
for lo, hi in ((0, 6), (6, 12), (12, 18), (18, 24)):
    f = ps[:, lo:hi][:, long_h[lo:hi].any(1)] if False else None
    sub = np.zeros(N)
    for ii in range(N):
        a = ps[ii, lo:hi][long_h[lo:hi]]; sub[ii] = a.mean() if a.size else np.nan
    print(f"  layers {lo}-{hi-1}: long heads {long_h[lo:hi].sum()}  AUC {auc(fc[ti], sub[ti]) if np.isfinite(sub[ti]).all() else float('nan'):.3f}")
json.dump(dict(model=name, N=int(N), H1=H1, ctrl=res["AUC(B3+P1) - AUC(B3+shuffled P1)"], H1_met=bool(ok), decision=dec, partial_rank_corr=pr), open(f"{HERE}/result_{name}.json", "w"), indent=1)
