"""EXPLORATORY analyses (not pre-registered).  numpy only.  usage: python exploratory_task.py <model>"""
import sys, glob, numpy as np
name = sys.argv[1]; HERE = "./exp_task"
Z = [np.load(f) for f in sorted(glob.glob(f"{HERE}/out/{name}_shard*.npz"))]
meta = np.concatenate([z["meta"] for z in Z]); psi_span = np.concatenate([z["psi_span"] for z in Z]); psi_all = np.concatenate([z["psi_all"] for z in Z])
n, j, age, fc, cl = meta[:, 0], meta[:, 1], meta[:, 2], meta[:, 5], meta[:, 7]; N = len(n); rng = np.random.default_rng(1)
long_h = np.load(f"{HERE}/out/{name}_halflife_cal.npy") >= 256; ps = psi_span.mean(1)
P1 = ps[:, long_h].mean(1); P1b = ps.reshape(N, -1).mean(1); P_short = ps[:, ~long_h].mean(1); Pall_prefix = psi_all[:, long_h].mean(1)
shuf = P1.copy()
for v in set(n):
    idx = np.where(n == v)[0]; shuf[idx] = P1[rng.permutation(idx)]
rk = lambda x: np.argsort(np.argsort(x)) / len(x)
oh = np.stack([(n == v).astype(float) for v in sorted(set(n))[1:]], 1)
def partial(x, y, mask, extra=None):
    cols = [np.ones(mask.sum()), age[mask]] + ([oh[mask]] if mask.all() or len(set(n[mask])) > 1 else []) + ([extra[mask]] if extra is not None else [])
    X = np.column_stack(cols); rx = rk(x[mask]); ry = rk(y[mask])
    res = lambda a: a - X @ np.linalg.lstsq(X, a, rcond=None)[0]; a, b = res(rx), res(ry); r = np.corrcoef(a, b)[0, 1]
    bs = [np.corrcoef(a[i], b[i])[0, 1] for i in (rng.integers(0, len(a), len(a)) for _ in range(1000))]; return r, np.percentile(bs, [2.5, 97.5])
allm = np.ones(N, bool); cal = j % 2 == 0
print(f"{name}: partial rank correlation with candidate log-prob (controls: age, one-hot n); negative = higher psi -> worse recall")
for lab, x in (("P1 (long-memory heads)", P1), ("P1 shuffled within n (null)", shuf), ("P1b (all heads)", P1b), ("short-memory heads", P_short), ("P1 on whole-prefix mean psi", Pall_prefix)):
    for sl, m in (("test", ~cal), ("calibration", cal), ("all", allm)):
        r, ci = partial(x, cl, m); print(f"  {lab:30s} {sl:12s} r = {r:+.3f}  [{ci[0]:+.3f}, {ci[1]:+.3f}]")
print("\nper load (all prompts, controls: age):")
for v in sorted(set(n)):
    m = n == v; X = np.column_stack([np.ones(m.sum()), age[m]]); res = lambda a: a - X @ np.linalg.lstsq(X, a, rcond=None)[0]
    r = np.corrcoef(res(rk(P1[m])), res(rk(cl[m])))[0, 1]; rs = np.corrcoef(res(rk(shuf[m])), res(rk(cl[m])))[0, 1]; ra = np.corrcoef(rk(P1[m]), rk(age[m]))[0, 1]
    print(f"  n={int(v):2d}: partial r(P1, logp) {r:+.3f}   null (shuffled) {rs:+.3f}   corr(P1, age) {ra:+.2f}")
print("\nby layer block (long-memory heads), all prompts, controls: age + n:")
for lo, hi in ((0, 6), (6, 12), (12, 18), (18, 24)):
    hh = long_h[lo:hi]
    if not hh.any(): continue
    sub = np.array([ps[i, lo:hi][hh].mean() for i in range(N)]); r, ci = partial(sub, cl, allm); print(f"  layers {lo}-{hi-1} ({int(hh.sum())} long heads): r = {r:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]")
print("\ncollinearity: Spearman(P1, age) =", round(float(np.corrcoef(rk(P1), rk(age))[0, 1]), 3), " Spearman(P1, n) =", round(float(np.corrcoef(rk(P1), rk(n))[0, 1]), 3))

# --- exact control for age and load: within-cell (n, queried position) fixed effects; age is identical inside a cell by construction
q = meta[:, 4]; cell = (n * 100 + q).astype(int)
def within_cell(x, y):
    xs, ys = np.zeros(N), np.zeros(N); keep = np.zeros(N, bool)
    for c in np.unique(cell):
        idx = np.where(cell == c)[0]
        if len(idx) >= 4: xs[idx] = rk(x[idx]) - rk(x[idx]).mean(); ys[idx] = rk(y[idx]) - rk(y[idx]).mean(); keep[idx] = True
    r = np.corrcoef(xs[keep], ys[keep])[0, 1]; ids = np.where(keep)[0]; bs = []
    for _ in range(1000):
        s = rng.choice(ids, len(ids)); bs.append(np.corrcoef(xs[s], ys[s])[0, 1])
    return r, np.percentile(bs, [2.5, 97.5]), int(keep.sum()), len(np.unique(cell[keep]))
print("\nEXACT age/load control: within-(n, query position) cells, rank correlation of psi with candidate log-prob (negative = higher psi -> worse recall):")
for lab, x in (("P1 (long-memory heads)", P1), ("P1 shuffled within n (null)", shuf), ("P1b (all heads)", P1b), ("short-memory heads", P_short)):
    r, ci, k, nc = within_cell(x, cl); print(f"  {lab:30s} r = {r:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]  ({k} prompts in {nc} cells)")
print("  same, outcome = forced-choice correct:")
for lab, x in (("P1 (long-memory heads)", P1), ("P1b (all heads)", P1b)):
    r, ci, k, nc = within_cell(x, fc); print(f"  {lab:30s} r = {r:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]")
