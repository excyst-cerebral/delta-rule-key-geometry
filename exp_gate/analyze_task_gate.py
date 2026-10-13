"""Part B statistics of exp_gate/PROTOCOL.md: does the gate-aware key-only predictor P1g add anything for task-level recall?  numpy only.
usage: python analyze_task_gate.py   -> exp_gate/result_task_gate.json   (uses figgen/data.py for the pre-registered task data and flex_auc)"""
import sys, glob, json, numpy as np
HERE = "."; sys.path.insert(0, HERE + "/figgen")
import data as D


def load_gate(tag, d):
    Z = [np.load(f) for f in sorted(glob.glob(f"{HERE}/exp_gate/out/{tag}_task_shard*.npz"))]
    nj = np.concatenate([z["nj"] for z in Z]); pos = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(d["n"], d["j"]))}; order = np.array([pos[(int(a), int(b))] for a, b in nj])
    out = {}
    for k in ("Dg", "Rg", "logA"):
        x = np.concatenate([z[k] for z in Z]); y = np.zeros_like(x); y[order] = x; out[k] = y            # re-ordered to the row order of the pre-registered data
    assert len(order) == len(d["n"]) == len(set(order.tolist()))
    return out


def rk(x): o = np.argsort(np.argsort(x)); return o / len(x)


def analyse(nm, tag):
    d = D.TASK[nm]; return analyse_data(nm, d, load_gate(tag, d))


def analyse_data(nm, d, g, extra=None):
    """d: task data (n, j, age, q, fc, clp, cal, long_h; as D.TASK[...]); g: gate-aware arrays Dg, Rg, logA (N, 3, NL, H) in the same row order; extra: further predictors, appended last."""
    N = len(d["n"]); n, j, age, fc, clp, q = d["n"], d["j"], d["age"], d["fc"], d["clp"], d["q"]; cal = d["cal"]; tst = ~cal; long_h = d["long_h"]
    sm = lambda x: x.mean(1)                                                       # mean over the 3 span tokens -> (N, NL, H)
    feats = {"P1g": sm(g["Dg"])[:, long_h].mean(1), "P1gb": sm(g["Dg"]).reshape(N, -1).mean(1), "P1r": sm(g["Rg"])[:, long_h].mean(1), "P1A": -sm(g["logA"])[:, long_h].mean(1)}
    rng = np.random.default_rng(0); shuf = feats["P1g"].copy()
    for v in set(n):
        for msk in (cal, tst):
            idx = np.where((n == v) & msk)[0]; shuf[idx] = feats["P1g"][rng.permutation(idx)]
    feats["P1g_shuffled"] = shuf
    if extra: feats.update(extra)                                                   # appended last: does not change the random streams of the predictors above
    res = dict(model=nm, N=int(N), n_long_heads=int(long_h.sum()), within_load_auc={}, corr_with_age={})
    # (1) pre-registered style gain over age + one-hot load (B3); fit and bootstrap copied from exp_task/analyze_task.py
    onehot = np.stack([(n == v).astype(float) for v in sorted(set(n))[1:]], 1)
    def fit(X, y, lam=1e-3, it=60):
        mu, sd = X[cal][:, 1:].mean(0), X[cal][:, 1:].std(0) + 1e-12; Xs = X.copy(); Xs[:, 1:] = (X[:, 1:] - mu) / sd; b = np.zeros(X.shape[1]); Xc, yc = Xs[cal], y[cal]
        for _ in range(it):
            p = 1 / (1 + np.exp(-Xc @ b)); W = p * (1 - p) + 1e-9; R = lam * np.eye(len(b)); R[0, 0] = 0
            b = b + np.linalg.solve(Xc.T @ (Xc * W[:, None]) + R, Xc.T @ (yc - p) - R @ b)
        return 1 / (1 + np.exp(-Xs @ b))
    design = lambda cols: np.column_stack([np.ones(N)] + cols); ti = np.where(tst)[0]
    pb3 = fit(design([age, onehot]), fc); res["auc_B3"] = float(D.auc(fc[ti], pb3[ti])); brng = np.random.default_rng(0)
    def boot(fn, B=2000):
        out = [fn(np.concatenate([brng.choice(np.where(tst & (n == v))[0], size=(tst & (n == v)).sum()) for v in sorted(set(n))])) for _ in range(B)]
        return np.nanpercentile(out, [2.5, 97.5])
    res["pre_registered_style"] = {}
    for k, x in feats.items():
        p1 = fit(design([age, onehot, x]), fc); fn = lambda s: D.auc(fc[s], p1[s]) - D.auc(fc[s], pb3[s]); v = fn(ti); ci = boot(fn)
        res["pre_registered_style"][k] = dict(auc_plus=float(D.auc(fc[ti], p1[ti])), delta=float(v), ci=[float(ci[0]), float(ci[1])])
    # (2) flexible categorical-age baseline (a fixed effect for every (load, queried position) cell), ridge weight 1, paired bootstrap over test prompts
    res["flexible_age"] = {}
    for k, x in feats.items():
        dd = dict(d); dd["P1"] = x; a0, a1, dl, ci = D.flex_auc(dd, lam=1.0); res["flexible_age"][k] = dict(auc_base=float(a0), auc_plus=float(a1), delta=float(dl), ci=ci)
    # (3) exact within-(load, queried position) cells: rank correlation with the candidate log-probability and with forced-choice correctness
    cell = (n * 100 + q).astype(int)
    def within_cell(x, y):
        xs, ys = np.zeros(N), np.zeros(N); keep = np.zeros(N, bool)
        for c in np.unique(cell):
            idx = np.where(cell == c)[0]
            if len(idx) >= 4: xs[idx] = rk(x[idx]) - rk(x[idx]).mean(); ys[idx] = rk(y[idx]) - rk(y[idx]).mean(); keep[idx] = True
        r = np.corrcoef(xs[keep], ys[keep])[0, 1]; ids = np.where(keep)[0]; bs = []
        for _ in range(1000):
            s = rng.choice(ids, len(ids)); bs.append(np.corrcoef(xs[s], ys[s])[0, 1])
        return float(r), [float(v) for v in np.percentile(bs, [2.5, 97.5])], int(keep.sum()), int(len(np.unique(cell[keep])))
    res["within_cell_logp"] = {k: dict(zip(("r", "ci", "prompts", "cells"), within_cell(x, clp))) for k, x in feats.items()}
    res["within_cell_correct"] = {k: dict(zip(("r", "ci", "prompts", "cells"), within_cell(x, fc))) for k, x in feats.items()}
    # context: collinearity with age, and test AUC alone within each load
    for k, x in feats.items():
        res["corr_with_age"][k] = float(np.corrcoef(rk(x), rk(age))[0, 1])
        res["within_load_auc"][k] = {int(v): float(D.auc(fc[tst & (n == v)], x[tst & (n == v)])) for v in sorted(set(n))}
    return res


if __name__ == "__main__":
    out = {nm: analyse(nm, tag) for nm, tag in (("GDN-340M", "model_gdn340m"), ("GDN-360M", "model_gdn360m"))}
    json.dump(out, open(f"{HERE}/exp_gate/result_task_gate.json", "w"), indent=1)
    for nm, r in out.items():
        print(f"\n{nm}: {r['N']} prompts; {r['n_long_heads']} long-memory heads; AUC(B3) {r['auc_B3']:.3f}")
        for k in ("P1g", "P1r", "P1A", "P1g_shuffled", "P1gb"):
            a, b, c = r["pre_registered_style"][k], r["flexible_age"][k], r["within_cell_logp"][k]
            print(f"  {k:13s} dAUC(B3) {a['delta']:+.3f} [{a['ci'][0]:+.3f},{a['ci'][1]:+.3f}] | dAUC(flex age) {b['delta']:+.3f} [{b['ci'][0]:+.3f},{b['ci'][1]:+.3f}] | within-cell r {c['r']:+.3f} [{c['ci'][0]:+.3f},{c['ci'][1]:+.3f}] | corr(age) {r['corr_with_age'][k]:+.2f}")
