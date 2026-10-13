import json, glob, numpy as np, sys
sys.path.insert(0, "."); from flatlib import window_metrics, reff, unit
runs = {}
for f in sorted(glob.glob("exp_mqar/runs/*.json")):
    r = json.load(open(f)); a = r["args"]; runs[(a["reg"], a["lam"], a["seed"])] = (f[:-5], r)
NS = sorted({k[2] for k in runs}); cfgs = [("none", 0.0, "baseline"), ("dec", 1.0, "key decorrelation"), ("match", 1.0, "Gram matching"), ("rank", 0.1, "rank matching")]
acc = lambda r, ns=(8, 16, 24, 32): float(np.mean([r["grid"][f"{n}_{L}"] for n in ns for L in (0, 32, 96)]))
def escape(r):                       # first logged step (multiple of 250) with CE < 2.0 ; None if never
    for st, ce, _ in r["hist"]:
        if st % 250 == 0 and ce < 2.0: return st
    return None
NSEEDS = len(NS)
print("per-seed mean recall accuracy (n=8..32, L=0/32/96) and plateau-escape step (first multiple of 250 with logged CE < 2)")
print(f"{'config':>18} | " + " ".join(f"s{s}".rjust(11) for s in NS) + " | mean   sd   failures(<0.8)")
tab = {}
for reg, lam, name in cfgs:
    vals = [acc(runs[(reg, lam, s)][1]) for s in NS if (reg, lam, s) in runs]; tab[name] = vals
    cells = []
    for s in NS:
        if (reg, lam, s) in runs:
            r = runs[(reg, lam, s)][1]; e = escape(r); cells.append(f"{acc(r):.2f}/{('-' if e is None else e):>4}")
        else: cells.append("      n/a")
    print(f"{name:>18} | " + " ".join(c.rjust(11) for c in cells) + f" | {np.mean(vals):.3f} {np.std(vals, ddof=1):.3f}   {sum(v < 0.8 for v in vals)}/{len(vals)}")
base = tab["baseline"]
print("\npaired comparisons with the baseline (same seed), mean accuracy over n = 8..32:")
for reg, lam, name in cfgs[1:]:
    seeds = [s for s in NS if (reg, lam, s) in runs and ('none', 0.0, s) in runs]
    d = np.array([acc(runs[(reg, lam, s)][1]) - acc(runs[("none", 0.0, s)][1]) for s in seeds])
    d32 = np.array([acc(runs[(reg, lam, s)][1], (32,)) - acc(runs[("none", 0.0, s)][1], (32,)) for s in seeds])
    t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan")
    # exact sign-flip permutation test for the mean difference
    from itertools import product
    perm = [np.mean(d * np.array(sg)) for sg in product([1, -1], repeat=len(d))]; pval = np.mean(np.abs(perm) >= abs(d.mean()) - 1e-12)
    print(f"  {name:>18}: mean diff {d.mean():+.3f} (n=32 only {d32.mean():+.3f}); wins {int((d > 0).sum())}/{len(d)}; t={t:+.2f}; exact sign-flip p={pval:.3f}")
# across ALL runs: does the geometry predict accuracy?
rows = []
for k, (f, r) in runs.items():
    c = np.load(f + "_cache.npz"); out = [acc(r)]
    for layer in (0, 1):
        K, V = c[f"k{layer}"], c[f"v{layer}"]
        out += [np.median([reff(K[b, :16, h]) for b in range(K.shape[0]) for h in range(K.shape[2])]),
                np.median([reff(unit(V[b, :16, h])) for b in range(K.shape[0]) for h in range(K.shape[2])])]
    rows.append((k[0],) + tuple(out))
A = np.array([r[1:] for r in rows], dtype=float); rk = lambda x: np.argsort(np.argsort(x)).astype(float); sp = lambda a, b: float(np.corrcoef(rk(a), rk(b))[0, 1])
print(f"\nacross all {len(rows)} runs (Spearman with mean accuracy): key rank layer0 {sp(A[:,0],A[:,1]):+.2f} | layer1 {sp(A[:,0],A[:,3]):+.2f} | value rank layer0 {sp(A[:,0],A[:,2]):+.2f} | layer1 {sp(A[:,0],A[:,4]):+.2f} | min key rank over layers {sp(A[:,0], np.minimum(A[:,1],A[:,3])):+.2f}")
