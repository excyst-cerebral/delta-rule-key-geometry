import json, glob, collections, sys, numpy as np
sys.path.insert(0, "."); from flatlib import window_metrics, reff, unit
R = collections.defaultdict(list)
for f in sorted(glob.glob("exp_mqar/runs/*.json")):
    r = json.load(open(f)); a = r["args"]; R[(a["reg"], a["lam"])].append((f[:-5], r))
ns = [8, 16, 24, 32]
def acc(r, n): return float(np.mean([r["grid"][f"{n}_{L}"] for L in (0, 32, 96)]))
print("recall accuracy (mean over distractor lengths 0/32/96), mean ± sd over seeds; paired difference vs baseline by seed")
print(f"{'reg':>6} {'lam':>5} | seeds | " + " | ".join(f"n={n:<2}".rjust(13) for n in ns) + " |   mean over n")
base = {r["args"]["seed"]: r for _, r in R.get(("none", 0.0), [])}
for k in sorted(R):
    v = R[k]; cells = []
    for n in ns:
        x = np.array([acc(r, n) for _, r in v]); cells.append(f"{x.mean():.3f}±{x.std(ddof=1) if len(x)>1 else 0:.3f}")
    mean_all = np.array([np.mean([acc(r, n) for n in ns]) for _, r in v])
    d = [np.mean([acc(r, n) for n in ns]) - np.mean([acc(base[r['args']['seed']], n) for n in ns]) for _, r in v if r["args"]["seed"] in base]
    extra = f" | diff {np.mean(d):+.3f} (wins {sum(x > 0 for x in d)}/{len(d)})" if k != ("none", 0.0) and d else ""
    print(f"{k[0]:>6} {k[1]:>5} | {len(v):5d} | " + " | ".join(c.rjust(13) for c in cells) + f" | {mean_all.mean():.3f}{extra}")
print("\ngeometry of the first 16 positions (eval batch n=16, L=32), per layer; medians over sequences and heads")
print(f"{'reg':>6} {'lam':>5} | layer | key r_eff | value r_eff | rank ratio | q | RMS/psi | mean beta | mean alpha")
for k in sorted(R):
    for layer in (0, 1):
        rows = []
        for f, r in R[k]:
            c = np.load(f + "_cache.npz"); K, V, be, al = c[f"k{layer}"], c[f"v{layer}"], c[f"beta{layer}"], c[f"alpha{layer}"]
            for b in range(K.shape[0]):
                for h in range(K.shape[2]):
                    m = window_metrics(K[b, :16, h], unit(V[b, :16, h]))
                    rows.append((reff(K[b, :16, h]), reff(unit(V[b, :16, h])), m["q"], m["ratio"], be[b, :, h].mean(), al[b, :, h].mean()))
        a = np.array(rows)
        print(f"{k[0]:>6} {k[1]:>5} | {layer:5d} | {np.median(a[:,0]):9.2f} | {np.median(a[:,1]):11.2f} | {np.median(a[:,1]/a[:,0]):10.2f} | {np.median(a[:,2]):5.2f} | {np.median(a[:,3]):7.2f} | {np.mean(a[:,4]):9.3f} | {np.mean(a[:,5]):10.3f}")
