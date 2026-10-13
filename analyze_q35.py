import pickle, numpy as np
q = lambda x, p=.9: float(np.quantile(x, p))
D = pickle.load(open("qwen35_flat.pkl", "rb")); R = D["rec"]; doms = D["doms"]; L = D["L"]
cols = ["dom", "seq", "layer", "head", "n", "q", "ratio", "rms", "psi", "gmax", "k_top", "k_bot", "lslope", "ovl", "chance", "botG", "botH", "c", "dGH", "A64", "A256", "rx", "rK", "rV", "rVrot", "rVg"]
A = {c: np.array([r.get(c, np.nan) for r in R], dtype=float) for c in cols}; A["var"] = np.array([r["var"] for r in R])
print("validation of the numpy reconstruction (max abs diff, K / V):", max(v[0] for v in D["validation"].values()).round(5), max(v[1] for v in D["validation"].values()).round(6))
# head classes by the same decay rule, from prose and code
cls = {}
for l in L:
    for h in range(16):
        m = lambda d: (A["layer"] == l) & (A["head"] == h) & (A["dom"] == d) & (A["var"] == "own") & (A["n"] == 16)
        a, b = [(np.median(A["A64"][m(d)]), np.median(A["A256"][m(d)])) for d in (0, 1)]
        cls[(l, h)] = 0 if (a[1] > .5 and b[1] > .5) else (2 if (a[0] <= .1 and b[0] <= .1) else 1)
cl = np.array([cls[(int(l), int(h))] for l, h in zip(A["layer"], A["head"])]); cname = ["long", "mid", "short"]
print("head classes (288 delta-rule heads):", {cname[c]: sum(v == c for v in cls.values()) for c in range(3)})
depth = np.minimum(3 * np.searchsorted(L, A["layer"]) // len(L), 2).astype(int)
own = lambda n, **kw: (A["var"] == "own") & (A["n"] == n) & np.logical_and.reduce([np.ones(len(A["n"]), bool)] + [(cl == v) if k == "cl" else (A[k] == v) for k, v in kw.items()])
NS = [16, 32, 64]
print("\n=== 1. ratio RMS/psi_rms by head class (all domains), median / p90 / p99, and measured levels ===")
for n in NS:
    for c in range(3):
        m = own(n, cl=c); r = A["ratio"][m]
        print(f"n={n} {cname[c]:>5} ({int(m.sum())} windows): ratio {np.median(r):.2f} / {q(r):.2f} / {q(r,.99):.2f} | psi {np.median(A['psi'][m]):.3f}  RMS {np.median(A['rms'][m]):.3f}")
print("\n=== 2. value-branch controls (heads with controls: every 4th head), by class; long heads then all ===")
sub = lambda v, n, c=None: (A["var"] == v) & (A["n"] == n) & ((cl == c) if c is not None else True)
med = lambda v, n, k, c=None: float(np.nanmedian(A[k][sub(v, n, c)]))
for lab, c in (("LONG heads", 0), ("ALL heads", None)):
    print(f"-- {lab}")
    print(f"{'n':>3} {'values':>6} | {'q':>5} {'RMS/psi':>7} {'kappa bot':>9} {'logslope':>8} | {'corr(G,H)':>9} {'||H-G||/||G-I||':>15} | {'top-8 ovl':>9} {'(chance)':>8} | windows")
    for n in NS:
        for v in ["own", "oh", "rot", "gauss", "shuf"]:
            if sub(v, n, c).sum() == 0: continue
            print(f"{n:3d} {v:>6} | {med(v,n,'q',c):5.2f} {med(v,n,'ratio',c):7.2f} {med(v,n,'k_bot',c):9.2f} {med(v,n,'lslope',c):8.2f} | {med(v,n,'c',c):9.2f} {med(v,n,'dGH',c):15.2f} | {med(v,n,'ovl',c):9.2f} {med(v,n,'chance',c):8.2f} | {int(sub(v,n,c).sum())}")
    for n in NS:
        if sub('rot', n, c).sum() == 0: continue
        f1 = lambda v: (med('shuf',n,'ratio',c)-med(v,n,'ratio',c))/(med('shuf',n,'ratio',c)-med('own',n,'ratio',c))
        f2 = lambda v: np.log(med('shuf',n,'q',c)/med(v,n,'q',c))/np.log(med('shuf',n,'q',c)/med('own',n,'q',c))
        print(f"   n={n}: share of shuffled->own improvement reproduced by rotated / Gaussian / other head (RMS/psi | log q): {f1('rot'):.2f}|{f2('rot'):.2f}  {f1('gauss'):.2f}|{f2('gauss'):.2f}  {f1('oh'):.2f}|{f2('oh'):.2f}")
print("\n=== 3. effective ranks (heads with controls), median ===")
for n in NS:
    for lab, c in (("long", 0), ("all", None)):
        m = sub("own", n, c)
        print(f"n={n} {lab:>4}: x={np.nanmedian(A['rx'][m]):.1f} keys={np.nanmedian(A['rK'][m]):.1f} trained values={np.nanmedian(A['rV'][m]):.1f} rotated={np.nanmedian(A['rVrot'][m]):.1f} gaussian={np.nanmedian(A['rVg'][m]):.1f}   rank ratio values/keys = {np.nanmedian(A['rV'][m]/A['rK'][m]):.2f} (rotated: {np.nanmedian(A['rVrot'][m]/A['rK'][m]):.2f})")
print("\n=== 4. calibrated ratio: domain, depth, class (own values, all heads of the class) ===")
for n in NS:
    for c in (0, None):
        m = own(n, cl=c) if c is not None else own(n)
        r = A["ratio"][m]; d = A["dom"][m].astype(int); dp = depth[m]
        cov = [np.mean(r[d == b] <= q(r[d != b])) for b in range(5)]
        print(f"n={n} {('long' if c==0 else 'all'):>4}: r90 by domain " + " ".join(f"{doms[k]}={q(r[d==k]):.2f}" for k in range(5)) + " | LODO coverage " + " ".join(f"{x:.2f}" for x in cov) +
              " | r90 by depth early/mid/late " + "/".join(f"{q(r[dp==k]):.2f}" for k in range(3)) + f" | bound r90*psi median {np.median(q(r)*A['psi'][m]):.2f} vs RMS {np.median(A['rms'][m]):.3f}, %<1 {100*np.mean(q(r)*A['psi'][m]<1):.0f}%")
print("\n=== 5. cross-model calibration: pooled-over-heads r90 (all heads) ===")
g = {}
for name, f in (("GDN-1", "results_transfer.npz"), ("GDN-2", "results_transfer_model_gdn360m.npz")):
    a = np.load(f)["A"]; nn = a[:, 5].astype(int); g[name] = {k: a[nn == k, 6] / a[nn == k, 7] for k in NS}
g["Qwen3.5"] = {k: A["ratio"][own(k)] for k in NS}
for n in NS:
    print(f"n={n}: r90 " + "  ".join(f"{k}={q(v[n]):.2f}" for k, v in g.items()) + " | coverage of GDN-1/GDN-2 mean r90 on Qwen3.5: " +
          f"{np.mean(g['Qwen3.5'][n] <= np.mean([q(g['GDN-1'][n]), q(g['GDN-2'][n])])):.2f}; of Qwen3.5 r90 on GDN-1/GDN-2: {np.mean(g['GDN-1'][n] <= q(g['Qwen3.5'][n])):.2f}/{np.mean(g['GDN-2'][n] <= q(g['Qwen3.5'][n])):.2f}")
