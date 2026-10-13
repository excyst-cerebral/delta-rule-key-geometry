import pickle, numpy as np
q = lambda x, p=.9: float(np.quantile(x, p))
D = pickle.load(open("rwkv7_flat.pkl", "rb")); R = D["rec"]; doms = D["doms"]; ST = D["stat"]
cols = ["dom", "seq", "layer", "head", "n", "q", "ratio", "rms", "psi", "gmax", "k_top", "k_bot", "lslope", "ovl", "chance", "botG", "botH", "c", "dGH", "rx", "rK", "rV", "rVrot", "rVg", "rKw"]
A = {c: np.array([r.get(c, np.nan) for r in R], dtype=float) for c in cols}
A["var"] = np.array([r["var"] for r in R]); A["key"] = np.array([r["key"] for r in R])
NS = [16, 32]

print("=== 0. RWKV-7 layer characteristics (per head; median over heads and sequences) ===")
for l in (0, 3, 6, 9, 11):
    s = ST[l]; f = lambda k: np.median(np.concatenate([x[k] for x in s]))
    print(f"  layer {l:2d}: in-context learning rate a = {f('a'):.2f} | mean per-channel retention over 16 / 64 tokens = {f('A16'):.3f} / {f('A64'):.3f} | fraction of channels still > 0.5 after 64 tokens = {f('frac_slow'):.3f}")
allA64 = np.concatenate([np.concatenate([x['A64'] for x in ST[l]]) for l in ST]); allfs = np.concatenate([np.concatenate([x['frac_slow'] for x in ST[l]]) for l in ST])
print(f"  all heads: mean-channel retention@64 quantiles (10/50/90%): {q(allA64,.1):.3f} / {q(allA64,.5):.3f} / {q(allA64,.9):.3f};  slow-channel fraction: {q(allfs,.1):.3f} / {q(allfs,.5):.3f} / {q(allfs,.9):.3f}")
# head classes by slow-channel fraction (the decay is per channel, not per head)
fs = {(l, h): np.median([x['frac_slow'][h] for x in ST[l]]) for l in ST for h in range(12)}
cls = {k: (0 if v > 0.25 else (2 if v <= 0.02 else 1)) for k, v in fs.items()}
cl = np.array([cls[(int(l), int(h))] for l, h in zip(A["layer"], A["head"])]); cname = ["slow-channel-rich", "mid", "fast"]
print("  head classes by fraction of slow channels (>0.25 / 0.02-0.25 / <=0.02):", {cname[c]: sum(v == c for v in cls.values()) for c in range(3)}, "of 144")
own = lambda n, key="kappa", **kw: (A["var"] == "own") & (A["key"] == key) & (A["n"] == n) & np.logical_and.reduce([np.ones(len(A["n"]), bool)] + [(cl == v) if k == "cl" else (A[k] == v) for k, v in kw.items()])

print("\n=== 1. levels: ideal pass on RWKV-7 keys (removal key kappa vs normalised write key), own values ===")
for n in NS:
    for key in ("kappa", "write"):
        m = own(n, key); r = A["ratio"][m]
        print(f"n={n} {key:>6} key: RMS/psi median {np.median(r):.2f}, p90 {q(r):.2f}, p99 {q(r,.99):.2f} | psi {np.median(A['psi'][m]):.3f}, RMS {np.median(A['rms'][m]):.3f} | q median {np.median(A['q'][m]):.2f} | windows {int(m.sum())}")

print("\n=== 2. value-branch controls (kappa keys; every 3rd head), all heads ===")
sub = lambda v, n, key="kappa": (A["var"] == v) & (A["n"] == n) & (A["key"] == key)
med = lambda v, n, k, key="kappa": float(np.nanmedian(A[k][sub(v, n, key)]))
print(f"{'n':>3} {'values':>6} | {'q':>6} {'RMS/psi':>7} {'kappa top':>9} {'kappa bot':>9} {'logslope':>8} | {'corr(G,H)':>9} {'||H-G||/||G-I||':>15} | {'top-8 ovl':>9} {'chance':>6} | windows")
for n in NS:
    for v in ["own", "oh", "rot", "gauss", "shuf"]:
        print(f"{n:3d} {v:>6} | {med(v,n,'q'):6.2f} {med(v,n,'ratio'):7.2f} {med(v,n,'k_top'):9.2f} {med(v,n,'k_bot'):9.2f} {med(v,n,'lslope'):8.2f} | {med(v,n,'c'):9.2f} {med(v,n,'dGH'):15.2f} | {med(v,n,'ovl'):9.2f} {med(v,n,'chance'):6.2f} | {int(sub(v,n).sum())}")
    for key in ("write",):
        for v in ("own", "shuf"):
            print(f"{n:3d} {v:>6} | (write key) q {med(v,n,'q',key):6.2f}  RMS/psi {med(v,n,'ratio',key):5.2f}  kappa bot {med(v,n,'k_bot',key):6.2f}  top-8 ovl {med(v,n,'ovl',key):.2f}")
    f1 = lambda v: (med('shuf',n,'ratio')-med(v,n,'ratio'))/(med('shuf',n,'ratio')-med('own',n,'ratio'))
    f2 = lambda v: np.log(med('shuf',n,'q')/med(v,n,'q'))/np.log(med('shuf',n,'q')/med('own',n,'q'))
    print(f"   n={n}: share of shuffled->own improvement reproduced by rotated / Gaussian / other head (RMS/psi | log q): {f1('rot'):.2f}|{f2('rot'):.2f}  {f1('gauss'):.2f}|{f2('gauss'):.2f}  {f1('oh'):.2f}|{f2('oh'):.2f}")

print("\n=== 3. effective ranks (kappa keys; heads with controls), median ===")
for n in NS:
    m = own(n)
    m = m & np.isfinite(A["rV"])
    print(f"n={n}: x={np.nanmedian(A['rx'][m]):.1f} removal keys={np.nanmedian(A['rK'][m]):.1f} write keys={np.nanmedian(A['rKw'][m]):.1f} trained values={np.nanmedian(A['rV'][m]):.1f} rotated={np.nanmedian(A['rVrot'][m]):.1f} gaussian={np.nanmedian(A['rVg'][m]):.1f} | rank ratio values/removal keys {np.nanmedian(A['rV'][m]/A['rK'][m]):.2f} (rotated {np.nanmedian(A['rVrot'][m]/A['rK'][m]):.2f})")

print("\n=== 4. calibrated ratio (kappa keys, own values): domain, depth, class ===")
for n in NS:
    m = own(n); r = A["ratio"][m]; d = A["dom"][m].astype(int); L = A["layer"][m]; c = cl[m]
    cov = [np.mean(r[d == b] <= q(r[d != b])) for b in range(5)]
    dep = [q(r[(L >= lo) & (L < hi)]) for lo, hi in ((0, 4), (4, 8), (8, 12))]
    print(f"n={n}: r90 by domain " + " ".join(f"{doms[k]}={q(r[d==k]):.2f}" for k in range(5)) + " | LODO coverage " + " ".join(f"{x:.2f}" for x in cov) +
          " | r90 by depth (layers 0-3/4-7/8-11) " + "/".join(f"{x:.2f}" for x in dep) + " | by class " + "/".join(f"{q(r[c==k]):.2f}" for k in range(3)))
    print(f"       pooled r90 {q(r):.2f}; bound r90*psi median {np.median(q(r)*A['psi'][m]):.2f} vs RMS {np.median(A['rms'][m]):.3f}, % windows < 1: {100*np.mean(q(r)*A['psi'][m]<1):.0f}%")

print("\n=== 5. cross-model: pooled-over-heads r90 and coverage of the other models' r90 ===")
g = {}
for name, f in (("GDN-1", "results_transfer.npz"), ("GDN-2", "results_transfer_model_gdn360m.npz")):
    a = np.load(f)["A"]; nn = a[:, 5].astype(int); g[name] = {k: a[nn == k, 6] / a[nn == k, 7] for k in (16, 32)}
a = pickle.load(open("qwen35_flat.pkl", "rb"))["rec"]
g["Qwen3.5"] = {k: np.array([r["ratio"] for r in a if r["var"] == "own" and r["n"] == k]) for k in (16, 32)}
for nm, f in (("SmolLM2", "tf_flat_SmolLM2-135M-Instruct.pkl"), ("Qwen2.5", "tf_flat_Qwen2.5-1.5B-Instruct.pkl")):
    rr = pickle.load(open(f, "rb"))["rec"]; g[nm] = {k: np.array([r["ratio"] for r in rr if r["var"] == "own" and r["n"] == k]) for k in (16, 32)}
for n in NS:
    rw = A["ratio"][own(n)]
    print(f"n={n}: pooled r90: RWKV-7 {q(rw):.2f} | " + "  ".join(f"{k}={q(v[n]):.2f}" for k, v in g.items()))
    print("      coverage of each other model's r90 on RWKV-7 windows (nominal 0.90): " + "  ".join(f"{k}: {np.mean(rw <= q(v[n])):.2f}" for k, v in g.items()))
    print("      coverage of RWKV-7's r90 on each other model's windows:             " + "  ".join(f"{k}: {np.mean(v[n] <= q(rw)):.2f}" for k, v in g.items()))
