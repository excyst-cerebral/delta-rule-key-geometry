import pickle, sys, numpy as np
q = lambda x, p=.9: float(np.quantile(x, p))
def load(f):
    D = pickle.load(open(f, "rb")); R = D["rec"]
    cols = ["var", "dom", "layer", "head", "n", "q", "ratio", "rms", "psi", "gmax", "k_top", "k_bot", "lslope", "ovl", "chance", "botG", "botH", "c", "dGH", "rope", "rx", "rK", "rV", "rVrot", "rVg"]
    A = {c: np.array([r.get(c, np.nan) for r in R]) for c in cols if c != "var"}
    A["var"] = np.array([r["var"] for r in R])
    return D, A
name, f = sys.argv[1], sys.argv[2]
D, A = load(f); cfg = D["cfg"]; NL = cfg["NL"]; NS = cfg["NS"]; doms = D["doms"]
depth = np.minimum(3 * A["layer"] // NL, 2).astype(int)
print(f"##### {name}: {NL} layers, {cfg['Hkv']} KV heads of dim {cfg['dh']}; {int((A['var']=='own').sum()/len(NS))} windows per n (all KV heads, 5 domains x 3 sequences)")
med = lambda v, n, k: float(np.nanmedian(A[k][(A["var"] == v) & (A["n"] == n)]))
print("\n=== 1. flatness with different value branches (keys = trained pre-RoPE keys) ===")
print(f"{'n':>3} {'values':>6} | {'q':>5} {'RMS/psi':>7} {'lam_max(M)':>10} | {'kappa top':>9} {'kappa bottom':>12} {'log-slope':>9} | {'corr(G,H)':>9} {'||H-G||/||G-I||':>16} | {'top-8 overlap':>13} {'(chance)':>8}")
for n in NS:
    for v in ["own", "oh", "rot", "gauss", "shuf"]:
        print(f"{n:3d} {v:>6} | {med(v,n,'q'):5.2f} {med(v,n,'ratio'):7.2f} {med(v,n,'gmax'):10.2f} | {med(v,n,'k_top'):9.2f} {med(v,n,'k_bot'):12.2f} {med(v,n,'lslope'):9.2f} | {med(v,n,'c'):9.2f} {med(v,n,'dGH'):16.2f} | {med(v,n,'ovl'):13.2f} {med(v,n,'chance'):8.2f}")
print("\nshare of the shuffled->own improvement reproduced by random readouts (RMS/psi | log q):")
for n in NS:
    f_ = lambda k, v: (np.log(med('shuf',n,k)/med(v,n,k)) / np.log(med('shuf',n,k)/med('own',n,k))) if k == 'q' else (med('shuf',n,k)-med(v,n,k))/(med('shuf',n,k)-med('own',n,k))
    print(f"  n={n}: rot {f_('ratio','rot'):.2f} | {f_('q','rot'):.2f};   gauss {f_('ratio','gauss'):.2f} | {f_('q','gauss'):.2f};   other head {f_('ratio','oh'):.2f} | {f_('q','oh'):.2f}")
print("\n=== 2. effective ranks of the window Gram (median) ===")
for n in NS:
    m = (A["var"] == "own") & (A["n"] == n)
    print(f"  n={n}: x={np.nanmedian(A['rx'][m]):.1f}  keys={np.nanmedian(A['rK'][m]):.1f}  trained values={np.nanmedian(A['rV'][m]):.1f}  rotated values={np.nanmedian(A['rVrot'][m]):.1f}  Gaussian values={np.nanmedian(A['rVg'][m]):.1f}   (max {n})")
    print(f"        values' bottom-third trace share vs keys' (own / rot / gauss): " + " / ".join(f"{med(v,n,'botH'):.4f}" for v in ['own','rot','gauss']) + f" vs {med('own',n,'botG'):.4f}")
print("\n=== 3. calibrated ratio RMS/psi (own values): by domain, depth, and the pooled r90 ===")
for n in NS:
    m = (A["var"] == "own") & (A["n"] == n)
    r = A["ratio"][m]; d = A["dom"][m].astype(int); dp = depth[m]
    print(f"  n={n}: median {np.median(r):.2f}, p90 {q(r):.2f}, p99 {q(r,.99):.2f} | r90 by domain " + " ".join(f"{doms[k]}={q(r[d==k]):.2f}" for k in range(len(doms))) +
          " | by depth early/middle/late " + "/".join(f"{q(r[dp==k]):.2f}" for k in range(3)))
    cov = [np.mean(r[d == b] <= q(r[d != b])) for b in range(len(doms))]
    lowhigh = np.mean(r[dp == 2] <= q(r[dp == 0]))
    print(f"        leave-one-domain-out coverage " + " ".join(f"{c:.2f}" for c in cov) + f" | calibrate early -> test late: {lowhigh:.2f} | bound r90*psi (med) {np.median(q(r)*A['psi'][m]):.2f} vs RMS {np.median(A['rms'][m]):.3f}, % <1: {100*np.mean(q(r)*A['psi'][m]<1):.0f}%")
print("\n=== 4. sensitivity: post-RoPE keys, ratio RMS/psi (own values): median / p90 ===")
for n in NS:
    m = (A["var"] == "own") & (A["n"] == n)
    print(f"  n={n}: pre-RoPE {np.median(A['ratio'][m]):.2f} / {q(A['ratio'][m]):.2f}   post-RoPE {np.nanmedian(A['rope'][m]):.2f} / {q(A['rope'][m][np.isfinite(A['rope'][m])]):.2f}")
