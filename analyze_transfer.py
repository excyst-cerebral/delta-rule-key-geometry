import numpy as np
D = np.load("results_transfer.npz"); A = D["A"]; doms = list(D["domains"])
dom, seq, lay, head, cl, n, rms, psi, Fm, pm = A.T
dom, seq, lay, head, cl, n = [x.astype(int) for x in (dom, seq, lay, head, cl, n)]
ratio, rmax = rms / psi, Fm / pm
cname = ["long", "mid", "short"]
NS = [16, 32, 64]
q = lambda x, p=.9: float(np.quantile(x, p))
sel = lambda **kw: np.logical_and.reduce([np.ones(len(A), bool)] + [(globals()[k] == v) if not isinstance(v, (list, tuple)) else np.isin(globals()[k], v) for k, v in kw.items()])

print("=== 1. ratio RMS/psi_rms on LONG-memory heads, by domain (median / p90 / p99) and per-sequence spread of p90 ===")
print(f"{'domain':>7} | " + " | ".join(f"n={k:<2}: med   p90   p99  seq-p90 min..max" for k in NS))
for d, name in enumerate(doms):
    row = f"{name:>7} | "
    for k in NS:
        m = sel(dom=d, cl=0, n=k); r = ratio[m]
        sp = [q(ratio[m & (seq == s)]) for s in range(4)]
        row += f"      {np.median(r):.2f}  {q(r):.2f}  {q(r,.99):.2f}   {min(sp):.2f}..{max(sp):.2f}   | "
    print(row)
allm = {k: ratio[sel(cl=0, n=k)] for k in NS}
print(f"{'pooled':>7} | " + " | ".join(f"      {np.median(allm[k]):.2f}  {q(allm[k]):.2f}  {q(allm[k],.99):.2f}                | " for k in NS))

print("\n=== 2. cross-domain coverage: calibrate r90 on the ROW domain (long heads), test on the COLUMN domain; nominal 0.90 ===")
for k in NS:
    print(f"n={k}")
    print(f"{'calibrate on':>14} | " + " ".join(f"{d:>7}" for d in doms))
    for a, an in enumerate(doms):
        r90 = q(ratio[sel(dom=a, cl=0, n=k)])
        print(f"{an:>8} ({r90:.2f}) | " + " ".join(f"{np.mean(ratio[sel(dom=b, cl=0, n=k)] <= r90):7.2f}" for b in range(len(doms))))

print("\n=== 3. leave-one-DOMAIN-out (calibrate on the other four, test on the held-out domain), long heads ===")
print(f"{'n':>3} {'held-out':>8} | {'r90':>5} {'coverage':>8} | {'bound (med)':>11} {'measured RMS':>12} {'%bound<1':>9} | worst-item: {'rmax90':>6} {'coverage':>8} {'bound(med)':>10} {'%<1':>5}")
for k in NS:
    for d, name in enumerate(doms):
        tr, te = sel(cl=0, n=k) & (dom != d), sel(cl=0, n=k) & (dom == d)
        r90, m90 = q(ratio[tr]), q(rmax[tr])
        b = r90 * psi[te]; b2 = m90 * pm[te]
        print(f"{k:3d} {name:>8} | {r90:5.2f} {np.mean(ratio[te] <= r90):8.2f} | {np.median(b):11.2f} {np.median(rms[te]):12.3f} {100*np.mean(b<1):8.0f}% |             {m90:6.2f} {np.mean(rmax[te] <= m90):8.2f} {np.median(b2):10.2f} {100*np.mean(b2<1):4.0f}%")

print("\n=== 4. layer transfer (long heads, all domains): calibrate on layers 6-14 -> test 15-23, and the reverse ===")
for k in NS:
    lo, hi = sel(cl=0, n=k) & (lay < 15), sel(cl=0, n=k) & (lay >= 15)
    print(f"n={k}: r90(low layers)={q(ratio[lo]):.2f}  r90(high layers)={q(ratio[hi]):.2f} | coverage low->high {np.mean(ratio[hi] <= q(ratio[lo])):.2f}  high->low {np.mean(ratio[lo] <= q(ratio[hi])):.2f}  (n windows {lo.sum()}/{hi.sum()})")

print("\n=== 5. head-class transfer (all domains): ratio quantiles per class, and coverage when calibrated on LONG heads ===")
print(f"{'n':>3} {'class':>6} | {'med':>5} {'p90':>5} {'p99':>5} | {'cov. of long-r90':>16} | {'psi med':>7} {'RMS med':>7} {'bound=long-r90*psi (med)':>25} {'%<1':>5}")
for k in NS:
    r90L = q(ratio[sel(cl=0, n=k)])
    for c in range(3):
        m = sel(cl=c, n=k)
        b = r90L * psi[m]
        print(f"{k:3d} {cname[c]:>6} | {np.median(ratio[m]):5.2f} {q(ratio[m]):5.2f} {q(ratio[m],.99):5.2f} | {np.mean(ratio[m] <= r90L):16.2f} | {np.median(psi[m]):7.2f} {np.median(rms[m]):7.3f} {np.median(b):25.2f} {100*np.mean(b<1):4.0f}%")

print("\n=== 6. calibrating on ALL heads, pooled domains (a single global r90 per n) ===")
for k in NS:
    m = sel(n=k); r90 = q(ratio[m])
    print(f"n={k}: r90={r90:.2f}  coverage by class: " + "  ".join(f"{cname[c]} {np.mean(ratio[sel(cl=c, n=k)] <= r90):.2f}" for c in range(3)))

print("\n=== 7. PER-HEAD calibration, leave-one-domain-out: r90 of each (layer, head) estimated on the other four domains ===")
print(f"{'n':>3} {'class':>6} | {'coverage (med over heads)':>25} {'p10 over heads':>14} {'frac heads in [0.85,0.95]':>26} | {'bound med':>9} {'RMS med':>8} {'%<1':>5} | {'pooled-long r90 bound':>21}")
for k in NS:
    cov_all = {c: [] for c in range(3)}; bnd = {c: [] for c in range(3)}; rm = {c: [] for c in range(3)}
    for L in range(24):
        for H in range(4):
            c = int(cl[(lay == L) & (head == H)][0])
            for d in range(len(doms)):
                m = (lay == L) & (head == H) & (n == k)
                tr, te = m & (dom != d), m & (dom == d)
                r90 = q(ratio[tr])
                cov_all[c].append(np.mean(ratio[te] <= r90)); bnd[c].append(r90 * psi[te]); rm[c].append(rms[te])
    for c in range(3):
        cv = np.array(cov_all[c]); b = np.concatenate(bnd[c]); r = np.concatenate(rm[c])
        r90L = q(ratio[sel(cl=0, n=k)])
        print(f"{k:3d} {cname[c]:>6} | {np.mean(cv):25.2f} {np.quantile(cv,.1):14.2f} {np.mean((cv>=.85)&(cv<=.95)):26.2f} | {np.median(b):9.2f} {np.median(r):8.3f} {100*np.mean(b<1):4.0f}% | {r90L*np.median(psi[sel(cl=c,n=k)]):21.2f}")
