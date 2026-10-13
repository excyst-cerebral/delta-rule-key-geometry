import numpy as np
M = {}
for name, f, NL in (("M1 (linear-moe-hub 340M, 24L)", "results_transfer.npz", 24), ("M2 (puigde 360M, 21L)", "results_transfer_model_gdn360m.npz", 21)):
    D = np.load(f); A = D["A"]
    r = dict(dom=A[:, 0].astype(int), seq=A[:, 1].astype(int), lay=A[:, 2].astype(int), head=A[:, 3].astype(int), cl=A[:, 4].astype(int),
             n=A[:, 5].astype(int), rms=A[:, 6], psi=A[:, 7], Fm=A[:, 8], pm=A[:, 9], NL=NL)
    r["ratio"], r["rmax"] = r["rms"] / r["psi"], r["Fm"] / r["pm"]
    r["depth"] = np.minimum((3 * r["lay"]) // NL, 2)            # relative-depth third
    M[name] = r
names = list(M); a, b = M[names[0]], M[names[1]]
doms = ["prose", "code", "docs", "legal", "notes"]; cname = ["long", "mid", "short"]; NS = [16, 32, 64]
q = lambda x, p=.9: float(np.quantile(x, p))
S = lambda m, **kw: np.logical_and.reduce([np.ones(len(m["n"]), bool)] + [m[k] == v for k, v in kw.items()])

print("=== A. does model 2 replicate the within-model findings? (long heads) ===")
print(f"{'':>4} | {'ratio median / p90 / p99 per domain (prose, code, docs, legal, notes)':^100}")
for n in NS:
    for nm, m in ((" M1", a), (" M2", b)):
        print(f"n={n:<2}{nm} | " + "   ".join(f"{np.median(m['ratio'][S(m,dom=d,cl=0,n=n)]):.2f}/{q(m['ratio'][S(m,dom=d,cl=0,n=n)]):.2f}/{q(m['ratio'][S(m,dom=d,cl=0,n=n)],.99):.2f}" for d in range(5)))
print("\nM2 cross-domain coverage (calibrate on row, test on column; long heads, nominal 0.90), n = 32:")
for d1 in range(5):
    r90 = q(b["ratio"][S(b, dom=d1, cl=0, n=32)])
    print(f"  {doms[d1]:>6} ({r90:.2f}) | " + " ".join(f"{np.mean(b['ratio'][S(b,dom=d2,cl=0,n=32)] <= r90):.2f}" for d2 in range(5)))
print("M2 leave-one-domain-out (long heads): " + "; ".join(
    f"n={n}: cov " + "/".join(f"{np.mean(b['ratio'][S(b,cl=0,n=n,dom=d)] <= q(b['ratio'][S(b,cl=0,n=n) & (b['dom']!=d)])):.2f}" for d in range(5)) +
    f", bound {np.median(q(b['ratio'][S(b,cl=0,n=n)])*b['psi'][S(b,cl=0,n=n)]):.2f} vs RMS {np.median(b['rms'][S(b,cl=0,n=n)]):.3f}" for n in NS))

print("\n=== B. class and depth dependence in M2 (long heads, all domains) ===")
for n in NS:
    lo, hi = S(b, cl=0, n=n) & (b["lay"] < 14), S(b, cl=0, n=n) & (b["lay"] >= 14)
    print(f"n={n}: r90 layers<14 = {q(b['ratio'][lo]):.2f}, layers>=14 = {q(b['ratio'][hi]):.2f} | coverage low->high {np.mean(b['ratio'][hi] <= q(b['ratio'][lo])):.2f}; "
          f"class r90 long/mid/short = " + "/".join(f"{q(b['ratio'][S(b,cl=c,n=n)]):.2f}" for c in range(3)) +
          f"; coverage of long-r90 on mid/short = " + "/".join(f"{np.mean(b['ratio'][S(b,cl=c,n=n)] <= q(b['ratio'][S(b,cl=0,n=n)])):.2f}" for c in (1, 2)))

print("\n=== C. CROSS-MODEL: r90 by class and relative-depth third (all domains) ===")
print(f"{'n':>3} {'class':>6} {'depth':>6} | {'M1 r90':>7} {'M2 r90':>7} | {'M1 psi med':>10} {'M2 psi med':>10} | {'cov M1->M2':>10} {'cov M2->M1':>10} | windows M1/M2")
for n in NS:
    for c in range(3):
        for d in range(3):
            ma, mb = S(a, cl=c, n=n, depth=d), S(b, cl=c, n=n, depth=d)
            if ma.sum() < 50 or mb.sum() < 50: continue
            ra, rb = q(a["ratio"][ma]), q(b["ratio"][mb])
            print(f"{n:3d} {cname[c]:>6} {['early','middle','late'][d]:>6} | {ra:7.2f} {rb:7.2f} | {np.median(a['psi'][ma]):10.2f} {np.median(b['psi'][mb]):10.2f} | {np.mean(b['ratio'][mb] <= ra):10.2f} {np.mean(a['ratio'][ma] <= rb):10.2f} | {ma.sum()}/{mb.sum()}")

print("\n=== D. CROSS-MODEL, long heads pooled over depth and domains: calibrate on one model, test on the other (nominal 0.90) ===")
print(f"{'n':>3} | {'M1 r90':>6} {'M2 r90':>6} | {'M1->M2: coverage':>16} {'bound':>6} {'M2 RMS':>7} {'%<1':>5} | {'M2->M1: coverage':>16} {'bound':>6} {'M1 RMS':>7} {'%<1':>5}")
for n in NS:
    ma, mb = S(a, cl=0, n=n), S(b, cl=0, n=n)
    ra, rb = q(a["ratio"][ma]), q(b["ratio"][mb])
    b12, b21 = ra * b["psi"][mb], rb * a["psi"][ma]
    print(f"{n:3d} | {ra:6.2f} {rb:6.2f} | {np.mean(b['ratio'][mb] <= ra):16.2f} {np.median(b12):6.2f} {np.median(b['rms'][mb]):7.3f} {100*np.mean(b12<1):4.0f}% | "
          f"{np.mean(a['ratio'][ma] <= rb):16.2f} {np.median(b21):6.2f} {np.median(a['rms'][ma]):7.3f} {100*np.mean(b21<1):4.0f}%")
print("\nsame, calibrating on ALL heads of one model (pooled), testing on all heads of the other:")
for n in NS:
    ma, mb = S(a, n=n), S(b, n=n)
    ra, rb = q(a["ratio"][ma]), q(b["ratio"][mb])
    print(f"n={n}: M1 r90 {ra:.2f}, M2 r90 {rb:.2f}; coverage M1->M2 {np.mean(b['ratio'][mb] <= ra):.2f}, M2->M1 {np.mean(a['ratio'][ma] <= rb):.2f}")

print("\n=== E. key-side level: psi_rms (self-association error) and measured RMS, long heads ===")
for n in NS:
    ma, mb = S(a, cl=0, n=n), S(b, cl=0, n=n)
    print(f"n={n}: psi M1 {np.median(a['psi'][ma]):.3f} / M2 {np.median(b['psi'][mb]):.3f};  RMS M1 {np.median(a['rms'][ma]):.3f} / M2 {np.median(b['rms'][mb]):.3f};  "
          f"Spearman(psi,rms) M1 {np.corrcoef(np.argsort(np.argsort(a['psi'][ma])), np.argsort(np.argsort(a['rms'][ma])))[0,1]:.2f} / M2 {np.corrcoef(np.argsort(np.argsort(b['psi'][mb])), np.argsort(np.argsort(b['rms'][mb])))[0,1]:.2f}")

print("\n=== F. per-(class, relative-depth) cell calibration across models: window-weighted coverage, aggregated ===")
print(f"{'n':>3} | {'depth':>12} | {'M1->M2 coverage':>15} {'M2->M1 coverage':>15} | {'mean |log(r90_M2/r90_M1)|':>26} | {'cells':>5}")
for n in NS:
    for label, dset in (("early", [0]), ("middle+late", [1, 2]), ("all", [0, 1, 2])):
        cov12 = w12 = cov21 = w21 = 0.0; dif = []; cells = 0
        for c in range(3):
            for d in dset:
                ma, mb = S(a, cl=c, n=n, depth=d), S(b, cl=c, n=n, depth=d)
                if ma.sum() < 50 or mb.sum() < 50: continue
                ra, rb = q(a["ratio"][ma]), q(b["ratio"][mb])
                cov12 += np.sum(b["ratio"][mb] <= ra); w12 += mb.sum(); cov21 += np.sum(a["ratio"][ma] <= rb); w21 += ma.sum()
                dif.append(abs(np.log(rb / ra))); cells += 1
        print(f"{n:3d} | {label:>12} | {cov12/w12:15.2f} {cov21/w21:15.2f} | {np.mean(dif):26.2f} | {cells:5d}")

print("\n=== G. calibrate on ALL heads pooled in the other model; coverage by class in the test model ===")
for n in NS:
    ra, rb = q(a["ratio"][S(a, n=n)]), q(b["ratio"][S(b, n=n)])
    print(f"n={n}: M1-pooled r90={ra:.2f} -> M2 coverage by class long/mid/short: " + "/".join(f"{np.mean(b['ratio'][S(b,cl=c,n=n)] <= ra):.2f}" for c in range(3)) +
          f" | M2-pooled r90={rb:.2f} -> M1: " + "/".join(f"{np.mean(a['ratio'][S(a,cl=c,n=n)] <= rb):.2f}" for c in range(3)))
