"""Statistics of exp_scale/PROTOCOL.md.  numpy only.  usage: python -I exp_scale/analyze_scale.py [model_gdn13b]  -> exp_scale/result_scale_<model>.json"""
import sys, json, numpy as np
HERE = "."; sys.path.insert(0, HERE + "/figgen"); sys.path.insert(0, HERE + "/exp_gate")
import analyze_task_gate as A

name = sys.argv[1] if len(sys.argv) > 1 else "model_gdn13b"
z = np.load(f"{HERE}/exp_scale/out/{name}_scale.npz"); m = z["meta"]; N = len(m)
n, j, age, qpos, q, fc, rank, clp = (m[:, i] for i in (0, 1, 2, 3, 4, 5, 6, 7)); cal = (j % 2 == 0)
# long-memory heads: median per-token half-life over the calibration prompts with j even and j < 20 (as exp_task/halflife_cal.py)
sel = np.where((j % 2 == 0) & (j < 20))[0]; hl = z["hl"][sel]                                   # (k, NL, H, T)
med = np.array([[np.nanmedian(hl[:, li, h]) for h in range(hl.shape[2])] for li in range(hl.shape[1])]); long_h = med >= 256; fallback = False
if long_h.sum() < 8:                                                                              # fallback fixed in the protocol: the 20% of heads with the longest half-life
    thr = np.quantile(med, 0.8); long_h = med >= thr; fallback = True
ps = z["psi_span"].mean(1); P1 = ps[:, long_h].mean(1); P1b = ps.reshape(N, -1).mean(1)
rng = np.random.default_rng(0); P1s = P1.copy()
for v in set(n):
    for msk in (cal, ~cal):
        idx = np.where((n == v) & msk)[0]; P1s[idx] = P1[rng.permutation(idx)]
d = dict(n=n, j=j, age=age, qpos=qpos, q=q, fc=fc, rank=rank, clp=clp, cal=cal, long_h=long_h, P1=P1)
g = dict(Dg=z["Dg"], Rg=z["Rg"], logA=z["logA"])
res = A.analyse_data(name, d, g, extra={"P1": P1, "P1b": P1b, "P1_shuffled": P1s})
res["accuracy_by_load"] = {int(v): dict(acc=float(fc[n == v].mean()), chance=1 / v, n_prompts=int((n == v).sum())) for v in sorted(set(n))}
res["accuracy_overall"] = float(fc.mean()); res["halflife_median_range"] = [float(med.min()), float(med.max())]; res["long_head_fallback_used"] = bool(fallback)
res["n_long_heads"] = int(long_h.sum()); res["n_heads"] = int(long_h.size)
# verdict rule of the protocol
fx = {k: res["flexible_age"][k] for k in ("P1", "P1g")}; wc = {k: res["within_cell_logp"][k] for k in ("P1", "P1g")}
signal = any(fx[k]["delta"] >= 0.03 and fx[k]["ci"][0] > 0 for k in fx)
null = (not any(fx[k]["delta"] >= 0.03 for k in fx)) and all(fx[k]["ci"][0] <= 0 <= fx[k]["ci"][1] for k in fx) and all(wc[k]["ci"][0] <= 0 <= wc[k]["ci"][1] for k in wc)
res["verdict"] = "shows a signal" if signal else "confirms the null" if null else "mixed"
json.dump(res, open(f"{HERE}/exp_scale/result_scale_{name}.json", "w"), indent=1)
print(f"{name}: {N} prompts | accuracy {res['accuracy_overall']:.3f} | " + "  ".join(f"n={k}: {v['acc']:.2f} (chance {v['chance']:.2f})" for k, v in res["accuracy_by_load"].items()))
print(f"long-memory heads: {res['n_long_heads']} of {res['n_heads']} (fallback used: {fallback}); half-life range {res['halflife_median_range'][0]:.1f}..{res['halflife_median_range'][1]:.1f}; AUC(B3) {res['auc_B3']:.3f}")
for k in ("P1", "P1g", "P1r", "P1A", "P1_shuffled", "P1g_shuffled", "P1b", "P1gb"):
    a, b, c = res["pre_registered_style"][k], res["flexible_age"][k], res["within_cell_logp"][k]
    print(f"  {k:13s} dAUC(B3) {a['delta']:+.3f} [{a['ci'][0]:+.3f},{a['ci'][1]:+.3f}] | dAUC(flex age) {b['delta']:+.3f} [{b['ci'][0]:+.3f},{b['ci'][1]:+.3f}] | within-cell r {c['r']:+.3f} [{c['ci'][0]:+.3f},{c['ci'][1]:+.3f}] | corr(age) {res['corr_with_age'][k]:+.2f}")
print("VERDICT (protocol rule):", res["verdict"])
