import json, glob, collections, numpy as np
R = collections.defaultdict(list)
for f in sorted(glob.glob("exp_cl/runs/*.json")):
    r = json.load(open(f)); a = r["args"]; R[(a["arch"], a["method"], a["reg"], a["lam"])].append(r)
def ms(x): x = np.array(x); return f"{x.mean():.3f} ± {x.std(ddof=1) if len(x) > 1 else 0:.3f}"
print(f"{'arch':>4} {'method':>7} {'reg':>7} {'lam':>6} | {'seeds':>5} | {'final avg acc':>15} | {'forgetting':>15}")
for k in sorted(R):
    v = R[k]; print(f"{k[0]:>4} {k[1]:>7} {k[2]:>7} {k[3]:>6} | {len(v):5d} | {ms([r['final_avg'] for r in v]):>15} | {ms([r['forgetting'] for r in v]):>15}")
# paired comparison with fine-tuning, same arch & seed
print("\npaired (same seed) change in forgetting vs plain fine-tuning of the same architecture; negative = less forgetting")
for k in sorted(R):
    if k[1] == "joint" or (k[1] == "ft" and k[2] == "none"): continue
    base = {r["args"]["seed"]: r["forgetting"] for r in R.get((k[0], "ft", "none", 0.0), [])}
    d = [r["forgetting"] - base[r["args"]["seed"]] for r in R[k] if r["args"]["seed"] in base]
    if d: print(f"{k[0]:>4} {k[1]:>7} {k[2]:>7} {k[3]:>6} | mean diff {np.mean(d):+.3f}  (wins {sum(x < 0 for x in d)}/{len(d)})")
# overlap x surprise law (selective SSM, plain fine-tuning and regularised): drop of old task j while training task i vs gate overlap and initial surprise
rows = []
for k, v in R.items():
    if k[0] != "sel" or k[1] != "ft": continue
    for r in v:
        M = np.array(r["acc_matrix"]); g = r["gates"]; il = r["init_loss"]
        for i in range(1, M.shape[0]):
            for j in range(i):
                gj, gi = np.array(g[f"{i-1}_{j}"]), np.array(g[f"{i-1}_{i}"])          # gate profile of old task j and new task i under the pre-update model
                ov = float(gj @ gi / (np.linalg.norm(gj) * np.linalg.norm(gi) + 1e-12))
                rows.append((M[i - 1, j] - M[i, j], ov, il[str(i)], k[2]))
A = np.array([(r[0], r[1], r[2]) for r in rows])
rk = lambda x: np.argsort(np.argsort(x)).astype(float); sp = lambda a, b: float(np.corrcoef(rk(a), rk(b))[0, 1])
print(f"\noverlap x surprise law, selective SSM, {len(A)} (old task, new task) events from all runs:")
print(f"  Spearman(drop, gate overlap) = {sp(A[:,0], A[:,1]):+.2f} | Spearman(drop, initial surprise) = {sp(A[:,0], A[:,2]):+.2f} | Spearman(drop, overlap x surprise) = {sp(A[:,0], A[:,1]*A[:,2]):+.2f}")
