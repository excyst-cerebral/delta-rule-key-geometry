"""Fixed (never-trained) random poles: toy streams (exp_mem/runs3) and MNIST (exp_bench/runs_fixed).  numpy only."""
import json, glob, collections, numpy as np
ms = lambda x: f"{np.mean(x):.3f}±{np.std(x, ddof=1):.3f}" if len(x) > 1 else f"{np.mean(x):.3f}"
print("== toy deep-SSM stream (5 tasks, polarity-conflicting), poles fixed at initialisation ==")
R = collections.defaultdict(dict)
for f in sorted(glob.glob("exp_mem/runs3/*.json")):
    r = json.load(open(f)); a = r["args"]; R[(a["C"], a["method"], a["rho"] if a["method"] == "gpm" else "-")][a["seed"]] = r
for k in sorted(R, key=lambda k: (k[0], k[1] != "ft", str(k))):
    rs = R[k]; seeds = sorted(rs); base = R[(k[0], "ft", "-")]
    d = [rs[s]["forgetting"] - base[s]["forgetting"] for s in seeds if s in base]; dfa = [rs[s]["final_avg"] - base[s]["final_avg"] for s in seeds if s in base]
    same, diff = [], []
    for s in seeds:
        pol = rs[s]["pol"]; acc = rs[s]["acc"][-1]
        for t in range(1, len(pol)): (same if pol[t] == pol[0] else diff).append(acc[t])
    print(f"C={k[0]:<3d} {k[1]:4s} rho={k[2]!s:6s} n={len(seeds)} final {ms([rs[s]['final_avg'] for s in seeds])}  forgetting {ms([rs[s]['forgetting'] for s in seeds])}  last-task {ms([rs[s]['last_task'] for s in seeds])}  bits {int(np.mean([rs[s]['bits'] for s in seeds]))}  acc(same-polarity tasks) {np.mean(same):.2f} / (conflicting) {np.mean(diff):.2f}  d-final {np.mean(dfa):+.3f}  d-forgetting {np.mean(d):+.3f} ({sum(x < 0 for x in d)}/{len(d)})")
for bench in ("split", "perm"):
    R = collections.defaultdict(dict)
    for f in sorted(glob.glob(f"exp_bench/runs_fixed/{bench}_*.json")):
        r = json.load(open(f)); a = r["args"]; R[(a["method"], a["rho"] if a["method"] == "gpm" else "-")][a["seed"]] = r
    for v in (("full", "task") if bench == "split" else ("full",)):
        print(f"\n== {bench}-MNIST, poles fixed at initialisation, {v}-incremental ==" if bench == "split" else f"\n== perm-MNIST, poles fixed at initialisation (domain-incremental) ==")
        for k in sorted(R, key=lambda k: (k[0] != "joint", k[0] != "ft", str(k))):
            rs = R[k]; seeds = sorted(rs); base = R.get(("ft", "-"), {})
            d = [rs[s]["res"][v]["final_avg"] - base[s]["res"][v]["final_avg"] for s in seeds if s in base]
            print(f"{k[0]:6s} rho={k[1]!s:6s} n={len(seeds)} final avg {ms([rs[s]['res'][v]['final_avg'] for s in seeds])}  forgetting {ms([rs[s]['res'][v]['forgetting'] for s in seeds])}  last-task {ms([rs[s]['res'][v]['last_task'] for s in seeds])}  bits {int(np.mean([rs[s]['bits'] for s in seeds]))}  d-final vs ft {np.mean(d):+.3f} ({sum(x > 0 for x in d)}/{len(d)})")
