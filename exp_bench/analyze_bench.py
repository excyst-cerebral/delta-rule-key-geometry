"""Summarise exp_bench/runs.  numpy only."""
import json, glob, collections, math, numpy as np
R = collections.defaultdict(dict)
for f in sorted(glob.glob("exp_bench/runs/*.json")):
    r = json.load(open(f)); a = r["args"]; m = a["method"]
    key = (a["bench"], m, a["rho"] if m in ("gpm", "ogd") else "-", a["ewc"] if m == "ewc" else "-", a["k"] if m == "replay" else "-", a["ssm"] if m == "gpm" else "-")
    R[key][a["seed"]] = r
ms = lambda x: f"{np.mean(x):.3f}±{np.std(x, ddof=1):.3f}" if len(x) > 1 else f"{np.mean(x):.3f}"
for bench in ("split", "perm"):
    views = ("full", "task") if bench == "split" else ("full",)
    base = {v: {s: r["res"][v] for s, r in R[(bench, "ft", "-", "-", "-", "-")].items()} for v in views} if (bench, "ft", "-", "-", "-", "-") in R else {}
    for v in views:
        title = {"full": "domain-incremental (10-way)" if bench == "perm" else "class-incremental", "task": "task-incremental (task id at test time)"}[v]
        print(f"\n== {bench}-MNIST, {title} ==")
        print("learner | n | final avg acc | forgetting | last-task acc | bits stored | paired d-final-avg vs ft (wins) | paired d-forgetting vs ft (wins)")
        for key in sorted((k for k in R if k[0] == bench), key=lambda k: (k[1] != "ft", k[1], str(k))):
            rs = R[key]; seeds = sorted(rs); fa = [rs[s]["res"][v]["final_avg"] for s in seeds]; fg = [rs[s]["res"][v]["forgetting"] for s in seeds]; lt = [rs[s]["res"][v]["last_task"] for s in seeds]
            bits = np.mean([rs[s]["bits"] for s in seeds]); d1 = [rs[s]["res"][v]["final_avg"] - base[v][s]["final_avg"] for s in seeds if s in base[v]]; d2 = [rs[s]["res"][v]["forgetting"] - base[v][s]["forgetting"] for s in seeds if s in base[v]]
            lab = key[1] + (f" rho={key[2]}" if key[2] != "-" else "") + (f" lam={key[3]:g}" if key[3] != "-" else "") + (f" k={key[4]}" if key[4] != "-" else "") + (f" ssm={key[5]}" if key[5] != "-" else "")
            print(f"{lab:32s} | {len(seeds)} | {ms(fa)} | {ms(fg)} | {ms(lt)} | {int(bits):>9d} | {np.mean(d1):+.3f} ({sum(x > 0 for x in d1)}/{len(d1)}) | {np.mean(d2):+.3f} ({sum(x < 0 for x in d2)}/{len(d2)})")
