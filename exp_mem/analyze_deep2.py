"""Summarise exp_mem/runs2 (layerwise GPM variants, width control, learning-rate control).  numpy only."""
import json, glob, collections, numpy as np
R = collections.defaultdict(dict)
for f in sorted(glob.glob("exp_mem/runs2/*.json")):
    r = json.load(open(f)); a = r["args"]
    key = (a["method"], a["ssm"] if a["method"] == "gpm" else "-", a["rho"] if a["method"] in ("gpm", "ogd") else "-", a["q"] if a["method"] in ("gpm", "ogd") else "-", a["C"], a["lr"], a["k"] if a["method"] == "replay" else "-")
    R[key][a["seed"]] = r
ft = {(k[4], k[5]): v for k, v in R.items() if k[0] == "ft"}
ft[(12, 0.03)] = {}   # C=12, lr=0.03 fine-tuning baseline was run in the first grid (exp_mem/runs/)
for f in sorted(glob.glob("exp_mem/runs/ft_*_s*.json")):
    r = json.load(open(f)); ft[(12, 0.03)][r["args"]["seed"]] = r
m = lambda x: f"{np.mean(x):.3f}±{np.std(x, ddof=1):.3f}"
print("method ssm rho q C lr k | n | final avg | forgetting | last-task | acc on tasks whose polarity = task 0's / differs (final) | bits | d-forgetting vs ft (wins) | d-final-avg vs ft")
for key in sorted(R, key=lambda k: (k[4], k[5], str(k))):
    rs = R[key]; seeds = sorted(rs)
    fa = [rs[s]["final_avg"] for s in seeds]; fg = [rs[s]["forgetting"] for s in seeds]; lt = [rs[s]["last_task"] for s in seeds]
    same, diff = [], []
    for s in seeds:
        pol = rs[s]["pol"]; acc = rs[s]["acc"][-1]
        for t in range(1, len(pol)): (same if pol[t] == pol[0] else diff).append(acc[t])
    base = ft[(key[4], key[5])]; d = [rs[s]["forgetting"] - base[s]["forgetting"] for s in seeds if s in base]; dfa = [rs[s]["final_avg"] - base[s]["final_avg"] for s in seeds if s in base]
    wins = sum(x < 0 for x in d)
    print(f"{key[0]:6s} {key[1]:6s} {str(key[2]):6s} {str(key[3]):3s} C={key[4]:<3d} lr={key[5]:<6} k={key[6]!s:3s} | {len(seeds)} | {m(fa)} | {m(fg)} | {m(lt)} | {np.mean(same):.2f} (n={len(same)}) / {np.mean(diff):.2f} (n={len(diff)}) | {int(np.mean([rs[s]['bits'] for s in seeds]))} | {np.mean(d):+.3f} ({wins}/{len(d)}) | {np.mean(dfa):+.3f}")
