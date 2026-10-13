import json, glob, collections, math, numpy as np
R = collections.defaultdict(list)
for f in sorted(glob.glob("exp_mem/runs/*_s*.json")):
    r = json.load(open(f)); a = r["args"]
    key = (a["method"], a["rho"] if a["method"] == "ogd" else None, a["ewc"] if a["method"] == "ewc" else None, a["q"] if a["method"] == "ogd" else None, a["k"] if a["method"] == "replay" else None)
    R[key].append(r)
ms = lambda x: f"{np.mean(x):.3f}±{(np.std(x, ddof=1) if len(x) > 1 else 0):.3f}"
base = {r["args"]["seed"]: r for r in R.get(("ft", None, None, None, None), [])}
print(f"{'learner':>26} | seeds | final avg acc | forgetting | last-task acc | first-task final | bits stored | stored rank r (sum) | paired d-forgetting vs ft (wins)")
for key in sorted(R, key=lambda k: (k[0] != "ft", k[0], str(k[1:]))):
    v = R[key]; name = {"ft": "fine-tuning", "ogd": f"projection rho={key[1]}, q={key[3]}", "ewc": f"EWC lambda={key[2]}", "replay": f"replay k={key[4]}"}[key[0]]
    d = [r["forgetting"] - base[r["args"]["seed"]]["forgetting"] for r in v if r["args"]["seed"] in base]
    rk = [sum(r["ranks"]) for r in v] if key[0] == "ogd" else None
    print(f"{name:>26} | {len(v):5d} | {ms([r['final_avg'] for r in v]):>13} | {ms([r['forgetting'] for r in v]):>10} | {ms([r['last_task'] for r in v]):>13} | {ms([r['first_task_final'] for r in v]):>16} | {np.mean([r['bits'] for r in v]):11.0f} | "
          f"{(f'{np.mean(rk):.1f}') if rk else '-':>19} | " + (f"{np.mean(d):+.3f} ({sum(x < 0 for x in d)}/{len(d)})" if key[0] != 'ft' and d else ""))
print("\nTheorem-15 floor vs bits actually stored by the projection learner (delta' = 0.1): floor = r (p - r) log2(1/delta'),  p =", next(iter(R.values()))[0]["p"])
for key in sorted(k for k in R if k[0] == "ogd"):
    v = R[key]; p = v[0]["p"]; rk = np.array([sum(r["ranks"]) for r in v]); bits = np.mean([r["bits"] for r in v])
    floor = np.mean([r * (p - r) * math.log2(1 / 0.1) for r in rk])
    print(f"  rho={key[1]}, q={key[3]:>2}: r={rk.mean():5.1f}  stored {bits:9.0f} bits  floor {floor:8.0f} bits  ratio stored/floor {bits/floor:6.1f}")
