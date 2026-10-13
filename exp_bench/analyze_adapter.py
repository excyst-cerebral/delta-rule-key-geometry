"""Task-specific adapters vs the other learners (task id given).  numpy only."""
import json, glob, collections, numpy as np
ms = lambda x: f"{np.mean(x):.3f}±{np.std(x, ddof=1):.3f}"
A = collections.defaultdict(dict)
for f in sorted(glob.glob("exp_bench/runs_adapter/*.json")):
    r = json.load(open(f)); A[(r["args"]["bench"], r["args"]["rank"])][r["args"]["seed"]] = r
def ref(bench, method, **kw):
    out = []
    for f in glob.glob(f"exp_bench/runs/{bench}_{method}_*.json"):
        r = json.load(open(f)); a = r["args"]
        if all(a[k] == v for k, v in kw.items()): out.append(r)
    return out
for bench in ("split", "perm"):
    print(f"\n== {bench}-MNIST, task id given at test time (adapters) ==")
    print("learner | n | final avg acc (task-masked) | forgetting | last-task | bits stored | adapter params/task | max |dlogit| on an old task")
    for rk in sorted(r for (b, r) in A if b == bench):
        rs = A[(bench, rk)]; seeds = sorted(rs); v = "task" if bench == "split" else "full"
        print(f"adapter rank {rk:<2d} (+head)".ljust(32), f"| {len(seeds)} | {ms([rs[s]['res'][v]['final_avg'] for s in seeds])} | {ms([rs[s]['res'][v]['forgetting'] for s in seeds])} | {ms([rs[s]['res'][v]['last_task'] for s in seeds])} | {int(np.mean([rs[s]['bits'] for s in seeds]))} | {rs[seeds[0]]['adapter_params']} | {max(rs[s]['max_old_logit_change'] for s in seeds):.1e}")
    if bench == "split":
        print("reference rows (task-incremental), from exp_bench/runs:")
        for lab, m, kw in (("fine-tuning", "ft", {}), ("EWC lam=10", "ewc", {"ewc": 10.0}), ("replay k=20", "replay", {"k": 20}), ("replay k=200", "replay", {"k": 200}), ("layerwise frozen rho=0.99", "gpm", {"ssm": "freeze", "rho": 0.99})):
            R = ref(bench, m, **kw); 
            if R: print(f"  {lab:28s} n={len(R)} final {ms([r['res']['task']['final_avg'] for r in R])}  forgetting {ms([r['res']['task']['forgetting'] for r in R])}  bits {int(np.mean([r['bits'] for r in R]))}")
        J = [json.load(open(f)) for f in glob.glob("exp_bench/runs_joint/split_joint_s*.json")]; print(f"  joint training               n={len(J)} final {ms([r['res']['task']['final_avg'] for r in J])}")
    else:
        print("reference rows (domain-incremental, NO task id), from exp_bench/runs -- note adapters here receive the task id, the others do not:")
        for lab, m, kw in (("fine-tuning", "ft", {}), ("replay k=200", "replay", {"k": 200}), ("layerwise frozen rho=0.99", "gpm", {"ssm": "freeze", "rho": 0.99})):
            R = ref(bench, m, **kw)
            if R: print(f"  {lab:28s} n={len(R)} final {ms([r['res']['full']['final_avg'] for r in R])}  forgetting {ms([r['res']['full']['forgetting'] for r in R])}  bits {int(np.mean([r['bits'] for r in R]))}")
        J = [json.load(open(f)) for f in glob.glob("exp_bench/runs_joint/perm_joint_s*.json")]; print(f"  joint training (no task id)  n={len(J)} final {ms([r['res']['full']['final_avg'] for r in J])}")
