"""Statistics of exp_mqar/PROTOCOL_replication.md.  usage: figgen/.venv/bin/python -I exp_mqar/analyze_replication.py   (run from the repository root)  -> exp_mqar/result_replication.json
Replicates the paper's definitions (figgen/audit_lever.py): mean recall over n in {8,16,24,32} and L in {0,32,96}; paired bootstrap (20000 draws, seed 0); exact sign-flip and exact Wilcoxon tests; 'failed to learn' = mean recall < 0.9."""
import sys, os, json, itertools
import numpy as np
sys.path.insert(0, "figgen")
import data as D
from scipy.stats import fisher_exact

ALL = {**D.MQ, **D.MQ_REP}
acc = lambda reg, lam: {s: D.mq_acc(r[1]) for (rg, lm, s), r in ALL.items() if rg == reg and lm == lam}
BASE, DEC, MATCH, RANK = acc("none", 0.0), acc("dec", 1.0), acc("match", 1.0), acc("rank", 0.1)


def signflip_p(diff):
    """exact two-sided sign-flip permutation p for the mean (all 2^n sign patterns; meet in the middle for n > 20)."""
    n = len(diff); obs = abs(diff.sum()) - 1e-12; h = n // 2
    def sums(x): return np.array([np.dot(sg, x) for sg in itertools.product([-1.0, 1.0], repeat=len(x))]) if len(x) else np.zeros(1)
    a, b = sums(diff[:h]), sums(diff[h:]); cnt = 0
    for i in range(0, len(a), 512): cnt += int((np.abs(a[i:i + 512, None] + b[None, :]) >= obs).sum())
    return cnt / 2 ** n


def wilcoxon_p(diff):
    """exact two-sided Wilcoxon signed-rank p (distribution of the rank sum by dynamic programming)."""
    n = len(diff); r = np.argsort(np.argsort(abs(diff))) + 1; W = int(r[diff > 0].sum()); tot = int(r.sum())
    cnt = np.zeros(tot + 1); cnt[0] = 1
    for k in range(1, n + 1):
        new = cnt.copy(); new[k:] += cnt[:tot + 1 - k]; cnt = new
    mu = tot / 2; ws = np.arange(tot + 1); return float(cnt[np.abs(ws - mu) >= abs(W - mu) - 1e-12].sum() / 2 ** n)


def paired(a, b, seeds):
    seeds = [s for s in seeds if s in a and s in b]; diff = np.array([a[s] - b[s] for s in seeds]); rng = np.random.default_rng(0)
    bs = [rng.choice(diff, len(diff)).mean() for _ in range(20000)] if len(diff) else [np.nan]
    return dict(n=len(diff), mean=float(diff.mean()) if len(diff) else None, ci=[float(x) for x in np.percentile(bs, [2.5, 97.5])], higher=int((diff > 0).sum()), signflip_p=signflip_p(diff) if len(diff) else None,
                wilcoxon_p=wilcoxon_p(diff) if len(diff) else None, seeds=seeds)


def failures(a, seeds): return sum(1 for s in seeds if s in a and a[s] < 0.9)


def holm(ps):
    order = np.argsort(ps); m = len(ps); adj = np.zeros(m); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * ps[i])); adj[i] = run
    return [float(x) for x in adj]


ORIG, REP, POOL = list(range(12)), list(range(12, 24)), list(range(24))
res = dict(original=paired(DEC, BASE, ORIG), replication=paired(DEC, BASE, REP), pooled=paired(DEC, BASE, POOL))
for tag, seeds in (("original", ORIG), ("replication", REP), ("pooled", POOL)):
    sd = [s for s in seeds if s in BASE and s in DEC]; fb, fd = failures(BASE, sd), failures(DEC, sd)
    res[tag].update(base_mean=float(np.mean([BASE[s] for s in sd])) if sd else None, dec_mean=float(np.mean([DEC[s] for s in sd])) if sd else None, base_failed=fb, dec_failed=fd,
                    fisher_p=float(fisher_exact([[fb, len(sd) - fb], [fd, len(sd) - fd]])[1]) if sd else None,
                    gain_nonfailed_baseline=float(np.mean([DEC[s] - BASE[s] for s in sd if BASE[s] >= 0.9])) if any(BASE[s] >= 0.9 for s in sd) else None, base_failed_seeds=[s for s in sd if BASE[s] < 0.9])
reg = {"dec": paired(DEC, BASE, ORIG), "match": paired(MATCH, BASE, ORIG), "rank": paired(RANK, BASE, ORIG)}
adj = holm([reg[k]["signflip_p"] for k in reg if reg[k]["signflip_p"] is not None])
for (k, v), a in zip([(k, v) for k, v in reg.items() if v["signflip_p"] is not None], adj): v["holm_p"] = a
res["regularisers_seeds0_11"] = reg
json.dump(res, open("exp_mqar/result_replication.json", "w"), indent=1)
f = lambda d: f"mean diff {d['mean']:+.3f} CI [{d['ci'][0]:+.3f},{d['ci'][1]:+.3f}] higher {d['higher']}/{d['n']} signflip p {d['signflip_p']:.4f} wilcoxon p {d['wilcoxon_p']:.4f}" if d["n"] else "no pairs yet"
for tag in ("original", "replication", "pooled"): print(f"{tag:12s} {f(res[tag])}" + (f" | base {res[tag]['base_mean']:.3f} dec {res[tag]['dec_mean']:.3f} | failed base {res[tag]['base_failed']} dec {res[tag]['dec_failed']} (Fisher p {res[tag]['fisher_p']:.3f}) | gain where baseline learned {res[tag]['gain_nonfailed_baseline']}" if res[tag]["n"] else ""))
for k, v in reg.items(): print(f"{k:6s} (seeds 0-11, n={v['n']}) {f(v)}" + (f" | Holm p {v['holm_p']:.4f}" if "holm_p" in v else ""))

# ---- exploratory (post hoc): does key rank track success in the 24 new runs? (layer-1 median key effective rank of the first 16 positions, as in Appendix C.3)
sys.path.insert(0, ".")
def geom(f):
    from flatlib import window_metrics, reff, unit
    c = np.load(f + "_cache.npz"); out = []
    for layer in (0, 1):
        K, V = c[f"k{layer}"], c[f"v{layer}"]; a = [(reff(K[b, :16, h]), reff(unit(V[b, :16, h]))) for b in range(K.shape[0]) for h in range(K.shape[2])]; out.append(np.median(np.array(a), axis=0))
    return np.array(out)                                                                              # (layer, [key, value]) medians
rk_ = lambda x: np.argsort(np.argsort(x)) / len(x)
GCACHE = "exp_mqar/replication_geometry.json"        # per-run median key/value effective ranks of the new runs (the activation caches are large and not distributed)
gc = json.load(open(GCACHE)) if os.path.exists(GCACHE) else {}
for (reg, lam, s), r in D.MQ_REP.items():
    if reg in ("none", "dec") and f"{reg}|{s}" not in gc: gc[f"{reg}|{s}"] = geom(r[0]).tolist()
json.dump(gc, open(GCACHE, "w"))
new = [(reg, s, D.mq_acc(r[1]), np.array(gc[f"{reg}|{s}"])) for (reg, lam, s), r in D.MQ_REP.items() if reg in ("none", "dec")]
acc_ = np.array([x[2] for x in new]); k1 = np.array([x[3][1, 0] for x in new]); k0 = np.array([x[3][0, 0] for x in new]); v1 = np.array([x[3][1, 1] for x in new]); lab = np.array([x[0] for x in new])
sp = lambda a, b: float(np.corrcoef(rk_(a), rk_(b))[0, 1])
res["geometry_new_runs"] = dict(n=len(new), spearman_acc_key_rank_layer1=sp(k1, acc_), spearman_acc_key_rank_layer0=sp(k0, acc_), spearman_acc_value_rank_layer1=sp(v1, acc_),
    key_rank_layer1_median=dict(baseline=float(np.median(k1[lab == "none"])), decorrelation=float(np.median(k1[lab == "dec"]))),
    key_rank_layer1_failed_vs_learned=dict(failed=float(np.median(k1[acc_ < 0.9])), learned=float(np.median(k1[acc_ >= 0.9])), n_failed=int((acc_ < 0.9).sum()), n_learned=int((acc_ >= 0.9).sum())),
    spearman_within_baseline=sp(k1[lab == "none"], acc_[lab == "none"]), spearman_within_decorrelation=sp(k1[lab == "dec"], acc_[lab == "dec"]))
json.dump(res, open("exp_mqar/result_replication.json", "w"), indent=1)
g = res["geometry_new_runs"]; print("new runs (24): Spearman acc vs layer-1 key rank", round(g["spearman_acc_key_rank_layer1"], 2), "| layer 0", round(g["spearman_acc_key_rank_layer0"], 2), "| value rank L1", round(g["spearman_acc_value_rank_layer1"], 2),
      "| key rank L1 median: baseline", round(g["key_rank_layer1_median"]["baseline"], 2), "decorrelation", round(g["key_rank_layer1_median"]["decorrelation"], 2), "| failed vs learned", {k: round(v, 2) for k, v in g["key_rank_layer1_failed_vs_learned"].items()},
      "| within baseline", round(g["spearman_within_baseline"], 2), "within decorrelation", round(g["spearman_within_decorrelation"], 2))
