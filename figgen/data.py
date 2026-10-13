"""Single source of truth for every number in the paper's figures and tables.  numpy only.  All aggregates are computed from the raw run files."""
import json, glob, os, re, collections
import numpy as np
ROOT = "."
def J(p): return json.load(open(f"{ROOT}/{p}"))
def mean_sd(x): x = np.asarray(x, float); return float(x.mean()), float(x.std(ddof=1)) if len(x) > 1 else 0.0

# ---------------------------------------------------------------- MNIST benchmarks (exp_bench)
def load_runs(pattern):
    out = collections.defaultdict(dict)
    for f in sorted(glob.glob(f"{ROOT}/{pattern}")):
        r = json.load(open(f)); a = r["args"]; m = a["method"]
        key = (a["bench"], m, a["rho"] if m in ("gpm", "ogd") else None, a["ewc"] if m == "ewc" else None, a["k"] if m == "replay" else None, a["ssm"] if m == "gpm" else None, a.get("rank") if m == "adapter" else None)
        out[key][a["seed"]] = r
    return out
MN = load_runs("exp_bench/runs/*.json"); MJ = load_runs("exp_bench/runs_joint/*.json"); MF = load_runs("exp_bench/runs_fixed/*.json"); MA = load_runs("exp_bench/runs_adapter/*.json")
def agg(runs, view, metrics=("final_avg", "forgetting", "last_task")):
    seeds = sorted(runs); out = {}
    for m in metrics: out[m] = mean_sd([runs[s]["res"][view][m] for s in seeds])
    out["bits"] = float(np.mean([runs[s]["bits"] for s in seeds])); out["n"] = len(seeds); out["per_seed"] = {m: [runs[s]["res"][view][m] for s in seeds] for m in metrics}
    return out
def mnist(bench, method, view, rho=None, ewc=None, k=None, ssm=None):
    return agg(MN[(bench, method, rho, ewc, k, ssm, None)], view)
def joint(bench, view): return agg(MJ[(bench, "joint", None, None, None, None, None)], view)
def adapter(bench, rank, view): return agg(MA[(bench, "adapter", None, None, None, None, rank)], view)
def fixed(bench, method, view, rho=None): return agg(MF[(bench, method, rho, None, None, None, None)], view)
# paired differences against fine-tuning (same seed)
def paired(bench, method, view, metric, **kw):
    a = MN[(bench, method, kw.get("rho"), kw.get("ewc"), kw.get("k"), kw.get("ssm"), None)]; b = MN[(bench, "ft", None, None, None, None, None)]
    d = [a[s]["res"][view][metric] - b[s]["res"][view][metric] for s in sorted(a) if s in b]; return float(np.mean(d)), len(d)

# ---------------------------------------------------------------- deep SSM (exp_mem)
def load_deep(pattern, keyfn):
    out = collections.defaultdict(dict)
    for f in sorted(glob.glob(f"{ROOT}/{pattern}")):
        r = json.load(open(f)); out[keyfn(r["args"])][r["args"]["seed"]] = r
    return out
D1 = load_deep("exp_mem/runs/*_s*.json", lambda a: (a["method"], a["rho"] if a["method"] in ("ogd",) else None, a["ewc"] if a["method"] == "ewc" else None, a["q"] if a["method"] == "ogd" else None, a["k"] if a["method"] == "replay" else None))
D2 = load_deep("exp_mem/runs2/*.json", lambda a: (a["method"], a.get("ssm") if a["method"] == "gpm" else None, a["rho"] if a["method"] in ("gpm", "ogd") else None, a["q"] if a["method"] in ("gpm", "ogd") else None, a["C"], a["lr"], a["k"] if a["method"] == "replay" else None))
D3 = load_deep("exp_mem/runs3/*.json", lambda a: (a["method"], a["rho"] if a["method"] == "gpm" else None, a["C"]))
PROBE = load_deep("exp_mem/probe/*.json", lambda a: a["rho"])
DRIFT = [json.load(open(f))["drift"] for f in sorted(glob.glob(f"{ROOT}/exp_mem/drift/drift_s*.json"))]
def dagg(runs, keys=("final_avg", "forgetting", "last_task")):
    seeds = sorted(runs); out = {k: mean_sd([runs[s][k] for s in seeds]) for k in keys}; out["bits"] = float(np.mean([runs[s]["bits"] for s in seeds])); out["n"] = len(seeds)
    out["acc"] = np.mean([np.array(runs[s]["acc"]) for s in seeds], 0); out["per_seed"] = {k: [runs[s][k] for s in seeds] for k in keys}; return out

# ---------------------------------------------------------------- MQAR (exp_mqar)
MQ, MQ_REP = {}, {}                  # MQ: seeds 0-11 (the published comparison); MQ_REP: seeds >= 12 (replication, exp_mqar/PROTOCOL_replication.md)
for f in sorted(glob.glob(f"{ROOT}/exp_mqar/runs/*.json")):
    r = json.load(open(f)); a = r["args"]; (MQ if a["seed"] < 12 else MQ_REP)[(a["reg"], a["lam"], a["seed"])] = (f[:-5], r)
MQ_CFG = [("none", 0.0, "baseline"), ("dec", 1.0, "key decorrelation"), ("match", 1.0, "Gram matching"), ("rank", 0.1, "rank matching")]
def mq_acc(r, ns=(8, 16, 24, 32)): return float(np.mean([r["grid"][f"{n}_{L}"] for n in ns for L in (0, 32, 96)]))
def mq_seeds(reg, lam): return sorted(s for (rg, lm, s) in MQ if rg == reg and lm == lam)
def mq_by_n(reg, lam):
    seeds = mq_seeds(reg, lam); return {n: [float(np.mean([MQ[(reg, lam, s)][1]["grid"][f"{n}_{L}"] for L in (0, 32, 96)])) for s in seeds] for n in (8, 16, 24, 32)}
def mq_escape(r):
    for st, ce, _ in r["hist"]:
        if st % 250 == 0 and ce < 2.0: return st
    return None

# ---------------------------------------------------------------- task-level test (exp_task)
def load_task(name):
    Z = [np.load(f) for f in sorted(glob.glob(f"{ROOT}/exp_task/out/{name}_shard*.npz"))]
    meta = np.concatenate([z["meta"] for z in Z]); d = dict(n=meta[:, 0], j=meta[:, 1], age=meta[:, 2], qpos=meta[:, 3], q=meta[:, 4], fc=meta[:, 5], rank=meta[:, 6], clp=meta[:, 7],
        psi_span=np.concatenate([z["psi_span"] for z in Z]), mlog=np.concatenate([z["mlog"] for z in Z]))
    d["half"] = np.load(f"{ROOT}/exp_task/out/{name}_halflife_cal.npy"); return d
TASK = {"GDN-340M": load_task("model_gdn340m"), "GDN-360M": load_task("model_gdn360m")}
def task_features(d):
    long_h = d["half"] >= 256; ps = d["psi_span"].mean(1); d["P1"] = ps[:, long_h].mean(1); d["P1b"] = ps.reshape(len(ps), -1).mean(1); d["long_h"] = long_h
    rng = np.random.default_rng(0); sh = d["P1"].copy(); cal = d["j"] % 2 == 0
    for v in set(d["n"]):
        for msk in (cal, ~cal):
            idx = np.where((d["n"] == v) & msk)[0]; sh[idx] = d["P1"][rng.permutation(idx)]
    d["P1s"] = sh; d["cal"] = cal; return d
for k in TASK: task_features(TASK[k])
def auc(y, s):
    o = np.argsort(s, kind="mergesort"); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    for u in np.unique(s):
        m = s == u
        if m.sum() > 1: r[m] = r[m].mean()
    n1 = y.sum(); n0 = len(y) - n1; return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
def roc(y, s):
    o = np.argsort(-s, kind="mergesort"); y = y[o]; tp = np.cumsum(y) / y.sum(); fp = np.cumsum(1 - y) / (len(y) - y.sum()); return np.r_[0, fp], np.r_[0, tp]
def logistic(X, y, cal, lam=1e-3, it=60):
    mu, sd = X[cal][:, 1:].mean(0), X[cal][:, 1:].std(0) + 1e-12; Xs = X.copy(); Xs[:, 1:] = (X[:, 1:] - mu) / sd; b = np.zeros(X.shape[1]); Xc, yc = Xs[cal], y[cal]
    for _ in range(it):
        p = 1 / (1 + np.exp(-Xc @ b)); W = p * (1 - p) + 1e-9; R = lam * np.eye(len(b)); R[0, 0] = 0
        b = b + np.linalg.solve(Xc.T @ (Xc * W[:, None]) + R, Xc.T @ (yc - p) - R @ b)
    return 1 / (1 + np.exp(-Xs @ b))
def task_models(d):
    N = len(d["n"]); onehot = np.stack([(d["n"] == v).astype(float) for v in sorted(set(d["n"]))[1:]], 1)
    des = lambda cols: np.column_stack([np.ones(N)] + cols); fc = d["fc"]
    return {"age + load": logistic(des([d["age"], onehot]), fc, d["cal"]), "age + load + P1": logistic(des([d["age"], onehot, d["P1"]]), fc, d["cal"]), "age + load + shuffled P1": logistic(des([d["age"], onehot, d["P1s"]]), fc, d["cal"])}
def rk(x): return np.argsort(np.argsort(x)) / len(x)
def within_cell(d, x, y, B=1000, seed=1):
    rng = np.random.default_rng(seed); cell = (d["n"] * 100 + d["q"]).astype(int); N = len(x); xs, ys, keep = np.zeros(N), np.zeros(N), np.zeros(N, bool)
    for c in np.unique(cell):
        idx = np.where(cell == c)[0]
        if len(idx) >= 4: xs[idx] = rk(x[idx]) - rk(x[idx]).mean(); ys[idx] = rk(y[idx]) - rk(y[idx]).mean(); keep[idx] = True
    r = float(np.corrcoef(xs[keep], ys[keep])[0, 1]); ids = np.where(keep)[0]; bs = [float(np.corrcoef(xs[s], ys[s])[0, 1]) for s in (rng.choice(ids, len(ids)) for _ in range(B))]
    return r, np.percentile(bs, [2.5, 97.5]).tolist(), xs, ys, keep
def partial_linear(d, x, y, mask, B=1000, seed=1):
    rng = np.random.default_rng(seed); onehot = np.stack([(d["n"] == v).astype(float) for v in sorted(set(d["n"]))[1:]], 1)
    X = np.column_stack([np.ones(mask.sum()), d["age"][mask], onehot[mask]]); res = lambda a: a - X @ np.linalg.lstsq(X, a, rcond=None)[0]; a, b = res(rk(x[mask])), res(rk(y[mask]))
    r = float(np.corrcoef(a, b)[0, 1]); bs = [float(np.corrcoef(a[i], b[i])[0, 1]) for i in (rng.integers(0, len(a), len(a)) for _ in range(B))]; return r, np.percentile(bs, [2.5, 97.5]).tolist()

# ---------------------------------------------------------------- review
REV = {"coded": J("review/coded_records.json"), "prisma": J("review/prisma_counts.json")}

if __name__ == "__main__":
    print("MNIST split task-IL replay200:", mnist("split", "replay", "task", k=200)["final_avg"], " class-IL ft:", mnist("split", "ft", "full")["final_avg"], " joint:", joint("split", "task")["final_avg"])
    d = TASK["GDN-340M"]; m = task_models(d); ti = ~d["cal"]
    print("340M AUC age+load / +P1 / shuffled:", [round(auc(d["fc"][ti], m[k][ti]), 3) for k in m], " exact control r:", [round(x, 3) if not isinstance(x, list) else x for x in within_cell(d, d["P1"], d["clp"])[:2]])
    print("MQ baseline mean:", np.mean([mq_acc(MQ[("none", 0.0, s)][1]) for s in mq_seeds("none", 0.0)]), " dec:", np.mean([mq_acc(MQ[("dec", 1.0, s)][1]) for s in mq_seeds("dec", 1.0)]))
    k = ("ft", None, None, None, None); print("deep ft forgetting:", dagg(D1[k])["forgetting"])
    print("probe rhos:", sorted(PROBE), " drift seeds:", len(DRIFT))

# ---------------------------------------------------------------- toy testbed (exp_cl)
TOY = collections.defaultdict(list)
for f in sorted(glob.glob(f"{ROOT}/exp_cl/runs/*.json")):
    r = json.load(open(f)); a = r["args"]; TOY[(a["arch"], a["method"], a["reg"], a["lam"])].append(r)
def toy(arch, method, reg="none", lam=0.0):
    v = TOY[(arch, method, reg, lam)]; out = {m: mean_sd([r[m] for r in v]) for m in ("final_avg", "forgetting")}; out["n"] = len(v); out["seed"] = {r["args"]["seed"]: r for r in v}; return out
def toy_paired(arch, method, reg, lam):
    base = {r["args"]["seed"]: r["forgetting"] for r in TOY[(arch, "ft", "none", 0.0)]}
    d = [r["forgetting"] - base[r["args"]["seed"]] for r in TOY[(arch, method, reg, lam)] if r["args"]["seed"] in base]; return float(np.mean(d)), int(sum(x < 0 for x in d)), len(d)

# ---------------------------------------------------------------- MQAR geometry (computed once, cached): per run, per layer, rows of (key r_eff, value r_eff, q, RMS/psi) per (sequence, head)
def mq_geometry():
    cache = f"{ROOT}/figgen/cache_mq_geom.json"
    out = {tuple(json.loads(k)): np.array(v) for k, v in json.load(open(cache)).items()} if os.path.exists(cache) else {}
    if all(k in out for k in MQ): return out
    import sys; sys.path.insert(0, ROOT); from flatlib import window_metrics, reff, unit        # only runs missing from the cache are computed (from their activation caches)
    for (reg, lam, seed), (f, r) in sorted(MQ.items()):
        if (reg, lam, seed) in out: continue
        c = np.load(f + "_cache.npz"); layers = []
        for layer in (0, 1):
            K, V = c[f"k{layer}"], c[f"v{layer}"]; a = []
            for b in range(K.shape[0]):
                for h in range(K.shape[2]):
                    m = window_metrics(K[b, :16, h], unit(V[b, :16, h])); a.append((reff(K[b, :16, h]), reff(unit(V[b, :16, h])), m["q"], m["ratio"]))
            layers.append(np.array(a).tolist())
        out[(reg, lam, seed)] = np.array(layers)
    json.dump({json.dumps(list(k)): v.tolist() for k, v in out.items()}, open(cache, "w")); return out
def mq_geom_pooled(reg, lam, layer):
    G = mq_geometry(); a = np.concatenate([v[layer] for (rg, lm, s), v in G.items() if rg == reg and lm == lam])
    return dict(kr=float(np.median(a[:, 0])), vr=float(np.median(a[:, 1])), q=float(np.median(a[:, 2])), rms_psi=float(np.median(a[:, 3])), ratio=float(np.median(a[:, 1] / a[:, 0])))

# ---------------------------------------------------------------- measurement study: per-window RMS, psi for the six models (raw files at the repository root)
CERT_MODELS = [("GDN-340M", "GDN 340M"), ("GDN-360M", "GDN 360M"), ("Q3.5", "Qwen3.5-0.8B"), ("R7", "RWKV-7 0.1B"), ("Q2.5", "Qwen2.5-1.5B"), ("S2", "SmolLM2-135M")]
def load_cert():
    cache = f"{ROOT}/figgen/cache_cert.npz"
    if os.path.exists(cache):
        z = np.load(cache); return {k: {f: z[f"{k}|{f}"] for f in ("n", "rms", "psi", "dom", "layfrac", "seq")} for k, _ in CERT_MODELS}
    import pickle; out = {}
    for k, f, NL in (("GDN-340M", "results_transfer.npz", 24), ("GDN-360M", "results_transfer_model_gdn360m.npz", 21)):
        A = np.load(f"{ROOT}/{f}")["A"]; out[k] = dict(n=A[:, 5].astype(int), rms=A[:, 6], psi=A[:, 7], dom=A[:, 0].astype(int), layfrac=A[:, 2] / (NL - 1), seq=A[:, 0].astype(int) * 100 + A[:, 1].astype(int))
    for k, f, NL, sel in (("Q3.5", "qwen35_flat.pkl", 24, None), ("R7", "rwkv7_flat.pkl", 12, "kappa"), ("Q2.5", "tf_flat_Qwen2.5-1.5B-Instruct.pkl", 28, None), ("S2", "tf_flat_SmolLM2-135M-Instruct.pkl", 30, None)):
        r = pickle.load(open(f"{ROOT}/{f}", "rb"))["rec"]; r = [x for x in r if x["var"] == "own" and (sel is None or str(x.get("key")) == sel)]
        lay = np.array([x["layer"] for x in r], float); doms = ["prose", "code", "docs", "legal", "notes"]
        out[k] = dict(n=np.array([x["n"] for x in r]), rms=np.array([x["rms"] for x in r]), psi=np.array([x["psi"] for x in r]), dom=np.array([x["dom"] if isinstance(x["dom"], (int, np.integer)) else doms.index(x["dom"]) for x in r]), layfrac=lay / lay.max(), seq=np.array([doms.index(x["dom"]) * 100 + int(x["seq"]) if not isinstance(x["dom"], (int, np.integer)) else int(x["dom"]) * 100 + int(x["seq"]) for x in r]))
    np.savez_compressed(cache, **{f"{k}|{f}": v for k, d in out.items() for f, v in d.items()}); return out
def r90(k, n, q=0.9): d = CERT[k]; m = d["n"] == n; return float(np.quantile(d["rms"][m] / d["psi"][m], q))
CERT = load_cert()

# ---------------------------------------------------------------- task-level statistics as logged by the pre-registered pipeline (exp_task/*.txt, result_*.json)
def _task_log(tag):
    txt = open(f"{ROOT}/exp_task/exploratory_model_{tag}.txt").read(); res = json.load(open(f"{ROOT}/exp_task/result_model_{tag}.json")); num = r"r = ([+-]\d\.\d+)\s+\[([+-]\d\.\d+), ([+-]\d\.\d+)\]"
    g = lambda pat: tuple(float(x) for x in re.search(pat, txt).groups())
    return dict(lin=g(r"P1 \(long-memory heads\)\s+test\s+" + num), lin_s=g(r"P1 shuffled within n \(null\)\s+test\s+" + num), ex=g(r"P1 \(long-memory heads\)\s+" + num + r"\s+\(592"), ex_s=g(r"P1 shuffled within n \(null\)\s+" + num + r"\s+\(592"),
                H1=(res["H1"][0], res["H1"][1][0], res["H1"][1][1]), H1s=(res["ctrl"][0], res["ctrl"][1][0], res["ctrl"][1][1]))
TASK_LOG = {"GDN-340M": _task_log("gdn340m"), "GDN-360M": _task_log("gdn360m")}


def r90_ci(k, n, B=2000, seed=0, q=0.9):
    """Cluster bootstrap over (domain, sequence) of the pooled r90: all windows, heads and layers of a sequence are resampled together."""
    d = CERT[k]; m = d["n"] == n; ratio = (d["rms"] / d["psi"])[m]; seq = d["seq"][m]; ids = np.unique(seq); groups = [ratio[seq == s] for s in ids]; rng = np.random.default_rng(seed); out = []
    for _ in range(B):
        pick = rng.integers(0, len(ids), len(ids)); out.append(np.quantile(np.concatenate([groups[i] for i in pick]), q))
    return float(np.quantile(ratio, q)), [float(x) for x in np.percentile(out, [2.5, 97.5])], len(ids)


def flex_auc(d, B=2000, seed=0, lam=1.0):
    """Held-out AUC with a flexible age model: baseline = fixed effect of every (load, queried-position) cell (exactly age and load), fitted on the calibration prompts; the comparison adds P1.
    Returns (auc_base, auc_plus, delta, [lo, hi]) on the test prompts (paired bootstrap over test prompts)."""
    N = len(d["n"]); cell = (d["n"] * 100 + d["q"]).astype(int); cells = np.unique(cell[d["cal"]]); idx = {c: i for i, c in enumerate(cells)}
    # cells without calibration prompts fall back to their load's mean cell effect: add a one-hot for the load as well
    loads = sorted(set(d["n"])); X0 = np.zeros((N, len(cells) + len(loads)))
    for i in range(N):
        if cell[i] in idx: X0[i, idx[cell[i]]] = 1
        X0[i, len(cells) + loads.index(d["n"][i])] = 1
    y = d["fc"]; cal = d["cal"]
    def fit(X):
        mu = X[cal].mean(0); sd = X[cal].std(0) + 1e-9; Xs = (X - mu) / sd; Xs = np.column_stack([np.ones(N), Xs]); b = np.zeros(Xs.shape[1]); R = lam * np.eye(len(b)); R[0, 0] = 0
        for _ in range(60):
            p = 1 / (1 + np.exp(-Xs[cal] @ b)); W = p * (1 - p) + 1e-9; b = b + np.linalg.solve(Xs[cal].T @ (Xs[cal] * W[:, None]) + R, Xs[cal].T @ (y[cal] - p) - R @ b)
        return 1 / (1 + np.exp(-Xs @ b))
    s0 = fit(X0); s1 = fit(np.column_stack([X0, d["P1"]])); te = ~cal; yt = y[te]
    a0, a1 = auc(yt, s0[te]), auc(yt, s1[te]); rng = np.random.default_rng(seed); bs = []
    for _ in range(B):
        i = rng.integers(0, len(yt), len(yt))
        if yt[i].sum() in (0, len(i)): continue
        bs.append(auc(yt[i], s1[te][i]) - auc(yt[i], s0[te][i]))
    return a0, a1, a1 - a0, [float(x) for x in np.percentile(bs, [2.5, 97.5])]

# ---------------------------------------------------------------- gate-aware key-only analysis (exp_gate)
GATE_ACT = {nm: J(f"exp_gate/result_actual_{tag}.json") for nm, tag in (("GDN-340M", "model_gdn340m"), ("GDN-360M", "model_gdn360m"))}
GATE_TASK = J("exp_gate/result_task_gate.json")
def gate_scatter(tag, delta=64, n=2500, seed=0):
    """(D_gate, D_act) over all sequences, probes and long-memory heads of a model at one delay (the cells behind Table gate_actual), subsampled for plotting."""
    z = np.load(f"{ROOT}/exp_gate/out/{tag}_actual.npz", allow_pickle=True); lh = np.median(z["halflife"], 0) >= 256; di = list(z["deltas"]).index(delta)
    x = z["Dgate"][:, :, :, :, di][:, lh].reshape(-1).astype(float); y = z["Dact"][:, :, :, :, di][:, lh].reshape(-1).astype(float); ok = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x, y = x[ok], y[ok]; i = np.random.default_rng(seed).permutation(len(x))[:n]; return x[i], y[i], len(x)

# ---------------------------------------------------------------- scale replication of the task-level test on a 1.3B Gated DeltaNet (exp_scale)
SCALE = J("exp_scale/result_scale_model_gdn13b.json")

# ---------------------------------------------------------------- held-out check of the calibrated constants on new text (exp_heldout)
HELDOUT = J("exp_heldout/result_heldout.json")
