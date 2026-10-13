"""Part A statistics of exp_gate/PROTOCOL.md.  numpy only.  usage: python analyze_actual.py <model dir name> [npz path]  -> exp_gate/result_actual_<model>.json"""
import sys, json, numpy as np
name = sys.argv[1]; HERE = "./exp_gate"
Z = np.load(sys.argv[2] if len(sys.argv) > 2 else f"{HERE}/out/{name}_actual.npz", allow_pickle=True)
PRED = ["Dgate", "Dshuf", "Dfree", "Bkey", "Bsum", "A"]; DELTAS = list(Z["deltas"]); PROBES = list(Z["probes"]); nS = Z["Dact"].shape[0]
long_head = np.median(Z["halflife"], axis=0) >= 256                                    # (NL, H): median decay half-life >= 256 tokens over the sequences
rng = np.random.default_rng(0); NB = 500


def rank(x): return np.argsort(np.argsort(x, axis=-1), axis=-1).astype(np.float64)
def corr(a, b):
    a = a - a.mean(-1, keepdims=True); b = b - b.mean(-1, keepdims=True); return (a * b).sum(-1) / np.sqrt((a * a).sum(-1) * (b * b).sum(-1) + 1e-300)
def partial(rx, ry, rz):
    rxy, rxz, ryz = corr(rx, ry), corr(rx, rz), corr(ry, rz); return (rxy - rxz * ryz) / np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2) + 1e-300)


def stats(idx, di, sel):
    """for the resampled sequences idx, delay index di and head subset sel: within-head, partial within-head and between-head Spearman of D_act with each predictor."""
    ok = [pi for pi, s in enumerate(PROBES) if s + DELTAS[di] <= 1023]
    def grp(k):
        x = Z[k][idx][:, :, :, ok, di]; x = x[:, sel]; return np.moveaxis(x, 0, -2).reshape(-1, len(idx) * len(ok))   # (groups, cells)
    y = grp("Dact"); ry = rank(y); rA = rank(np.log(grp("A") + 1e-30)); out = {}
    for k in PRED:
        x = grp(k); rx = rank(x)
        out[k] = dict(within=float(corr(rx, ry).mean()), partial=float(partial(rx, ry, rA).mean()), between=float(corr(rank(np.median(x, -1)), rank(np.median(y, -1)))))
    return out


def run(subset):
    sel = {"all": np.ones(long_head.shape, bool), "long": long_head, "other": ~long_head}[subset]
    res = {}
    for di, dl in enumerate(DELTAS):
        base = stats(np.arange(nS), di, sel); boots = [stats(rng.integers(0, nS, nS), di, sel) for _ in range(NB)]
        res[int(dl)] = {k: {m: dict(est=base[k][m], lo=float(np.quantile([b[k][m] for b in boots], 0.025)), hi=float(np.quantile([b[k][m] for b in boots], 0.975))) for m in base[k]} for k in PRED}
    return res


# pooled rank correlation over all cells of a delay (no grouping)
pooled = {}
for di, dl in enumerate(DELTAS):
    ok = [pi for pi, s in enumerate(PROBES) if s + dl <= 1023]; y = Z["Dact"][..., ok, di].reshape(-1)
    pooled[int(dl)] = {k: float(np.corrcoef(rank(Z[k][..., ok, di].reshape(-1)), rank(y))[0, 1]) for k in PRED}
summary = {m: float(np.nanmedian(Z[m][..., 1])) for m in ("Dact", "Dgate", "Dfree")}
json.dump(dict(model=name, n_sequences=int(nS), long_heads=int(long_head.sum()), heads=int(long_head.size), pooled=pooled,
               subsets={s: run(s) for s in ("all", "long", "other")}, median_D_delta64=summary, max_identity_residual=float(np.nanmax(Z["lemma"]))),
          open(f"{HERE}/result_actual_{name}.json", "w"), indent=1)
r = json.load(open(f"{HERE}/result_actual_{name}.json"))
print(name, "sequences", nS, "| long-memory heads", r["long_heads"], "of", r["heads"], "| identity residual", f"{r['max_identity_residual']:.1e}")
for s in ("long", "all"):
    for dl in r["subsets"][s]:
        row = r["subsets"][s][dl]; print(f"[{s:4s}] Delta={dl:>3s} " + "  ".join(f"{k}: w {row[k]['within']['est']:+.2f} p {row[k]['partial']['est']:+.2f} b {row[k]['between']['est']:+.2f}" for k in ("Dgate", "Dfree", "Bkey")))
