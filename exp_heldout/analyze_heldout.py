"""Statistics of exp_heldout/PROTOCOL.md: coverage of the published r90 constants on held-out sequences.  numpy only.  usage: python -I exp_heldout/analyze_heldout.py
Writes exp_heldout/result_heldout.json.  Run from the repository root after the held-out runs (HELDOUT=1 on the measurement pipelines)."""
import sys, json, pickle, os
import numpy as np
sys.path.insert(0, "figgen")
import data as D

MODELS = [("GDN-340M", "results_transfer_model_gdn340m_heldout.npz", None), ("GDN-360M", "results_transfer_model_gdn360m_heldout.npz", None), ("Q3.5", "qwen35_flat_heldout.pkl", None),
          ("R7", "rwkv7_flat_heldout.pkl", "kappa"), ("Q2.5", "tf_flat_Qwen2.5-1.5B-Instruct_heldout.pkl", None), ("S2", "tf_flat_SmolLM2-135M-Instruct_heldout.pkl", None)]
DOMS = ["prose", "code", "docs", "legal", "notes"]
rng = np.random.default_rng(0)


def load(f, sel):
    if f.endswith(".npz"):
        A = np.load(f)["A"]; return dict(n=A[:, 5].astype(int), rms=A[:, 6], psi=A[:, 7], seq=A[:, 0].astype(int) * 100 + A[:, 1].astype(int))
    r = [x for x in pickle.load(open(f, "rb"))["rec"] if x["var"] == "own" and (sel is None or str(x.get("key")) == sel)]
    dom = lambda x: x["dom"] if isinstance(x["dom"], (int, np.integer)) else DOMS.index(x["dom"])
    return dict(n=np.array([x["n"] for x in r]), rms=np.array([x["rms"] for x in r]), psi=np.array([x["psi"] for x in r]), seq=np.array([dom(x) * 100 + int(x["seq"]) for x in r]))


def boot_cov(ratio, seq, r90, B=2000):
    ids = np.unique(seq); groups = [ratio[seq == s] for s in ids]; out = []
    for _ in range(B):
        pick = rng.integers(0, len(ids), len(ids)); out.append(float((np.concatenate([groups[i] for i in pick]) <= r90).mean()))
    return [float(x) for x in np.percentile(out, [2.5, 97.5])]


CACHE = "exp_heldout/heldout_windows.npz"          # per-window arrays (n, rms, psi, sequence id) of the held-out runs; the raw pickles are large and not distributed
cached = dict(np.load(CACHE)) if os.path.exists(CACHE) else {}
res = {}; store = {}
for k, f, sel in MODELS:
    if os.path.exists(f): d = load(f, sel)
    elif f"{k}|n" in cached: d = {a: cached[f"{k}|{a}"] for a in ("n", "rms", "psi", "seq")}
    else: print("missing", f); continue
    store.update({f"{k}|{a}": v for a, v in d.items()}); res[k] = {}
    for n in (16, 32):
        m = d["n"] == n
        if not m.any(): continue
        ratio = d["rms"][m] / d["psi"][m]; seq = d["seq"][m]; r_pub = D.r90(k, n)
        cov = float((ratio <= r_pub).mean()); lo, hi = boot_cov(ratio, seq, r_pub)
        res[k][str(n)] = dict(r90_published=r_pub, r90_heldout=float(np.quantile(ratio, 0.9)), coverage=cov, ci=[lo, hi], windows=int(m.sum()), sequences=int(len(np.unique(seq))),
                              transfers=bool((lo <= 0.9 <= hi) or (lo >= 0.85 and hi <= 0.95)))
    print(k, {n: f"cov {v['coverage']:.3f} [{v['ci'][0]:.3f},{v['ci'][1]:.3f}] r90 pub {v['r90_published']:.2f} held {v['r90_heldout']:.2f} ({v['sequences']} seq) {'ok' if v['transfers'] else 'NO'}" for n, v in res[k].items()})
json.dump(res, open("exp_heldout/result_heldout.json", "w"), indent=1)
if store and not os.path.exists(CACHE): np.savez_compressed(CACHE, **store)
