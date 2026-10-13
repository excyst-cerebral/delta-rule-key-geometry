"""Different family: softmax-attention Transformers (local cache, offline).  Same analysis as gdn_flat.py / gdn_transfer.py.

An *idealised* delta-rule memory (alpha = beta = 1, one pass) is built from each KV head's keys and values over a window of n consecutive
tokens.  The model does not use such a memory; this tests the geometry (shared hidden-state readouts), not the model's own memory.
Keys: pre-RoPE content keys (primary) and post-RoPE keys (sensitivity).  Rows are L2-normalised.  GQA: one head per KV head.

Usage:  python tf_flat.py <hf_model_dir_name>  e.g.  models--HuggingFaceTB--SmolLM2-135M-Instruct
"""
import sys, os, glob, json, hashlib
os.environ["HF_HUB_OFFLINE"] = "1"
import numpy as np

NAME = sys.argv[1]
MODE = sys.argv[2] if len(sys.argv) > 2 else "both"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); from heldout import HELD, SUFFIX, APPS_RANGE, starts          # opt-in held-out switch (exp_heldout/PROTOCOL.md); off by default
SCRATCH = "scratch/" + NAME.split("--")[-1] + SUFFIX
CACHE = os.path.expanduser("~/.cache/huggingface/hub")
PATH = glob.glob(f"{CACHE}/{NAME}/snapshots/*")[0]
APPS = glob.glob(f"{CACHE}/datasets--codeparrot--apps/snapshots/*/train.jsonl")[0]
T = 1024
ENDS = list(range(255, 1024, 128))
VARS = ["own", "oh", "rot", "gauss", "shuf"]
rng = np.random.default_rng(29)
unit = lambda x: x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def texts():
    prose, code = [], []
    with open(APPS) as f:
        for ln, line in enumerate(f):
            if ln < APPS_RANGE[0]: continue
            if ln >= APPS_RANGE[1]: break
            r = json.loads(line); sols = json.loads(r["solutions"]) if r.get("solutions") else []
            if r["question"] and sols:
                prose.append(r["question"]); code.append(sols[0])
    import pydoc_data.topics as t
    seen, parts = set(), []
    for f in sorted(glob.glob(HERE + "/.venv/lib/python*/site-packages/**/LICEN[CS]E*", recursive=True) + glob.glob(HERE + "/.venv/lib/python*/site-packages/**/COPYING*", recursive=True)):
        if os.path.isfile(f):
            b = open(f, "rb").read(); h = hashlib.sha1(b).hexdigest()
            if h not in seen: seen.add(h); parts.append(b.decode("utf-8", "ignore"))
    return {"prose": "\n\n".join(prose + ["\n"]), "code": "\n\n".join(code), "docs": "\n\n".join(t.topics.values()),
            "legal": "\n\n".join(parts), **({} if HELD else {"notes": open(HERE + "/notes.md").read()})}


def err_op(K):
    n = len(K); G = K @ K.T; L = np.tril(G, -1)
    return L.T @ np.linalg.inv(np.eye(n) + L)


def offdiag(M):
    return M[np.triu_indices(len(M), 1)]


def reff(X):
    s2 = np.linalg.svd(unit(X), compute_uv=False) ** 2
    return s2.sum() ** 2 / (s2 ** 2).sum()


def window_metrics(Kw, Vw):
    n = len(Kw)
    G = Kw @ Kw.T; H = Vw @ Vw.T
    lam, U = np.linalg.eigh(G); lam, U = np.clip(lam[::-1], 1e-12, None), U[:, ::-1]
    Tm = err_op(Kw); TK = Tm @ Kw
    psi = np.linalg.norm(TK) / np.sqrt(n); rms = np.linalg.norm(Tm @ Vw) / np.sqrt(n)
    UtV = U.T @ Vw
    kap = (UtV ** 2).sum(1) / lam
    Mt = (UtV / np.sqrt(lam)[:, None]) @ (UtV / np.sqrt(lam)[:, None]).T
    third = n // 3
    lamH, UH = np.linalg.eigh(H); lamH, UH = np.clip(lamH[::-1], 0, None), UH[:, ::-1]
    m8 = min(8, n // 4)
    og, oh = offdiag(G), offdiag(H)
    return dict(q=kap.mean(), ratio=rms / psi, rms=rms, psi=psi, gmax=np.linalg.eigvalsh(Mt).max(), k_top=kap[:third].mean(),
                k_bot=kap[2 * third:].mean(), lslope=np.polyfit(np.log(lam), np.log(np.clip(kap, 1e-12, None)), 1)[0],
                ovl=float(np.linalg.norm(U[:, :m8].T @ UH[:, :m8]) ** 2 / m8), chance=m8 / n,
                botG=lam[2 * third:].sum() / lam.sum(), botH=lamH[2 * third:].sum() / lamH.sum(),
                c=np.corrcoef(og, oh)[0, 1] if og.std() > 0 and oh.std() > 0 else np.nan,
                dGH=np.linalg.norm(H - G) / np.linalg.norm(G - np.eye(n)))


def rotate_half(x):
    h = x.shape[-1] // 2
    return np.concatenate([-x[..., h:], x[..., :h]], -1)


def extract():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    os.makedirs(SCRATCH, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(PATH)
    model = AutoModelForCausalLM.from_pretrained(PATH, dtype=torch.float32).eval()
    cfg = model.config
    NL, Hq, Hkv = cfg.num_hidden_layers, cfg.num_attention_heads, cfg.num_key_value_heads
    dh = getattr(cfg, "head_dim", None) or cfg.hidden_size // Hq
    f64 = lambda t: t.detach().numpy().astype(np.float64)
    W = {}
    for li in range(NL):
        at = model.model.layers[li].self_attn
        Wk, Wv = f64(at.k_proj.weight), f64(at.v_proj.weight)
        W[f"Wk{li}"], W[f"Wv{li}"] = Wk, Wv
        W[f"bk{li}"] = f64(at.k_proj.bias) if at.k_proj.bias is not None else np.zeros(Wk.shape[0])
        W[f"bv{li}"] = f64(at.v_proj.bias) if at.v_proj.bias is not None else np.zeros(Wv.shape[0])
    seqs = {}
    for dom, txt in texts().items():
        ids = tok(txt, add_special_tokens=False)["input_ids"]
        assert len(ids) >= 3 * T, (dom, len(ids))
        seqs[dom] = [ids[s:s + T] for s in starts(len(ids), T, 3)]
    xs = {}
    for i in range(NL):
        model.model.layers[i].self_attn.k_proj.register_forward_pre_hook(lambda m, a, i=i: xs.__setitem__(i, a[0][0].detach().numpy().copy()))
    with torch.no_grad():
        for di, (dom, ss) in enumerate(seqs.items()):
            for si, seq in enumerate(ss):
                model(torch.tensor([seq]))
                np.save(f"{SCRATCH}/x_{di}_{si}.npy", np.stack([xs[i] for i in range(NL)]).astype(np.float32))
                if di == 0 and si == 0:
                    cos, sin = model.model.rotary_emb(torch.from_numpy(xs[0])[None], torch.arange(T)[None])
                    W["cos"], W["sin"] = cos[0].numpy().astype(np.float64), sin[0].numpy().astype(np.float64)
                print("extracted", dom, si, flush=True)
    np.savez(f"{SCRATCH}/weights.npz", **W)
    json.dump(dict(NL=NL, Hq=Hq, Hkv=Hkv, dh=dh, hidden=cfg.hidden_size, doms=list(seqs), type=cfg.model_type), open(f"{SCRATCH}/meta.json", "w"))


def analyze():
    meta = json.load(open(f"{SCRATCH}/meta.json"))
    NL, Hkv, dh, hidden, doms = meta["NL"], meta["Hkv"], meta["dh"], meta["hidden"], meta["doms"]
    NS = [16, 32] if dh <= 64 else [16, 32, 64]
    print(f"{NAME}: type={meta['type']} layers={NL} kv-heads={Hkv} head_dim={dh} hidden={hidden} windows n={NS}", flush=True)
    Wz = np.load(f"{SCRATCH}/weights.npz")
    cos, sin = Wz["cos"], Wz["sin"]
    prng = np.random.default_rng(5)
    Wd, Ps, Wgs = {}, {}, {}
    for li in range(NL):
        Wk, bk, Wv, bv = Wz[f"Wk{li}"], Wz[f"bk{li}"], Wz[f"Wv{li}"], Wz[f"bv{li}"]
        Wd[li] = (Wk, bk, Wv, bv)
        Ps[li] = np.linalg.qr(prng.normal(size=(hidden, hidden)))[0]
        Wgs[li] = prng.normal(size=Wv.shape) * Wv.std()
    rec = []
    for di, dom in enumerate(doms):
        for si in range(3):
            X = np.load(f"{SCRATCH}/x_{di}_{si}.npy").astype(np.float64)
            for li in range(NL):
                Wk, bk, Wv, bv = Wd[li]
                x = X[li]
                P, Wg = Ps[li], Wgs[li]
                k_pre = (x @ Wk.T + bk).reshape(T, Hkv, dh)
                vs = {"own": x @ Wv.T + bv, "rot": (x @ P) @ Wv.T + bv, "gauss": x @ Wg.T + bv}
                vs = {k: unit(v.reshape(T, Hkv, dh)) for k, v in vs.items()}
                assert np.isfinite(k_pre).all() and all(np.isfinite(v).all() for v in vs.values()), ("non-finite", dom, si, li)
                K = unit(k_pre)
                Krope = unit(k_pre * cos[:, None, :] + rotate_half(k_pre) * sin[:, None, :])
                xn = x
                for h in range(Hkv):
                    h2 = (h + 1 + rng.integers(max(Hkv - 1, 1))) % Hkv if Hkv > 1 else h
                    for n in NS:
                        for end in ENDS:
                            sl = slice(end - n + 1, end + 1)
                            Kw = K[sl, h]
                            Vd = {"own": vs["own"][sl, h], "oh": vs["own"][sl, h2], "rot": vs["rot"][sl, h], "gauss": vs["gauss"][sl, h],
                                  "shuf": vs["own"][sl, h][rng.permutation(n)]}
                            for v in VARS:
                                m = window_metrics(Kw, Vd[v]); m.update(dom=di, seq=si, layer=li, head=h, n=n, var=v)
                                if v == "own":
                                    m["rope"] = window_metrics(Krope[sl, h], Vd["own"])["ratio"]
                                    m["rx"], m["rK"], m["rV"] = reff(xn[sl]), reff(Kw), reff(Vd["own"])
                                    m["rVrot"], m["rVg"] = reff(Vd["rot"]), reff(Vd["gauss"])
                                rec.append(m)
            print(dom, si, "done", flush=True)
    import pickle
    pickle.dump(dict(rec=rec, cfg=dict(NL=NL, Hkv=Hkv, dh=dh, NS=NS), doms=doms), open(f"{HERE}/tf_flat_{NAME.split('--')[-1]}{SUFFIX}.pkl", "wb"))
    print("saved", len(rec))


if __name__ == "__main__":
    if MODE in ("extract", "both"): extract()
    if MODE in ("analyze", "both"): analyze()
