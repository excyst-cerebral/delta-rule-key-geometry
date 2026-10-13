"""Qwen3.5-0.8B-Base (hybrid; 18 Gated DeltaNet layers x 16 heads, dk = dv = 128): same analysis as gdn_flat.py / gdn_transfer.py.
Stage 'extract' (torch only): wraps the native delta-rule function to capture the exact (L2-normalised) keys, values, log-decay g and
beta, plus each layer's input x.  Stage 'analyze' (numpy only): recomputes keys/values from x and the weights (validated against the
captured ones), builds the control value branches (rotated hidden state, Gaussian weights, other head, shuffled) and the window metrics.
Usage: python qwen35_flat.py extract|analyze
"""
import sys, os, json, glob, hashlib, pickle
import numpy as np

MODE = sys.argv[1]
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); from heldout import HELD, SUFFIX, APPS_RANGE, starts          # opt-in held-out switch (exp_heldout/PROTOCOL.md); off by default
PATH = HERE + "/model_qwen35_08b"
SCRATCH = "scratch/qwen35" + SUFFIX
APPS = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/datasets--codeparrot--apps/snapshots/*/train.jsonl"))[0]
T, NSEQ = 1024, 2
ENDS = list(range(255, 1024, 128))
NS = [16, 32, 64]
VARS = ["own", "oh", "rot", "gauss", "shuf"]


def texts():
    prose, code = [], []
    with open(APPS) as f:
        for ln, line in enumerate(f):
            if ln < APPS_RANGE[0]: continue
            if ln >= APPS_RANGE[1]: break
            r = json.loads(line); sols = json.loads(r["solutions"]) if r.get("solutions") else []
            if r["question"] and sols: prose.append(r["question"]); code.append(sols[0])
    import pydoc_data.topics as t
    seen, parts = set(), []
    for f in sorted(glob.glob(HERE + "/.venv/lib/python*/site-packages/**/LICEN[CS]E*", recursive=True) + glob.glob(HERE + "/.venv/lib/python*/site-packages/**/COPYING*", recursive=True)):
        if os.path.isfile(f):
            b = open(f, "rb").read(); h = hashlib.sha1(b).hexdigest()
            if h not in seen: seen.add(h); parts.append(b.decode("utf-8", "ignore"))
    return {"prose": "\n\n".join(prose), "code": "\n\n".join(code), "docs": "\n\n".join(t.topics.values()),
            "legal": "\n\n".join(parts), **({} if HELD else {"notes": open(HERE + "/notes.md").read()})}


def extract():
    os.environ["HF_HUB_OFFLINE"] = "1"
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import transformers.models.qwen3_5.modeling_qwen3_5 as mq
    os.makedirs(SCRATCH, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(PATH)
    model = AutoModelForCausalLM.from_pretrained(PATH, dtype=torch.float32).eval()
    cfg = model.config.get_text_config()
    gdn_layers = [i for i, t in enumerate(cfg.layer_types) if t == "linear_attention"]
    cur = {"layer": None}; cap = {}
    orig = mq.torch_chunk_gated_delta_rule

    def wrapper(query, key, value, g=None, beta=None, **kw):
        kn = key * torch.rsqrt((key * key).sum(-1, keepdim=True) + 1e-6)             # same l2norm as the native kernel
        cap[cur["layer"]] = dict(K=kn[0].numpy().copy(), V=value[0].numpy().copy(), g=g[0].numpy().copy(), beta=beta[0].numpy().copy())
        return orig(query, key, value, g=g, beta=beta, **kw)
    mq.torch_chunk_gated_delta_rule = wrapper
    for i in gdn_layers:
        def pre(mod, args, kwargs, i=i):
            cur["layer"] = i
            cap.setdefault(("x", i), (kwargs.get("hidden_states") if "hidden_states" in kwargs else args[0])[0].numpy().copy())
        model.model.layers[i].linear_attn.register_forward_pre_hook(pre, with_kwargs=True)
    # weights needed by the numpy stage
    W = {}
    for i in gdn_layers:
        la = model.model.layers[i].linear_attn
        W[f"qkv{i}"] = la.in_proj_qkv.weight.detach().numpy().copy()
        W[f"conv{i}"] = la.conv1d.weight.detach().numpy()[:, 0, :].copy()
    np.savez(f"{SCRATCH}/weights.npz", **W)
    seqs = {}
    for dom, txt in texts().items():
        ids = tok(txt, add_special_tokens=False)["input_ids"]
        assert len(ids) >= NSEQ * T, (dom, len(ids))
        seqs[dom] = [ids[s:s + T] for s in starts(len(ids), T, NSEQ)]
    with torch.no_grad():
        for di, (dom, ss) in enumerate(seqs.items()):
            for si, seq in enumerate(ss):
                cap.clear()
                out = model(torch.tensor([seq]), labels=torch.tensor([seq]))
                d = dict(loss=np.array(out.loss.item()))
                for i in gdn_layers:
                    d[f"x{i}"] = cap[("x", i)]
                    d[f"g{i}"], d[f"b{i}"] = cap[i]["g"], cap[i]["beta"]
                    if di == 0 and si == 0:
                        d[f"K{i}"], d[f"V{i}"] = cap[i]["K"], cap[i]["V"]
                np.savez(f"{SCRATCH}/seq_{di}_{si}.npz", **d)
                print("extracted", dom, si, "ppl", round(float(np.exp(out.loss.item())), 2), flush=True)
    json.dump(dict(gdn_layers=gdn_layers, doms=list(seqs), H=cfg.linear_num_value_heads, dk=cfg.linear_key_head_dim, dv=cfg.linear_value_head_dim,
                   hidden=cfg.hidden_size, kdim=cfg.linear_num_key_heads * cfg.linear_key_head_dim), open(f"{SCRATCH}/meta.json", "w"))


def analyze():
    sys.path.insert(0, HERE)
    from flatlib import unit, err_op, offdiag, reff, window_metrics
    meta = json.load(open(f"{SCRATCH}/meta.json")); Wz = np.load(f"{SCRATCH}/weights.npz")
    L, doms, H, dk, dv, D, kdim = meta["gdn_layers"], meta["doms"], meta["H"], meta["dk"], meta["dv"], meta["hidden"], meta["kdim"]
    rng = np.random.default_rng(31); prng = np.random.default_rng(5)
    silu = lambda a: a / (1 + np.exp(-a))

    def conv_silu(u, w):            # causal depthwise conv (kernel 4) + SiLU; u (T, C), w (C, 4)
        up = np.concatenate([np.zeros((3, u.shape[1])), u])
        return silu(sum(up[j:j + len(u)] * w[:, j] for j in range(4)))

    Ps = {i: np.linalg.qr(prng.normal(size=(D, D)))[0] for i in L}
    Wg = {i: prng.normal(size=(Wz[f"qkv{i}"].shape[0] - 2 * kdim, D)) * Wz[f"qkv{i}"][2 * kdim:].std() for i in L}
    rec, val = [], {}
    for di, dom in enumerate(doms):
        for si in range(NSEQ):
            S = np.load(f"{SCRATCH}/seq_{di}_{si}.npz")
            for i in L:
                x = S[f"x{i}"].astype(np.float64); Wq, Wc = Wz[f"qkv{i}"].astype(np.float64), Wz[f"conv{i}"].astype(np.float64)
                kc = conv_silu(x @ Wq[kdim:2 * kdim].T, Wc[kdim:2 * kdim])
                K = unit(kc.reshape(T, H, dk))
                v_own = conv_silu(x @ Wq[2 * kdim:].T, Wc[2 * kdim:])
                v_rot = conv_silu((x @ Ps[i]) @ Wq[2 * kdim:].T, Wc[2 * kdim:])
                v_g = conv_silu(x @ Wg[i].T, Wc[2 * kdim:])
                if di == 0 and si == 0:
                    val[i] = (np.abs(K - S[f"K{i}"]).max(), np.abs(v_own.reshape(T, H, dv) - S[f"V{i}"]).max())
                Vs = {"own": unit(v_own.reshape(T, H, dv)), "rot": unit(v_rot.reshape(T, H, dv)), "gauss": unit(v_g.reshape(T, H, dv))}
                lg = S[f"g{i}"].astype(np.float64); cs = np.concatenate([np.zeros((1, H)), lg.cumsum(0)])
                A = {}
                for Dd in (64, 256):
                    t = np.arange(64, T - 256); A[Dd] = np.median(np.exp(cs[t + Dd + 1] - cs[t + 1]), axis=0)
                for h in range(H):
                    h2 = (h + 1 + rng.integers(H - 1)) % H
                    sub = (h % 4 == 0)                                     # controls on every 4th head (cost)
                    for n in NS:
                        for end in ENDS:
                            sl = slice(end - n + 1, end + 1)
                            Kw = K[sl, h]
                            Vd = {"own": Vs["own"][sl, h]}
                            if sub:
                                Vd.update({"oh": Vs["own"][sl, h2], "rot": Vs["rot"][sl, h], "gauss": Vs["gauss"][sl, h], "shuf": Vs["own"][sl, h][rng.permutation(n)]})
                            for v, Vw in Vd.items():
                                m = window_metrics(Kw, Vw); m.update(dom=di, seq=si, layer=i, head=h, n=n, var=v, A64=A[64][h], A256=A[256][h])
                                if v == "own" and sub:
                                    m["rx"], m["rK"], m["rV"] = reff(x[sl]), reff(Kw), reff(Vd["own"])
                                    m["rVrot"], m["rVg"] = reff(Vd["rot"]), reff(Vd["gauss"])
                                rec.append(m)
            print(dom, si, "done", flush=True)
    pickle.dump(dict(rec=rec, validation=val, L=L, doms=doms), open(f"{HERE}/qwen35_flat{SUFFIX}.pkl", "wb"))
    print("max |K - captured K|, |V - captured V| over layers (seq 0):", max(v[0] for v in val.values()), max(v[1] for v in val.values()))
    print("saved", len(rec))


if __name__ == "__main__":
    extract() if MODE == "extract" else analyze()
