"""RWKV-7 'Goose' (fla-hub/rwkv7-0.1B-g1: 12 layers, hidden 768, 12 heads x 64): same analysis as gdn_flat.py, from scratch in NumPy.

RWKV-7 time-mix (per head, state S in R^{64 x 64}, rows = value dim, columns = key dim):
    S_t = S_{t-1} ( diag(w_t) - kk_t (a_t * kk_t)^T ) + v_t k~_t^T ,        o_t = S_t r_t
    w_t = exp(-0.6065 * sigmoid(lora_w(xw)))         per key channel,  in (0.545, 1)
    a_t = sigmoid(lora_a(xa))                        in-context learning rate (per channel)
    kk_t = L2norm_head(k_t * k_k)                    REMOVAL key (unit norm)           <- 'kappa' keys
    k~_t = k_t * (1 + (a_t - 1) * k_a)               WRITE key (not normalised)        <- 'write' keys
    v_t  = v_proj(xv) (+ lerp toward layer-0 v with sigmoid(lora_v(xv)) for layers >= 1)
The ideal alpha = beta = 1 delta-rule pass is run with a unit key (kappa or normalised write key) and unit-norm values.
No torch in the analysis process; weights are converted once with torch ('convert').
Usage: python rwkv7_flat.py convert | check | run
"""
import sys, os, json, glob, hashlib, ast, pickle, time
import numpy as np

MODE = sys.argv[1]
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); from heldout import HELD, SUFFIX, APPS_RANGE, starts          # opt-in held-out switch (exp_heldout/PROTOCOL.md); off by default
PATH = HERE + "/model_rwkv7_01b"
SCRATCH = "scratch/rwkv7"
APPS = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/datasets--codeparrot--apps/snapshots/*/train.jsonl"))[0]
T, NSEQ = 1024, 2
ENDS = list(range(255, 1024, 128))
NS = [16, 32]
VARS = ["own", "oh", "rot", "gauss", "shuf"]
NL, D, H, N = 12, 768, 12, 64
DECAY_C = 0.6065306597126334
BONUS = True
REMOVE = True
VRES = True


def convert():
    import torch
    from safetensors import safe_open
    os.makedirs(SCRATCH, exist_ok=True)
    f = safe_open(f"{PATH}/model.safetensors", "pt")
    np.savez(f"{SCRATCH}/weights.npz", **{k.replace(".", "__"): f.get_tensor(k).float().numpy() for k in f.keys()})
    print("converted", len(list(f.keys())), "tensors")


# ------------------------------------------------------------------ tokenizer (RWKV World, greedy longest match; own implementation)
def load_tokenizer():
    trie = {}
    for line in open(f"{PATH}/rwkv_vocab_v20230424.txt", encoding="utf-8"):
        idx = int(line[:line.index(" ")]); x = ast.literal_eval(line[line.index(" "):line.rindex(" ")])
        x = x.encode("utf-8") if isinstance(x, str) else x
        assert len(x) == int(line[line.rindex(" "):])
        node = trie
        for b in x: node = node.setdefault(b, {})
        node[None] = idx
    return trie


def encode(trie, text):
    b = text.encode("utf-8"); out = []; i = 0
    while i < len(b):
        node, j, last = trie, i, None
        while j < len(b) and b[j] in node:
            node = node[b[j]]; j += 1
            if None in node: last = (node[None], j)
        out.append(last[0]); i = last[1]
    return out


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
    for fn in sorted(glob.glob(HERE + "/.venv/lib/python*/site-packages/**/LICEN[CS]E*", recursive=True) + glob.glob(HERE + "/.venv/lib/python*/site-packages/**/COPYING*", recursive=True)):
        if os.path.isfile(fn):
            bb = open(fn, "rb").read(); h = hashlib.sha1(bb).hexdigest()
            if h not in seen: seen.add(h); parts.append(bb.decode("utf-8", "ignore"))
    return {"prose": "\n\n".join(prose), "code": "\n\n".join(code), "docs": "\n\n".join(t.topics.values()),
            "legal": "\n\n".join(parts), **({} if HELD else {"notes": open(HERE + "/notes.md").read()})}


# ------------------------------------------------------------------ model (float64)
sigmoid = lambda a: 1 / (1 + np.exp(-a))


def ln(x, w, b, eps=1e-5):
    m = x.mean(-1, keepdims=True); v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps) * w + b


class Model:
    def __init__(self):
        z = np.load(f"{SCRATCH}/weights.npz")
        self.W = {k.replace("__", "."): z[k].astype(np.float64) for k in z.files}

    def p(self, l, name):
        return self.W[f"model.layers.{l}.{name}"]

    def forward(self, ids, capture=False):
        W = self.W
        h = W["model.embeddings.weight"][ids]
        h = ln(h, W["model.layers.0.pre_norm.weight"], W["model.layers.0.pre_norm.bias"])
        Tn = len(ids); v_first = None; cap = {}
        for l in range(NL):
            P = lambda n: self.p(l, "attn." + n)
            xin = ln(h, self.p(l, "attn_norm.weight"), self.p(l, "attn_norm.bias"))
            sh = np.concatenate([np.zeros((1, D)), xin[:-1]]); dl = sh - xin
            mix = {k: xin + dl * P("x_" + k)[0, 0] for k in "rwkvag"}
            r = mix["r"] @ P("r_proj.weight").T; k = mix["k"] @ P("k_proj.weight").T; v = mix["v"] @ P("v_proj.weight").T
            wl = -DECAY_C * sigmoid(np.tanh(mix["w"] @ P("w_lora.lora.0.weight").T) @ P("w_lora.lora.2.weight").T + P("w_lora.lora.2.bias"))
            a = sigmoid((mix["a"] @ P("a_lora.lora.0.weight").T) @ P("a_lora.lora.2.weight").T + P("a_lora.lora.2.bias"))
            g = sigmoid(mix["g"] @ P("g_lora.lora.0.weight").T) @ P("g_lora.lora.2.weight").T
            v_pre = v
            if l == 0: v_first = v
            elif VRES: v = v + (v_first - v) * sigmoid((mix["v"] @ P("v_lora.lora.0.weight").T) @ P("v_lora.lora.2.weight").T + P("v_lora.lora.2.bias"))
            kk = (k * P("k_k")).reshape(Tn, H, N); kk = kk / np.maximum(np.linalg.norm(kk, axis=-1, keepdims=True), 1e-12)
            kt = k * (1 + (a - 1) * P("k_a"))
            r_, kt_, v_, a_, w_ = (z.reshape(Tn, H, N) for z in (r, kt, v, a, wl))
            S = np.zeros((H, N, N)); o = np.empty((Tn, H, N)); dec = np.exp(w_)
            for t in range(Tn):
                sa = np.einsum("hik,hk->hi", S, -kk[t]) if REMOVE else np.zeros((H, N))
                S = S * dec[t][:, None, :] + sa[:, :, None] * (kk[t] * a_[t])[:, None, :] + v_[t][:, :, None] * kt_[t][:, None, :]
                o[t] = np.einsum("hij,hj->hi", S, r_[t])
            # group norm over each head's 64 channels, per token (eps = head_dim * 1e-5)
            mu = o.mean(-1, keepdims=True); var = ((o - mu) ** 2).mean(-1, keepdims=True)
            o = ((o - mu) / np.sqrt(var + N * 1e-5)).reshape(Tn, D) * P("g_norm.weight") + P("g_norm.bias")
            if BONUS: o = o + ((r_ * kt_ * P("r_k")).sum(-1, keepdims=True) * v_).reshape(Tn, D)
            h = h + (o * g) @ P("o_proj.weight").T
            if capture:
                cap[l] = dict(kappa=kk, write=kt_ / np.maximum(np.linalg.norm(kt_, axis=-1, keepdims=True), 1e-12), v=v_, v_pre=v_pre.reshape(Tn, H, N),
                              xin=xin, xv=mix["v"], logw=w_, a=a_)
            xf = ln(h, self.p(l, "ffn_norm.weight"), self.p(l, "ffn_norm.bias"))
            sh = np.concatenate([np.zeros((1, D)), xf[:-1]])
            kx = xf + (sh - xf) * self.p(l, "ffn.x_k")
            h = h + (np.maximum(kx @ self.p(l, "ffn.key.weight").T, 0) ** 2) @ self.p(l, "ffn.value.weight").T
        h = ln(h, W["model.norm.weight"], W["model.norm.bias"])
        return h, cap

    def loss(self, ids):
        h, _ = self.forward(ids)
        Wl = self.W["lm_head.weight"]; tot = 0.0
        for s in range(0, len(ids) - 1, 128):
            lg = h[s:min(s + 128, len(ids) - 1)] @ Wl.T
            lg = lg - lg.max(-1, keepdims=True)
            lse = np.log(np.exp(lg).sum(-1))
            tot += (lse - lg[np.arange(len(lg)), ids[s + 1:s + 1 + len(lg)]]).sum()
        return tot / (len(ids) - 1)


def check():
    trie = load_tokenizer(); M = Model(); tx = texts()
    for dom in ("prose", "code", "notes"):
        ids = encode(trie, tx[dom])[:512]
        t0 = time.time(); l = M.loss(np.array(ids))
        print(f"{dom}: {len(ids)} tokens, loss {l:.3f}, ppl {np.exp(l):.1f}  [{time.time()-t0:.0f}s]", flush=True)
    print("tokenizer round-trip example:", encode(trie, "Hello, world! 你好")[:12])


def run():
    sys.path.insert(0, HERE)
    from flatlib import unit, window_metrics, reff
    trie = load_tokenizer(); M = Model(); tx = texts()
    rng = np.random.default_rng(37); prng = np.random.default_rng(5)
    Ps = {l: np.linalg.qr(prng.normal(size=(D, D)))[0] for l in range(NL)}
    Wg = {l: prng.normal(size=(D, D)) * M.p(l, "attn.v_proj.weight").std() for l in range(NL)}
    seqs = {}
    for dom, txt in tx.items():
        ids = encode(trie, txt)
        assert len(ids) >= NSEQ * T, (dom, len(ids))
        seqs[dom] = [ids[s:s + T] for s in starts(len(ids), T, NSEQ)]
    rec, stat = [], {}
    for di, (dom, ss) in enumerate(seqs.items()):
        for si, seq in enumerate(ss):
            t0 = time.time()
            _, cap = M.forward(np.array(seq), capture=True)
            for l in range(NL):
                c = cap[l]
                Kk, Kw = c["kappa"], c["write"]
                Vs = {"own": unit(c["v"]), "rot": unit(((c["xv"] @ Ps[l]) @ M.p(l, "attn.v_proj.weight").T).reshape(T, H, N)),
                      "gauss": unit((c["xv"] @ Wg[l].T).reshape(T, H, N))}
                A = {}
                lw = c["logw"]                                                   # (T,H,N) log-decay per key channel
                for Dd in (16, 64):
                    cs = np.concatenate([np.zeros((1, H, N)), lw.cumsum(0)]); t = np.arange(64, T - 64)
                    A[Dd] = np.exp(cs[t + Dd + 1] - cs[t + 1])                  # (t,H,N) per-channel retention
                stat.setdefault(l, []).append(dict(a=c["a"].mean((0, 2)), A16=A[16].mean((0, 2)), A64=A[64].mean((0, 2)),
                                                   A64_med=np.median(A[64], axis=(0, 2)), frac_slow=(A[64] > 0.5).mean((0, 2))))
                for h in range(H):
                    h2 = (h + 1 + rng.integers(H - 1)) % H
                    sub = (h % 3 == 0)
                    for n in NS:
                        for end in ENDS:
                            sl = slice(end - n + 1, end + 1)
                            Vd = {"own": Vs["own"][sl, h]}
                            if sub:
                                Vd.update({"oh": Vs["own"][sl, h2], "rot": Vs["rot"][sl, h], "gauss": Vs["gauss"][sl, h], "shuf": Vs["own"][sl, h][rng.permutation(n)]})
                            for kname, Kfull in (("kappa", Kk), ("write", Kw)):
                                if kname == "write" and not (sub or True): continue
                                for v, Vw in Vd.items():
                                    if kname == "write" and v not in ("own", "shuf"): continue
                                    m = window_metrics(Kfull[sl, h], Vw); m.update(dom=di, seq=si, layer=l, head=h, n=n, var=v, key=kname)
                                    if v == "own" and kname == "kappa" and sub:
                                        m["rx"], m["rK"], m["rV"] = reff(c["xin"][sl]), reff(Kfull[sl, h]), reff(Vd["own"])
                                        m["rVrot"], m["rVg"], m["rKw"] = reff(Vd["rot"]), reff(Vd["gauss"]), reff(Kw[sl, h])
                                    rec.append(m)
            print(dom, si, f"done [{time.time()-t0:.0f}s]", flush=True)
    pickle.dump(dict(rec=rec, stat=stat, doms=list(seqs)), open(f"{HERE}/rwkv7_flat{SUFFIX}.pkl", "wb"))
    print("saved", len(rec))


if __name__ == "__main__":
    {"convert": convert, "check": check, "run": run}[MODE]()
