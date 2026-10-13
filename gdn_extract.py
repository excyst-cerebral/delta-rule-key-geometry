"""Measure the quantities in notes.md on a trained Gated-DeltaNet (340M, 24 layers, 4 heads, dk=dv=256).

Pure-PyTorch re-implementation of the fla GatedDeltaNet forward (no repo code is executed):
  q,k,v = silu(causal_dwconv4(proj(x)));  q,k L2-normalised per head;  q *= dk^-0.5
  beta = sigmoid(b_proj x);  alpha = exp(-exp(A_log) * softplus(a_proj x + dt_bias))
  S_t = alpha S_{t-1} + beta (v - alpha S_{t-1} k) k^T     (== alpha S (I - beta k k^T) + beta v k^T)
  o_t = S_t q_t;   out = o_proj( RMSNorm_head(o) * silu(g_proj x) )

Validation: (i) perplexity on held-out text must be sane; (ii) Lemma 1 residual on the real recurrence.
"""
import json, sys, time, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from heldout import APPS_RANGE          # opt-in held-out switch (exp_heldout/PROTOCOL.md); off by default
import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoTokenizer

torch.set_num_threads(10)
import os
MD = os.environ.get("GDN_MD", "model_gdn340m")
APPS = __import__("glob").glob(__import__("os").path.expanduser("~/.cache/huggingface/hub/datasets--codeparrot--apps/snapshots/*/train.jsonl"))[0]
T = 1024
PROBES = [64, 192, 320, 448, 576, 704]
DELTAS = [16, 64, 256]
WIN_ENDS = [255, 511, 767, 1023]
TAUS = [1, 2, 4, 8, 16, 64, 256]

cfg = json.load(open(f"{MD}/config.json"))
H, DK = cfg["num_heads"], cfg["head_dim"]
DV = DK * cfg["expand_v"]
NL, EPS = cfg["num_hidden_layers"], cfg["norm_eps"]
W = {k: v.float() for k, v in load_file(f"{MD}/model.safetensors").items()}


def rms(x, w):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * w


def sconv(u, w):                                    # causal depthwise conv, kernel 4, then SiLU
    y = F.conv1d(u.t().unsqueeze(0), w, padding=w.shape[-1] - 1, groups=u.shape[1])[0, :, :u.shape[0]].t()
    return F.silu(y)


def gdn(x, i, collect):
    p = lambda n: W[f"model.layers.{i}.attn.{n}"]
    n_t = x.shape[0]
    def qkv(nm):                                            # short causal conv + SiLU, or plain SiLU when the checkpoint has no short convolution (use_short_conv = false)
        u = x @ p(f"{nm}_proj.weight").T; cw = f"model.layers.{i}.attn.{nm}_conv1d.weight"
        return sconv(u, W[cw]) if cw in W else F.silu(u)
    q = qkv("q").view(n_t, H, DK)
    k = qkv("k").view(n_t, H, DK)
    v = qkv("v").view(n_t, H, DV)
    q = F.normalize(q, dim=-1, eps=1e-6) * DK ** -0.5
    k = F.normalize(k, dim=-1, eps=1e-6)
    beta = torch.sigmoid(x @ p("b_proj.weight").T)                                   # (T,H)
    g = -torch.exp(p("A_log")) * F.softplus(x @ p("a_proj.weight").T + p("dt_bias"))
    alpha = torch.exp(g)                                                              # (T,H)

    sched = {}
    if collect:
        for s in PROBES:
            sched.setdefault(s, []).append((s, 0))
            for d in DELTAS:
                sched.setdefault(s + d, []).append((s, d))
    snaps = {}
    S = torch.zeros(H, DV, DK)
    outs, Es = [], []
    for t in range(n_t):
        S = S * alpha[t][:, None, None]
        Sk = torch.bmm(S, k[t][:, :, None])[:, :, 0]                                  # alpha S_{t-1} k_t
        e = Sk - v[t]                                                                 # e_t
        S = S - beta[t][:, None, None] * e[:, :, None] * k[t][:, None, :]
        outs.append(torch.bmm(S, q[t][:, :, None])[:, :, 0])
        if collect:
            Es.append(e)
            for (s, d) in sched.get(t, []):
                snaps[(s, d)] = torch.bmm(S, k[s][:, :, None])[:, :, 0].clone()       # S_t k_s
    o = torch.stack(outs)                                                             # (T,H,DV)
    gate = F.silu(x @ p("g_proj.weight").T).view(n_t, H, DV)
    o = rms(o, p("o_norm.weight")) * gate
    y = o.reshape(n_t, H * DV) @ p("o_proj.weight").T
    if not collect:
        return y, None
    data = dict(K=k.numpy().astype(np.float64), V=v.numpy().astype(np.float64),
                E=torch.stack(Es).numpy().astype(np.float64),
                alpha=alpha.numpy().astype(np.float64), beta=beta.numpy().astype(np.float64),
                snaps={kk: vv.numpy().astype(np.float64) for kk, vv in snaps.items()})
    return y, data


def mlp(m, i):
    """SwiGLU MLP; handles a fused gate_proj (gate and up stacked) or separate gate_proj / up_proj."""
    up_key = f"model.layers.{i}.mlp.up_proj.weight"
    if up_key in W:
        gate, up = m @ W[f"model.layers.{i}.mlp.gate_proj.weight"].T, m @ W[up_key].T
    else:
        gate, up = (m @ W[f"model.layers.{i}.mlp.gate_proj.weight"].T).chunk(2, -1)
    return (F.silu(gate) * up) @ W[f"model.layers.{i}.mlp.down_proj.weight"].T


def forward(tokens, collect_layers=()):
    h = W["model.embeddings.weight"][tokens]
    collected = {}
    for i in range(NL):
        a, d = gdn(rms(h, W[f"model.layers.{i}.attn_norm.weight"]), i, i in collect_layers)
        h = h + a
        if d is not None:
            collected[i] = d
        m = rms(h, W[f"model.layers.{i}.mlp_norm.weight"])
        h = h + mlp(m, i)
    h = rms(h, W["model.norm.weight"])
    return h @ W.get("lm_head.weight", W["model.embeddings.weight"]).T, collected


# ----------------------------------------------------------------------------- data
def load_sequences(tok, n_per_domain=4):
    prose, code = [], []
    with open(APPS) as f:
        for ln, line in enumerate(f):
            if ln < APPS_RANGE[0]:
                continue
            if ln >= APPS_RANGE[1]:
                break
            r = json.loads(line)
            sols = json.loads(r["solutions"]) if r.get("solutions") else []
            if r["question"] and sols:
                prose.append(r["question"]); code.append(sols[0])
    def pack(texts, offset):
        seqs, cur = [], []
        for tx in texts[offset:]:
            cur += tok(tx + "\n\n", add_special_tokens=False)["input_ids"]
            while len(cur) >= T - 1:
                seqs.append([tok.bos_token_id] + cur[:T - 1]); cur = cur[T - 1:]
            if len(seqs) >= n_per_domain:
                break
        return seqs[:n_per_domain]
    return {"prose": pack(prose, 0), "code": pack(code, 0)}


# ----------------------------------------------------------------------------- stats
def window_stats(Kh, Vh, n):
    rows = []
    for end in WIN_ENDS:
        Kw, Vw = Kh[end - n + 1:end + 1], Vh[end - n + 1:end + 1]
        G = Kw @ Kw.T
        off = G - np.eye(n)
        mu = np.abs(off).max()
        lam = np.linalg.eigvalsh(G)
        er = lam.sum() ** 2 / (lam ** 2).sum()
        Vn = Vw / np.maximum(np.linalg.norm(Vw, axis=1, keepdims=True), 1e-12)
        sv0 = np.linalg.norm(Vn, 2)
        S = np.zeros((Vn.shape[1], Kw.shape[1]))
        for t in range(n):
            S += np.outer(Vn[t] - S @ Kw[t], Kw[t])
        Fi = np.linalg.norm(Kw @ S.T - Vn, axis=1)
        rho = np.array([np.linalg.norm(off[i, i + 1:]) for i in range(n)])
        lmin = max(lam.min(), 1e-12)
        rows.append(dict(mu=mu, mean_abs=np.abs(off).sum() / (n * (n - 1)), lam_min=lam.min(), eff_rank=er,
                         sv0=sv0, F_ideal=Fi.max(), thm5=(rho * sv0 / np.sqrt(lmin)).max(),
                         thm4_ok=bool(n * mu <= 1)))
    return {k: float(np.median([r[k] for r in rows])) for k in rows[0]}


def probe_stats(d, h, probes_data):
    K, E, al, be = d["K"][:, h], d["E"][:, h], d["alpha"][:, h], d["beta"][:, h]
    out = {dl: [] for dl in DELTAS}
    for s in PROBES:
        r0 = d["snaps"][(s, 0)][h]
        for dl in DELTAS:
            r1 = d["snaps"][(s, dl)][h]
            ts = np.arange(s + 1, s + dl + 1)
            G = K[ts] @ K[s]
            en = np.linalg.norm(E[ts], axis=1)
            A = float(np.prod(al[ts]))
            D = np.linalg.norm(r1 - A * r0)
            B = float((be[ts] * np.abs(G) * en).sum())
            suf = np.concatenate([np.cumprod(al[ts][::-1])[::-1][1:], [1.0]])        # prod_{u>t} alpha_u
            pred = A * r0 - ((suf * be[ts] * G)[:, None] * E[ts]).sum(0)
            n0 = np.linalg.norm(r0)
            out[dl].append(dict(A=A, ret=np.linalg.norm(r1) / n0, cos=float(r1 @ r0 / (np.linalg.norm(r1) * n0 + 1e-30)),
                                D=D / n0, B=B / n0, lemma_res=np.linalg.norm(pred - r1) / n0,
                                bound_ok=bool(D <= B * (1 + 1e-4) + 1e-9)))
    return out


def head_summary(d, h):
    K, V, E, al, be = d["K"][:, h], d["V"][:, h], d["E"][:, h], d["alpha"][:, h], d["beta"][:, h]
    res = {}
    res["alpha_mean"] = float(al.mean()); res["alpha_med"] = float(np.median(al))
    res["halflife_med"] = float(np.median(np.log(2) / np.maximum(-np.log(al), 1e-9)))
    res["beta_mean"] = float(be.mean())
    vn = np.linalg.norm(V, axis=1)
    res["surprise_rel"] = float(np.median(np.linalg.norm(E[32:], axis=1) / np.maximum(vn[32:], 1e-9)))
    for tau in TAUS:
        res[f"c{tau}"] = float(np.abs((K[tau:] * K[:-tau]).sum(1)).mean())
    for n in (16, 64):
        for kk, vv in window_stats(K, V, n).items():
            res[f"w{n}_{kk}"] = vv
    pr = probe_stats(d, h, None)
    for dl in DELTAS:
        for key in ("A", "ret", "cos", "D", "B", "lemma_res"):
            res[f"p{dl}_{key}"] = float(np.median([r[key] for r in pr[dl]]))
        res[f"p{dl}_bound_ok"] = float(np.mean([r["bound_ok"] for r in pr[dl]]))
        res[f"p{dl}_lemma_max"] = float(np.max([r["lemma_res"] for r in pr[dl]]))
    return res


if __name__ == "__main__":
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MD)
    data = load_sequences(tok)
    print({k: [len(s) for s in v] for k, v in data.items()}, flush=True)
    results, ppl = {}, {}
    with torch.no_grad():
        for dom, seqs in data.items():
            ls = []
            for si, seq in enumerate(seqs):
                ids = torch.tensor(seq)
                logits, col = forward(ids, collect_layers=set(range(NL)))
                ls.append(F.cross_entropy(logits[:-1], ids[1:]).item())
                for li, d in col.items():
                    for h in range(H):
                        results.setdefault(f"{dom}|{li}|{h}", []).append(head_summary(d, h))
                print(f"{dom} seq{si} loss={ls[-1]:.3f} ppl={np.exp(ls[-1]):.1f}  [{time.time()-t0:.0f}s]", flush=True)
            ppl[dom] = float(np.exp(np.mean(ls)))
    agg = {k: {m: float(np.median([r[m] for r in v])) for m in v[0]} for k, v in results.items()}
    json.dump(dict(ppl=ppl, per_head=agg), open("results_gdn.json", "w"), indent=1)
    print("ppl", ppl)
