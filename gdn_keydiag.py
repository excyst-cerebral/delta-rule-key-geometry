"""Why are trained keys so coherent?  Dominant direction vs genuinely low rank.

For each layer/head and each n=64 window: raw keys vs keys after subtracting the per-sequence mean
direction and re-normalising.  Reports top-eigenvalue share, mean-direction energy, mu, effective rank,
lambda_min, and the worst recall error of an *ideal* (alpha=beta=1) delta-rule pass on unit-norm values.
Also a random-key reference at the same (n, dk).
"""
import sys, json
sys.path.insert(0, ".")
import numpy as np
import torch
import torch.nn.functional as F
import gdn_extract as g

N = 64
ENDS = g.WIN_ENDS


def keys_of(x, i):
    n = x.shape[0]
    k = g.sconv(x @ g.W[f"model.layers.{i}.attn.k_proj.weight"].T, g.W[f"model.layers.{i}.attn.k_conv1d.weight"])
    return F.normalize(k.view(n, g.H, g.DK), dim=-1, eps=1e-6)


def forward_keys(ids):
    h = g.W["model.embeddings.weight"][ids]
    out = {}
    for i in range(g.NL):
        x = g.rms(h, g.W[f"model.layers.{i}.attn_norm.weight"])
        out[i] = keys_of(x, i).numpy().astype(np.float64)
        a, _ = g.gdn(x, i, False)
        h = h + a
        m = g.rms(h, g.W[f"model.layers.{i}.mlp_norm.weight"])
        gate, up = (m @ g.W[f"model.layers.{i}.mlp.gate_proj.weight"].T).chunk(2, -1)
        h = h + (F.silu(gate) * up) @ g.W[f"model.layers.{i}.mlp.down_proj.weight"].T
    return out


def stats(Kfull, rng):
    """Kfull (T,DK) unit rows -> dict of medians over windows."""
    rows = []
    for end in ENDS:
        Kw = Kfull[end - N + 1:end + 1]
        G = Kw @ Kw.T
        lam = np.linalg.eigvalsh(G)
        off = G - np.eye(N)
        V = rng.normal(size=(N, g.DV)); V /= np.linalg.norm(V, axis=1, keepdims=True)
        S = np.zeros((g.DV, g.DK))
        for t in range(N):
            S += np.outer(V[t] - S @ Kw[t], Kw[t])
        rows.append(dict(top_share=lam[-1] / N, mu=np.abs(off).max(), eff_rank=lam.sum() ** 2 / (lam ** 2).sum(),
                         lam_min=lam[0], F_ideal=np.linalg.norm(Kw @ S.T - V, axis=1).max()))
    return {k: float(np.median([r[k] for r in rows])) for k in rows[0]}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    tok = g.AutoTokenizer.from_pretrained(g.MD)
    data = g.load_sequences(tok, n_per_domain=2)
    long_heads = {d: {tuple(map(int, k.split("|")[1:])) for k, v in json.load(open("results_gdn.json"))["per_head"].items()
                      if k.startswith(d + "|") and v["p256_A"] > 0.5} for d in data}
    res = {"raw": {}, "centered": {}}
    with torch.no_grad():
        for dom, seqs in data.items():
            for seq in seqs:
                Ks = forward_keys(torch.tensor(seq))
                for li, K in Ks.items():
                    for h in range(g.H):
                        Kh = K[:, h]
                        mean_dir = Kh.mean(0)
                        meanE = float(mean_dir @ mean_dir)                       # avg pairwise overlap
                        Kc = Kh - mean_dir
                        Kc /= np.maximum(np.linalg.norm(Kc, axis=1, keepdims=True), 1e-9)
                        for name, KK in (("raw", Kh), ("centered", Kc)):
                            r = stats(KK, rng); r["mean_energy"] = meanE
                            res[name].setdefault((dom, li, h), []).append(r)
                print(dom, "seq done", flush=True)
    # random-key reference
    ref = stats(np.linalg.qr(rng.normal(size=(g.DK, g.DK)))[0][:1024 // 1 if False else g.DK].T[:0] if False else
                (lambda A: A / np.linalg.norm(A, axis=1, keepdims=True))(rng.normal(size=(1024, g.DK))), rng)
    def agg(name, sel):
        keys = [k for k in res[name] if sel(k)]
        return {m: float(np.median([np.median([r[m] for r in res[name][k]]) for k in keys])) for m in res[name][keys[0]][0]}, len(keys)
    print(f"\nWindow n={N}, dk={g.DK}.  Random unit-key reference: " + ", ".join(f"{k}={v:.3g}" for k, v in ref.items()))
    for label, sel in [("ALL heads", lambda k: True),
                       ("LONG-memory heads", lambda k: (k[1], k[2]) in long_heads[k[0]])]:
        print(f"\n--- {label} (medians over heads) ---")
        for name in ("raw", "centered"):
            a, n_h = agg(name, sel)
            print(f"  {name:9s} n_heads={n_h:3d}  top-eig share={a['top_share']:.2f}  mean-direction energy={a['mean_energy']:.2f}  "
                  f"mu={a['mu']:.2f}  eff_rank={a['eff_rank']:.1f}  lam_min={a['lam_min']:.2e}  ideal worst err={a['F_ideal']:.2f}")
