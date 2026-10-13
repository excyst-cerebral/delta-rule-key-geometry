"""Fair re-test of the joint-low-rank prediction: centred windows + canonical correlations.
Per window: centre K and V columns, reduce each to d principal components, compute the canonical correlations of the two score matrices.
Shared m-dim latent  =>  ~m large canonical correlations that exceed the null (shuffled pairing / other head / other layer).
Also the centred participation-ratio rank of [K V] vs r(K)+r(V)."""
import sys, json
sys.path.insert(0, ".")
import numpy as np
import torch
import gdn_extract as g
from gdn_kv import keys_values
from gdn_latent import reff

NS = [32, 64]
ENDS = list(range(255, 1024, 64))
D = 8
rng = np.random.default_rng(41)
unitv = lambda V: V / np.maximum(np.linalg.norm(V, axis=-1, keepdims=True), 1e-12)


def pcs(M, d):
    M = M - M.mean(0, keepdims=True)
    U, s, _ = np.linalg.svd(M, full_matrices=False)
    return U[:, :d]                                   # orthonormal scores


def cancorr(K, V, d=D):
    return np.linalg.svd(pcs(K, d).T @ pcs(V, d), compute_uv=False)


if __name__ == "__main__":
    tok = g.AutoTokenizer.from_pretrained(g.MD)
    data = g.load_sequences(tok, n_per_domain=3)
    per = json.load(open("results_gdn.json"))["per_head"]
    long_heads = {d: {tuple(map(int, k.split("|")[1:])) for k, v in per.items() if k.startswith(d + "|") and v["p256_A"] > 0.5} for d in data}
    res = {n: {s: dict(cc=[], rj=[]) for s in ("real", "shuf", "oh", "ol")} for n in NS}
    with torch.no_grad():
        for dom, seqs in data.items():
            for seq in seqs:
                KV = keys_values(torch.tensor(seq))
                for (li, h) in sorted(long_heads[dom]):
                    K = KV[li][0][:, h]; V = unitv(KV[li][1][:, h])
                    h2 = (h + 1 + rng.integers(g.H - 1)) % g.H; lj = (li + 1 + rng.integers(g.NL - 1)) % g.NL
                    Voh = unitv(KV[li][1][:, h2]); Vol = unitv(KV[lj][1][:, rng.integers(g.H)])
                    for n in NS:
                        for end in ENDS:
                            s0 = end - n + 1
                            Kw = K[s0:end + 1]
                            for name, Vw in (("real", V[s0:end + 1]), ("shuf", V[s0:end + 1][rng.permutation(n)]), ("oh", Voh[s0:end + 1]), ("ol", Vol[s0:end + 1])):
                                res[n][name]["cc"].append(cancorr(Kw, Vw))
                                Kc, Vc = Kw - Kw.mean(0), Vw - Vw.mean(0)
                                res[n][name]["rj"].append(reff(np.hstack([Kc, Vc])) / (reff(Kc) + reff(Vc)))
                print(dom, "seq done", flush=True)
    print(f"\nCentred windows, PCA to d = {D} per side, canonical correlations rho_1..rho_{D} (median over windows)")
    print(f"{'n':>3} {'pairing':>8} | " + " ".join(f"rho{i+1}".rjust(5) for i in range(D)) + " | mean(rho^2) | centred r([K V])/(r(K)+r(V))")
    for n in NS:
        for s in ("real", "shuf", "oh", "ol"):
            cc = np.array(res[n][s]["cc"])
            print(f"{n:3d} {s:>8} | " + " ".join(f"{np.median(cc[:,i]):5.2f}" for i in range(D)) + f" | {np.median((cc**2).mean(1)):11.3f} | {np.median(res[n][s]['rj']):.2f}")
