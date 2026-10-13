"""Is a Lipschitz-type constant estimable outside the window?
For pairs with key distance d_k = ||k_i-k_j|| >= 0.1 define ratio = ||v_i-v_j|| / d_k  (unit values).
Compare the 50th/90th percentile of the ratio in a window with the same statistic on the preceding 256 tokens."""
import sys, json
sys.path.insert(0, ".")
import numpy as np
import torch
import gdn_extract as g
from gdn_kv import keys_values

NS = [16, 32, 64]
ENDS = (511, 767, 1023)
CTX = 256


def ratios(K, V, sub=None, rng=None):
    n = len(K)
    iu = np.triu_indices(n, 1)
    dk = np.sqrt(np.clip(2 - 2 * (K @ K.T), 0, None))[iu]
    dv = np.linalg.norm(V[:, None] - V[None], axis=2)[iu]
    m = dk >= 0.1
    r = dv[m] / dk[m]
    if sub is not None and len(r) > sub:
        r = rng.choice(r, sub, replace=False)
    return r, dk[m], dv[m]


if __name__ == "__main__":
    rng = np.random.default_rng(3)
    tok = g.AutoTokenizer.from_pretrained(g.MD)
    data = g.load_sequences(tok, n_per_domain=2)
    per = json.load(open("results_gdn.json"))["per_head"]
    long_heads = {d: {tuple(map(int, k.split("|")[1:])) for k, v in per.items() if k.startswith(d + "|") and v["p256_A"] > 0.5} for d in data}
    unitv = lambda V: V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)
    out = {n: [] for n in NS}
    with torch.no_grad():
        for dom, seqs in data.items():
            for seq in seqs:
                KV = keys_values(torch.tensor(seq))
                for (li, h) in sorted(long_heads[dom]):
                    K, V = KV[li][0][:, h], unitv(KV[li][1][:, h])
                    for n in NS:
                        for end in ENDS:
                            s0 = end - n + 1
                            rw, dkw, dvw = ratios(K[s0:end + 1], V[s0:end + 1])
                            rc, _, _ = ratios(K[s0 - CTX:s0], V[s0 - CTX:s0], sub=2000, rng=rng)
                            # shuffled-pairing window for reference
                            Vs = V[s0:end + 1][rng.permutation(n)]
                            rs, _, _ = ratios(K[s0:end + 1], Vs)
                            out[n].append((np.median(rw), np.quantile(rw, .9), np.median(rc), np.quantile(rc, .9), np.median(rs), np.quantile(rs, .9)))
                print(dom, "seq done", flush=True)
    rank = lambda a: np.argsort(np.argsort(a)).astype(float)
    spearman = lambda a, b: float(np.corrcoef(rank(a), rank(b))[0, 1])
    print(f"\nlong-memory heads, {len(out[NS[0]])} windows per n, medians (ratio = ||dv||/||dk|| over pairs with ||dk||>=0.1)")
    print(f"{'n':>3} | {'window med':>10} {'window p90':>10} | {'context med':>11} {'context p90':>11} | {'shuffled med':>12} {'shuffled p90':>12} | "
          f"{'spearman(win p90, ctx p90)':>26} {'spearman(med)':>13} | {'median |log(ctx p90 / win p90)|':>30}")
    for n in NS:
        a = np.array(out[n])
        print(f"{n:3d} | {np.median(a[:,0]):10.2f} {np.median(a[:,1]):10.2f} | {np.median(a[:,2]):11.2f} {np.median(a[:,3]):11.2f} | "
              f"{np.median(a[:,4]):12.2f} {np.median(a[:,5]):12.2f} | {spearman(a[:,1], a[:,3]):26.2f} {spearman(a[:,0], a[:,2]):13.2f} | "
              f"{np.median(np.abs(np.log(a[:,3] / a[:,1]))):30.2f}")
