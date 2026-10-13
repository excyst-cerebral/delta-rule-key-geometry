"""Confound controls for the within-window key-value consistency (long-memory heads, n = 32, 64).
1. lag-binned correlation: corr(key dist, value dist) among pairs at the same position gap (removes temporal smoothness)
2. cross-head control: key distances of head h vs value distances of another head h' of the same layer, same positions
3. closure of the real-vs-shuffled error gap by structure-preserving shuffles:
     token-preserving | class-preserving (ws/digit/punct/alpha) | local blocks of 4 and 8 | global
"""
import sys, json
sys.path.insert(0, ".")
import numpy as np
import torch
import gdn_extract as g
from gdn_kv import keys_values
from gdn_kv_tok import err_op, rms, tok_class, corr

NS = [32, 64]
ENDS = list(range(255, 1024, 64))
BINS = [(1, 1), (2, 3), (4, 8), (9, 32), (33, 10 ** 6)]
rng = np.random.default_rng(5)


def group_shuffle(V, groups, reps=4):
    out = []
    for _ in range(reps):
        V2 = V.copy()
        for gid in np.unique(groups):
            idx = np.where(groups == gid)[0]
            if len(idx) > 1:
                V2[idx] = V[rng.permutation(idx)]
        out.append(V2)
    return out


def dists(K, V):
    n = len(K)
    iu = np.triu_indices(n, 1)
    return np.sqrt(np.clip(2 - 2 * (K @ K.T), 0, None))[iu], np.linalg.norm(V[:, None] - V[None], axis=2)[iu], (iu[1] - iu[0])


if __name__ == "__main__":
    tok = g.AutoTokenizer.from_pretrained(g.MD)
    data = g.load_sequences(tok, n_per_domain=3)
    per = json.load(open("results_gdn.json"))["per_head"]
    long_heads = {d: {tuple(map(int, k.split("|")[1:])) for k, v in per.items() if k.startswith(d + "|") and v["p256_A"] > 0.5} for d in data}
    unitv = lambda V: V / np.maximum(np.linalg.norm(V, axis=-1, keepdims=True), 1e-12)
    lagc = {n: {b: [] for b in BINS} for n in NS}; cross = {n: [] for n in NS}; own = {n: [] for n in NS}
    sh = {n: {k: [] for k in ("real", "global", "token", "class", "block4", "block8")} for n in NS}
    with torch.no_grad():
        for dom, seqs in data.items():
            for seq in seqs:
                ids = np.array(seq)
                cls_all = np.array([tok_class(t) for t in tok.convert_ids_to_tokens(seq)])
                KV = keys_values(torch.tensor(seq))
                for (li, h) in sorted(long_heads[dom]):
                    Kall, Vall = KV[li]
                    K, V = Kall[:, h], unitv(Vall[:, h])
                    h2 = (h + 1 + rng.integers(g.H - 1)) % g.H
                    V2all = unitv(Vall[:, h2])
                    for n in NS:
                        for end in ENDS:
                            s0 = end - n + 1
                            Kw, Vw = K[s0:end + 1], V[s0:end + 1]
                            dk, dv, lag = dists(Kw, Vw)
                            own[n].append(corr(dk, dv))
                            for b in BINS:
                                m = (lag >= b[0]) & (lag <= b[1])
                                lagc[n][b].append(corr(dk[m], dv[m]))
                            _, dv2, _ = dists(Kw, V2all[s0:end + 1])
                            cross[n].append(corr(dk, dv2))
                            T = err_op(Kw)
                            sh[n]["real"].append(rms(T, Vw))
                            tw, cw = ids[s0:end + 1], cls_all[s0:end + 1]
                            pos = np.arange(n)
                            for name, grp in (("global", np.zeros(n, int)), ("token", tw), ("class", np.searchsorted(["alpha", "digit", "punct", "ws"], cw)),
                                              ("block4", pos // 4), ("block8", pos // 8)):
                                sh[n][name].append(np.mean([rms(T, V2) for V2 in group_shuffle(Vw, grp)]))
                print(dom, "seq done", flush=True)
    nm = lambda a: float(np.nanmedian(a))
    print("\n=== 1. correlation(key distance, value distance) within position-gap bins (median over windows) ===")
    print(f"{'n':>3} | {'all pairs':>9} | " + " ".join(f"{('gap '+str(b[0])+('-'+str(b[1]) if b[1]<1e5 else '+')):>10}" for b in BINS) + f" | {'other head values':>17}")
    for n in NS:
        print(f"{n:3d} | {nm(own[n]):9.2f} | " + " ".join(f"{nm(lagc[n][b]):10.2f}" for b in BINS) + f" | {nm(cross[n]):17.2f}")
    print("\n=== 3. structure-preserving shuffles: RMS error and closure of the (global shuffle - real) gap ===")
    print(f"{'n':>3} | {'real':>6} | {'global':>7} | {'token-pres':>10} {'closure':>8} | {'class-pres':>10} {'closure':>8} | {'block-4':>8} {'closure':>8} | {'block-8':>8} {'closure':>8}")
    for n in NS:
        m = {k: float(np.mean(v)) for k, v in sh[n].items()}
        gap = m["global"] - m["real"]
        cl = lambda k: (m["global"] - m[k]) / gap
        print(f"{n:3d} | {m['real']:6.3f} | {m['global']:7.3f} | {m['token']:10.3f} {cl('token'):8.2f} | {m['class']:10.3f} {cl('class'):8.2f} | "
              f"{m['block4']:8.3f} {cl('block4'):8.2f} | {m['block8']:8.3f} {cl('block8'):8.2f}")
