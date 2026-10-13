"""Is the within-window key-value consistency just repeated tokens?   (long-memory heads, ideal alpha=beta=1 pass)

A. pair decomposition   corr(pairwise key distance, pairwise value distance) over: all / different-token / different-token & same
                         coarse class / different-token & different class / same-token pairs.
B. selective shuffles   R = positions whose token occurs >=2x in the window, U = unique-token positions.
                         shuffle-all | keep R real (shuffle U among U) | keep U real (shuffle R among R) | shuffle within R and U separately
C. de-duplicated        windows made only of distinct tokens (first occurrences in a 3n-token span): real vs shuffled error, correlation
D. token lookup         predict a position's value by the mean value of earlier same-token positions in the preceding 256 tokens
"""
import sys, json, re
sys.path.insert(0, ".")
import numpy as np
import torch
import gdn_extract as g
from gdn_kv import keys_values
from gdn_kv_nl import nw_predict, choose_kappa

NS = [16, 32, 64]
ENDS = list(range(255, 1024, 64))
CTX = 256
rng = np.random.default_rng(21)


def tok_class(t):
    t = t.replace("▁", "")
    if t == "" or t.startswith("<0x"):
        return "ws"
    if t.isdigit():
        return "digit"
    if not any(ch.isalnum() for ch in t):
        return "punct"
    return "alpha"


def err_op(K):
    n = len(K)
    G = K @ K.T
    L = np.tril(G, -1)
    return L.T @ np.linalg.inv(np.eye(n) + L)


def rms(T, V):
    return np.linalg.norm(T @ V) / np.sqrt(len(V))


def shuffled(V, idx, reps=4):
    """mean RMS-ready list of value matrices with positions `idx` permuted among themselves."""
    out = []
    for _ in range(reps):
        V2 = V.copy()
        if len(idx) > 1:
            V2[idx] = V[rng.permutation(idx)]
        out.append(V2)
    return out


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1]) if len(a) >= 8 and a.std() > 0 and b.std() > 0 else np.nan


def pair_stats(K, V, toks, cls):
    n = len(K)
    iu = np.triu_indices(n, 1)
    dk = np.sqrt(np.clip(2 - 2 * (K @ K.T), 0, None))[iu]
    dv = np.linalg.norm(V[:, None] - V[None], axis=2)[iu]
    same = (toks[:, None] == toks[None, :])[iu]
    samec = (cls[:, None] == cls[None, :])[iu]
    D = ~same
    near = (K @ K.T)[iu] > 0.9
    return dict(p_same=same.mean(), c_all=corr(dk, dv), c_diff=corr(dk[D], dv[D]),
                c_diff_samecls=corr(dk[D & samec], dv[D & samec]), c_diff_diffcls=corr(dk[D & ~samec], dv[D & ~samec]),
                c_same=corr(dk[same], dv[same]), dk_same=dk[same].mean() if same.any() else np.nan, dk_diff=dk[D].mean(),
                dv_same=dv[same].mean() if same.any() else np.nan, dv_diff=dv[D].mean(),
                near_same=float(same[near].mean()) if near.any() else np.nan, p_near=near.mean())


if __name__ == "__main__":
    tok = g.AutoTokenizer.from_pretrained(g.MD)
    data = g.load_sequences(tok, n_per_domain=3)
    per = json.load(open("results_gdn.json"))["per_head"]
    long_heads = {d: {tuple(map(int, k.split("|")[1:])) for k, v in per.items() if k.startswith(d + "|") and v["p256_A"] > 0.5} for d in data}
    unitv = lambda V: V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)
    cls_of = {}
    A = {n: [] for n in NS}; B = {n: [] for n in NS}; C = {n: [] for n in NS}; D = {n: [] for n in NS}
    with torch.no_grad():
        for dom, seqs in data.items():
            for seq in seqs:
                ids = np.array(seq)
                names = tok.convert_ids_to_tokens(seq)
                cls_all = np.array([tok_class(t) for t in names])
                KV = keys_values(torch.tensor(seq))
                for (li, h) in sorted(long_heads[dom]):
                    K, V = KV[li][0][:, h], unitv(KV[li][1][:, h])
                    for n in NS:
                        for end in ENDS:
                            s0 = end - n + 1
                            Kw, Vw, tw, cw = K[s0:end + 1], V[s0:end + 1], ids[s0:end + 1], cls_all[s0:end + 1]
                            T = err_op(Kw)
                            real = rms(T, Vw)
                            # --- A
                            A[n].append(pair_stats(Kw, Vw, tw, cw))
                            # --- B
                            vals, counts = np.unique(tw, return_counts=True)
                            cnt = dict(zip(vals, counts))
                            Rm = np.array([cnt[t] >= 2 for t in tw])
                            Ri, Ui = np.where(Rm)[0], np.where(~Rm)[0]
                            f = lambda idx: np.mean([rms(T, V2) for V2 in shuffled(Vw, idx)])
                            all_idx = np.arange(n)
                            B[n].append(dict(real=real, shuf=f(all_idx), keepR=f(Ui), keepU=f(Ri),
                                             both=np.mean([rms(T, shuffled(shuffled(Vw, Ri, 1)[0], Ui, 1)[0]) for _ in range(4)]),
                                             fracR=Rm.mean()))
                            # --- C: de-duplicated window from a 3n-token span ending at `end`
                            lo = max(1, end - 3 * n + 1)
                            seen, sel = set(), []
                            for p in range(lo, end + 1):
                                if ids[p] not in seen:
                                    seen.add(ids[p]); sel.append(p)
                            if len(sel) >= n:
                                sel = np.array(sel[-n:])
                                Kd, Vd, td, cd = K[sel], V[sel], ids[sel], cls_all[sel]
                                Td = err_op(Kd)
                                ps = pair_stats(Kd, Vd, td, cd)
                                C[n].append(dict(real=rms(Td, Vd), shuf=np.mean([rms(Td, V2) for V2 in shuffled(Vd, np.arange(n))]),
                                                 c_all=ps["c_all"], c_diff_samecls=ps["c_diff_samecls"], c_diff_diffcls=ps["c_diff_diffcls"]))
                            # --- D: token-lookup predictor from the preceding CTX tokens
                            if s0 >= CTX:
                                Kc, Vc, tc = K[s0 - CTX:s0], V[s0 - CTX:s0], ids[s0 - CTX:s0]
                                kap = choose_kappa(Kc, Vc)
                                Fnw = nw_predict(Kw @ Kc.T, Vc, kap)
                                Fmean = np.tile(Vc.mean(0), (n, 1))
                                Ftok = Fmean.copy(); matched = np.zeros(n, bool)
                                for i, t in enumerate(tw):
                                    m = tc == t
                                    if m.any():
                                        Ftok[i] = Vc[m].mean(0); matched[i] = True
                                e = lambda F, m: np.linalg.norm((F - Vw)[m], axis=1).mean() if m.any() else np.nan
                                D[n].append(dict(frac=matched.mean(), tok_m=e(Ftok, matched), nw_m=e(Fnw, matched), mean_m=e(Fmean, matched),
                                                 nw_u=e(Fnw, ~matched), mean_u=e(Fmean, ~matched)))
                print(dom, "seq done", flush=True)
    nm = lambda lst, k: float(np.nanmedian([r[k] for r in lst]))
    nmean = lambda lst, k: float(np.nanmean([r[k] for r in lst]))
    print(f"\n=== A. pair decomposition (medians over {len(A[NS[0]])} windows) ===")
    print(f"{'n':>3} | {'P(same tok)':>11} | {'corr all':>8} {'diff tok':>8} {'diff,same cls':>13} {'diff,diff cls':>13} {'same tok':>8} | {'dk same/diff':>13} {'dv same/diff':>13} | {'P(cos>.9)':>9} {'..same tok':>10}")
    for n in NS:
        a = A[n]
        print(f"{n:3d} | {nm(a,'p_same'):11.3f} | {nm(a,'c_all'):8.2f} {nm(a,'c_diff'):8.2f} {nm(a,'c_diff_samecls'):13.2f} {nm(a,'c_diff_diffcls'):13.2f} {nm(a,'c_same'):8.2f} | "
              f"{nm(a,'dk_same'):6.2f}/{nm(a,'dk_diff'):5.2f} {nm(a,'dv_same'):6.2f}/{nm(a,'dv_diff'):5.2f} | {nm(a,'p_near'):9.3f} {nm(a,'near_same'):10.2f}")
    print(f"\n=== B. selective shuffles: RMS error (R = repeated-token positions, U = unique-token positions) ===")
    print(f"{'n':>3} | {'frac R':>6} | {'real':>6} {'shuf all':>8} | {'keep R real':>11} {'closure':>8} | {'keep U real':>11} {'closure':>8} | {'shuf R,U separately':>19} {'closure':>8}")
    for n in NS:
        b = B[n]
        gap = nmean(b, 'shuf') - nmean(b, 'real')
        cl = lambda k: (nmean(b, 'shuf') - nmean(b, k)) / gap
        print(f"{n:3d} | {nmean(b,'fracR'):6.2f} | {nmean(b,'real'):6.3f} {nmean(b,'shuf'):8.3f} | {nmean(b,'keepR'):11.3f} {cl('keepR'):8.2f} | "
              f"{nmean(b,'keepU'):11.3f} {cl('keepU'):8.2f} | {nmean(b,'both'):19.3f} {cl('both'):8.2f}")
    print(f"\n=== C. de-duplicated windows (all tokens distinct), vs contiguous windows of the same n ===")
    print(f"{'n':>3} | {'#win':>5} | {'real':>6} {'shuffled':>8} {'gap':>6} | {'corr all':>8} {'same cls':>8} {'diff cls':>8} | contiguous: {'real':>6} {'shuf':>6} {'gap':>6}")
    for n in NS:
        c, b = C[n], B[n]
        print(f"{n:3d} | {len(c):5d} | {nmean(c,'real'):6.3f} {nmean(c,'shuf'):8.3f} {nmean(c,'shuf')-nmean(c,'real'):6.3f} | {nm(c,'c_all'):8.2f} "
              f"{nm(c,'c_diff_samecls'):8.2f} {nm(c,'c_diff_diffcls'):8.2f} | {'':11}{nmean(b,'real'):6.3f} {nmean(b,'shuf'):6.3f} {nmean(b,'shuf')-nmean(b,'real'):6.3f}")
    print(f"\n=== D. token-lookup predictor (mean value of earlier same-token positions in the preceding {CTX} tokens); mean ||F-v|| per position ===")
    print(f"{'n':>3} | {'matched':>7} | on matched positions: {'token':>6} {'NW':>6} {'mean':>6} | on unmatched: {'NW':>6} {'mean':>6}")
    for n in NS:
        d = D[n]
        print(f"{n:3d} | {nmean(d,'frac'):7.2f} | {'':20}{nmean(d,'tok_m'):6.3f} {nmean(d,'nw_m'):6.3f} {nmean(d,'mean_m'):6.3f} | {'':13}{nmean(d,'nw_u'):6.3f} {nmean(d,'mean_u'):6.3f}")
