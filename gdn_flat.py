"""Why is the gain profile flat?   (model chosen by GDN_MD; long-memory heads; ideal alpha = beta = 1; unit-normalised rows)

Exact:  kappa_j = ||u_j^T V||^2 / lam_j = 1 + delta_j / lam_j,  delta_j = u_j^T (H - G) u_j,  and  sum_j lam_j kappa_j = n = sum_j lam_j.
Hypothesis: keys and values are two feature maps of the same hidden state x_t, so H ~ G (random-feature / Johnson-Lindenstrauss picture).

Value variants (the keys are always the trained keys of head h):
  own   : the head's trained value branch
  oh    : another head's trained value branch (same layer)
  rot   : the head's own value branch applied to a randomly ROTATED hidden state  x -> x P  (P orthogonal): same weights' spectrum, alignment with x destroyed
  gauss : value projection replaced by i.i.d. Gaussian weights of the same scale (conv weights unchanged)
  shuf  : own values, pairing shuffled within the window
Stages (own head): cosine Gram at x, after conv (pre-SiLU), after SiLU+normalisation.
"""
import sys, json, os
sys.path.insert(0, ".")
import numpy as np
import torch
import torch.nn.functional as F
import gdn_extract as g
from gdn_kv_tok import err_op

NS = [32, 64]
ENDS = list(range(255, 1024, 128))
VARS = ["own", "oh", "rot", "gauss", "shuf"]
rng = np.random.default_rng(23)
unit = lambda x: x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def conv_pre(u, w):
    return F.conv1d(u.t().unsqueeze(0), w, padding=w.shape[-1] - 1, groups=u.shape[1])[0, :, :u.shape[0]].t()


def layer_streams(ids):
    """per layer: x (T,D), Kpre/Vpre variants (T,H,d) pre-SiLU, and post-SiLU unit rows."""
    h = g.W["model.embeddings.weight"][ids]
    out = {}
    gen = torch.Generator().manual_seed(7)
    for i in range(g.NL):
        x = g.rms(h, g.W[f"model.layers.{i}.attn_norm.weight"])
        T_ = x.shape[0]
        Wk, Wv = g.W[f"model.layers.{i}.attn.k_proj.weight"], g.W[f"model.layers.{i}.attn.v_proj.weight"]
        ck, cv = g.W[f"model.layers.{i}.attn.k_conv1d.weight"], g.W[f"model.layers.{i}.attn.v_conv1d.weight"]
        P = torch.linalg.qr(torch.randn(x.shape[1], x.shape[1], generator=gen))[0]
        Wg = torch.randn(Wv.shape, generator=gen) * Wv.std()
        kpre = conv_pre(x @ Wk.T, ck).view(T_, g.H, g.DK)
        pre = {"own": conv_pre(x @ Wv.T, cv), "rot": conv_pre((x @ P) @ Wv.T, cv), "gauss": conv_pre(x @ Wg.T, cv)}
        pre = {k: v.view(T_, g.H, g.DV) for k, v in pre.items()}
        post = lambda t: t * torch.sigmoid(t)
        out[i] = dict(x=x.numpy().astype(np.float64), kpre=kpre.numpy().astype(np.float64),
                      K=unit(post(kpre).numpy().astype(np.float64)),
                      Vpre={k: v.numpy().astype(np.float64) for k, v in pre.items()},
                      V={k: unit(post(v).numpy().astype(np.float64)) for k, v in pre.items()})
        a, _ = g.gdn(x, i, False)
        h = h + a
        h = h + g.mlp(g.rms(h, g.W[f"model.layers.{i}.mlp_norm.weight"]), i)
    return out


def offdiag(M):
    return M[np.triu_indices(len(M), 1)]


def window_metrics(Kw, Vw):
    n = len(Kw)
    G = Kw @ Kw.T; H = Vw @ Vw.T
    lam, U = np.linalg.eigh(G); lam, U = np.clip(lam[::-1], 1e-12, None), U[:, ::-1]
    T = err_op(Kw); TK = T @ Kw
    psi = np.linalg.norm(TK) / np.sqrt(n); rms = np.linalg.norm(T @ Vw) / np.sqrt(n)
    UtV = U.T @ Vw
    kap = (UtV ** 2).sum(1) / lam
    delta = np.einsum("ij,ik,kj->j", U, H - G, U) if False else np.diag(U.T @ (H - G) @ U)
    Mt = (UtV / np.sqrt(lam)[:, None]) @ (UtV / np.sqrt(lam)[:, None]).T
    third = n // 3
    og, oh = offdiag(G), offdiag(H)
    c = np.corrcoef(og, oh)[0, 1] if og.std() > 0 and oh.std() > 0 else np.nan
    slope = np.polyfit(og, oh, 1)[0] if og.std() > 0 else np.nan
    lamH, UH = np.linalg.eigh(H); lamH, UH = np.clip(lamH[::-1], 0, None), UH[:, ::-1]
    m8 = min(8, n // 4)
    overlap = float(np.linalg.norm(U[:, :m8].T @ UH[:, :m8]) ** 2 / m8)
    shareG = lam[:third].sum() / lam.sum(); shareH = lamH[:third].sum() / lamH.sum()
    botG = lam[2 * third:].sum() / lam.sum(); botH = lamH[2 * third:].sum() / lamH.sum()
    return dict(ovl=overlap, ovl_chance=m8 / n, topG=shareG, topH=shareH, botG=botG, botH=botH,
                q=kap.mean(), ratio=rms / psi, rms=rms, psi=psi, gmax=np.linalg.eigvalsh(Mt).max(),
                k_top=kap[:third].mean(), k_bot=kap[2 * third:].mean(),
                lslope=np.polyfit(np.log(lam), np.log(np.clip(kap, 1e-12, None)), 1)[0],
                dGH=np.linalg.norm(H - G) / np.linalg.norm(G - np.eye(n)), c=c, s=slope,
                wmean=(lam * kap).sum() / lam.sum(),
                d_top=delta[:third].mean(), d_mid=delta[third:2 * third].mean(), d_bot=delta[2 * third:].mean(),
                dcorr=np.corrcoef(delta, 1 / lam)[0, 1], kmax_idx=int(np.argmax(kap)) / n)


if __name__ == "__main__":
    tok = g.AutoTokenizer.from_pretrained(g.MD)
    data = g.load_sequences(tok, n_per_domain=3)
    cls = json.load(open(f"classes_{os.path.basename(g.MD)}.json"))
    long_heads = sorted((tuple(map(int, k.split("|"))) for k, v in cls.items() if v == "long"))
    R = {n: {v: [] for v in VARS} for n in NS}
    ST = {n: {k: [] for k in ("x-kpre", "x-vpre", "kpre-vpre", "kpre-kpost", "vpre-vpost", "G-H")} for n in NS}
    with torch.no_grad():
        for dom, seqs in data.items():
            for seq in seqs:
                S = layer_streams(torch.tensor(seq))
                for (li, h) in long_heads:
                    st = S[li]
                    h2 = (h + 1 + rng.integers(g.H - 1)) % g.H
                    for n in NS:
                        for end in ENDS:
                            sl = slice(end - n + 1, end + 1)
                            Kw = st["K"][sl, h]
                            Vs = {"own": st["V"]["own"][sl, h], "oh": st["V"]["own"][sl, h2], "rot": st["V"]["rot"][sl, h],
                                  "gauss": st["V"]["gauss"][sl, h], "shuf": st["V"]["own"][sl, h][rng.permutation(n)]}
                            for v in VARS:
                                R[n][v].append(window_metrics(Kw, Vs[v]))
                            cosg = lambda M: offdiag(unit(M) @ unit(M).T)
                            xg, kp, vp = cosg(st["x"][sl]), cosg(st["kpre"][sl, h]), cosg(st["Vpre"]["own"][sl, h])
                            kq, vq = offdiag(Kw @ Kw.T), offdiag(Vs["own"] @ Vs["own"].T)
                            cc = lambda a, b: np.corrcoef(a, b)[0, 1]
                            for k, (a, b) in {"x-kpre": (xg, kp), "x-vpre": (xg, vp), "kpre-vpre": (kp, vp), "kpre-kpost": (kp, kq),
                                              "vpre-vpost": (vp, vq), "G-H": (kq, vq)}.items():
                                ST[n][k].append(cc(a, b))
                print(dom, "seq done", flush=True)
    med = lambda n, v, k: float(np.nanmedian([r[k] for r in R[n][v]]))
    print(f"\nmodel {g.MD}: {len(long_heads)} long heads, {len(R[NS[0]]['own'])} windows per n and variant")
    print("\n=== 1. gain-profile flatness with different value branches (keys = trained keys) ===")
    print(f"{'n':>3} {'values':>6} | {'q':>5} {'RMS/psi':>7} {'lam_max(M)':>10} | {'kappa top':>9} {'kappa bottom':>12} {'log-slope':>9} | {'corr(G_ij,H_ij)':>15} {'slope':>6} {'||H-G||/||G-I||':>16}")
    for n in NS:
        for v in VARS:
            print(f"{n:3d} {v:>6} | {med(n,v,'q'):5.2f} {med(n,v,'ratio'):7.2f} {med(n,v,'gmax'):10.2f} | {med(n,v,'k_top'):9.2f} {med(n,v,'k_bot'):12.2f} {med(n,v,'lslope'):9.2f} | "
                  f"{med(n,v,'c'):15.2f} {med(n,v,'s'):6.2f} {med(n,v,'dGH'):16.2f}")
    print("\n=== 2. exact identity and the origin of the deviations (own values) ===")
    for n in NS:
        print(f"n={n}: key-energy-weighted mean gain sum(lam*kappa)/sum(lam) = {med(n,'own','wmean'):.4f} (exactly 1);  "
              f"delta_j = u_j^T(H-G)u_j by thirds (large / mid / small lam): {med(n,'own','d_top'):+.3f} / {med(n,'own','d_mid'):+.3f} / {med(n,'own','d_bot'):+.3f};  "
              f"median corr(delta_j, 1/lam_j) = {med(n,'own','dcorr'):+.2f};  q-1 = {med(n,'own','q')-1:+.2f}")
    print("\n=== 2b. shared dominant structure: top-8 eigenspace overlap of G and H, and spectral shape (own / variants) ===")
    print(f"{'n':>3} {'values':>6} | {'overlap of top-8 eigenspaces':>28} {'(chance)':>9} | {'top-third share of trace: G':>27} {'H':>6} | {'bottom-third share: G':>21} {'H':>6}")
    for n in NS:
        for v in VARS:
            print(f"{n:3d} {v:>6} | {med(n,v,'ovl'):28.2f} {med(n,v,'ovl_chance'):9.2f} | {med(n,v,'topG'):27.3f} {med(n,v,'topH'):6.3f} | {med(n,v,'botG'):21.4f} {med(n,v,'botH'):6.4f}")
    print("\n=== 3. where in the pipeline do key and value similarity structures agree? (correlation of off-diagonal cosines, own head, medians) ===")
    print(f"{'n':>3} | " + " ".join(f"{k:>11}" for k in ST[NS[0]]))
    for n in NS:
        print(f"{n:3d} | " + " ".join(f"{np.nanmedian(ST[n][k]):11.2f}" for k in ST[n]))
