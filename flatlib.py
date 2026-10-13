"""Shared numpy helpers for the flatness analyses (identical to tf_flat.py)."""
import numpy as np

unit = lambda x: x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


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
