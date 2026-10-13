"""Nonlinear local comparators.

A predictor F = f(window keys; preceding 256 tokens) that never looks at the window's values induces a comparator
    Z = ridge(window keys -> F),   R = K Z^T - V = (K Z^T - F) + (F - V)
so  ||R|| <= ||K Z^T - F|| (keys+context only)  +  ||F - V|| (predictor's error on the window).
Theorems 6 / 6' then apply unchanged.  Predictors:
  mean : context mean value                         (value redundancy only)
  lin  : ridge map fitted on the context            (baseline from 3d(c))
  nw   : Nadaraya-Watson kernel regression on key cosine, temperature chosen by leave-one-out INSIDE the context
Reports the predictor error, the certificate computed with the true residual, and a fully V-free "calibrated" bound
in which ||F - V|| is replaced by the 90th percentile of the same quantity over chunks of the context.
"""
import sys, json
sys.path.insert(0, ".")
import numpy as np
import torch
import gdn_extract as g
from gdn_kv import keys_values

NS = [8, 16, 32, 64]
ENDS = (511, 767, 1023)
CTX = 256
KAPPAS = [3, 10, 30, 100, 300]
LAMS = np.logspace(-5, 1, 25)
rng = np.random.default_rng(13)


def softmax_rows(S):
    S = S - S.max(1, keepdims=True)
    E = np.exp(S)
    return E / E.sum(1, keepdims=True)


def nw_predict(sim, Vc, kappa, exclude=None):
    S = kappa * sim
    if exclude is not None:
        S = S.copy(); S[:, exclude] = -1e9
    return softmax_rows(S) @ Vc


def choose_kappa(Kc, Vc):
    simc = Kc @ Kc.T
    best = (np.inf, KAPPAS[0])
    for kap in KAPPAS:
        S = kap * simc; np.fill_diagonal(S, -1e9)
        err = np.mean(np.linalg.norm(softmax_rows(S) @ Vc - Vc, axis=1) ** 2)
        if err < best[0]:
            best = (err, kap)
    return best[1]


def calibrate(Kc, Vc, kappa, n):
    """90th-percentile Frobenius-RMS and op-norm error of the NW predictor over held-out chunks of size n in the context."""
    simc = Kc @ Kc.T
    fr, op = [], []
    for a in range(0, len(Kc) - n + 1, n):
        idx = np.arange(a, a + n)
        S = kappa * simc[idx]; S[:, idx] = -1e9
        Rm = softmax_rows(S) @ Vc - Vc[idx]
        fr.append(np.linalg.norm(Rm) / np.sqrt(n)); op.append(np.linalg.norm(Rm, 2))
    return np.quantile(fr, 0.9), np.quantile(op, 0.9)


def certificate(lam, U, Top, Tf, psi_rms, F, Vw, n, eps_cal=None):
    UtF = U.T @ F
    best_s = (np.inf, None)
    outs = {}
    for l in LAMS:
        d = lam / (lam + l)
        Fhat = U @ (d[:, None] * UtF)
        zr = np.linalg.norm((np.sqrt(lam) / (lam + l))[:, None] * UtF, 2)
        Rfit = Fhat - F
        rfit_op, rfit_f = np.linalg.norm(Rfit, 2), np.linalg.norm(Rfit)
        sur = zr * psi_rms + min(Top * rfit_f, Tf * rfit_op) / np.sqrt(n)     # V-free surrogate used to pick lambda
        outs[l] = (zr, Fhat, rfit_op, rfit_f)
        if sur < best_s[0]:
            best_s = (sur, l)
    zr, Fhat, rfit_op, rfit_f = outs[best_s[1]]
    R = Fhat - Vw
    rr, rf = np.linalg.norm(R, 2), np.linalg.norm(R)
    res = dict(zr=zr, rr=rr, rf=rf / np.sqrt(n), fit_f=rfit_f / np.sqrt(n),
               rms_b=zr * psi_rms + min(Top * rf, Tf * rr) / np.sqrt(n), rr_ub=rfit_op + np.linalg.norm(F - Vw, 2))
    if eps_cal is not None:
        ef, eo = eps_cal
        res["rms_pred"] = zr * psi_rms + min(Top * (rfit_f + np.sqrt(n) * ef), Tf * (rfit_op + eo)) / np.sqrt(n)
        res["cover_f"] = float(np.linalg.norm(F - Vw) / np.sqrt(n) <= ef)
        res["cover_o"] = float(np.linalg.norm(F - Vw, 2) <= eo)
    return res, best_s[1], zr, Fhat


def pair_corr(K, V, rng):
    n = len(K)
    iu = np.triu_indices(n, 1)
    dk = np.sqrt(np.clip(2 - 2 * (K @ K.T), 0, None))[iu]
    dv = np.linalg.norm(V[:, None] - V[None], axis=2)[iu]
    c_real = np.corrcoef(dk, dv)[0, 1]
    Vs = V[rng.permutation(n)]
    c_shuf = np.corrcoef(dk, np.linalg.norm(Vs[:, None] - Vs[None], axis=2)[iu])[0, 1]
    return c_real, c_shuf


if __name__ == "__main__":
    tok = g.AutoTokenizer.from_pretrained(g.MD)
    data = g.load_sequences(tok, n_per_domain=2)
    per = json.load(open("results_gdn.json"))["per_head"]
    long_heads = {d: {tuple(map(int, k.split("|")[1:])) for k, v in per.items() if k.startswith(d + "|") and v["p256_A"] > 0.5} for d in data}
    unitv = lambda V: V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)
    rows = {n: [] for n in NS}
    with torch.no_grad():
        for dom, seqs in data.items():
            for seq in seqs:
                KV = keys_values(torch.tensor(seq))
                for (li, h) in sorted(long_heads[dom]):
                    K, V = KV[li][0][:, h], unitv(KV[li][1][:, h])
                    for n in NS:
                        for end in ENDS:
                            s0 = end - n + 1
                            Kc, Vc = K[s0 - CTX:s0], V[s0 - CTX:s0]
                            Kw, Vw = K[s0:end + 1], V[s0:end + 1]
                            G = Kw @ Kw.T
                            lam, U = np.linalg.eigh(G); lam = np.clip(lam, 1e-12, None)
                            L = np.tril(G, -1)
                            T = L.T @ np.linalg.inv(np.eye(n) + L)
                            Top, Tf = np.linalg.norm(T, 2), np.linalg.norm(T)
                            psi_rms = np.linalg.norm(T @ Kw) / np.sqrt(n)
                            rms_real = np.linalg.norm(T @ Vw) / np.sqrt(n)
                            # predictors
                            kap = choose_kappa(Kc, Vc)
                            C = Kc.T @ Kc / CTX; B = Kc.T @ Vc / CTX
                            Zl = np.linalg.solve(C + 1e-2 * np.eye(g.DK), B).T
                            preds = {"mean": np.tile(Vc.mean(0), (n, 1)), "lin": Kw @ Zl.T, "nw": nw_predict(Kw @ Kc.T, Vc, kap)}
                            row = dict(rms=rms_real, kappa=kap)
                            cal = calibrate(Kc, Vc, kap, n)
                            row["cal_f"], row["cal_o"] = cal
                            for name, F in preds.items():
                                res, lam_sel, zr, _ = certificate(lam, U, Top, Tf, psi_rms, F, Vw, n, cal if name == "nw" else None)
                                row[name] = dict(eps=np.linalg.norm(F - Vw) / np.sqrt(n), **res)
                            row["lip_real"], row["lip_shuf"] = pair_corr(Kw, Vw, rng) if n >= 16 else (np.nan, np.nan)
                            rows[n].append(row)
                print(dom, "seq done", flush=True)
    med = lambda n, f: float(np.nanmedian([f(r) for r in rows[n]]))
    json.dump({str(n): {"rms": med(n, lambda r: r["rms"])} for n in NS}, open("results_kv_nl.json", "w"))
    print(f"\nlong-memory heads, {len(rows[NS[0]])} windows per n; medians. Context = preceding {CTX} tokens; no window values used to build F.")
    print(f"{'n':>3} | {'measured':>8} | predictor error eps=||F-V||_F/sqrt n: {'mean':>5} {'lin':>5} {'NW':>5} | NW comparator: {'|Z|op':>6} {'fit':>5} {'|R|F/sqn':>8} {'|R|op':>6} | "
          f"{'cert RMS':>8} {'<1':>5} | {'calib RMS':>9} {'<1':>5} {'covF':>5} {'covOp':>5} | {'lin cert':>8} {'mean cert':>9} | corr(key d, val d) real/shuf")
    for n in NS:
        lt1 = lambda f: np.mean([f(r) < 1 for r in rows[n]])
        print(f"{n:3d} | {med(n,lambda r:r['rms']):8.3f} | {'':37}{med(n,lambda r:r['mean']['eps']):5.2f} {med(n,lambda r:r['lin']['eps']):5.2f} {med(n,lambda r:r['nw']['eps']):5.2f} | "
              f"{'':15}{med(n,lambda r:r['nw']['zr']):6.2f} {med(n,lambda r:r['nw']['fit_f']):5.2f} {med(n,lambda r:r['nw']['rf']):8.3f} {med(n,lambda r:r['nw']['rr']):6.2f} | "
              f"{med(n,lambda r:r['nw']['rms_b']):8.3f} {lt1(lambda r:r['nw']['rms_b']):5.2f} | {med(n,lambda r:r['nw']['rms_pred']):9.3f} {lt1(lambda r:r['nw']['rms_pred']):5.2f} "
              f"{np.mean([r['nw']['cover_f'] for r in rows[n]]):5.2f} {np.mean([r['nw']['cover_o'] for r in rows[n]]):5.2f} | "
              f"{med(n,lambda r:r['lin']['rms_b']):8.3f} {med(n,lambda r:r['mean']['rms_b']):9.3f} | "
              f"{med(n,lambda r:r['lip_real']):.2f}/{med(n,lambda r:r['lip_shuf']):.2f}")
    print("\nvalidity of the certificate with true residual (RMS_real <= cert):",
          {n: float(np.mean([r['rms'] <= r['nw']['rms_b'] * (1 + 1e-9) for r in rows[n]])) for n in NS})
    print("validity of the calibrated bound where its two coverage conditions hold:",
          {n: float(np.mean([r['rms'] <= r['nw']['rms_pred'] * (1 + 1e-9) for r in rows[n] if r['nw']['cover_f'] and r['nw']['cover_o']])) for n in NS})
    print("calibrated bound valid overall:", {n: float(np.mean([r['rms'] <= r['nw']['rms_pred'] * (1 + 1e-9) for r in rows[n]])) for n in NS})
    print("median chosen kappa:", {n: med(n, lambda r: r['kappa']) for n in NS})
