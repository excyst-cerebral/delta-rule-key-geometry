"""Memory (bits) vs forgetting for readout-only continual learning on SSM features (numpy only).
Methods: ft (nothing stored) | proj(rho,q): GPM-style stored basis of old feature subspace, q bits/entry | replay(k): k stored examples/task | ewc: theta* + diag Fisher per task.
Features: final-state features (+bias) of a random-init SSM, 4 tasks, 2000 samples each; squared loss on +-1 targets, 300 GD steps/task."""
import numpy as np
M, STEPS = 4, 300
rk = lambda x: np.argsort(np.argsort(x)).astype(float)


def load(arch, seed):
    D = np.load(f"exp_cl/runs/feat_{arch}_s{seed}.npz")
    return [np.hstack([D[f"phi{t}"], np.ones((len(D[f"phi{t}"]), 1))]) for t in range(M)], [D[f"y{t}"] for t in range(M)]


def acc(th, Phi, Y): return float(np.mean(np.sign(Phi @ th) == Y))


def quantize(B, q):
    if q >= 32: return B
    lv = 2 ** (q - 1) - 1; Bq = np.round(np.clip(B, -1, 1) * lv) / lv
    return np.linalg.qr(Bq)[0]


def run(Phi, Y, method, rho=0.99, q=8, k=32, lam=50.0, rng=None):
    p = Phi[0].shape[1]; th = np.zeros(p); basis = np.zeros((p, 0)); bits = 0; accm = np.zeros((M, M)); buf = []; anchors = []
    for t in range(M):
        X, y = Phi[t], Y[t]
        curv = np.linalg.eigvalsh(X.T @ X / len(X)).max() + (lam * sum(F.max() for _, F in anchors) if method == 'ewc' else 0) + (np.linalg.eigvalsh(np.vstack([b[0] for b in buf]).T @ np.vstack([b[0] for b in buf]) / sum(len(b[0]) for b in buf)).max() if method == 'replay' and buf else 0)
        lr = 0.5 / curv
        for _ in range(STEPS):
            g = X.T @ (X @ th - y) / len(X)
            if method == "replay" and buf:
                bx = np.vstack([b[0] for b in buf]); by = np.concatenate([b[1] for b in buf]); g = g + bx.T @ (bx @ th - by) / len(bx)
            if method == "ewc":
                for (ts, F) in anchors: g = g + lam * F * (th - ts)
            if method == "proj" and basis.shape[1]: g = g - basis @ (basis.T @ g)
            th = th - lr * g
        if method == "replay":
            idx = rng.choice(len(X), k, replace=False); buf.append((X[idx], y[idx])); bits += k * (p * 32 + 1)
        if method == "ewc": anchors.append((th.copy(), (X ** 2).mean(0))); bits += 2 * p * 32
        if method == "proj":
            R = X - X @ basis @ basis.T if basis.shape[1] else X
            U, s, Vt = np.linalg.svd(R, full_matrices=False); e = np.cumsum(s ** 2) / np.sum(s ** 2); r = int(np.searchsorted(e, rho) + 1)
            new = quantize(Vt[:r].T, q); new = new - basis @ (basis.T @ new) if basis.shape[1] else new
            new = np.linalg.qr(new)[0][:, :r]; basis = np.hstack([basis, new]); bits += r * p * q
        for j in range(M): accm[t, j] = acc(th, Phi[j], Y[j])
    forget = float(np.mean([accm[:M - 1, j].max() - accm[-1, j] for j in range(M - 1)]))
    return accm[-1].mean(), forget, bits, accm


rng = np.random.default_rng(0)
configs = [("ft", {}), ("ewc", {}), ("replay", dict(k=8)), ("replay", dict(k=64))] + [("proj", dict(rho=r_, q=8)) for r_ in (0.5, 0.8, 0.95, 0.99)] + [("proj", dict(rho=0.95, q=q_)) for q_ in (2, 4)]
print(f"{'method':>8} {'setting':>14} | {'final acc':>9} {'forgetting':>10} {'new-task acc (last)':>19} | {'bits stored (total)':>20} | stored rank r (mean)")
for method, kw in configs:
    res = []; ranks = []
    for arch in ("lti", "sel"):
        for seed in range(5):
            Phi, Y = load(arch, seed); fa, fg, bits, accm = run(Phi, Y, method, rng=rng, **kw); res.append((fa, fg, accm[-1, -1], bits))
            if method == "proj": ranks.append(bits / (Phi[0].shape[1] * kw.get("q", 8)))
    a = np.array(res); setting = ",".join(f"{k}={v}" for k, v in kw.items()) or "-"
    print(f"{method:>8} {setting:>14} | {a[:,0].mean():9.3f} {a[:,1].mean():10.3f} {a[:,2].mean():19.3f} | {a[:,3].mean():20.0f} | {np.mean(ranks) if ranks else float('nan'):6.1f}")
print("\nnumerical rank of each task's standardised feature matrix (bias column excluded; singular values > 1% / 10% of the largest), p = 24:")
for arch in ("lti", "sel"):
    out = []
    for seed in range(5):
        Phi, Y = load(arch, seed)
        for X in Phi:
            Z = X[:, :-1]; Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-9); sv = np.linalg.svd(Z, compute_uv=False); out.append((int((sv > .01 * sv[0]).sum()), int((sv > .1 * sv[0]).sum())))
    o = np.array(out); print(f"  {arch}: rank@1% median {np.median(o[:,0]):.0f} (range {o[:,0].min()}-{o[:,0].max()}), rank@10% median {np.median(o[:,1]):.0f} (range {o[:,1].min()}-{o[:,1].max()})")
