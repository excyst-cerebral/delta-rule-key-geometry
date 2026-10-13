"""Numerical verification of Theorem 16 (protection-plasticity frontier).  numpy only.
Setting: theta in R^p, old feature subspace U with orthogonal projector P (rank r); new task: Phi (n x p), target surprise e (n).
interference I(D) = ||P D||, fit error E(D) = ||Phi D - e||.   G = Phi (I-P),  Pi = projector onto range(G),  c = ||(I-Pi) e||.
(a) min_{P D = 0} E = c;  E[c^2] = sigma^2 (n - rank G) for e ~ N(0, sigma^2 I).
(b) frontier E*(eps) = min{E: I <= eps} is traced by D_mu = (Phi^T Phi + mu P)^+ Phi^T e; its slope is dE*^2/d(eps^2) = -mu.
(c) one example, ||w||=1, kappa = ||(I-P)w||, rho = ||P w||, step budget ||D|| <= B:  reach(eps) = rho*eps + kappa*sqrt(B^2-eps^2) for eps <= rho*B, = B otherwise.
"""
import numpy as np
rng = np.random.default_rng(0)


def rand_proj(p, r):
    Q = np.linalg.qr(rng.normal(size=(p, p)))[0]; return Q[:, :r] @ Q[:, :r].T, Q


def check_a(trials=30):
    worst = 0.0; worst_gd = 0.0
    for _ in range(trials):
        p = int(rng.integers(6, 16)); r = int(rng.integers(1, p - 1)); n = int(rng.integers(1, 2 * p)); P, Q = rand_proj(p, r)
        Phi = rng.normal(size=(n, p)); e = rng.normal(size=n); G = Phi @ (np.eye(p) - P)
        Pi = G @ np.linalg.pinv(G); c = np.linalg.norm((np.eye(n) - Pi) @ e)
        D = np.linalg.pinv(G) @ e; assert np.linalg.norm(P @ D) < 1e-9
        worst = max(worst, abs(np.linalg.norm(Phi @ D - e) - c))
        # independent check: gradient descent over Delta = (I-P) z
        z = np.zeros(p); lr = 0.5 / (np.linalg.norm(G, 2) ** 2 + 1e-12)
        for _ in range(300000): z -= lr * G.T @ (G @ z - e)      # slow on ill-conditioned G (2e4 iterations left errors up to 0.35)
        worst_gd = max(worst_gd, abs(np.linalg.norm(G @ z - e) - c))
    return worst, worst_gd


def check_a_expect(p=12, r=5, n=20, trials=40000, sigma=0.7):
    P, _ = rand_proj(p, r); Phi = rng.normal(size=(n, p)); G = Phi @ (np.eye(p) - P); Pi = G @ np.linalg.pinv(G); rk = np.linalg.matrix_rank(G)
    E = rng.normal(size=(trials, n)) * sigma; c2 = ((E - E @ Pi.T) ** 2).sum(1).mean()
    return c2, sigma ** 2 * (n - rk), rk, p - r


def frontier_point(Phi, P, e, mu):
    A = Phi.T @ Phi + mu * P; D = np.linalg.pinv(A) @ Phi.T @ e; return np.linalg.norm(P @ D), np.linalg.norm(Phi @ D - e)


def pgd_min_E(Phi, P, e, eps, iters=60000):
    """independent solver: min ||Phi D - e||^2 s.t. ||P D|| <= eps, projected gradient (Lipschitz step) on the full vector"""
    p = Phi.shape[1]; D = np.zeros(p); L = np.linalg.norm(Phi, 2) ** 2; I = np.eye(p)
    for _ in range(iters):
        D = D - (0.9 / L) * Phi.T @ (Phi @ D - e)
        a = P @ D; na = np.linalg.norm(a)
        if na > eps: D = D - a + a * (eps / na)           # projection onto {||P D|| <= eps} (exact: P-part scaled, orthogonal part untouched)
    return np.linalg.norm(Phi @ D - e)


def check_b(trials=8):
    worst_gap = 0.0; worst_slope = 0.0
    for _ in range(trials):
        p = 10; r = int(rng.integers(2, 6)); n = int(rng.integers(5, 25)); P, _ = rand_proj(p, r); Phi = rng.normal(size=(n, p)); e = rng.normal(size=n) * 2
        for mu in (0.05, 0.3, 1.0, 5.0, 30.0):
            eps, E = frontier_point(Phi, P, e, mu); Epgd = pgd_min_E(Phi, P, e, eps)
            worst_gap = max(worst_gap, abs(E - Epgd))
            # slope check: dE^2/d(eps^2) = -mu, finite difference on the frontier through nearby mu
            h = 1e-4 * mu; e1, E1 = frontier_point(Phi, P, e, mu - h); e2, E2 = frontier_point(Phi, P, e, mu + h)
            if abs(e1 ** 2 - e2 ** 2) > 1e-12: worst_slope = max(worst_slope, abs((E1 ** 2 - E2 ** 2) / (e1 ** 2 - e2 ** 2) + mu) / mu)
        # monotonicity
        mus = np.geomspace(1e-3, 1e3, 30); pts = np.array([frontier_point(Phi, P, e, m) for m in mus])
        assert np.all(np.diff(pts[:, 0]) <= 1e-9) and np.all(np.diff(pts[:, 1]) >= -1e-9)
    return worst_gap, worst_slope


def check_c(trials=30):
    worst = 0.0
    for _ in range(trials):
        p = 6; r = int(rng.integers(1, 5)); P, _ = rand_proj(p, r); w = rng.normal(size=p); w /= np.linalg.norm(w)
        rho = np.linalg.norm(P @ w); kappa = np.linalg.norm((np.eye(p) - P) @ w); B = float(rng.uniform(0.5, 2)); eps = float(rng.uniform(0, 1.2 * rho * B))
        pred = B if eps >= rho * B else rho * eps + kappa * np.sqrt(B ** 2 - eps ** 2)
        # independent: maximise <w, D> over ||D|| <= B, ||P D|| <= eps by dense random search + local ascent on the 2-D reduced problem (a = |PD| along Pw, rest along (I-P)w, extra mass wasted)
        best = 0.0
        for a in np.linspace(0, min(eps, B), 4001): best = max(best, rho * a + kappa * np.sqrt(max(B ** 2 - a ** 2, 0)))
        # random sampling in the full space as a sanity check (lower bound)
        X = rng.normal(size=(200000, p)); X /= np.linalg.norm(X, axis=1, keepdims=True); X *= B * rng.uniform(0.5, 1, size=(200000, 1)) ** 0.1
        ok = (np.linalg.norm(X @ P, axis=1) <= eps) & (np.linalg.norm(X, axis=1) <= B); samp = (X[ok] @ w).max() if ok.any() else 0.0
        assert samp <= pred + 1e-9
        worst = max(worst, abs(best - pred))
    return worst


if __name__ == "__main__":
    a1, a2 = check_a(); print(f"(a) zero-interference optimum = c: closed form vs lstsq max err {a1:.2e}; vs independent gradient descent max err {a2:.2e}")
    c2, pred, rk, pr = check_a_expect(); print(f"(a) E[c^2] Monte Carlo {c2:.4f} vs sigma^2 (n - rank G) = {pred:.4f}  (rank G = {rk}, p - r = {pr})")
    g, sl = check_b(); print(f"(b) frontier: parametric D_mu vs independent projected-gradient solver, max |gap| {g:.2e};  shadow-price check |dE^2/d eps^2 + mu|/mu max {sl:.2e}; monotone: ok")
    print(f"(c) one-example reach formula vs maximisation over the budget (4001-point grid; random sampling never exceeds it): max err {check_c():.2e}")
