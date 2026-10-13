"""Checks for the tightened single-pass bounds (alpha = beta = 1, unit keys).

G = K K^T (Gram), L = strict lower triangle of G, W = (I+L)^{-1} V  (rows w_t = -e_t).
Exact:      R = L^T (I+L)^{-1} V,  i.e. err_i = sum_{j>i} G_ij w_j
Thm 4:      F_i <= (1+mu)^n - (1+mu)^i   (i = 1..n)       (worst case, elementary)
Thm 4 tight: equal values, G_ij = mu  ->  F_1 = (1-mu) - (1-mu)^n  ~ mu*n  for n*mu <= 1
Thm 5:      F_i <= rho_i * ||V||_op / sqrt(lambda_min(G)),  rho_i = ||(G_ij)_{j>i}||_2
"""
import numpy as np

rng = np.random.default_rng(1)


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def run_pass(K, V):
    n, dk = K.shape
    S = np.zeros((V.shape[1], dk))
    for t in range(n):
        S = S + np.outer(V[t] - S @ K[t], K[t])
    return S


def errors(K, V):
    S = run_pass(K, V)
    return np.linalg.norm(K @ S.T - V, axis=1)          # F_i, i = 0..n-1


def exact_formula(K, V):
    G = K @ K.T
    L = np.tril(G, -1)
    W = np.linalg.solve(np.eye(len(G)) + L, V)
    R = L.T @ W
    return np.linalg.norm(R, axis=1), W


def coherence(G):
    H = G - np.eye(len(G))
    return np.abs(H).max()


# --- 1. exact closed form ----------------------------------------------------
def check_exact(trials=200):
    worst = 0.0
    for _ in range(trials):
        n, dk, dv = 15, 24, 5
        K = unit(rng.normal(size=(n, dk)))
        V = rng.normal(size=(n, dv))
        F = errors(K, V)
        F2, _ = exact_formula(K, V)
        worst = max(worst, np.abs(F - F2).max())
    return worst


# --- 2. worst-case bound + tightness example ----------------------------------
def check_worst_case(trials=200):
    viol = -np.inf
    for _ in range(trials):
        n, dk, dv = 12, 200, 4
        K = unit(rng.normal(size=(n, dk)))
        V = unit(rng.normal(size=(n, dv)))
        F = errors(K, V)
        mu = coherence(K @ K.T)
        bound = (1 + mu) ** n - (1 + mu) ** (np.arange(n) + 1)
        viol = max(viol, (F - bound).max())
    return viol                                          # must be <= 0


def tightness_example(n=30, mu=0.02, dv=3):
    # keys with all pairwise inner products = mu:  k_j = sqrt(mu) u + sqrt(1-mu) e_j
    dk = n + 1
    u = np.zeros(dk); u[-1] = 1.0
    K = np.array([np.sqrt(mu) * u + np.sqrt(1 - mu) * np.eye(dk)[j] for j in range(n)])
    assert np.allclose(K @ K.T - np.eye(n), mu * (1 - np.eye(n)))
    v = unit(rng.normal(size=dv))
    V = np.tile(v, (n, 1))                               # all values equal
    F = errors(K, V)
    pred = (1 - mu) ** (np.arange(1, n + 1)) - (1 - mu) ** n   # item index i = 1..n
    return F[0], pred[0], mu * (n - 1), np.abs(F - pred).max()


# --- 3. spectral/Frobenius-coherence bound ------------------------------------
def thm5(K, V):
    G = K @ K.T
    n = len(G)
    lam = np.linalg.eigvalsh(G).min()
    vop = np.linalg.norm(V, 2)
    rho = np.array([np.linalg.norm(G[i, i + 1:]) for i in range(n)])
    return rho * vop / np.sqrt(lam), lam, vop, rho


def check_thm5(trials=300):
    maxratio = 0.0
    for _ in range(trials):
        n, dk, dv = 20, int(rng.choice([48, 96, 256])), int(rng.choice([2, 8, 64]))
        K = unit(rng.normal(size=(n, dk)))
        V = unit(rng.normal(size=(n, dv)))
        F = errors(K, V)
        B, *_ = thm5(K, V)
        # last item has F = B = 0 exactly (up to float noise), so compare with a tolerance
        assert (F <= B + 1e-12).all()
        maxratio = max(maxratio, (F[:-1] / B[:-1]).max())
    return maxratio                                      # must be <= 1


def comparison(n=20, dvs=(8, 64), dks=(32, 64, 128, 256, 512), reps=100):
    print(f"{'dv':>4} {'dk':>5} {'mu':>6} {'measured':>9} {'old bound':>10} {'new bound':>10} {'new/meas':>9}")
    for dv in dvs:
        for dk in dks:
            m, o, nb, mu_ = [], [], [], []
            for _ in range(reps):
                K = unit(rng.normal(size=(n, dk)))
                V = unit(rng.normal(size=(n, dv)))
                F = errors(K, V)
                G = K @ K.T
                Sstar = V.T @ np.linalg.solve(G, K)       # min-norm S*, dv x dk
                old = coherence(G) * np.sqrt(n) * np.linalg.norm(Sstar)
                new, *_ = thm5(K, V)
                i = F.argmax()
                m.append(F.max()); o.append(old); nb.append(new[i]); mu_.append(coherence(G))
            print(f"{dv:4d} {dk:5d} {np.mean(mu_):6.3f} {np.mean(m):9.3f} {np.mean(o):10.3f} "
                  f"{np.mean(nb):10.3f} {np.mean(nb)/np.mean(m):9.2f}")


if __name__ == "__main__":
    print("exact closed form  max |F - formula|           :", check_exact())
    print("Thm 4 worst-case   max (F - bound) (must <= 0) :", check_worst_case())
    f, p, lin, dev = tightness_example()
    print(f"Tightness example  F_1 = {f:.5f}  closed form = {p:.5f}  mu*(n-1) = {lin:.5f}  max dev = {dev:.1e}")
    print("Thm 5              max F/bound (must <= 1)     :", check_thm5())
    print()
    comparison()
