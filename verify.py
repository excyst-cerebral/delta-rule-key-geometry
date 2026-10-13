"""Numerical checks for the claims in notes.md (Option A: learning in the SSM state).

State update (gated delta rule), S in R^{dv x dk}, unit-norm keys:
    S_t = alpha_t * S_{t-1} (I - beta_t k_t k_t^T) + beta_t v_t k_t^T
"""
import numpy as np

rng = np.random.default_rng(0)


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def step(S, k, v, alpha, beta):
    d = k.shape[0]
    return alpha * S @ (np.eye(d) - beta * np.outer(k, k)) + beta * np.outer(v, k)


def coherence(K):
    G = K @ K.T
    np.fill_diagonal(G, 0)
    return np.abs(G).max()


# ---- Lemma 1: exact interference identity ---------------------------------
# S_t k_i = alpha_t S_{t-1} k_i - beta_t <k_t,k_i> e_t,  e_t = alpha_t S_{t-1} k_t - v_t
def check_lemma1(dk=32, dv=8, T=200):
    S = rng.normal(size=(dv, dk))
    worst = 0.0
    probes = unit(rng.normal(size=(5, dk)))
    for _ in range(T):
        k = unit(rng.normal(size=dk))
        v = rng.normal(size=dv)
        a, b = rng.uniform(0.7, 1.0), rng.uniform(0.2, 1.0)
        e = a * S @ k - v
        Snew = step(S, k, v, a, b)
        for p in probes:
            pred = a * S @ p - b * (k @ p) * e
            worst = max(worst, np.linalg.norm(Snew @ p - pred))
        S = Snew
    return worst


# ---- Thm 2: Fejer monotonicity / bounded cumulative surprise (alpha=beta=1) -
def check_kaczmarz(dk=64, dv=8, n=40, T=400):
    K = unit(rng.normal(size=(n, dk)))
    V = rng.normal(size=(n, dv))
    Sstar = V.T @ np.linalg.pinv(K.T)            # min-norm S with S k_i = v_i
    assert np.allclose(Sstar @ K.T, V.T)
    S = np.zeros((dv, dk))
    E0 = np.linalg.norm(Sstar)
    prev = E0
    mono_viol, sum_e2 = 0.0, 0.0
    for _ in range(T):
        i = rng.integers(n)
        e = S @ K[i] - V[i]
        sum_e2 += e @ e
        S = step(S, K[i], V[i], 1.0, 1.0)
        cur = np.linalg.norm(S - Sstar)
        mono_viol = max(mono_viol, cur - prev)
        prev = cur
    return mono_viol, sum_e2, E0**2


# ---- Thm 1: retention bound (decay + mu * surprise) -------------------------
def check_retention(dk=256, dv=8, n=24, gap=60, alpha=0.99, beta=1.0, trials=200):
    ratios = []
    for _ in range(trials):
        K = unit(rng.normal(size=(n + gap, dk)))
        V = rng.normal(size=(n + gap, dv))
        S = np.zeros((dv, dk))
        surprise_after = 0.0
        # write n items, probe item 0 right after its write, then keep writing
        for t in range(n + gap):
            e = alpha * S @ K[t] - V[t]
            S = step(S, K[t], V[t], alpha, beta)
            if t == 0:
                r0 = S @ K[0]                     # = beta*v_0 (S_0 = 0)
                assert np.allclose(r0, beta * V[0])
            elif t >= 1:
                mu_i = abs(K[t] @ K[0])
                surprise_after += beta * mu_i * np.linalg.norm(e)
        T_after = n + gap - 1
        retained = S @ K[0]
        ideal = (alpha ** T_after) * r0
        err = np.linalg.norm(retained - ideal)
        # bound: err <= sum_t beta |<k_t,k_0>| ||e_t||   (computed exactly above)
        ratios.append(err / max(surprise_after, 1e-12))
    return max(ratios)


# ---- Thm 3: capacity lower bound for any linear state ----------------------
def check_capacity(dk=16, dv=4, extra=6, trials=100):
    """n = dk+extra keys must be linearly dependent. Choose values from the
    dependency; best linear S still has max error >= ||w|| ||c||_2^2 / ||c||_1."""
    n = dk + extra
    worst_gap = np.inf
    for _ in range(trials):
        K = unit(rng.normal(size=(n, dk)))
        # null-space vector c with K^T c = 0
        _, _, Vt = np.linalg.svd(K.T)
        c = Vt[-1]                                # K.T has dk rows, n cols -> null dim >= extra
        assert np.allclose(K.T @ c, 0, atol=1e-9)
        w = unit(rng.normal(size=dv))
        V = np.outer(c, w)                        # v_i = c_i w
        # least-squares optimum of max-error is >= LS lower bound; use min-max via LS as probe
        S = V.T @ np.linalg.pinv(K.T)             # best Frobenius-fit linear map
        errs = np.linalg.norm(K @ S.T - V, axis=1)
        bound = (c @ c) / np.abs(c).sum()         # ||w||=1
        worst_gap = min(worst_gap, errs.max() - bound)
    return worst_gap                              # should be >= 0


# ---- Empirical vs bound: forgetting vs coherence ---------------------------
def forgetting_vs_mu(n=20, dvs=8):
    rows = []
    for dk in (32, 64, 128, 256, 512):
        fg, mus, bnds = [], [], []
        for _ in range(100):
            K = unit(rng.normal(size=(n, dk)))
            V = unit(rng.normal(size=(n, dvs)))
            S = np.zeros((dvs, dk))
            for t in range(n):
                S = step(S, K[t], V[t], 1.0, 1.0)
            err = np.linalg.norm(K @ S.T - V, axis=1)   # per-item recall error
            Sstar = V.T @ np.linalg.pinv(K.T)
            mu = coherence(K)
            fg.append(err.max())
            mus.append(mu)
            sig = np.linalg.svd(K, compute_uv=False).min()
            bnds.append(mu * np.sqrt(n) * np.linalg.norm(Sstar))   # Thm: mu*sqrt(n)*||S*||_F
        rows.append((dk, np.mean(mus), np.mean(fg), np.mean(bnds)))
    return rows


if __name__ == "__main__":
    print("Lemma 1 max identity residual      :", check_lemma1())
    mv, se2, E02 = check_kaczmarz()
    print("Thm 2 monotonicity violation (<=0)  :", mv)
    print("Thm 2 sum ||e_t||^2 <= ||S*||_F^2   :", se2, "<=", E02, se2 <= E02 + 1e-9)
    print("Thm 1 max (err / sum beta|<k,k0>|e) :", check_retention(), "(must be <= 1)")
    print("Thm 3 min (max_err - lower bound)   :", check_capacity(), "(must be >= 0)")
    print("\nforgetting vs key dimension (n=20 items, one pass, alpha=beta=1)")
    print(f"{'dk':>5} {'mu':>8} {'max recall err':>16} {'proved bound':>14}")
    for dk, mu, fg, b in forgetting_vs_mu():
        print(f"{dk:5d} {mu:8.3f} {fg:16.4f} {b:14.3f}")
