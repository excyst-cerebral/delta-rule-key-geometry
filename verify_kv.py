"""Synthetic checks for the key-value-aware bounds (alpha = beta = 1, unit keys, single pass).

T = L^T (I+L)^{-1}  (n x n): error operator, error matrix = T V  (row i = S_n k_i - v_i).

K1  Lemma 9 (comparator form):   for ANY Z (dv x dk), R = K Z^T - V:
        ||W||_op <= sqrt(||Z||_op^2 + ||R||_op^2) + ||R||_op          (W = (I+L)^{-1} V)
K2  Thm 6 (item-wise):           F_i <= rho_i * [ sqrt(||Z||^2 + ||R||^2) + ||R|| ]
    Thm 6 (RMS):                 RMS <= sqrt((n/r_eff - 1)/2) * [ same bracket ]
K3  Prop 7 (transfer):           V = K Z^T  =>  F_i = ||Z p_i|| <= ||Z||_op psi_i <= ||Z||_op rho_i,
                                 p_i = S^{id} k_i - k_i  (state after storing the identity map k -> k)
K4  psi_i = || Q_1 ... Q_n k_i ||,  Q_t = I - k_t k_t^T
K5  duplicates: identical keys + identical values => no interference
"""
import numpy as np

rng = np.random.default_rng(11)
unit = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True)


def lowrank_keys(n, dk, r, noise):
    Q = np.linalg.qr(rng.normal(size=(dk, r)))[0]
    A = rng.normal(size=(n, r)) + 0.4 * rng.normal(size=(1, r))
    return unit(A @ Q.T + noise * rng.normal(size=(n, dk)) / np.sqrt(dk))


def ops(K):
    G = K @ K.T
    n = len(G)
    L = np.tril(G, -1)
    A = np.eye(n) + L
    T = L.T @ np.linalg.inv(A)
    rho = np.array([np.linalg.norm(G[i, i + 1:]) for i in range(n)])
    r_eff = n ** 2 / np.linalg.norm(G) ** 2
    return G, L, A, T, rho, r_eff


def bracket(Z, K, V):
    R = K @ Z.T - V
    zr, rr = np.linalg.norm(Z, 2), np.linalg.norm(R, 2)
    return np.sqrt(zr ** 2 + rr ** 2) + rr, R


# ---- K1 / K2 -----------------------------------------------------------------
v1 = v2a = v2b = -np.inf
slack = []
for _ in range(400):
    n, dk, dv = int(rng.choice([8, 24, 48])), 96, int(rng.choice([4, 32, 96]))
    K = lowrank_keys(n, dk, int(rng.choice([3, 8, 30])), float(rng.choice([0.05, 0.3])))
    Z0 = rng.normal(size=(dv, dk)) / np.sqrt(dk) * rng.uniform(0.3, 3)
    V = K @ Z0.T + float(rng.choice([0, 0.05, 0.3, 1.0])) * rng.normal(size=(n, dv)) / np.sqrt(dv)
    G, L, A, T, rho, r_eff = ops(K)
    W = np.linalg.solve(A, V)
    E = T @ V
    F = np.linalg.norm(E, axis=1)
    kind = rng.integers(3)
    if kind == 0:   Z = Z0
    elif kind == 1:
        lam = 10 ** rng.uniform(-4, 1); Z = (K.T @ np.linalg.solve(G + lam * np.eye(n), V)).T
    else:           Z = rng.normal(size=(dv, dk)) / np.sqrt(dk)
    br, R = bracket(Z, K, V)
    v1 = max(v1, np.linalg.norm(W, 2) - br)
    v2a = max(v2a, (F - rho * br).max())
    v2b = max(v2b, np.linalg.norm(E) / np.sqrt(n) - np.sqrt(max(n / r_eff - 1, 0) / 2) * br)
    slack.append(rho[:-1].max() * br / max(F.max(), 1e-12))
print(f"K1 ||W||_op - bracket (max)              : {v1:.1e}   (<= 0 expected)")
print(f"K2 item-wise  F_i - rho_i*bracket (max)  : {v2a:.1e}   (<= 0 expected)")
print(f"K2 RMS        RMS - fac*bracket (max)    : {v2b:.1e}   (<= 0 expected)")

# ---- K3 / K4 -----------------------------------------------------------------
eq3 = v3 = v3b = e4 = 0.0
for _ in range(200):
    n, dk, dv = int(rng.choice([6, 20, 40])), 64, 12
    K = lowrank_keys(n, dk, int(rng.choice([2, 6, 20])), float(rng.choice([0.05, 0.3])))
    Z = rng.normal(size=(dv, dk)) / np.sqrt(dk) * rng.uniform(0.3, 3)
    G, L, A, T, rho, r_eff = ops(K)
    V = K @ Z.T
    F = np.linalg.norm(T @ V, axis=1)
    P = T @ K                                                  # rows p_i
    psi = np.linalg.norm(P, axis=1)
    eq3 = max(eq3, np.abs(F - np.linalg.norm(P @ Z.T, axis=1)).max())
    v3 = max(v3, (F - np.linalg.norm(Z, 2) * psi).max())
    v3b = max(v3b, (psi - rho).max())
    M = np.eye(dk)
    for t in range(n):
        M = M @ (np.eye(dk) - np.outer(K[t], K[t]))
    e4 = max(e4, np.abs(psi - np.linalg.norm(K @ M.T, axis=1)).max())
print(f"K3 F_i = ||Z p_i|| (max abs dev)         : {eq3:.1e}")
print(f"K3 F_i - ||Z||psi_i (max)                : {v3:.1e}   (<= 0 expected)")
print(f"K3 psi_i - rho_i (max)                   : {v3b:.1e}   (<= 0 expected)")
print(f"K4 psi_i = ||Q_1..Q_n k_i|| (max dev)    : {e4:.1e}")

# ---- K5: duplicates ------------------------------------------------------------
dk, dv = 40, 8
base = np.linalg.qr(rng.normal(size=(dk, 5)))[0].T          # orthonormal base keys: only duplicates can interfere
K = np.vstack([base, base[[2]], base[[2]], base[[0]]])         # repeated keys
Z = rng.normal(size=(dv, dk)) / np.sqrt(dk)
V = K @ Z.T                                                    # same key -> same value
_, _, _, T, _, _ = ops(K)
Vc = np.linalg.norm(T @ V, axis=1)
print("K5 duplicates, consistent values:     errors", np.round(Vc, 12), "(all ~0 expected)")
V2 = V.copy(); V2[6] += rng.normal(size=dv)                   # same key, DIFFERENT value (an overwrite)
print("K5 duplicates, one conflicting value: errors", np.round(np.linalg.norm(T @ V2, axis=1), 3),
      f"(items 2,5 overwritten by item 6: expect ||conflict|| = {np.linalg.norm(V2[6]-V[6]):.3f}, others 0)")
