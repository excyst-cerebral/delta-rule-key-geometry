"""Synthetic checks for the rank-based restatement (alpha = beta = 1, unit keys, single pass).

Notation: G = K K^T (n x n), eigenvalues lam_j, L = strict lower triangle of G, W = (I+L)^{-1} V,
error matrix E = L^T W (row i = recall error of item i).

  r_eff(G)  = n^2 / ||G||_F^2                       (participation-ratio effective rank)
  r_tau(G)  = #{ j : lam_j >= tau }                  (threshold / numerical rank)

R1  identity   ||L||_F^2 = (n^2/r_eff - n)/2           and  n/r_eff - 1 <= (n-1) mu^2
R2  Thm 5'     RMS_i F_i <= sqrt((n/r_eff - 1)/2) * ||W||_op
R3  Lemma 6    ||W||_op <= sqrt(||Pi V||_op^2/tau + ||N||_op^2) + ||N||_op,  Pi = spectral proj (lam>=tau), N=(I-Pi)V
R4  Lemma 7    ||S_t||_op <= ||V_{<=t}||_F   (delta-rule state norm is automatically bounded)
R5  Thm 3'     any S with ||S||_op <= B:  ||K S^T - V||_F^2 >= sum_{j in J} (1 - B sqrt(lam_j))_+^2
               for V = sum_{j in J} u_j w_j^T (w_j orthonormal); tight; applies to the delta rule with B = ||V||_F
R6  Thm 4'     Lambda := ||L||_F < 1  =>  RMS_i F_i <= Lambda/(1-Lambda)  for unit-norm values
"""
import numpy as np

rng = np.random.default_rng(3)


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def lowrank_keys(n, dk, r, noise):
    Q = np.linalg.qr(rng.normal(size=(dk, r)))[0]
    A = rng.normal(size=(n, r)) + 0.4 * rng.normal(size=(1, r))        # shared component, like trained keys
    return unit(A @ Q.T + noise * rng.normal(size=(n, dk)) / np.sqrt(dk))


def delta_pass(K, V, track_norm=False):
    n, dk = K.shape
    S = np.zeros((V.shape[1], dk))
    worst = 0.0
    for t in range(n):
        S = S + np.outer(V[t] - S @ K[t], K[t])
        if track_norm:
            worst = max(worst, np.linalg.norm(S, 2) / max(np.linalg.norm(V[:t + 1]), 1e-12))
    return S, worst


def parts(K, V):
    G = K @ K.T
    n = len(G)
    L = np.tril(G, -1)
    W = np.linalg.solve(np.eye(n) + L, V)
    E = L.T @ W
    return G, L, W, E


r_eff = lambda G: len(G) ** 2 / np.linalg.norm(G) ** 2

# ---- R1 / R2 -----------------------------------------------------------------
res1, viol2, slack2, viol1b = 0.0, -np.inf, [], -np.inf
for _ in range(300):
    n, dk = int(rng.choice([8, 24, 64])), 128
    r, noise = int(rng.choice([2, 6, 20, 80])), float(rng.choice([0.05, 0.3, 1.0]))
    K = lowrank_keys(n, dk, min(r, dk), noise)
    V = unit(rng.normal(size=(n, int(rng.choice([4, 32, 128])))))
    G, L, W, E = parts(K, V)
    re = r_eff(G)
    res1 = max(res1, abs(np.linalg.norm(L) ** 2 - (n ** 2 / re - n) / 2))
    mu = np.abs(G - np.eye(n)).max()
    viol1b = max(viol1b, (n / re - 1) - (n - 1) * mu ** 2)
    rms = np.linalg.norm(E) / np.sqrt(n)
    bound = np.sqrt((n / re - 1) / 2) * np.linalg.norm(W, 2)
    viol2 = max(viol2, rms - bound); slack2.append(bound / max(rms, 1e-12))
print(f"R1 identity residual                      : {res1:.1e}")
print(f"R1 n/r_eff-1 <= (n-1)mu^2 (max violation) : {viol1b:.1e}   (<= 0 expected)")
print(f"R2 RMS - bound (max)                      : {viol2:.1e}   (<= 0 expected); median bound/RMS = {np.median(slack2):.2f}")

# ---- R3 ------------------------------------------------------------------------
viol3, slack3 = -np.inf, []
for _ in range(300):
    n, dk = int(rng.choice([16, 40])), 96
    K = lowrank_keys(n, dk, int(rng.choice([3, 8, 30])), float(rng.choice([0.05, 0.3])))
    V = unit(rng.normal(size=(n, int(rng.choice([4, 32])))))
    G, L, W, E = parts(K, V)
    lam, U = np.linalg.eigh(G)
    tau = float(rng.choice([1e-3, 1e-2, 0.1, 1.0]))
    if (lam >= tau).sum() == 0:
        continue
    Pi = (U[:, lam >= tau]) @ U[:, lam >= tau].T
    N = V - Pi @ V
    nop = np.linalg.norm(N, 2)
    b = np.sqrt(np.linalg.norm(Pi @ V, 2) ** 2 / tau + nop ** 2) + nop
    viol3 = max(viol3, np.linalg.norm(W, 2) - b); slack3.append(b / np.linalg.norm(W, 2))
print(f"R3 ||W||_op - bound (max)                 : {viol3:.1e}   (<= 0 expected); median bound/||W|| = {np.median(slack3):.2f}")

# ---- R4 ------------------------------------------------------------------------
w4 = 0.0
for _ in range(100):
    n, dk = 60, 48
    K = lowrank_keys(n, dk, int(rng.choice([2, 10, 40])), 0.05)
    V = rng.normal(size=(n, 16)) * rng.uniform(0.2, 3)
    _, worst = delta_pass(K, V, track_norm=True)
    w4 = max(w4, worst)
print(f"R4 max ||S_t||_op / ||V_<=t||_F           : {w4:.3f}   (<= 1 expected)")

# ---- R5 ------------------------------------------------------------------------
rows = []
for _ in range(200):
    n, dk, dv = 40, 96, 40
    K = lowrank_keys(n, dk, int(rng.choice([4, 8, 12])), float(rng.choice([0.1, 0.3, 0.6])))
    G = K @ K.T
    lam, U = np.linalg.eigh(G)
    m = int(rng.choice([2, 4, 8, 16]))
    J = np.arange(m)                                              # m smallest eigen-directions
    Wo = np.linalg.qr(rng.normal(size=(dv, m)))[0]                 # orthonormal w_j
    V = U[:, J] @ Wo.T
    S, _ = delta_pass(K, V)
    err2 = np.linalg.norm(K @ S.T - V) ** 2
    B = np.linalg.norm(V)                                          # = sqrt(m)
    floor = ((1 - B * np.sqrt(np.maximum(lam[J], 0))).clip(min=0) ** 2).sum()
    rows.append((err2 - floor, floor / m, err2 / m))
rows = np.array(rows)
print(f"R5 min (err^2 - floor)                    : {rows[:,0].min():.3e}  (>= 0 expected)")
print(f"R5 relative err^2: floor median {np.median(rows[:,1]):.3f}   delta-rule median {np.median(rows[:,2]):.3f}")

# R5 tightness: best B-bounded S attains the floor (construct it)
n, dk, dv, m = 30, 80, 30, 6
K = lowrank_keys(n, dk, 6, 0.3)
G = K @ K.T; lam, U = np.linalg.eigh(G)
Wo = np.linalg.qr(rng.normal(size=(dv, m)))[0]
V = U[:, :m] @ Wo.T
B = 3.0
# right singular vector paired with eigenvalue lam_j:  y_j = K^T u_j / sqrt(lam_j)
Y = (K.T @ U[:, :m]) / np.sqrt(lam[:m])
# S = sum_j a_j w_j y_j^T:  K S^T = sum_j sqrt(lam_j) a_j u_j w_j^T,  ||S||_op = max_j |a_j|
a = np.minimum(B, 1 / np.sqrt(lam[:m]))
St = sum(np.outer(Y[:, j], a[j] * Wo[:, j]) for j in range(m))     # S^T  (dk x dv)
assert np.linalg.norm(St.T, 2) <= B + 1e-9
opt = np.linalg.norm(K @ St - V) ** 2
floor = ((1 - B * np.sqrt(np.maximum(lam[:m], 0))).clip(min=0) ** 2).sum()
print(f"R5 tightness: budget-B optimum err^2 = {opt:.6f}, floor = {floor:.6f}")

# ---- R6 ------------------------------------------------------------------------
viol6, used = -np.inf, 0
for _ in range(400):
    n, dk = int(rng.choice([4, 8, 12])), int(rng.choice([300, 800, 2000]))
    K = unit(rng.normal(size=(n, dk)))
    V = unit(rng.normal(size=(n, int(rng.choice([1, 4, 32])))))
    G, L, W, E = parts(K, V)
    Lam = np.linalg.norm(L)
    if Lam < 1:
        used += 1
        viol6 = max(viol6, np.linalg.norm(E) / np.sqrt(n) - Lam / (1 - Lam))
print(f"R6 RMS - Lambda/(1-Lambda) (max, {used} cases with Lambda<1): {viol6:.1e}  (<= 0 expected)")
