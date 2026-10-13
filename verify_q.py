"""Synthetic checks for the q-identities (alpha = beta = 1, unit keys, linearly independent keys).
H = V V^T, M = G^{-1/2} H G^{-1/2}, q = tr(M)/n = tr(V^T G^-1 V)/n, B = G^{1/2} T^T T G^{1/2},  T = L^T (I+L)^-1.
Q1  n RMS^2 = tr(M B),   tr B = n psi_rms^2
Q2  n RMS^2 <= min( lam_max(M) tr B , ||B||_op tr M )   i.e.  RMS <= ||Z||_op psi_rms  and  RMS <= ||TK||_op sqrt(q)
Q3  H = g^2 G  =>  F_i = g psi_i exactly, q = g^2
Q4  random pairing:  E_pi[q] = [ tr(G^-1)(1-hbar) + hbar 1^T G^-1 1 ] / n,  hbar = (||sum v||^2 - n)/(n(n-1))   (exhaustive, n = 6)
"""
import itertools
import numpy as np

rng = np.random.default_rng(2)
unit = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True)


def lowrank_keys(n, dk, r, noise):
    Q = np.linalg.qr(rng.normal(size=(dk, r)))[0]
    A = rng.normal(size=(n, r)) + 0.4 * rng.normal(size=(1, r))
    return unit(A @ Q.T + noise * rng.normal(size=(n, dk)) / np.sqrt(dk))


def setup(K):
    n = len(K); G = K @ K.T
    L = np.tril(G, -1); T = L.T @ np.linalg.inv(np.eye(n) + L)
    lam, U = np.linalg.eigh(G)
    Gh = U @ np.diag(np.sqrt(lam)) @ U.T; Gmh = U @ np.diag(lam ** -0.5) @ U.T
    return G, T, Gh, Gmh


e1 = v2a = v2b = e3 = e3q = -np.inf
e1 = e3 = e3q = 0.0
for _ in range(300):
    n, dk, dv = int(rng.choice([6, 16, 40])), 96, int(rng.choice([4, 32, 96]))
    K = lowrank_keys(n, dk, int(rng.choice([3, 10, 40])), float(rng.choice([0.1, 0.4])))
    V = unit(rng.normal(size=(n, dv)) + float(rng.choice([0, 1, 3])) * rng.normal(size=(1, dv)) + float(rng.choice([0, 1])) * (K @ rng.normal(size=(dk, dv)) / 4))
    G, T, Gh, Gmh = setup(K)
    H = V @ V.T
    M = Gmh @ H @ Gmh
    B = Gh @ T.T @ T @ Gh
    rms2 = np.linalg.norm(T @ V) ** 2 / n
    q = np.trace(M) / n
    psi2 = np.linalg.norm(T @ K) ** 2 / n
    e1 = max(e1, abs(n * rms2 - np.trace(M @ B)) / (n * rms2 + 1e-12), abs(np.trace(B) - n * psi2) / (n * psi2))
    e3q = max(e3q, abs(q - np.trace(V.T @ np.linalg.solve(G, V)) / n))
    lm = np.linalg.eigvalsh(M).max(); Bop = np.linalg.norm(T @ K, 2) ** 2
    v2a = max(v2a, n * rms2 - lm * np.trace(B)); v2b = max(v2b, n * rms2 - Bop * np.trace(M))
print(f"Q1 identity n*RMS^2 = tr(MB), tr B = n psi^2  (max rel. dev) : {e1:.1e};   q = tr(V^T G^-1 V)/n : {e3q:.1e}")
print(f"Q2 n RMS^2 - lam_max(M) tr B (max, <= 0)                      : {v2a:.1e}")
print(f"Q2 n RMS^2 - ||TK||_op^2 tr M (max, <= 0)                     : {v2b:.1e}")

dev, devq = 0.0, 0.0
for _ in range(100):
    n, dk = int(rng.choice([6, 20])), 64
    K = lowrank_keys(n, dk, 8, 0.3); G, T, *_ = setup(K)
    g = rng.uniform(0.3, 2.5)
    V = g * K                                               # H = g^2 G  (dv = dk)
    F = np.linalg.norm(T @ V, axis=1); psi = np.linalg.norm(T @ K, axis=1)
    dev = max(dev, np.abs(F - g * psi).max()); devq = max(devq, abs(np.trace(V.T @ np.linalg.solve(G, V)) / n - g ** 2))
print(f"Q3 flat gain: max |F_i - g psi_i| = {dev:.1e},  max |q - g^2| = {devq:.1e}")

worst = 0.0
for _ in range(10):
    n, dk, dv = 6, 40, 12
    K = lowrank_keys(n, dk, 4, 0.3); G = K @ K.T; Gi = np.linalg.inv(G)
    V = unit(rng.normal(size=(n, dv)) + 1.0 * rng.normal(size=(1, dv)))
    qs = [np.trace(V[list(p)].T @ Gi @ V[list(p)]) / n for p in itertools.permutations(range(n))]
    hbar = (np.linalg.norm(V.sum(0)) ** 2 - n) / (n * (n - 1))
    formula = (np.trace(Gi) * (1 - hbar) + hbar * np.ones(n) @ Gi @ np.ones(n)) / n
    worst = max(worst, abs(np.mean(qs) - formula) / formula)
print(f"Q4 exhaustive E_pi[q] vs formula (n=6, 720 permutations): max rel. dev = {worst:.1e}")
