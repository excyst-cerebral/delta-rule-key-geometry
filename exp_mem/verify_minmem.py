"""Numerical checks for the minimal-memory theorem (numpy only).
Setting: f(x)=<theta,phi(x)>, theta in R^p.  Old task: unknown r-dim feature subspace U (orthonormal Q, P_U = Q Q^T), theta_0 fits it.
New task: one example (feature w, unit norm, with ||(I-P_U) w|| >= kappa) requiring Delta-theta with <w, Delta-theta> = e.
Interference of the old task:  delta' = ||P_U Delta-theta||  (max change over unit old features).
Learner with memory: stores a codeword U^ (subspace) chosen from the stored message, then Delta-theta = e (I-P_U^) w / ||(I-P_U^) w||^2.
 M1  upper bound:   delta' <= |e| ||P_U - P_U^||_op / ||(I-P_U^) w||^2, and <w,Delta-theta> = e exactly
 M2  lower-bound construction: for U1,U2 with largest principal vectors a1 in U1, a2 in U2, w=(a1-a2)/||a1-a2||:
        ||(I-P_U1) w|| = cos(theta_max/2)  (realisable),  |<w,D>| <= (||P_U1 D|| + ||P_U2 D||)/(2 sin(theta_max/2)) for every D
 M3  scaling: with a random codebook of 2^m subspaces, the median interference falls like 2^(-m/(r(p-r)))
"""
import numpy as np
rng = np.random.default_rng(0)


def haar(p, r, n=None):
    shp = (p, r) if n is None else (n, p, r)
    A = rng.normal(size=shp); Q, _ = np.linalg.qr(A); return Q


def gap(Q1, Q2):                                   # ||P1 - P2||_op = sin(theta_max)
    s = np.linalg.svd(Q1.T @ Q2, compute_uv=False)
    return float(np.sqrt(max(0.0, 1 - s.min() ** 2)))


# M1
worst, worst_fit = -np.inf, 0.0
for _ in range(2000):
    p, r = int(rng.choice([4, 6, 10])), int(rng.choice([1, 2, 3])); 
    Q = haar(p, r); E = 0.15 * rng.normal(size=(p, r)); Qh = np.linalg.qr(Q + E)[0]
    w = rng.normal(size=p); w /= np.linalg.norm(w); e = rng.normal()
    wp = w - Qh @ (Qh.T @ w)
    if np.linalg.norm(wp) < 0.2: continue
    D = e * wp / (wp @ wp); worst_fit = max(worst_fit, abs(w @ D - e))
    delta = np.linalg.norm(Q.T @ D)                # ||P_U D|| (Q orthonormal)
    worst = max(worst, delta - abs(e) * gap(Q, Qh) / (wp @ wp))
print(f"M1 upper bound: max(delta' - bound) = {worst:.2e} (<= 0 expected); max |<w,D> - e| = {worst_fit:.1e}")

# M2
viol, real_err = -np.inf, 0.0
for _ in range(2000):
    p, r = int(rng.choice([4, 6, 10])), int(rng.choice([1, 2, 3]))
    Q1, Q2 = haar(p, r), haar(p, r)
    U, s, Vt = np.linalg.svd(Q1.T @ Q2)              # principal vectors: a1 = Q1 U[:,k], a2 = Q2 V[:,k]; angle = arccos(s_k)
    k = int(np.argmin(s)); a1, a2 = Q1 @ U[:, k], Q2 @ Vt[k]
    th = np.arccos(np.clip(s[k], -1, 1)); w = (a1 - a2); nw = np.linalg.norm(w); w /= nw
    real_err = max(real_err, abs(np.linalg.norm(w - Q1 @ (Q1.T @ w)) - np.cos(th / 2)))
    for _ in range(20):
        D = rng.normal(size=p)
        lhs = abs(w @ D); rhs = (np.linalg.norm(Q1.T @ D) + np.linalg.norm(Q2.T @ D)) / (2 * np.sin(th / 2))
        viol = max(viol, lhs - rhs)
print(f"M2 pair construction: max(|<w,D>| - bound) = {viol:.2e} (<= 0 expected); |kappa - cos(theta/2)| = {real_err:.1e}")

# M3
print("\nM3 scaling: bits m -> median interference delta' (|e| = 1, random Haar old subspace and new feature, nearest of 2^m random codewords)")
print(f"{'(r,p)':>7} {'r(p-r)':>6} | " + " ".join(f"m={m:<2}" .rjust(8) for m in (2, 4, 6, 8, 10)) + " | fitted decay exponent (per bit)  vs  1/(r(p-r))")
for (r, p) in [(1, 3), (1, 5), (2, 5), (2, 6), (3, 6), (1, 8)]:
    ms = (2, 4, 6, 8, 10); med = []
    for m in ms:
        cb = haar(p, r, 2 ** m); trials = 300 if m < 10 else 120; ds = []
        for _ in range(trials):
            Q = haar(p, r); S = np.linalg.svd(np.einsum("pr,npq->nrq", Q, cb), compute_uv=False)
            j = int(np.argmax(S.min(axis=1)))        # codeword with smallest gap
            Qh = cb[j]; w = rng.normal(size=p); w /= np.linalg.norm(w); wp = w - Qh @ (Qh.T @ w)
            if np.linalg.norm(wp) < 0.3: continue
            D = wp / (wp @ wp); ds.append(np.linalg.norm(Q.T @ D))
        med.append(np.median(ds))
    slope = np.polyfit(ms, np.log2(med), 1)[0]
    print(f"({r},{p})".rjust(7) + f" {r*(p-r):6d} | " + " ".join(f"{x:8.3f}" for x in med) + f" | {slope:+.3f}   vs  {-1/(r*(p-r)):+.3f}")
