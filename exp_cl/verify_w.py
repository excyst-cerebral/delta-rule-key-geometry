"""Numerical checks for the weight-space statements (numpy only).
W1  one SGD step on a linear-in-features model: Delta f(x) = -eta * l'(f(x')) <phi(x), phi(x')>   (squared and logistic loss)
W2  converged GD on an interpolable new task from theta0:  Delta f_A = Phi_A Phi_B^+ (y_B - Phi_B theta0);  zero for all y_B  <=>  Phi_A P_B = 0
W3  2-D no-memory construction: any theta' with theta'_1 ~ 1 has max old loss >= (theta1^2 + theta2^2)/2  (checked for random learners)
W4  gated features z_c = g_c psi_c, readout-only SGD:  |Delta f(x_old)| <= eta Psi^2 sum_s |l'_s| sum_c g_c(x_old) g_c(x_s)
"""
import numpy as np
rng = np.random.default_rng(0)
# W1
d, eta = 12, 0.1
th = rng.normal(size=d); phi_old, phi_new = rng.normal(size=d), rng.normal(size=d)
y = 0.7
f = lambda t, p: t @ p
dsq = f(th - eta * (f(th, phi_new) - y) * phi_new, phi_old) - f(th, phi_old)
pred_sq = -eta * (f(th, phi_new) - y) * (phi_old @ phi_new)
sig = lambda a: 1 / (1 + np.exp(-a))
yb = 1.0; lp = sig(f(th, phi_new)) - yb
dlog = f(th - eta * lp * phi_new, phi_old) - f(th, phi_old); pred_log = -eta * lp * (phi_old @ phi_new)
print(f"W1 identity (squared / logistic): max |actual - predicted| = {abs(dsq - pred_sq):.1e} / {abs(dlog - pred_log):.1e}")
# W2
worst, worst0 = 0.0, 0.0
for _ in range(200):
    p, nA, nB = 10, 6, 5
    PhiA, PhiB = rng.normal(size=(nA, p)), rng.normal(size=(nB, p)); th0 = rng.normal(size=p); yB = rng.normal(size=nB)
    t = th0.copy()
    for _ in range(4000): t -= 0.01 * PhiB.T @ (PhiB @ t - yB)               # gradient descent to convergence (interpolates: nB < p)
    pred = PhiA @ np.linalg.pinv(PhiB) @ (yB - PhiB @ th0)
    worst = max(worst, np.abs(PhiA @ (t - th0) - pred).max())
# orthogonal feature subspaces => no change for arbitrary yB
p = 10; Q = np.linalg.qr(rng.normal(size=(p, p)))[0]; PhiA, PhiB = rng.normal(size=(4, 4)) @ Q[:, :4].T, rng.normal(size=(5, 5)) @ Q[:, 4:9].T
for _ in range(50):
    th0, yB = rng.normal(size=p), rng.normal(size=5); t = th0.copy()
    for _ in range(4000): t -= 0.01 * PhiB.T @ (PhiB @ t - yB)
    worst0 = max(worst0, np.abs(PhiA @ (t - th0)).max())
print(f"W2 converged-GD closed form: max error {worst:.1e};  orthogonal subspaces: max |Delta f_A| = {worst0:.1e}")
# W3: learners that fit B = (input (1,0), label 1) from theta0 = 0 and then do anything else (random extra moves in the null direction of B)
viol = []
for _ in range(2000):
    eps = rng.uniform(0, 0.05)
    t1 = 1 - np.sqrt(eps) * rng.uniform(0, 1); t2 = rng.normal() * rng.uniform(0, 3)        # fits B to within eps; arbitrary second coordinate
    lossP, lossM = (t1 + t2) ** 2 / 2, (t1 - t2) ** 2 / 2                                   # A_+ : input (1,1)/sqrt2 label 0 ; A_- : (1,-1)/sqrt2 label 0
    viol.append(max(lossP, lossM) - (1 - np.sqrt(eps)) ** 2 / 2)
print(f"W3 min over learners of [max old loss - (1-sqrt(eps))^2/2] = {min(viol):.2e}  (>= 0 expected)")
# W4
C, ns = 20, 30
worst4 = -np.inf
for _ in range(200):
    g_old = rng.uniform(0, 1, C) * (rng.uniform(size=C) < rng.uniform(0.1, 0.9)); psi_old = rng.uniform(-1, 1, C)
    w = rng.normal(size=C); tot = 0.0; bound = 0.0; wc = w.copy()
    G_new = rng.uniform(0, 1, (ns, C)) * (rng.uniform(size=(ns, C)) < 0.3); Psi_new = rng.uniform(-1, 1, (ns, C)); yv = rng.normal(size=ns)
    z_old = g_old * psi_old
    for s in range(ns):
        z = G_new[s] * Psi_new[s]; lp = (wc @ z - yv[s]); before = wc @ z_old
        wc = wc - 0.05 * lp * z; tot += abs(wc @ z_old - before)
        bound += 0.05 * abs(lp) * float((g_old * G_new[s]).sum()) * 1.0
    worst4 = max(worst4, np.abs(wc @ z_old - w @ z_old) - bound)
print(f"W4 gate-overlap bound: max (|Delta f| - bound) = {worst4:.2e}  (<= 0 expected)")
