"""Stage B (numpy only): weight-space law on SSM features.  Readout-only sequential learning: fit task A by least squares, then fine-tune on B by
gradient descent (squared loss, +-1 targets).  Test whether the accuracy drop on A is explained by feature-subspace overlap x surprise (Prop. 13)."""
import numpy as np, glob
rk = lambda x: np.argsort(np.argsort(x)).astype(float); sp = lambda a, b: float(np.corrcoef(rk(a), rk(b))[0, 1])
rows = []
for arch in ("lti", "sel"):
    for seed in range(5):
        D = np.load(f"exp_cl/runs/feat_{arch}_s{seed}.npz"); M = 4
        Phi = {t: np.hstack([D[f"phi{t}"], np.ones((len(D[f"phi{t}"]), 1))]) for t in range(M)}; Y = {t: D[f"y{t}"] for t in range(M)}
        for A in range(M):
            th_A = np.linalg.lstsq(Phi[A], Y[A], rcond=None)[0]
            acc0 = np.mean(np.sign(Phi[A] @ th_A) == Y[A])
            for B in range(M):
                if B == A: continue
                lr = 0.5 / np.linalg.eigvalsh(Phi[B].T @ Phi[B] / len(Phi[B])).max()
                th = th_A.copy()
                for _ in range(200): th -= lr * Phi[B].T @ (Phi[B] @ th - Y[B]) / len(Phi[B])
                drop = acc0 - np.mean(np.sign(Phi[A] @ th) == Y[A])
                dfA = Phi[A] @ (th - th_A)                                              # actual change of the old predictions
                U, s, Vt = np.linalg.svd(Phi[B], full_matrices=False); P = Vt[s > 1e-2 * s[0]]
                overlap = np.linalg.norm(Phi[A] @ P.T) / np.linalg.norm(Phi[A])          # ||Phi_A P_B||_F / ||Phi_A||_F
                surprise = np.mean((Phi[B] @ th_A - Y[B]) ** 2)
                # first-order law: sum over GD steps of -lr * Phi_A Phi_B^T e/n   (exact for the first step; here compare magnitude with the realised change)
                step1 = lr * Phi[A] @ Phi[B].T @ (Phi[B] @ th_A - Y[B]) / len(Phi[B])
                rows.append((arch, seed, A, B, drop, overlap, surprise, np.linalg.norm(dfA) / np.linalg.norm(Phi[A] @ th_A), np.linalg.norm(step1) / np.linalg.norm(Phi[A] @ th_A), acc0))
for arch in ("lti", "sel", "both"):
    R = [r for r in rows if arch == "both" or r[0] == arch]; d, o, s_, ch, st = (np.array([r[i] for r in R]) for i in (4, 5, 6, 7, 8))
    print(f"{arch:>4}: {len(R)} (A -> B) events | Spearman(drop, overlap) {sp(d, o):+.2f} | (drop, surprise) {sp(d, s_):+.2f} | (drop, overlap x surprise) {sp(d, o * s_):+.2f}"
          f" | relative change of old predictions vs first-step law: Spearman {sp(ch, st):+.2f}, median ratio {np.median(ch / np.maximum(st, 1e-12)):.1f}")
