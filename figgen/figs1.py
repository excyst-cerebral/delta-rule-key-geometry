"""Figures: overview, theory verification, measurement study."""
import math
import numpy as np
from tz import Fig, Ax, cmap, rgb, VIRIDIS, DIVERGE
import data as D

# ---- r90 is computed from the raw per-window files; RANKRATIO, INTERF, DECAY are transcribed from the analysis output (notes.md)
MODELS = ["GDN-340M", "GDN-360M", "Qwen3.5-0.8B", "RWKV-7 0.1B", "Qwen2.5-1.5B", "SmolLM2-135M"]
SHORT = ["G1", "G2", "Q3.5", "R7", "Q2.5", "S2"]
FAMILY = ["delta", "delta", "delta", "rwkv", "tf", "tf"]
R90 = {n: [D.r90(k, n) for k, _ in D.CERT_MODELS] for n in (16, 32)}
RANKRATIO = [0.55, 0.85, 0.47, 2.2, 5.5, 6.4]
INTERF = [0.12, 0.23, 0.42]; DECAY = [0.006, 0.04, 0.16]
FAMCOL = {"delta": "cBlue", "rwkv": "cGreen", "tf": "cOrange"}


def sci(x, nd=1):
    m, e = f"{x:.{nd}e}".split("e"); return rf"${m}\times10^{{{int(e)}}}$"


def fig_overview():
    f = Fig(16.0, 7.2); f.rect(0.05, 0.05, 15.95, 7.15, "cGray", "white", 0.6, r=9)
    for x in (5.45, 10.75): f.line([(x, 0.5), (x, 6.9)], "cLight", 0.6)
    top, sub = 6.78, 6.38
    # ---------------- panel A: measure
    f.text(2.75, top, "measure", size="normalsize", bold=True)
    f.text(2.75, sub, r"pooled key-only constant $r_{90}$ ($n{=}32$)", size="tiny", color="cGray")
    A = Ax(f, 1.95, 1.7, 3.15, 3.7, (0, 4), (0, 6)); A.frame(xticks=[0, 1, 2, 3, 4], xtl=["0", "1", "2", "3", "4"], yticks=[], grid="x", xlabel=r"$\mathrm{RMS}\le r_{90}\,\psi$")
    for i, (m, fam, v) in enumerate(zip(MODELS, FAMILY, R90[32])):
        y = 5.5 - i; f.rect(A.X(0), A.Y(y) - 0.2, A.X(v), A.Y(y) + 0.2, None, FAMCOL[fam], 0)
        f.text(A.X(0) - 0.06, A.Y(y), m, anchor="east", size="tiny"); f.text(A.X(v) + 0.05, A.Y(y), f"{v:.2f}", anchor="west", size="tiny")
    f.text(2.75, 5.85, r"\textcolor{cBlue}{$\blacksquare$} delta-rule\quad\textcolor{cGreen}{$\blacksquare$} RWKV-7\quad\textcolor{cOrange}{$\blacksquare$} softmax", size="tiny")
    f.text(2.75, 0.62, r"calibrated on text; carries across domains", size="tiny", color="cGray")
    f.text(2.75, 0.38, r"but not across model families", size="tiny", color="cGray")
    # ---------------- panel B: the law
    f.text(8.1, top, "the interference law", size="normalsize", bold=True)
    f.text(8.1, sub, r"$S_tp=\alpha_tS_{t-1}p-\beta_t\,\textcolor{cGreen}{\langle k_t,p\rangle}\,\textcolor{cGold}{e_t}$", size="small")
    B = Ax(f, 6.75, 1.7, 3.0, 3.7, (0, 1), (0, 1)); N = 20; ed = np.linspace(0, 1, N + 1); ctr = (ed[:-1] + ed[1:]) / 2
    B.heat(np.outer(ctr, ctr), ed, ed, 0, 1)
    for c in (0.1, 0.25, 0.5):
        xs = np.linspace(c, 1, 60); B.line(xs, c / xs, "white", 0.7, "3pt on 2pt")
        B.text(0.97, c / 0.97 + 0.045, f"{c:g}", anchor="east", size="tiny", color="white")
    B.frame(xticks=[0, 0.5, 1], yticks=[0, 0.5, 1], xtl=["0", "0.5", "1"], ytl=["0", "0.5", "1"], grid=None, xlabel=r"\textcolor{cGreen}{key overlap $|\langle k_t,p\rangle|$}", ylabel=r"\textcolor{cGold}{surprise $\|e_t\|$}")
    B.colorbar(0, 1, 9.95, 1.7, 0.16, 3.7, ticks=[0, 0.5, 1], nd=1, label=r"interference$/\beta_t$")
    f.text(8.1, 0.62, r"exact on every real recurrence tested", size="tiny", color="cGray")
    f.text(8.1, 0.38, r"(residual $\le1.6\times10^{-6}$)", size="tiny", color="cGray")
    f.text(8.1, 5.85, r"a write disturbs an old association by overlap $\times$ surprise", size="tiny")
    # ---------------- panel C: predict
    f.text(13.35, top, "predict", size="normalsize", bold=True)
    f.text(13.35, sub, r"recall vs the key-only $\psi$, two Gated DeltaNets", size="tiny", color="cGray")
    C = Ax(f, 11.2, 1.7, 4.3, 3.7, (-0.6, 0.25), (0, 5)); C.frame(xticks=[-0.4, -0.2, 0, 0.2], xtl=["$-0.4$", "$-0.2$", "0", "0.2"], yticks=[], grid="x", xlabel=r"rank correlation of $\psi$ with log-prob.")
    rows = [(nm, *D.TASK_LOG[nm]["lin"], "cOrange") for nm in ("GDN-340M", "GDN-360M")] + [(nm, *D.TASK_LOG[nm]["ex"], "cBlue") for nm in ("GDN-340M", "GDN-360M")]
    C.vline(0, "cGray", 0.6)
    ys = [3.85, 3.05, 1.5, 0.7]
    for (lab, r, lo, hi, c), y in zip(rows, ys):
        C.herr(y, lo, hi, c, 0.08, 1.0); C.marker(r, y, c, "o", 0.09); C.text(-0.585, y + 0.37, lab, anchor="west", size="tiny")
    C.line([-0.6, 0.25], [2.55, 2.55], "cLight", 0.5)
    C.text(-0.585, 4.65, "age as a linear term (pre-registered)", anchor="west", size="tiny", color="cOrange", bold=True); C.text(-0.585, 2.3, "age matched exactly", anchor="west", size="tiny", color="cBlue", bold=True)
    f.text(13.35, 0.62, r"age held fixed exactly", size="tiny", color="cGray"); f.text(13.35, 0.38, r"$\Rightarrow$ no signal beyond distance", size="tiny", color="cGray")
    return f.out()


def theory_data():
    rng = np.random.default_rng(3); out = {}
    xs, ys = [], []
    for _ in range(300):
        n = int(rng.integers(4, 24)); d = int(rng.integers(n + 2, 48)); dv = int(rng.integers(3, 12))
        K = rng.normal(size=(n, d)); K /= np.linalg.norm(K, axis=1, keepdims=True); V = rng.normal(size=(n, dv)); V /= np.linalg.norm(V, axis=1, keepdims=True)
        S = np.zeros((dv, d))
        for t in range(n): S += np.outer(V[t] - S @ K[t], K[t])
        sim = float(((K @ S.T - V) ** 2).sum())
        G = K @ K.T; L = np.tril(G, -1); T = L.T @ np.linalg.inv(np.eye(n) + L); lam, U = np.linalg.eigh(G); Gh = U @ np.diag(np.sqrt(lam)) @ U.T; Gmh = U @ np.diag(1 / np.sqrt(lam)) @ U.T
        M = Gmh @ (V @ V.T) @ Gmh; B_ = Gh @ T.T @ T @ Gh; xs.append(float(np.trace(M @ B_))); ys.append(sim)
    out["xs"], out["ys"] = np.array(xs), np.array(ys); out["relerr"] = float(np.max(np.abs(out["xs"] - out["ys"]) / out["ys"]))
    n, d = 12, 16; lam = 1.6 * 0.6 ** np.arange(n); lam = lam / lam.sum() * n
    U, _ = np.linalg.qr(rng.normal(size=(n, n))); Y, _ = np.linalg.qr(rng.normal(size=(d, n))); K = U @ np.diag(np.sqrt(lam)) @ Y.T
    Wv, _ = np.linalg.qr(rng.normal(size=(n, n))); V = U @ Wv.T
    S = np.zeros((n, d))
    for t in range(n): S += np.outer(V[t] - S @ K[t], K[t])
    out["Bop"] = float(np.linalg.norm(S, 2)); out["err_delta"] = float(((K @ S.T - V) ** 2).sum()); out["floor_at_Bop"] = float(np.sum(np.clip(1 - out["Bop"] * np.sqrt(lam), 0, None) ** 2))
    Bs = np.linspace(0.05, 6, 120); out["Bs"] = Bs; out["floor"] = [float(np.sum(np.clip(1 - b * np.sqrt(lam), 0, None) ** 2)) for b in Bs]
    out["att"] = [float(((K @ sum(min(b, 1 / np.sqrt(l)) * np.outer(Wv[:, j], Y[:, j]) for j, l in enumerate(lam)).T - V) ** 2).sum()) for b in Bs]
    coh = {}
    nn = 30
    for mu in (0.01, 0.03, 0.06):
        Kc = np.array([math.sqrt(mu) * np.eye(nn + 1)[0] + math.sqrt(1 - mu) * np.eye(nn + 1)[i + 1] for i in range(nn)]); v = np.zeros(nn + 1); v[0] = 1.0
        Sm = np.zeros((nn + 1, nn + 1))
        for t in range(nn): Sm += np.outer(v - Sm @ Kc[t], Kc[t])
        Fi = [float(np.linalg.norm(Sm @ Kc[i] - v)) for i in range(nn)]; ii = np.arange(1, nn + 1); coh[mu] = (Fi, (1 - mu) ** ii - (1 - mu) ** nn, float(np.max(np.abs(np.array(Fi) - ((1 - mu) ** ii - (1 - mu) ** nn)))))
    out["coh"] = coh; return out


def fig_theory():
    t = theory_data(); f = Fig(16.0, 5.7)
    xs, ys = t["xs"], t["ys"]; lo = 10 ** math.floor(math.log10(min(xs.min(), ys.min()))); hi = 10 ** math.ceil(math.log10(max(xs.max(), ys.max())))
    a = Ax(f, 1.45, 1.1, 3.9, 3.7, (lo, hi), (lo, hi), xlog=True, ylog=True)
    decs = list(range(int(math.log10(lo)), int(math.log10(hi)) + 1, 2)); tk = [10.0 ** e for e in decs]; tl = [rf"$10^{{{e}}}$" for e in decs]
    a.frame(xticks=tk, xtl=tl, yticks=tk, ytl=tl, grid="xy", panel="a", title="gain-weighted identity", xlabel=r"$\mathrm{tr}(MB)$ (closed form)", ylabel=r"$n\,\mathrm{RMS}^2$ (simulated pass)")
    a.line([lo, hi], [lo, hi], "cGray", 0.6, "3pt on 2pt"); a.scatter(xs, ys, "cBlue", r=0.05, opacity=0.55)
    a.rtext(0.05, 0.92, "300 random instances", anchor="west", size="tiny"); a.rtext(0.05, 0.83, rf"max. relative error {sci(t['relerr'])}", anchor="west", size="tiny", color="cGreen")
    b_ = Ax(f, 6.55, 1.1, 3.9, 3.7, (0, 6), (0, 12)); b_.frame(xticks=[0, 2, 4, 6], yticks=[0, 4, 8, 12], grid="xy", panel="b", title="capacity floor", xlabel=r"operator-norm budget $B$", ylabel=r"squared recall error")
    b_.line(t["Bs"], t["floor"], "cRed", 1.4); b_.line(t["Bs"], t["att"], "cBlue", 0.9, "4pt on 2pt"); b_.marker(t["Bop"], t["err_delta"], "cGold", "d", 0.09)
    b_.legend([(r"floor $\sum_j(1-B\sqrt{\lambda_j})_+^2$", "cRed", "line"), ("attaining memory", "cBlue", "dash"), ("one delta-rule pass", "cGold", "d")], 10.35, 4.75, anchor="north east", size="tiny", rowh=0.27, cw=0.15)
    c_ = Ax(f, 11.65, 1.1, 3.9, 3.7, (1, 30), (0, 1.0)); c_.frame(xticks=[1, 10, 20, 30], yticks=[0, 0.25, 0.5, 0.75, 1.0], ytl=["0", "0.25", "0.5", "0.75", "1"], grid="xy", panel="c", title=r"coherence alone: $\Theta(\mu n)$", xlabel=r"item index $i$ (of $n=30$ writes)", ylabel=r"error $F_i$ of item $i$")
    for mu, cc in zip((0.01, 0.03, 0.06), ("cBlue", "cGreen", "cOrange")):
        Fi, form, dev = t["coh"][mu]; ii = np.arange(1, 31); c_.line(ii, Fi, cc, 1.2); c_.scatter(ii, form, "cInk", r=0.03, opacity=0.9)
        f.text(c_.X(3), c_.Y(Fi[2]) + 0.14, rf"$\mu={mu}$", anchor="south west", size="tiny", color=cc)
    c_.legend([("simulated pass", "cGray", "line"), (r"formula $(1-\mu)^i-(1-\mu)^n$", "cInk", "o")], 15.5, 4.7, anchor="north east", size="tiny", rowh=0.27, cw=0.13)
    return f.out(), dict(relerr=t["relerr"], floor_gap=t["err_delta"] - t["floor_at_Bop"], coh_dev=max(v[2] for v in t["coh"].values()))


def fig_measure():
    f = Fig(16.0, 5.7)
    a = Ax(f, 1.5, 1.3, 4.2, 3.5, (0, 6), (1.0, 4.0)); a.frame(xticks=[i + 0.5 for i in range(6)], xtl=SHORT, yticks=[1, 2, 3, 4], grid="y", panel="a", title="certificate constant", ylabel=r"pooled $r_{90}$")
    for i in range(6):
        c = FAMCOL[FAMILY[i]]; a.line([i + 0.5, i + 0.5], [R90[16][i], R90[32][i]], c, 1.4); a.marker(i + 0.5, R90[16][i], c, "o", 0.08, fill=False, lw=1.0); a.marker(i + 0.5, R90[32][i], c, "o", 0.08)
    f.text(a.x0 + 0.1, a.y0 + 0.2, r"$\circ\ n{=}16\quad\bullet\ n{=}32$", anchor="west", size="tiny")
    f.text(8.0, 0.22, r"G1, G2: Gated DeltaNet 340M, 360M;\quad Q3.5: Qwen3.5-0.8B;\quad R7: RWKV-7 0.1B;\quad Q2.5: Qwen2.5-1.5B;\quad S2: SmolLM2-135M", size="tiny", color="cGray")
    b = Ax(f, 6.9, 1.3, 4.2, 3.5, (0.3, 9), (1.2, 3.9), xlog=True); b.frame(xticks=[0.5, 1, 2, 5], xtl=["0.5", "1", "2", "5"], yticks=[1.5, 2, 2.5, 3, 3.5], grid="xy", panel="b", title="what sets the level", xlabel=r"values-to-keys effective-rank ratio ($n{=}32$)", ylabel=r"pooled $r_{90}$ ($n{=}32$)")
    shift = {"G1": (0.0, -0.12, "north"), "G2": (0.12, -0.04, "west"), "Q3.5": (-0.1, 0.12, "south east"), "R7": (0.12, 0.0, "west"), "Q2.5": (-0.12, 0.0, "east"), "S2": (0.12, -0.04, "west")}
    for i in range(6):
        c = FAMCOL[FAMILY[i]]; b.marker(RANKRATIO[i], R90[32][i], c, "o", 0.09); dx, dy, an = shift[SHORT[i]]; px, py = b.P(RANKRATIO[i], R90[32][i]); f.text(px + dx, py + dy, SHORT[i], anchor=an, size="tiny", color=c)
    b.vline(1.0, "cGray", 0.5, "2pt on 2pt"); b.rtext(0.03, 0.94, r"ratio $\le1$: values no richer than keys", anchor="west", size="tiny", color="cGray")
    c = Ax(f, 12.3, 1.3, 3.3, 3.5, (0, 3), (0, 0.55)); c.frame(xticks=[0.5, 1.5, 2.5], xtl=["16", "64", "256"], yticks=[0, 0.1, 0.2, 0.3, 0.4, 0.5], grid="y", panel="c", title="interference vs decay", xlabel="tokens after the write", ylabel="relative loss")
    c.bars([0.5 - 0.17, 1.5 - 0.17, 2.5 - 0.17], INTERF, 0.5, "cRed"); c.bars([0.5 + 0.17, 1.5 + 0.17, 2.5 + 0.17], DECAY, 0.5, "cBlue")
    for i in range(3):
        c.text(i + 0.5 - 0.17, INTERF[i] + 0.012, f"{INTERF[i]:g}", anchor="south", size="tiny"); c.text(i + 0.5 + 0.17, DECAY[i] + 0.012, f"{DECAY[i]:g}", anchor="south", size="tiny")
    c.legend([("later writes", "cRed", "box"), ("designed decay", "cBlue", "box")], 12.38, 4.72, anchor="north west", size="tiny", rowh=0.27)
    return f.out()
