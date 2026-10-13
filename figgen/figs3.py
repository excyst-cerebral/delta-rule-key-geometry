"""Figures: weight-space theory (frontier, minimal memory, no-memoryless), MNIST benchmarks, deep SSM, adapters, toy testbed."""
import math
import numpy as np
from tz import Fig, Ax, cmap, VIRIDIS, DIVERGE
import data as D


# ------------------------------------------------------------------------------------------------ frontier theory
def frontier_instance(p=12, r=5, n=10, seed=7):
    rng = np.random.default_rng(seed); Q = np.linalg.qr(rng.normal(size=(p, p)))[0]; P = Q[:, :r] @ Q[:, :r].T
    Phi = rng.normal(size=(n, p)); e = rng.normal(size=n); e /= np.linalg.norm(e); return Phi, P, e


def frontier_point(Phi, P, e, mu):
    D_ = np.linalg.pinv(Phi.T @ Phi + mu * P) @ Phi.T @ e; return float(np.linalg.norm(P @ D_)), float(np.linalg.norm(Phi @ D_ - e))


def frontier_numbers():
    p, r, n = 12, 5, 10; Phi, P, e = frontier_instance(p, r, n); G = Phi @ (np.eye(p) - P); Pi = G @ np.linalg.pinv(G); c = float(np.linalg.norm((np.eye(n) - Pi) @ e))
    mus = np.geomspace(1e-3, 1e4, 60); pts = [frontier_point(Phi, P, e, m) for m in mus]
    return dict(p=p, r=r, n=n, c=c, mus=mus, pts=pts, Phi=Phi, P=P, e=e)


def fig_frontier():
    F = frontier_numbers(); f = Fig(16.0, 5.7); pts = np.array(F["pts"]); mus = F["mus"]
    a = Ax(f, 1.45, 1.1, 3.9, 3.7, (0, 1.6), (0, 0.8)); a.frame(xticks=[0, 0.4, 0.8, 1.2, 1.6], yticks=[0, 0.2, 0.4, 0.6, 0.8], grid="xy", panel="a", title="the frontier $E^\\ast(\\varepsilon)$", xlabel=r"allowed interference $\varepsilon=\|P\Delta\|$", ylabel=r"best fit error $E^\ast$  ($\|e\|=1$)")
    a.hline(F["c"], "cRed", 0.8, "4pt on 2pt")
    a.text(1.55, F["c"] + 0.03, rf"unavoidable error $c={F['c']:.3f}$", anchor="south east", size="tiny", color="cRed")
    a.line(pts[:, 0], pts[:, 1], "cBlue", 1.5)
    idx = np.linspace(0, len(mus) - 1, 40).astype(int)
    for i in idx[::2]: a.marker(pts[i, 0], pts[i, 1], "cBlue", "o", 0.04)
    for mu, col in ((0.3, "cGreen"), (2.0, "cOrange"), (15.0, "cPurple")):
        e_, E_ = frontier_point(F["Phi"], F["P"], F["e"], mu); sl = -mu * e_ / E_ if E_ > 0 else 0; dx = 0.12
        a.line([e_ - dx, e_ + dx], [E_ - sl * dx, E_ + sl * dx], col, 1.0); a.marker(e_, E_, col, "o", 0.07)
        f.text(a.X(e_) + 0.14, a.Y(E_) + 0.05, rf"$\mu={mu:g}$", anchor="west", size="tiny", color=col)
    # (b) unavoidable error versus n
    b = Ax(f, 6.65, 1.1, 3.9, 3.7, (0, 25), (0, 1.02)); b.frame(xticks=[0, 5, 10, 15, 20, 25], yticks=[0, 0.25, 0.5, 0.75, 1], ytl=["0", ".25", ".5", ".75", "1"], grid="xy", panel="b", title="zero interference costs plasticity", xlabel="new examples $n$", ylabel=r"unavoidable error$^2$ $/\|e\|^2$")
    rng = np.random.default_rng(2); p = 12
    for r_, col in ((2, "cGreen"), (5, "cBlue"), (8, "cRed")):
        ns = np.arange(1, 26); th = [max(n - min(n, p - r_), 0) / n for n in ns]; b.line(ns, th, col, 1.2)
        Q = np.linalg.qr(rng.normal(size=(p, p)))[0]; P = Q[:, :r_] @ Q[:, :r_].T; mc = []
        for n in ns:
            vals = []
            for _ in range(60):
                Phi = rng.normal(size=(n, p)); G = Phi @ (np.eye(p) - P); Pi = G @ np.linalg.pinv(G); e = rng.normal(size=n); vals.append(np.sum(((np.eye(n) - Pi) @ e) ** 2) / np.sum(e ** 2))
            mc.append(np.mean(vals))
        b.scatter(ns, mc, col, r=0.045, opacity=0.9)
    b.legend([(f"$p-r={p - r_}$", col, "line") for r_, col in ((2, "cGreen"), (5, "cBlue"), (8, "cRed"))] + [("Monte Carlo", "cInk", "o")], 6.75, 4.7, anchor="north west", size="tiny", rowh=0.27)
    # (c) reach
    c = Ax(f, 11.95, 1.1, 3.6, 3.7, (0, 1), (0.0, 1.05)); c.frame(xticks=[0, 0.25, 0.5, 0.75, 1], xtl=["0", "", ".5", "", "1"], yticks=[0, 0.25, 0.5, 0.75, 1], ytl=["0", ".25", ".5", ".75", "1"], grid="xy", panel="c", title="one example: reach", xlabel=r"allowed interference $\varepsilon/B$", ylabel="largest output change $/B$")
    for rho, col in ((0.3, "cGreen"), (0.6, "cBlue"), (0.9, "cRed")):
        kap = math.sqrt(1 - rho ** 2); es = np.linspace(0, 1, 201); ys = [rho * e_ + kap * math.sqrt(max(1 - e_ * e_, 0)) if e_ <= rho else 1.0 for e_ in es]
        c.line(es, ys, col, 1.4); c.marker(0, kap, col, "o", 0.07); c.vline(rho, col, 0.4, "2pt on 2pt")
        f.text(c.X(0) + 0.1, c.Y(kap) - 0.12, rf"$\kappa={kap:.2f}$", anchor="west", size="tiny", color=col)
    c.legend([(rf"$\rho={r_}$", col, "line") for r_, col in ((0.3, "cGreen"), (0.6, "cBlue"), (0.9, "cRed"))], 15.5, 1.2, anchor="south east", size="tiny", rowh=0.27)
    return f.out(), F


def minmem_data():
    rng = np.random.default_rng(0)
    def haar(p, r, n=None):
        shp = (p, r) if n is None else (n, p, r); return np.linalg.qr(rng.normal(size=shp))[0]
    out = []
    for (r, p) in [(1, 3), (1, 5), (2, 5), (2, 6), (3, 6), (1, 8)]:
        ms = (2, 4, 6, 8, 10); med = []
        for m in ms:
            cb = haar(p, r, 2 ** m); trials = 250 if m < 10 else 120; ds = []
            for _ in range(trials):
                Q = haar(p, r); S = np.linalg.svd(np.einsum("pr,npq->nrq", Q, cb), compute_uv=False); j = int(np.argmax(S.min(axis=1))); Qh = cb[j]
                w = rng.normal(size=p); w /= np.linalg.norm(w); wp = w - Qh @ (Qh.T @ w)
                if np.linalg.norm(wp) < 0.3: continue
                ds.append(np.linalg.norm(Q.T @ (wp / (wp @ wp))))
            med.append(float(np.median(ds)))
        out.append(dict(r=r, p=p, ms=ms, med=med, slope=float(np.polyfit(ms, np.log2(med), 1)[0]), pred=-1 / (r * (p - r))))
    return out


def fig_minmem():
    M = minmem_data(); f = Fig(16.0, 5.7); cols = ["cBlue", "cGreen", "cOrange", "cPurple", "cTeal", "cRed"]
    a = Ax(f, 1.45, 1.1, 3.9, 3.7, (1, 11), (0.03, 12), ylog=True); a.frame(xticks=[2, 4, 6, 8, 10], yticks=[0.05, 0.1, 0.2, 0.5, 1], ytl=["0.05", "0.1", "0.2", "0.5", "1"], grid="xy", panel="a", title="interference versus stored bits", xlabel="stored bits $m$ (random codebook of $2^m$)", ylabel="median interference $\\delta'$")
    for d, col in zip(M, cols):
        a.line(d["ms"], d["med"], col, 1.2)
        for x, y in zip(d["ms"], d["med"]): a.marker(x, y, col, "o", 0.05)
        y0 = d["med"][0] * 2 ** (d["pred"] * (np.array([2, 10]) - 2)); a.line([2, 10], list(y0), col, 0.6, "3pt on 2pt")
    a.legend([(f"$(r,p)=({d['r']},{d['p']})$", col, "line") for d, col in zip(M, cols)], 1.5, 4.75, anchor="north west", size="tiny", rowh=0.25, ncol=2, cw=0.1)
    b = Ax(f, 6.65, 1.1, 3.9, 3.7, (-0.55, 0), (-0.55, 0)); tk = [-0.5, -0.4, -0.3, -0.2, -0.1, 0]; b.frame(xticks=tk, xtl=["$-0.5$", "$-0.4$", "$-0.3$", "$-0.2$", "$-0.1$", "0"], yticks=tk, ytl=["$-0.5$", "$-0.4$", "$-0.3$", "$-0.2$", "$-0.1$", "0"], grid="xy", panel="b", title="decay exponent per bit", xlabel=r"predicted $-1/(r(p-r))$", ylabel="fitted slope of $\\log_2\\delta'$", ylab_dx=-0.2)
    b.line([-0.55, 0], [-0.55, 0], "cGray", 0.6, "3pt on 2pt")
    for d, col in zip(M, cols): b.marker(d["pred"], d["slope"], col, "o", 0.09)
    b.rtext(0.96, 0.06, "random codebooks lie below", anchor="east", size="tiny", color="cGray")
    # (c) Theorem 14 geometry: max loss vs theta2 along theta1 = 1
    c = Ax(f, 11.95, 1.1, 3.6, 3.7, (-2, 2), (0, 2.1)); c.frame(xticks=[-2, -1, 0, 1, 2], xtl=["$-2$", "$-1$", "0", "1", "2"], yticks=[0, 0.5, 1, 1.5, 2], ytl=["0", ".5", "1", "1.5", "2"], grid="xy", panel="c", title=r"no memoryless guarantee", xlabel=r"$\theta_2$ after fitting $B$ ($\theta_1=1$)", ylabel="old-task loss")
    ts = np.linspace(-2, 2, 200); c.line(ts, (1 + ts) ** 2 / 2, "cBlue", 1.1, "4pt on 2pt"); c.line(ts, (1 - ts) ** 2 / 2, "cGreen", 1.1, "4pt on 2pt"); c.line(ts, np.maximum((1 + ts) ** 2 / 2, (1 - ts) ** 2 / 2), "cRed", 1.8)
    c.marker(0, 0.5, "cRed", "d", 0.09); c.text(0.2, 0.28, "worst case $\\ge\\frac12$", anchor="west", size="tiny", color="cRed", extra="fill=white")
    c.marker(-1, 0, "cBlue", "o", 0.07); c.marker(1, 0, "cGreen", "o", 0.07)
    c.legend([("loss on $A_+$", "cBlue", "dash"), ("loss on $A_-$", "cGreen", "dash"), ("worst case", "cRed", "line")], 15.5, 4.72, anchor="north east", size="tiny", rowh=0.27)
    return f.out(), M


# ------------------------------------------------------------------------------------------------ MNIST
BENCH = [("split", "full", "Split, class-incremental"), ("split", "task", "Split, task-incremental"), ("perm", "full", "Permuted, domain-inc.")]
FAM = {"ft": ("fine-tuning", "cGray", "o"), "ewc": ("EWC", "cOrange", "t"), "replay": ("replay", "cRed", "s"), "lw": ("layerwise, frozen", "cBlue", "o"), "lwf": ("layerwise, free", "cSky", "o"), "ogd": ("gradient subspace", "cGreen", "d"), "ad": ("task adapters", "cPurple", "d")}


def mnist_points(bench, view):
    pts = []
    pts.append(("ft", "fine-tuning", D.mnist(bench, "ft", view)))
    for l in (10.0, 100.0, 1000.0): pts.append(("ewc", rf"EWC $\lambda={l:g}$", D.mnist(bench, "ewc", view, ewc=l)))
    for k in (20, 200): pts.append(("replay", f"replay {k}", D.mnist(bench, "replay", view, k=k)))
    for r in (0.95, 0.99, 0.999): pts.append(("lw", rf"frozen $\rho={r}$", D.mnist(bench, "gpm", view, rho=r, ssm="freeze")))
    for r in (0.95, 0.99): pts.append(("lwf", rf"free $\rho={r}$", D.mnist(bench, "gpm", view, rho=r, ssm="free")))
    for r in (0.9, 0.99): pts.append(("ogd", rf"OGD $\rho={r}$", D.mnist(bench, "ogd", view, rho=r)))
    if not (bench == "split" and view == "full"):
        for rk in (0, 1, 4, 16): pts.append(("ad", f"adapter $r={rk}$", D.adapter(bench, rk, view)))
    return pts


def joint_ref(bench, view): return D.joint(bench, view)["final_avg"]


def fig_mnist_pareto():
    f = Fig(16.0, 5.9)
    for i, (bench, view, ttl) in enumerate(BENCH):
        x0 = 1.45 + i * 5.2; a = Ax(f, x0, 1.2, 3.9, 3.7, (1e4, 3e8), (0, 1.0), xlog=True); pts = mnist_points(bench, view)
        a.frame(xticks=[1e4, 1e5, 1e6, 1e7, 1e8], xtl=[r"$10^4$", r"$10^5$", r"$10^6$", r"$10^7$", r"$10^8$"], yticks=[0, 0.25, 0.5, 0.75, 1.0], ytl=["0", ".25", ".5", ".75", "1"], grid="xy", panel="abc"[i], title=ttl, xlabel="stored bits", ylabel="final average accuracy" if i == 0 else None)
        jm, js = joint_ref(bench, view); a.hline(jm, "cInk", 0.6, "4pt on 2pt")
        a.text(1.3e4, jm - 0.02, "joint training", anchor="north west", size="tiny", color="cInk")
        ft = pts[0][2]["final_avg"]; a.hline(ft[0], "cGray", 0.6, "2pt on 2pt"); a.text(1.3e4, ft[0] - 0.02, "fine-tuning (0 bits)", anchor="north west", size="tiny", color="cGray")
        for fam, lab, v in pts[1:]:
            if v["bits"] <= 0: continue
            col, sym = FAM[fam][1], FAM[fam][2]; a.err(v["bits"], v["final_avg"][0] - v["final_avg"][1], v["final_avg"][0] + v["final_avg"][1], col, 0.03, 0.5); a.marker(v["bits"], v["final_avg"][0], col, sym, 0.08)
        if i == 0:
            seen = []
            for fam, lab, v in pts[1:]:
                if fam not in [s[0] for s in seen]: seen.append((fam, FAM[fam][0], FAM[fam][1], FAM[fam][2]))
            a.legend([(n, c, s) for _, n, c, s in seen], x0 + 3.85, 1.3, anchor="south east", size="tiny", rowh=0.25)
        if i == 1:
            a.legend([(FAM["ad"][0], "cPurple", "d")], x0 + 3.85, 1.3, anchor="south east", size="tiny", rowh=0.25)
    return f.out()


def fig_mnist_bars():
    f = Fig(16.0, 5.9); sel = [("ft", "fine-tune", lambda b, v: D.mnist(b, "ft", v)), ("ewc", "EWC 10", lambda b, v: D.mnist(b, "ewc", v, ewc=10.0)), ("replay", "replay 20", lambda b, v: D.mnist(b, "replay", v, k=20)),
                               ("replay", "replay 200", lambda b, v: D.mnist(b, "replay", v, k=200)), ("lw", "layerwise", lambda b, v: D.mnist(b, "gpm", v, rho=0.99, ssm="freeze")), ("ogd", "OGD", lambda b, v: D.mnist(b, "ogd", v, rho=0.99)), ("ad", "adapter 4", lambda b, v: D.adapter(b, 4, v))]
    for i, (bench, view, ttl) in enumerate(BENCH):
        x0 = 1.45 + i * 5.2; a = Ax(f, x0, 1.7, 3.9, 3.2, (0, len(sel)), (0, 1.05)); a.frame(xticks=[j + 0.5 for j in range(len(sel))], xtl=[""] * len(sel), yticks=[0, 0.25, 0.5, 0.75, 1.0], ytl=["0", ".25", ".5", ".75", "1"], grid="y", panel="abc"[i], title=ttl, ylabel="final avg. accuracy (bars) / forgetting ($\\diamond$)" if i == 0 else None)
        jm, _ = joint_ref(bench, view); a.hline(jm, "cInk", 0.6, "4pt on 2pt")
        for j, (fam, lab, g) in enumerate(sel):
            if fam == "ad" and bench == "split" and view == "full": f.text(a.X(j + 0.5), a.Y(0.04), "n/a", size="tiny", color="cGray", rot=90); f.text(a.X(j + 0.5), a.y0 - 0.1, lab, anchor="north east", size="tiny", rot=45); continue
            v = g(bench, view); col = FAM[fam][1]; m, s = v["final_avg"]; a.bars([j + 0.5], [m], 0.62, col); a.err(j + 0.5, m - s, m + s, "cInk", 0.05, 0.6); a.marker(j + 0.5, v["forgetting"][0], "white", "d", 0.07, edge="cInk", lw=0.7, fill=True)
            f.text(a.X(j + 0.5), a.y0 - 0.1, lab, anchor="north east", size="tiny", rot=45)
    return f.out()


def fig_mnist_heat():
    f = Fig(16.0, 5.7)
    rows = [("fine-tuning", lambda b, v: D.mnist(b, "ft", v)), ("EWC $\\lambda=10$", lambda b, v: D.mnist(b, "ewc", v, ewc=10.0)), ("EWC $\\lambda=1000$", lambda b, v: D.mnist(b, "ewc", v, ewc=1000.0)), ("replay 20", lambda b, v: D.mnist(b, "replay", v, k=20)), ("replay 200", lambda b, v: D.mnist(b, "replay", v, k=200)),
            (r"layerwise frozen $\rho{=}0.99$", lambda b, v: D.mnist(b, "gpm", v, rho=0.99, ssm="freeze")), (r"layerwise free $\rho{=}0.99$", lambda b, v: D.mnist(b, "gpm", v, rho=0.99, ssm="free")), (r"gradient subspace $\rho{=}0.99$", lambda b, v: D.mnist(b, "ogd", v, rho=0.99)),
            ("adapter $r{=}4$", lambda b, v: D.adapter(b, 4, v)), ("joint training", lambda b, v: dict(final_avg=D.joint(b, v)["final_avg"], forgetting=(0.0, 0.0), last_task=D.joint(b, v)["last_task"] if "last_task" in D.joint(b, v) else (np.nan, 0)))]
    metrics = [("final_avg", "final average accuracy", VIRIDIS, 0, 1), ("forgetting", "forgetting", [(240, 240, 240), (233, 180, 120), (192, 100, 43), (110, 40, 30)], 0, 1), ("last_task", "last-task accuracy", VIRIDIS, 0, 1)]
    for mi, (key, ttl, tab, lo, hi) in enumerate(metrics):
        x0 = 3.1 + mi * 4.6 if mi else 3.1; x0 = [3.2, 7.9, 12.3][mi]; ax = Ax(f, x0, 1.0, 3.5, 3.9, (0, 3), (0, len(rows)))
        Z = np.full((len(rows), 3), np.nan)
        for ri, (lab, g) in enumerate(rows):
            for bi, (bench, view, _t) in enumerate(BENCH):
                if "adapter" in lab and bench == "split" and view == "full": continue
                try: v = g(bench, view); Z[len(rows) - 1 - ri, bi] = v[key][0]
                except Exception: pass
        ax.heat(Z, [0, 1, 2, 3], list(range(len(rows) + 1)), lo, hi, table=tab, annotate=lambda v: f"{v:.2f}".lstrip("0") if v < 1 else "1.0", textsize="tiny", seams=True)
        ax.frame(xticks=[0.5, 1.5, 2.5], xtl=[r"\shortstack{Split\\class}", r"\shortstack{Split\\task}", "Permuted"], yticks=[len(rows) - 1 - i + 0.5 for i in range(len(rows))], ytl=[r_[0] for r_ in rows] if mi == 0 else [""] * len(rows), grid=None, panel="abc"[mi], title=ttl)
    return f.out()


def fig_mnist_plane():
    f = Fig(16.0, 5.7)
    for i, (bench, view, ttl) in enumerate(BENCH):
        x0 = 1.45 + i * 5.2; a = Ax(f, x0, 1.1, 3.9, 3.7, (0, 1.0), (0, 1.0)); a.frame(xticks=[0, 0.25, 0.5, 0.75, 1], xtl=["0", "", ".5", "", "1"], yticks=[0, 0.25, 0.5, 0.75, 1], ytl=["0", ".25", ".5", ".75", "1"], grid="xy", panel="abc"[i], title=ttl, xlabel="forgetting (worse $\\to$)", ylabel="last-task accuracy (plasticity)" if i == 0 else None)
        for fam, lab, v in mnist_points(bench, view):
            col, sym = FAM[fam][1], FAM[fam][2]; fm, fs = v["forgetting"]; lm, ls = v["last_task"]
            a.herr(lm, fm - fs, fm + fs, col, 0.03, 0.5); a.err(fm, lm - ls, lm + ls, col, 0.03, 0.5); a.marker(fm, lm, col, sym, 0.08)
        a.rtext(0.04, 0.93, r"ideal: top left", anchor="west", size="tiny", color="cGray")
        if i == 0: a.legend([(FAM[k][0], FAM[k][1], FAM[k][2]) for k in ("ft", "ewc", "replay", "lw", "lwf", "ogd")], x0 + 3.85, 1.2, anchor="south east", size="tiny", rowh=0.25)
        if i == 1: a.legend([(FAM["ad"][0], "cPurple", "d")], x0 + 3.85, 1.2, anchor="south east", size="tiny", rowh=0.25)
    return f.out()


# ------------------------------------------------------------------------------------------------ deep SSM
def deep_learners():
    d1, d2 = D.D1, D.D2; G = D.dagg
    L = [("fine-tuning", "ft", G(d1[("ft", None, None, None, None)])), ("EWC $\\lambda=1$", "ewc", G(d1[("ewc", None, 1.0, None, None)])), ("EWC $\\lambda=10$", "ewc", G(d1[("ewc", None, 10.0, None, None)])), ("EWC $\\lambda=100$", "ewc", G(d1[("ewc", None, 100.0, None, None)])),
         ("replay 4", "replay", G(d1[("replay", None, None, None, 4)])), ("replay 32", "replay", G(d1[("replay", None, None, None, 32)])),
         (r"grad. subspace $\rho{=}0.9$", "ogd", G(d1[("ogd", 0.9, None, 32, None)])), (r"grad. subspace $\rho{=}0.99$", "ogd", G(d1[("ogd", 0.99, None, 32, None)])), (r"grad. subspace $\rho{=}0.9999$", "ogd", G(d1[("ogd", 0.9999, None, 32, None)])),
         (r"layerwise frozen $\rho{=}0.99$", "lw", G(d2[("gpm", "freeze", 0.99, 32, 12, 0.03, None)])), (r"layerwise frozen $\rho{=}0.999$", "lw", G(d2[("gpm", "freeze", 0.999, 32, 12, 0.03, None)])),
         (r"layerwise free $\rho{=}0.99$", "lwf", G(d2[("gpm", "free", 0.99, 32, 12, 0.03, None)])), (r"layerwise free $\rho{=}0.999$", "lwf", G(d2[("gpm", "free", 0.999, 32, 12, 0.03, None)]))]
    return L


def fig_deep1():
    L = deep_learners(); f = Fig(16.0, 5.7)
    a = Ax(f, 1.45, 1.1, 3.4, 3.7, (0, 0.8), (0, 1.05)); a.frame(xticks=[0, 0.2, 0.4, 0.6, 0.8], yticks=[0, 0.25, 0.5, 0.75, 1], ytl=["0", ".25", ".5", ".75", "1"], grid="xy", panel="a", title="protection costs plasticity", xlabel="forgetting", ylabel="last-task accuracy")
    for lab, fam, v in L:
        col, sym = FAM[fam][1], FAM[fam][2]; fm, fs = v["forgetting"]; lm, ls = v["last_task"]; a.herr(lm, fm - fs, fm + fs, col, 0.03, 0.5); a.err(fm, lm - ls, lm + ls, col, 0.03, 0.5); a.marker(fm, lm, col, sym, 0.08)
    a.legend([(FAM[k][0], FAM[k][1], FAM[k][2]) for k in ("ft", "ewc", "replay", "ogd", "lw", "lwf")], 4.8, 1.2, anchor="south east", size="tiny", rowh=0.25)
    b = Ax(f, 6.0, 1.1, 3.4, 3.7, (3e4, 1e6), (0, 0.8), xlog=True); b.frame(xticks=[1e5, 1e6], xtl=[r"$10^5$", r"$10^6$"], yticks=[0, 0.2, 0.4, 0.6, 0.8], grid="xy", panel="b", title="memory against forgetting", xlabel="stored bits", ylabel="forgetting")
    for lab, fam, v in L:
        if v["bits"] <= 0: continue
        col, sym = FAM[fam][1], FAM[fam][2]; fm, fs = v["forgetting"]; b.err(v["bits"], fm - fs, fm + fs, col, 0.03, 0.5); b.marker(v["bits"], fm, col, sym, 0.08)
    ftf = L[0][2]["forgetting"][0]; b.hline(ftf, "cGray", 0.6, "3pt on 2pt"); b.text(3.5e4, ftf + 0.015, "fine-tuning", anchor="south west", size="tiny", color="cGray")
    # (c) paired differences in forgetting versus fine-tuning, per seed
    c = Ax(f, 12.4, 1.1, 3.2, 3.7, (-0.8, 0.2), (-0.5, len(L) - 1.5)); c.frame(xticks=[-0.6, -0.4, -0.2, 0], xtl=["$-0.6$", "$-0.4$", "$-0.2$", "0"], yticks=[], grid="x", panel="c", title="paired change", xlabel="forgetting $-$ fine-tuning (same seed)")
    base = L[0][2]["per_seed"]["forgetting"]; c.vline(0, "cGray", 0.6)
    for k, (lab, fam, v) in enumerate(L[1:]):
        y = len(L) - 2 - k; col = FAM[fam][1]; diffs = [a_ - b_ for a_, b_ in zip(v["per_seed"]["forgetting"], base)]
        c.line([min(diffs), max(diffs)], [y, y], col, 0.6); [c.marker(d_, y, col, "o", 0.04) for d_ in diffs]; c.marker(float(np.mean(diffs)), y, col, "d", 0.07, edge="cInk", lw=0.5)
        f.text(c.x0 - 0.1, c.Y(y), lab, anchor="east", size="tiny")
    return f.out()


def fig_deep2():
    f = Fig(16.0, 5.0); items = [("fine-tuning", D.D1[("ft", None, None, None, None)]), ("EWC $\\lambda=10$", D.D1[("ewc", None, 10.0, None, None)]), ("replay 32", D.D1[("replay", None, None, None, 32)]), (r"layerwise frozen $\rho{=}0.99$", D.D2[("gpm", "freeze", 0.99, 32, 12, 0.03, None)])]
    for i, (lab, runs) in enumerate(items):
        v = D.dagg(runs); A = np.array(v["acc"]); x0 = 1.0 + i * 3.95; ax = Ax(f, x0, 1.1, 3.0, 3.0, (0, 5), (0, 5)); ax.heat(A[::-1], list(range(6)), list(range(6)), 0.4, 1.0, annotate=lambda z: f"{z * 100:.0f}", textsize="tiny", seams=True)
        ax.frame(xticks=[j + 0.5 for j in range(5)], xtl=[str(j + 1) for j in range(5)], yticks=[j + 0.5 for j in range(5)], ytl=[str(5 - j) for j in range(5)], grid=None, panel="abcd"[i], title=lab, xlabel="evaluated task", ylabel="after learning task" if i == 0 else None)
    f.text(8.0, 0.12, "accuracy (\\%) of each task (columns) after each stage of the stream (rows); mean of 5 seeds; the diagonal is learning, the lower-left triangle is retention", size="tiny", color="cGray")
    return f.out()


def fig_deep3():
    f = Fig(16.0, 5.7)
    a = Ax(f, 1.45, 1.1, 3.9, 3.7, (0, 4), (0, 1.0)); a.frame(xticks=[0.5, 1.5, 2.5, 3.5], xtl=["4", "3", "2", "1"], yticks=[0, 0.25, 0.5, 0.75, 1], ytl=["0", ".25", ".5", ".75", "1"], grid="y", panel="a", title="subspace rotation", xlabel="tasks learned before the end", ylabel="mean $\\cos^2$ of principal angles")
    vals = np.array([[s[t]["mean_cos2"] for t in range(4)] for s in D.DRIFT])
    a.bars([0.5, 1.5, 2.5, 3.5], vals.mean(0), 0.55, "cPurple")
    for s in range(len(vals)):
        for t in range(4): a.marker(t + 0.5 + (s - 2) * 0.07, vals[s, t], "cInk", "o", 0.035, opacity=0.8)
    for t in range(4): a.text(t + 0.5, vals[:, t].max() + 0.04, f"{vals[:, t].mean():.2f}", anchor="south", size="tiny")
    a.hline(0.005, "cRed", 0.7, "3pt on 2pt"); a.text(3.95, 0.97, "dashed: random subspace (0.005)", anchor="north east", size="tiny", color="cRed")
    rhos = sorted(D.PROBE); xs = [1 - r for r in rhos]
    b = Ax(f, 6.65, 1.1, 3.9, 3.7, (5e-6, 0.3), (1e-3, 30), xlog=True, ylog=True); b.frame(xticks=[1e-5, 1e-4, 1e-3, 1e-2, 1e-1], xtl=[r"$10^{-5}$", r"$10^{-4}$", r"$10^{-3}$", r"$10^{-2}$", r"$10^{-1}$"], yticks=[1e-3, 1e-2, 1e-1, 1, 5], ytl=[r"$10^{-3}$", r"$10^{-2}$", r"$10^{-1}$", "1", "5"], grid="xy", panel="b", title="layerwise exactness", xlabel=r"energy left outside the stored subspace $1-\rho$", ylabel="mean change of old-task logits")
    mean_ch, max_ch = [], []
    for r in rhos:
        ch = [np.mean([p["mean"] for p in run["probe"]]) for run in D.PROBE[r].values()]; mx = [np.max([p["max"] for p in run["probe"]]) for run in D.PROBE[r].values()]; mean_ch.append(ch); max_ch.append(mx)
    for x, ch, mx in zip(xs, mean_ch, max_ch):
        b.err(x, min(ch), max(ch), "cBlue", 0.0, 0.8)
        for c_ in ch: b.marker(x, c_, "cBlue", "o", 0.035, opacity=0.5)
        b.marker(x, float(np.mean(ch)), "cBlue", "o", 0.08); b.marker(x, float(np.mean(mx)), "cRed", "t", 0.07)
    b.line(xs[::-1], [float(np.mean(c)) for c in mean_ch][::-1], "cBlue", 1.0)
    b.legend([("mean over probes", "cBlue", "o"), ("maximum", "cRed", "t")], 10.45, 4.72, anchor="north east", size="tiny", rowh=0.27)
    c = Ax(f, 11.95, 1.1, 3.6, 3.7, (0, 4), (0, 0.7)); c.frame(xticks=[0.5, 1.5, 2.5, 3.5], xtl=[r"$0.9$", r"$0.99$", r"$0.999$", r"$0.99999$"], yticks=[0, 0.2, 0.4, 0.6], grid="y", panel="c", title="forgetting, stored directions", xlabel=r"energy threshold $\rho$", ylabel="forgetting")
    for i, r in enumerate(rhos):
        fg = [run["forgetting"] for run in D.PROBE[r].values()]; nd = np.mean([sum(sum(x) for x in run["lranks"]) for run in D.PROBE[r].values()])
        c.bars([i + 0.5], [np.mean(fg)], 0.55, "cBlue"); c.err(i + 0.5, min(fg), max(fg), "cInk", 0.05, 0.6); c.text(i + 0.5, 0.64, f"{nd:.0f} dir.", anchor="south", size="tiny", color="cGray")
    return f.out()


# ------------------------------------------------------------------------------------------------ adapters, toy testbed
def fig_adapters():
    f = Fig(16.0, 5.7); ranks = [0, 1, 4, 16]
    a = Ax(f, 1.45, 1.1, 3.9, 3.7, (-0.5, 3.5), (0.45, 1.02)); a.frame(xticks=[0, 1, 2, 3], xtl=[str(r) for r in ranks], yticks=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0], grid="y", panel="a", title="accuracy against adapter rank", xlabel="adapter rank $r$ (0: head only)", ylabel="accuracy (task id given)")
    for bench, view, col, lab in (("split", "task", "cBlue", "Split-MNIST"), ("perm", "full", "cOrange", "Permuted-MNIST")):
        v = [D.adapter(bench, r, view)["final_avg"] for r in ranks]; a.line(range(4), [x[0] for x in v], col, 1.3)
        for i, (m, s) in enumerate(v): a.err(i, m - s, m + s, col, 0.05, 0.6); a.marker(i, m, col, "o", 0.07)
        jm = D.joint(bench, view)["final_avg"][0]; a.hline(jm, col, 0.6, "4pt on 2pt")
    a.legend([("Split task-inc.", "cBlue", "line"), ("Permuted (task id)", "cOrange", "line"), ("joint training", "cGray", "dash")], 5.3, 1.2, anchor="south east", size="tiny", rowh=0.27)
    b = Ax(f, 6.65, 1.1, 3.9, 3.7, (5e4, 2e7), (0.55, 1.0), xlog=True); b.frame(xticks=[1e5, 1e6, 1e7], xtl=[r"$10^5$", r"$10^6$", r"$10^7$"], yticks=[0.6, 0.7, 0.8, 0.9, 1.0], grid="xy", panel="b", title="Split task-incremental", xlabel="stored bits", ylabel="final average accuracy")
    for rk in ranks: v = D.adapter("split", rk, "task"); b.marker(v["bits"], v["final_avg"][0], "cPurple", "d", 0.09)
    for k in (20, 200): v = D.mnist("split", "replay", "task", k=k); b.marker(v["bits"], v["final_avg"][0], "cRed", "s", 0.08)
    v = D.mnist("split", "ewc", "task", ewc=10.0); b.marker(v["bits"], v["final_avg"][0], "cOrange", "t", 0.09)
    for r in (0.95, 0.99): v = D.mnist("split", "gpm", "task", rho=r, ssm="freeze"); b.marker(v["bits"], v["final_avg"][0], "cBlue", "o", 0.08)
    b.legend([("adapters", "cPurple", "d"), ("replay", "cRed", "s"), ("EWC", "cOrange", "t"), ("layerwise", "cBlue", "o")], 10.45, 1.2, anchor="south east", size="tiny", rowh=0.25)
    c = Ax(f, 11.95, 1.1, 3.6, 3.7, (1, 10), (1e4, 3e7), ylog=True); c.frame(xticks=[1, 2, 4, 6, 8, 10], yticks=[1e4, 1e5, 1e6, 1e7], ytl=[r"$10^4$", r"$10^5$", r"$10^6$", r"$10^7$"], grid="xy", panel="c", title="memory against stream length", xlabel="tasks $T$", ylabel="stored bits")
    ad = D.adapter("split", 4, "task")["bits"] / 4; r20 = D.mnist("split", "replay", "task", k=20)["bits"] / 5; r200 = D.mnist("split", "replay", "task", k=200)["bits"] / 5;     Ts = np.arange(1, 11)
    c.line(Ts[1:], [ad * (t - 1) for t in Ts[1:]], "cPurple", 1.4); c.line(Ts, [r20 * t for t in Ts], "cRed", 1.4, "4pt on 2pt"); c.line(Ts, [r200 * t for t in Ts], "cRed", 1.4)
    c.legend([("adapters, rank 4", "cPurple", "line"), ("replay 200 / task", "cRed", "line"), ("replay 20 / task", "cRed", "dash")], 15.5, 1.2, anchor="south east", size="tiny", rowh=0.25)
    return f.out()


def fig_toy():
    f = Fig(16.0, 5.7)
    spec = {"lti": [("fine-tune", ("lti", "ft", "none", 0.0), "cGray"), ("EWC", ("lti", "ewc", "none", 0.0), "cOrange"), ("replay", ("lti", "replay", "none", 0.0), "cRed")],
            "sel": [("fine-tune", ("sel", "ft", "none", 0.0), "cGray"), ("gate sparsity .03", ("sel", "ft", "sparse", 0.03), "cSky"), ("gate decorr. 2", ("sel", "ft", "decor", 2.0), "cSky"), ("EWC", ("sel", "ewc", "none", 0.0), "cOrange"), ("replay", ("sel", "replay", "none", 0.0), "cRed")]}
    for pi, (arch, ttl) in enumerate((("lti", "diagonal LTI SSM"), ("sel", "selective SSM"))):
        x0 = 1.45 + pi * 5.2; items = spec[arch]; a = Ax(f, x0, 1.7, 3.9, 3.1, (0, len(items)), (0, 0.6)); a.frame(xticks=[i + 0.5 for i in range(len(items))], xtl=[""] * len(items), yticks=[0, 0.2, 0.4, 0.6], grid="y", panel="ab"[pi], title=ttl, ylabel="forgetting" if pi == 0 else None)
        for i, (lab, key, col) in enumerate(items):
            t = D.toy(*key); fs = [r["forgetting"] for r in t["seed"].values()]; a.bars([i + 0.5], [np.mean(fs)], 0.6, col); a.err(i + 0.5, np.mean(fs) - np.std(fs, ddof=1), np.mean(fs) + np.std(fs, ddof=1), "cInk", 0.05, 0.6)
            for k, v in enumerate(fs): a.marker(i + 0.5 + (k - 2) * 0.09, v, "cInk", "o", 0.03)
            f.text(a.X(i + 0.5), a.y0 - 0.1, lab, anchor="north east", size="tiny", rot=45)
    c = Ax(f, 11.95, 1.7, 3.6, 3.1, (0.5, 1.0), (0, 0.5)); c.frame(xticks=[0.6, 0.7, 0.8, 0.9, 1.0], yticks=[0, 0.1, 0.2, 0.3, 0.4, 0.5], grid="xy", panel="c", title="forgetting against accuracy", xlabel="final average accuracy", ylabel="forgetting")
    for arch, sym in (("lti", "o"), ("sel", "s")):
        for lab, key, col in spec[arch]:
            t = D.toy(*key); c.marker(t["final_avg"][0], t["forgetting"][0], col, sym, 0.08)
        j = D.toy(arch, "joint", "none", 0.0); c.marker(j["final_avg"][0], 0.0, "cInk", sym, 0.08, fill=False)
    c.legend([("LTI", "cInk", "o"), ("selective", "cInk", "s")], 15.5, 4.72, anchor="north east", size="tiny", rowh=0.25)
    f.text(13.7, 0.35, "open markers: joint training (reference)", size="tiny", color="cGray")
    return f.out()
