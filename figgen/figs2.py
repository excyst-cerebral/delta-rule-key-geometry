"""Figures: task-level test, training-time lever."""
import math
import numpy as np
from tz import Fig, Ax, cmap, VIRIDIS, DIVERGE
import data as D

MODELS2 = ["GDN-340M", "GDN-360M"]
MC = {"GDN-340M": "cBlue", "GDN-360M": "cOrange"}
LOADS = [4, 8, 16, 24]


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d; return c - h, c + h


def boot_auc(y, s, B=600, seed=3):
    rng = np.random.default_rng(seed); out = []
    for _ in range(B):
        i = rng.integers(0, len(y), len(y))
        if y[i].sum() in (0, len(i)): continue
        out.append(D.auc(y[i], s[i]))
    return np.percentile(out, [2.5, 97.5])


def boot_dauc(y, a, b, B=800, seed=5):
    rng = np.random.default_rng(seed); out = []
    for _ in range(B):
        i = rng.integers(0, len(y), len(y))
        if y[i].sum() in (0, len(i)): continue
        out.append(D.auc(y[i], b[i]) - D.auc(y[i], a[i]))
    return D.auc(y, b) - D.auc(y, a), np.percentile(out, [2.5, 97.5])


def task_stats():
    S = {}
    for nm in MODELS2:
        d = D.TASK[nm]; ti = ~d["cal"]; m = D.task_models(d); y = d["fc"][ti]
        S[nm] = dict(m=m, y=y, ti=ti, auc={k: D.auc(y, v[ti]) for k, v in m.items()}, aucci={k: boot_auc(y, v[ti]) for k, v in m.items()},
                     h1=boot_dauc(y, m["age + load"][ti], m["age + load + P1"][ti]), h1s=boot_dauc(y, m["age + load + shuffled P1"][ti], m["age + load + P1"][ti]))
        S[nm]["lin"] = D.partial_linear(d, d["P1"], d["clp"], ti); S[nm]["lin_s"] = D.partial_linear(d, d["P1s"], d["clp"], ti)
        S[nm]["ex"] = D.within_cell(d, d["P1"], d["clp"]); S[nm]["ex_s"] = D.within_cell(d, d["P1s"], d["clp"])
    return S


_TS = None
def TS():
    global _TS
    if _TS is None: _TS = task_stats()
    return _TS


def fig_task():
    S = TS(); f = Fig(16.0, 5.7)
    def xl(gi):
        nm = MODELS2[gi]; dl = D.TASK_LOG[nm]["H1"][0]; sg = "$-$" if dl <= -0.0005 else ("$+$" if dl >= 0.0005 else "$\\pm$")
        return r"\shortstack{" + nm.replace("GDN-", "GDN ") + r"\\$\Delta$AUC " + f"{sg}{abs(dl):.3f}" + "}"

    a = Ax(f, 1.45, 1.1, 3.9, 3.7, (-0.5, 3.5), (0, 0.6)); a.frame(xticks=[0, 1, 2, 3], xtl=[str(n) for n in LOADS], yticks=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6], grid="y", panel="a", title="recall falls with load", xlabel="facts in the prompt $n$", ylabel="forced-choice accuracy")
    a.line([0, 1, 2, 3], [1 / n for n in LOADS], "cGray", 1.0, "3pt on 2pt")
    for nm in MODELS2:
        d = D.TASK[nm]; ks = [(int(d["fc"][d["n"] == n].sum()), int((d["n"] == n).sum())) for n in LOADS]; p = [k / m for k, m in ks]; ci = [wilson(k, m) for k, m in ks]
        a.band([0, 1, 2, 3], [c[0] for c in ci], [c[1] for c in ci], MC[nm], 0.18); a.line([0, 1, 2, 3], p, MC[nm], 1.3)
        for i, v in enumerate(p): a.marker(i, v, MC[nm], "o" if nm == "GDN-340M" else "s", 0.07)
    a.legend([("GDN 340M", "cBlue", "o"), ("GDN 360M", "cOrange", "s"), ("chance $1/n$", "cGray", "dash")], 5.3, 4.72, anchor="north east", size="tiny", rowh=0.27)
    b = Ax(f, 6.65, 1.1, 3.9, 3.7, (0, 6), (0.35, 0.85)); keys = [("age + load", "cGray"), ("age + load + P1", "cBlue"), ("age + load + shuffled P1", "cLight")]
    b.frame(xticks=[1.5, 4.5], xtl=[xl(0), xl(1)], yticks=[0.4, 0.5, 0.6, 0.7, 0.8], grid="y", panel="b", title="what $P_1$ adds to age and load", ylabel="test AUC")
    b.hline(0.5, "cGray", 0.5)
    for gi, nm in enumerate(MODELS2):
        for ki, (k, c) in enumerate(keys):
            x = gi * 3 + 0.5 + ki * 0.8; v = S[nm]["auc"][k]; lo, hi = S[nm]["aucci"][k]
            b.bars([x], [v], 0.65, c if c != "cLight" else "cSky", base=0.35); b.err(x, lo, hi, "cInk", 0.05, 0.6)
    b.legend([("age + load", "cGray", "box"), (r"$+P_1$", "cBlue", "box"), (r"$+$ shuffled $P_1$", "cSky", "box")], 6.72, 4.72, anchor="north west", size="tiny", rowh=0.27)
    c = Ax(f, 11.95, 1.1, 3.6, 3.7, (-0.6, 0.25), (0, 6)); c.frame(xticks=[-0.4, -0.2, 0, 0.2], xtl=["$-0.4$", "$-0.2$", "0", "0.2"], yticks=[], grid="x", panel="c", title="age matched exactly", xlabel=r"rank correlation of $P_1$ with log-prob.")
    c.vline(0, "cGray", 0.6)
    rows = [("linear age", "lin", "cOrange"), ("shuffled null", "lin_s", "cGray"), ("exact cells", "ex", "cBlue"), ("exact, shuffled", "ex_s", "cGray")]
    y = 5.55
    for lab, key, col in rows:
        c.text(-0.585, y, lab, anchor="west", size="tiny", color=col, bold=True); y -= 0.42
        for nm in MODELS2:
            r, lo, hi = D.TASK_LOG[nm][key]; ci = (lo, hi)
            c.herr(y, ci[0], ci[1], col, 0.06, 0.9); c.marker(r, y, col, "o" if nm == "GDN-340M" else "s", 0.07); y -= 0.3
        y -= 0.1
    c.legend([("340M", "cInk", "o"), ("360M", "cInk", "s")], 15.5, 4.72, anchor="north east", size="tiny", rowh=0.25)
    return f.out()


def cell_matrix(d, v):
    M = np.full((4, 24), np.nan)
    for i, n in enumerate(LOADS):
        for q in range(n):
            m = (d["n"] == n) & (d["q"] == q)
            if m.sum() >= 1: M[i, q] = np.mean(v[m])
    return M


def fig_task_app():
    S = TS(); f = Fig(16.0, 9.6); d = D.TASK["GDN-340M"]
    ed = np.arange(25) - 0.5 + 0.5; ey = [0, 1, 2, 3, 4]
    specs = [("mean age (tokens)", d["age"], "viridis"), (r"mean $P_1$", d["P1"], "viridis"), ("mean log-prob of the true value", d["clp"], "viridis")]
    for i, (ttl, v, _) in enumerate(specs):
        x0 = 1.0 + i * 5.2; ax = Ax(f, x0, 6.35, 3.6, 2.2, (0, 24), (0, 4)); M = cell_matrix(d, v)
        lo, hi = np.nanmin(M), np.nanmax(M); ax.heat(M[::-1], list(range(25)), [0, 1, 2, 3, 4], lo, hi)
        ax.frame(xticks=[0.5, 4.5, 9.5, 14.5, 19.5, 23.5], xtl=["0", "4", "9", "14", "19", "23"], yticks=[0.5, 1.5, 2.5, 3.5], ytl=["24", "16", "8", "4"], grid=None, panel="abc"[i], title=ttl, xlabel="queried fact index", ylabel="facts $n$")
        ax.colorbar(lo, hi, x0 + 3.72, 6.35, 0.13, 2.2, nd=2 if hi < 10 else 0, ticks=[lo, (lo + hi) / 2, hi])
    f.text(8.0, 9.35, r"GDN 340M: each cell is a (load, queried position) pair of prompts at which age is identical", size="tiny", color="cGray")
    # (d) Spearman P1 vs age within load
    ax = Ax(f, 1.0, 1.15, 3.6, 3.3, (0, 4), (0.4, 1.25)); ax.frame(xticks=[0.5, 1.5, 2.5, 3.5], xtl=[str(n) for n in LOADS], yticks=[0.4, 0.6, 0.8, 1.0], grid="y", panel="d", title=r"$P_1$ tracks age within a load", xlabel="facts $n$", ylabel=r"Spearman$(P_1,\mathrm{age})$")
    for gi, nm in enumerate(MODELS2):
        dd = D.TASK[nm]; vals = []
        for n in LOADS:
            m = dd["n"] == n; vals.append(float(np.corrcoef(D.rk(dd["P1"][m]), D.rk(dd["age"][m]))[0, 1]))
        xs = [i + 0.3 + gi * 0.4 for i in range(4)]; ax.bars(xs, vals, 0.36, MC[nm], base=0.4)
        for x, v in zip(xs, vals): ax.text(x, v + 0.012, f"{v:.2f}", anchor="south", size="tiny")
    ax.legend([("GDN 340M", "cBlue", "box"), ("GDN 360M", "cOrange", "box")], 1.07, 4.4, anchor="north west", size="tiny", rowh=0.25)
    for i, nm in enumerate(MODELS2):
        dd = D.TASK[nm]; r, ci, xs_, ys_, keep = S[nm]["ex"]
        x0 = 6.2 + i * 5.2; ax = Ax(f, x0, 1.15, 3.6, 3.3, (-0.5, 0.5), (-0.5, 0.5))
        ax.frame(xticks=[-0.4, -0.2, 0, 0.2, 0.4], xtl=["$-0.4$", "$-0.2$", "0", "0.2", "0.4"], yticks=[-0.4, -0.2, 0, 0.2, 0.4], ytl=["$-0.4$", "$-0.2$", "0", "0.2", "0.4"], grid="xy", panel="ef"[i], title=f"exact control, {nm.replace('GDN-', 'GDN ')}", xlabel=r"$P_1$ rank, centred in its cell", ylabel="log-prob rank, centred")
        ax.scatter(xs_[keep], ys_[keep], MC[nm], r=0.035, opacity=0.35)
        sl = float(np.polyfit(xs_[keep], ys_[keep], 1)[0]); ax.line([-0.5, 0.5], [-0.5 * sl, 0.5 * sl], "cInk", 1.0)
        ax.rtext(0.04, 0.93, rf"$r={r:.3f}$", anchor="west", size="tiny")
        ax.rtext(0.04, 0.84, f"{int(keep.sum())} prompts in 48 cells", anchor="west", size="tiny", color="cGray")
    return f.out()


def fig_task_roc():
    S = TS(); f = Fig(16.0, 5.7); cols = ["cGray", "cBlue", "cSky"]; nm_keys = ["age + load", "age + load + P1", "age + load + shuffled P1"]
    for gi, nm in enumerate(MODELS2):
        ax = Ax(f, 1.45 + gi * 5.2, 1.1, 3.9, 3.7, (0, 1), (0, 1)); ax.frame(xticks=[0, 0.25, 0.5, 0.75, 1], xtl=["0", "", "0.5", "", "1"], yticks=[0, 0.25, 0.5, 0.75, 1], ytl=["0", "", "0.5", "", "1"], grid="xy", panel="ab"[gi], title=f"ROC, {nm.replace('GDN-', 'GDN ')}", xlabel="false-positive rate", ylabel="true-positive rate")
        ax.line([0, 1], [0, 1], "cGray", 0.6, "3pt on 2pt")
        items = []
        for k, c in zip(nm_keys, cols):
            s = S[nm]["m"][k][S[nm]["ti"]]; fp, tp = D.roc(S[nm]["y"], s); ax.line(fp, tp, c, 1.3 if c != "cSky" else 1.0, None if c != "cSky" else "4pt on 2pt"); items.append((f"{k.replace('age + load', 'B3')} ({S[nm]['auc'][k]:.3f})", c, "line" if c != "cSky" else "dash"))
        ax.legend(items, 1.45 + gi * 5.2 + 3.85, 1.2, anchor="south east", size="tiny", rowh=0.27, cw=0.1)
    ax = Ax(f, 11.95, 1.1, 3.6, 3.7, (0, 4), (0.15, 0.8)); ax.frame(xticks=[0.5, 1.5, 2.5, 3.5], xtl=[str(n) for n in LOADS], yticks=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7], grid="y", panel="c", title="AUC within each load", xlabel="facts $n$", ylabel="test AUC")
    ax.hline(0.5, "cGray", 0.5)
    for nm, sym in zip(MODELS2, ("o", "s")):
        dd = D.TASK[nm]; ti = ~dd["cal"]
        for lab, v, dash, col in ((r"$P_1$", dd["P1"], None, MC[nm]), ("age", dd["age"], "4pt on 2pt", MC[nm])):
            ys = []
            for n in LOADS:
                m = ti & (dd["n"] == n); ys.append(D.auc(dd["fc"][m], v[m]))
            ax.line([0.5, 1.5, 2.5, 3.5], ys, col, 1.1, dash)
            if dash is None:
                for x, y_ in zip([0.5, 1.5, 2.5, 3.5], ys): ax.marker(x, y_, col, sym, 0.07)
    ax.legend([(r"GDN 340M, $P_1$", "cBlue", "line"), ("GDN 340M, age", "cBlue", "dash"), (r"GDN 360M, $P_1$", "cOrange", "line"), ("GDN 360M, age", "cOrange", "dash")], 15.5, 4.72, anchor="north east", size="tiny", rowh=0.25)
    return f.out()


# ------------------------------------------------------------------------------------------------ training-time lever
LCFG = [("none", 0.0, "baseline", "cGray", "o"), ("dec", 1.0, "key decorrelation", "cRed", "s"), ("match", 1.0, "Gram matching", "cOrange", "d"), ("rank", 0.1, "rank matching", "cGreen", "t")]


def fig_lever():
    f = Fig(16.0, 5.7)
    a = Ax(f, 1.45, 1.1, 3.9, 3.7, (6, 34), (0.2, 1.02)); a.frame(xticks=[8, 16, 24, 32], yticks=[0.2, 0.4, 0.6, 0.8, 1.0], grid="y", panel="a", title="capacity-limited recall", xlabel="stored pairs $n$", ylabel="mean recall accuracy")
    for reg, lam, lab, col, sym in LCFG:
        byn = D.mq_by_n(reg, lam); ns = [8, 16, 24, 32]; m = [np.mean(byn[n]) for n in ns]; se = [np.std(byn[n], ddof=1) / math.sqrt(len(byn[n])) if len(byn[n]) > 1 else 0 for n in ns]
        a.band(ns, [x - s for x, s in zip(m, se)], [x + s for x, s in zip(m, se)], col, 0.10); a.line(ns, m, col, 1.3)
        for n, v in zip(ns, m): a.marker(n, v, col, sym, 0.07)
    a.legend([(f"{lab} ({len(D.mq_seeds(reg, lam))})", col, sym) for reg, lam, lab, col, sym in LCFG], 5.3, 1.2, anchor="south east", size="tiny", rowh=0.27)
    b = Ax(f, 6.65, 1.1, 3.9, 3.7, (-0.3, 1.3), (0, 1.05)); b.frame(xticks=[0, 1], xtl=["baseline", "key decorr."], yticks=[0, 0.25, 0.5, 0.75, 1.0], ytl=["0", ".25", ".5", ".75", "1"], grid="y", panel="b", title="12 paired seeds", ylabel="mean recall (seed)")
    sb = D.mq_seeds("none", 0.0); sd = D.mq_seeds("dec", 1.0); pa = []
    for s in sorted(set(sb) & set(sd)):
        x = D.mq_acc(D.MQ[("none", 0.0, s)][1]); y = D.mq_acc(D.MQ[("dec", 1.0, s)][1]); pa.append((x, y)); col = "cRed" if x < 0.9 else "cGray"
        b.line([0, 1], [x, y], col, 0.8, None, 0.8 if x < 0.9 else 0.5); b.marker(0, x, "cGray", "o", 0.045); b.marker(1, y, "cRed", "o", 0.045)
    mb, md = np.mean([p[0] for p in pa]), np.mean([p[1] for p in pa]); b.line([0, 1], [mb, md], "cInk", 1.8); b.marker(0, mb, "cInk", "s", 0.08); b.marker(1, md, "cInk", "s", 0.08)
    b.text(0.5, 0.18, rf"mean {mb:.3f} $\to$ {md:.3f}", size="tiny"); b.text(0.5, 0.08, "red: seeds that failed to learn", size="tiny", color="cRed")
    c = Ax(f, 11.95, 1.1, 3.6, 3.7, (0, 3050), (0, 1.02)); c.frame(xticks=[0, 1000, 2000, 3000], yticks=[0, 0.25, 0.5, 0.75, 1.0], ytl=["0", ".25", ".5", ".75", "1"], grid="xy", panel="c", title="leaving the chance plateau", xlabel="training step", ylabel="fraction of runs with CE $<2$")
    for reg, lam, lab, col, sym in LCFG[:2]:
        esc = [D.mq_escape(D.MQ[(reg, lam, s)][1]) for s in D.mq_seeds(reg, lam)]; N = len(esc); xs = [0] + sorted(e for e in esc if e is not None) + [3000]; ys = [0] + [(i + 1) / N for i in range(sum(e is not None for e in esc))]; ys.append(ys[-1])
        pts = []
        for i in range(len(xs) - 1): pts += [(xs[i], ys[i]), (xs[i + 1], ys[i])]
        c.line([p[0] for p in pts], [p[1] for p in pts], col, 1.5)
    c.legend([("baseline", "cGray", "line"), ("key decorrelation", "cRed", "line")], 15.5, 1.2, anchor="south east", size="tiny", rowh=0.27)
    return f.out()


def lever_run_table():
    rows = {}
    for reg, lam, lab, col, sym in LCFG:
        rows[reg] = {s: (D.mq_acc(D.MQ[(reg, lam, s)][1]), D.mq_escape(D.MQ[(reg, lam, s)][1])) for s in D.mq_seeds(reg, lam)}
    return rows


def spear(x, y): return float(np.corrcoef(D.rk(np.asarray(x)), D.rk(np.asarray(y)))[0, 1])


def fig_lever_app():
    f = Fig(16.0, 9.8); R = lever_run_table(); col_of = {reg: c for reg, _, _, c, _ in LCFG}
    M = np.full((4, 12), np.nan)
    for i, (reg, *_r) in enumerate(LCFG):
        for s, (v, e) in R[reg].items(): M[3 - i, s] = v
    a = Ax(f, 2.2, 6.1, 3.9, 2.6, (0, 12), (0, 4)); a.heat(M, list(range(13)), [0, 1, 2, 3, 4], 0, 1, annotate=lambda v: f"{v * 100:.0f}", textsize="tiny")
    a.frame(xticks=[i + 0.5 for i in range(12)], xtl=[str(i) if i % 2 == 0 else "" for i in range(12)], yticks=[0.5, 1.5, 2.5, 3.5], ytl=["rank", "Gram", "decorr.", "baseline"], grid=None, panel="a", title="accuracy per seed (\\%)", xlabel="seed")
    a.colorbar(0, 1, 6.25, 6.1, 0.14, 2.6, nd=1, ticks=[0, 0.5, 1])
    b = Ax(f, 8.9, 6.1, 3.0, 2.6, (0, 3000), (0, 5.4)); b.frame(xticks=[0, 1000, 2000, 3000], yticks=[0, 1, 2, 3, 4, 5], grid="xy", panel="b", title="training cross-entropy", xlabel="step", ylabel="CE")
    for reg, lam, lab, col, sym in LCFG[:2]:
        for s in D.mq_seeds(reg, lam):
            h = np.array(D.MQ[(reg, lam, s)][1]["hist"]); b.line(h[:, 0], h[:, 1], col, 0.5, None, 0.4)
    b.legend([("baseline seeds", "cGray", "line"), ("key decorr. seeds", "cRed", "line")], 11.85, 8.65, anchor="north east", size="tiny", rowh=0.25)
    G = D.mq_geometry(); xs, ys, cs = [], [], []
    for (reg, lam, s), arr in G.items():
        xs.append(float(np.median(arr[1][:, 0]))); ys.append(D.mq_acc(D.MQ[(reg, lam, s)][1])); cs.append(col_of[reg])
    c = Ax(f, 13.2, 6.1, 2.4, 2.6, (1, 14), (0, 1.03), xlog=True); c.frame(xticks=[1, 2, 5, 10], xtl=["1", "2", "5", "10"], yticks=[0, 0.25, 0.5, 0.75, 1.0], ytl=["0", ".25", ".5", ".75", "1"], grid="xy", panel="c", title="accuracy vs rank", xlabel="layer-1 key $r_{\\mathrm{eff}}$", ylabel="mean recall")
    for x, y, col in zip(xs, ys, cs): c.marker(x, y, col, "o", 0.06, opacity=0.8)
    c.rtext(0.96, 0.08, rf"$\rho_s={spear(xs, ys):.2f}$", anchor="east", size="tiny")
    # row 2
    d = Ax(f, 1.45, 1.1, 3.9, 3.3, (0, 4), (0, 13)); d.frame(xticks=[0.5, 1.5, 2.5, 3.5], xtl=["baseline", "decorr.", "Gram", "rank"], yticks=[0, 4, 8, 12], grid="y", panel="d", title="layer-1 effective ranks", ylabel="median $r_{\\mathrm{eff}}$")
    for i, (reg, lam, lab, col, sym) in enumerate(LCFG):
        g = D.mq_geom_pooled(reg, lam, 1); d.bars([i + 0.3], [g["kr"]], 0.36, "cBlue"); d.bars([i + 0.7], [g["vr"]], 0.36, "cOrange")
        d.text(i + 0.3, g["kr"] + 0.25, f"{g['kr']:.1f}", anchor="south", size="tiny"); d.text(i + 0.7, g["vr"] + 0.25, f"{g['vr']:.1f}", anchor="south", size="tiny")
    d.legend([("keys", "cBlue", "box"), ("values", "cOrange", "box")], 5.3, 4.35, anchor="north east", size="tiny", rowh=0.25)
    e = Ax(f, 6.65, 1.1, 3.9, 3.3, (1, 20), (0.8, 3.0), xlog=True); e.frame(xticks=[1, 2, 5, 10, 20], xtl=["1", "2", "5", "10", "20"], yticks=[1, 1.5, 2, 2.5, 3], grid="xy", panel="e", title="gain and the certificate ratio", xlabel="mean gain $q$, layer 1", ylabel=r"$\mathrm{RMS}/\psi$")
    qs = np.logspace(0, math.log10(20), 40); e.line(qs, np.sqrt(qs), "cGray", 0.8, "3pt on 2pt")
    for (reg, lam, s_), arr in G.items(): e.marker(float(np.median(arr[1][:, 2])), float(np.median(arr[1][:, 3])), col_of[reg], "o", 0.06, opacity=0.8)
    e.rtext(0.04, 0.92, r"dashed: $\sqrt q$ (flat-gain law)", anchor="west", size="tiny", color="cGray")
    e.legend([(lab, col, "o") for reg, lam, lab, col, sym in LCFG], 10.5, 1.2, anchor="south east", size="tiny", rowh=0.25)
    g_ = Ax(f, 12.1, 1.1, 3.5, 3.3, (0, 1.02), (0, 1.02)); g_.frame(xticks=[0, 0.25, 0.5, 0.75, 1], xtl=["0", "", ".5", "", "1"], yticks=[0, 0.25, 0.5, 0.75, 1], ytl=["0", "", ".5", "", "1"], grid="xy", panel="f", title="no distractor effect", xlabel="accuracy at $L=0$", ylabel="accuracy at $L=96$")
    g_.line([0, 1], [0, 1], "cGray", 0.6, "3pt on 2pt"); dmax = 0
    for (reg, lam, s_), (fn, r) in D.MQ.items():
        pass
    for key, (fn, r) in D.MQ.items():
        reg, lam, s_ = key
        for n in (8, 16, 24, 32): g_.marker(r["grid"][f"{n}_0"], r["grid"][f"{n}_96"], col_of[reg], "o", 0.04, opacity=0.5)
    learned = [r for (reg, lam, s_), (fn, r) in D.MQ.items() if D.mq_acc(r) > 0.5]; dmax = max(abs(np.mean([r["grid"][f"{n}_0"] for n in (8, 16, 24, 32)]) - np.mean([r["grid"][f"{n}_96"] for n in (8, 16, 24, 32)])) for r in learned)
    g_.rtext(0.96, 0.15, rf"max gap {dmax:.3f}", anchor="east", size="tiny")
    return f.out()
