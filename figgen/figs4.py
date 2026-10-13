"""Figures: certificate (six models), literature review."""
import math
import numpy as np
from tz import Fig, Ax, cmap, VIRIDIS, DIVERGE
import data as D

NAMES = [k for k, _ in D.CERT_MODELS]
SHORT = ["G1", "G2", "Q3.5", "R7", "Q2.5", "S2"]
FAMILY = ["delta", "delta", "delta", "rwkv", "tf", "tf"]
FAMCOL = {"delta": "cBlue", "rwkv": "cGreen", "tf": "cOrange"}
MCOL = ["cBlue", "cSky", "cTeal", "cGreen", "cOrange", "cRed"]
DOMS = ["prose", "code", "docs", "legal", "notes"]


def ratio(k, n):
    d = D.CERT[k]; m = d["n"] == n; return d["rms"][m] / d["psi"][m]


def coverage_matrix(n):
    M = np.zeros((6, 6))
    for i, a in enumerate(NAMES):
        ra = D.r90(a, n)
        for j, b in enumerate(NAMES): M[i, j] = float(np.mean(ratio(b, n) <= ra))
    return M


def fig_cert():
    f = Fig(16.0, 5.7); n = 32
    a = Ax(f, 1.45, 1.1, 3.9, 3.7, (0, 6), (0, 1.02)); a.frame(xticks=[0, 1, 2, 3, 4, 5, 6], yticks=[0, 0.25, 0.5, 0.75, 0.9, 1], ytl=["0", ".25", ".5", ".75", ".9", "1"], grid="xy", panel="a", title="distribution of RMS$/\\psi$", xlabel=r"window ratio $\mathrm{RMS}/\psi$ ($n=32$)", ylabel="cumulative fraction of windows")
    a.hline(0.9, "cGray", 0.6, "3pt on 2pt")
    for k, c, s in zip(NAMES, MCOL, SHORT):
        r = np.sort(ratio(k, n)); idx = np.linspace(0, len(r) - 1, 160).astype(int); a.line(r[idx], (idx + 1) / len(r), c, 1.2)
        a.marker(D.r90(k, n), 0.9, c, "o", 0.06)
    a.legend([(nm, c, "line") for nm, c in zip(["GDN 340M", "GDN 360M", "Qwen3.5", "RWKV-7", "Qwen2.5", "SmolLM2"], MCOL)], 5.3, 1.2, anchor="south east", size="tiny", rowh=0.25)
    M = coverage_matrix(n); b = Ax(f, 7.0, 1.1, 3.5, 3.7, (0, 6), (0, 6)); b.heat(M[::-1], list(range(7)), list(range(7)), 0, 1, annotate=lambda v: f"{v:.2f}".lstrip("0") if v < 0.995 else "1.0", textsize="tiny", seams=True)
    b.frame(xticks=[i + 0.5 for i in range(6)], xtl=SHORT, yticks=[i + 0.5 for i in range(6)], ytl=SHORT[::-1], grid=None, panel="b", title="borrowed constant", xlabel="tested on", ylabel="$r_{90}$ calibrated on")
    b.colorbar(0, 1, 10.62, 1.1, 0.14, 3.7, nd=1, ticks=[0, 0.5, 1])
    c = Ax(f, 12.25, 1.1, 3.35, 3.7, (0, 6), (0, 3.4)); c.frame(xticks=[i + 0.5 for i in range(6)], xtl=SHORT, yticks=[0, 0.5, 1, 1.5, 2, 2.5, 3], grid="y", panel="c", title="bound vs measurement", ylabel="median error / value norm")
    for i, k in enumerate(NAMES):
        d = D.CERT[k]; m = d["n"] == n; rms = float(np.median(d["rms"][m])); bd = float(np.median(D.r90(k, n) * d["psi"][m]))
        c.bars([i + 0.3], [rms], 0.34, "cBlue"); c.bars([i + 0.7], [bd], 0.34, "cOrange")
        c.text(i + 0.7, bd + 0.05, f"{bd:.2f}", anchor="south", size="tiny")
    c.hline(1.0, "cGray", 0.6, "3pt on 2pt"); c.legend([("measured RMS", "cBlue", "box"), (r"certificate $r_{90}\psi$", "cOrange", "box")], 12.3, 4.72, anchor="north west", size="tiny", rowh=0.25)
    return f.out()


def fig_cert2():
    f = Fig(16.0, 5.9); n = 32
    a = Ax(f, 1.45, 1.3, 3.9, 3.5, (0, 6), (1.0, 4.2)); a.frame(xticks=[i + 0.5 for i in range(6)], xtl=SHORT, yticks=[1, 2, 3, 4], grid="y", panel="a", title="stable across text domains", ylabel=r"$r_{90}$ per domain ($n=32$)")
    for i, k in enumerate(NAMES):
        d = D.CERT[k]; m = d["n"] == n; vals = [float(np.quantile(d["rms"][m & (d["dom"] == j)] / d["psi"][m & (d["dom"] == j)], 0.9)) for j in range(5)]
        a.line([i + 0.2, i + 0.8], [D.r90(k, n)] * 2, MCOL[i], 1.6)
        for j, v in enumerate(vals): a.marker(i + 0.2 + 0.15 * j, v, MCOL[i], "o", 0.045)
    f.text(a.x0 + a.w / 2, a.y0 - 0.65, "dots: the five domains; line: pooled", size="tiny", color="cGray")
    b = Ax(f, 6.65, 1.3, 3.9, 3.5, (0, 6), (0.7, 1.02)); b.frame(xticks=[i + 0.5 for i in range(6)], xtl=SHORT, yticks=[0.7, 0.8, 0.9, 1.0], grid="y", panel="b", title="leave-one-domain-out coverage", ylabel="coverage (nominal 0.90)")
    b.hline(0.9, "cGray", 0.7, "3pt on 2pt")
    for i, k in enumerate(NAMES):
        d = D.CERT[k]; m = d["n"] == n; cv = []
        for j in range(5):
            tr = m & (d["dom"] != j); te = m & (d["dom"] == j); rr = d["rms"][tr] / d["psi"][tr]; cv.append(float(np.mean(d["rms"][te] / d["psi"][te] <= np.quantile(rr, 0.9))))
        for j, v in enumerate(cv): b.marker(i + 0.2 + 0.15 * j, v, MCOL[i], "o", 0.045)
        b.line([i + 0.2, i + 0.8], [np.mean(cv)] * 2, MCOL[i], 1.4)
    c = Ax(f, 11.95, 1.3, 3.6, 3.5, (0, 3), (1.0, 7.5)); c.frame(xticks=[0.5, 1.5, 2.5], xtl=["early", "middle", "late"], yticks=[1, 2, 3, 4, 5, 6], grid="y", panel="c", title="$r_{90}$ by depth", xlabel="relative depth (thirds)", ylabel=r"$r_{90}$ ($n=32$)")
    for i, k in enumerate(NAMES):
        d = D.CERT[k]; m = d["n"] == n; v = []
        for t in range(3):
            lo, hi = t / 3, (t + 1) / 3; mm = m & (d["layfrac"] >= lo - 1e-9) & ((d["layfrac"] < hi) if t < 2 else (d["layfrac"] <= 1))
            v.append(float(np.quantile(d["rms"][mm] / d["psi"][mm], 0.9)))
        vv = [min(y, 6.6) for y in v]; c.line([0.5, 1.5, 2.5], vv, MCOL[i], 1.3, "3pt on 2pt" if v[0] > 6.6 else None)
        for x, y, y0 in zip([0.5, 1.5, 2.5], vv, v):
            c.marker(x, y, MCOL[i], "o" if y0 <= 6.6 else "t", 0.06 if y0 <= 6.6 else 0.09)
            if y0 > 6.6: c.text(x + 0.12, y - 0.05, f"{y0:.0f}", anchor="west", size="tiny", color=MCOL[i])
    c.legend([(nm, col, "line") for nm, col in zip(["GDN 340M", "GDN 360M", "Qwen3.5", "RWKV-7", "Qwen2.5", "SmolLM2"], MCOL)], 15.5, 4.78, anchor="north east", size="tiny", rowh=0.23, ncol=3, cw=0.07, boxw=0.3)
    return f.out()


# ------------------------------------------------------------------------------------------------ review
def level_class(x):
    if x is None: return None
    if x.startswith("SAME") or x.startswith("CLOSE"): return "same / close"
    if x == "SUPPORT": return "supporting"
    return "context"


def fig_review():
    f = Fig(16.0, 5.9); P = D.REV["prisma"]; C = D.REV["coded"]; inc = [x for x in C if x["level"]]
    # (a) year x level stacked bars
    yrs = list(range(2015, 2027)); cls = ["same / close", "supporting", "context"]; cc = ["cRed", "cGold", "cSky"]
    cnt = {(y, c): 0 for y in yrs for c in cls}
    for x in inc:
        y = int(x["year"]) if x["year"] and str(x["year"]).isdigit() else None
        if y is None: continue
        y = max(y, 2015); cnt[(y, level_class(x["level"]))] += 1
    a = Ax(f, 1.45, 1.3, 4.7, 3.5, (2014.4, 2026.6), (0, 140)); a.frame(xticks=[2015, 2018, 2021, 2024, 2026], xtl=[r"$\le$2015", "2018", "2021", "2024", "2026"], yticks=[0, 40, 80, 120], grid="y", panel="a", title="included records by year", ylabel="records")
    for y in yrs:
        base = 0
        for c, col in zip(cls, cc):
            v = cnt[(y, c)]
            if v: a.bars([y], [base + v], 0.7, col, base=base); base += v
    a.legend([(c, col, "box") for c, col in zip(cls, cc)], 1.5, 4.75, anchor="north west", size="tiny", rowh=0.25)
    # (b) records per claim, stacked by overlap level
    claims = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"]; cntc = {(k, c): 0 for k in claims for c in cls[:2]}
    for x in inc:
        lv = level_class(x["level"])
        if lv not in cls[:2]: continue
        for k in str(x["claims"]).split(","):
            k = k.strip()
            if k in claims: cntc[(k, lv)] += 1
    b = Ax(f, 7.4, 1.3, 3.9, 3.5, (0, 8), (0, 36)); b.frame(xticks=[i + 0.5 for i in range(8)], xtl=claims, yticks=[0, 10, 20, 30], grid="y", panel="b", title="prior work per claim", xlabel="claim (protocol C1--C8)", ylabel="records")
    for i, k in enumerate(claims):
        base = 0
        for lv, col in zip(cls[:2], cc[:2]):
            v = cntc[(k, lv)]
            if v: b.bars([i + 0.5], [base + v], 0.62, col, base=base); base += v
        b.text(i + 0.5, base + 0.5, str(base), anchor="south", size="tiny")
    b.legend([(c, col, "box") for c, col in zip(cls[:2], cc[:2])], 11.25, 4.75, anchor="north east", size="tiny", rowh=0.25)
    # (c) recall funnel
    stages = [("S1--14", 27), ("S1--17", 29), ("keyword", 23), ("title", 21), ("final", 25)]
    c = Ax(f, 12.4, 1.3, 3.2, 3.5, (0, 5), (0, 36)); c.frame(xticks=[i + 0.5 for i in range(5)], xtl=[""] * 5, yticks=[0, 11, 22, 33], grid="y", panel="c", title="recall, 33 known papers", ylabel="found")
    for i, (lab, v) in enumerate(stages):
        c.bars([i + 0.5], [v], 0.62, "cTeal" if i != 4 else "cGreen"); c.text(i + 0.5, v + 0.5, str(v), anchor="south", size="tiny"); f.text(c.X(i + 0.5), c.y0 - 0.1, lab, anchor="north east", size="tiny", rot=45)
    c.hline(33, "cGray", 0.5, "3pt on 2pt")
    return f.out()


def fig_prisma():
    P = D.REV["prisma"]; f = Fig(16.0, 8.6); N = lambda x: f"{x:,}".replace(",", "{,}")
    def box(x, y, w, h, title, lines, fill="cLight"):
        f.rect(x - w / 2, y - h / 2, x + w / 2, y + h / 2, "cGray", fill, 0.6, r=3)
        f.text(x, y + h / 2 - 0.22, title, size="scriptsize", bold=True)
        for i, l in enumerate(lines): f.text(x, y + h / 2 - 0.55 - 0.27 * i, l, size="tiny")
    def arrow(x0, y0, x1, y1): f.line([(x0, y0), (x1, y1)], "cGray", 0.8, arrow="-{Stealth[length=2mm]}")
    cx, w = 5.2, 7.6
    ys = [7.9, 6.55, 5.2, 3.95, 2.7, 1.45]
    box(cx, ys[0], w, 1.0, "Records retrieved", [f"S1--S14: {N(P['raw_main'])} (arXiv {N(P['raw_main_arxiv'])}; OpenAlex {N(P['raw_main_oa'])});  S15--S17: {N(P['raw_supp'])};  chasing: {P['snow_fwd']} + {P['snow_bwd']}"], "cSky")
    box(cx, ys[1], w, 0.95, "After de-duplication (new records)", [f"{N(P['main_unique'])} + {N(P['supp_new_unique'])} + {P['snow_new']}"], "cSky")
    box(cx, ys[2], w, 0.95, "Stage 1: keyword screen, passed", [f"{N(P['main_stage1'])} + {P['supp_stage1']} + {P['snow_stage1']}"], "cSky")
    box(cx, ys[3], w, 0.95, "Amendment 2: S1--S14 rejects re-screened by title", [f"{P['amend2_screened']} screened, {P['amend2_kept']} kept"], "cGold")
    box(cx, ys[4], w, 0.95, "Stage 2: title screening, kept", [f"{P['main_title_kept']} + {P['supp_title_kept']} + {P['snow_title_kept']} + {P['amend2_kept']} (re-screened)"], "cSky")
    box(cx, ys[5], w, 0.95, "Stage 3: abstract screening, included", [f"{P['included']} records"], "cSky")
    for i in range(5): arrow(cx, ys[i] - (0.5 if i == 0 else 0.475), cx, ys[i + 1] + 0.475)
    sx = 12.3; sw = 5.0
    box(sx, ys[2], sw, 0.95, "auto-excluded", [f"{N(P['main_auto_excluded'])} (S1--S14), {P['supp_auto_excluded']} (S15--S17), {P['snow_new'] - P['snow_stage1']} (chasing)"], "cLight")
    box(sx, ys[4], sw, 0.95, "dropped at title", [f"{N(P['main_title_dropped'])} (S1--S14), {P['supp_title_dropped']} (S15--S17), {P['snow_title_dropped']} (chasing)"], "cLight")
    box(sx, ys[5], sw, 0.95, "excluded at abstract", [f"{P['abstract_excluded']}: E1 2, E2 7, E4 8, no abstract 6"], "cLight")
    for i in (2, 4, 5): arrow(cx + w / 2, ys[i], sx - sw / 2, ys[i])
    L = P["levels"]; same = sum(v for k, v in L.items() if k.startswith("SAME") or k.startswith("CLOSE")); sup = L["SUPPORT"]; ctx = L["CONTEXT"]
    f.text(cx, 0.45, rf"coded: {same} same or close to a claim, {sup} supporting, {ctx} context; {sum(1 for x in D.REV['coded'] if x['level'] and x['peer'].startswith('non-peer'))} non-peer-reviewed (Zenodo)", size="scriptsize")
    return f.out(), dict(same=same, sup=sup, ctx=ctx)
