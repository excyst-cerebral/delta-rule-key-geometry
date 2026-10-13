"""Matplotlib figures: literature review."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from mp import *
import data as D
import figs4 as F4


def fig_review():
    P = D.REV["prisma"]; Cd = D.REV["coded"]; inc = [x for x in Cd if x["level"]]; fig, ax = new(2.6, 1, 3, widths=[1.15, 1.0, 0.85], wspace=0.5, left=0.075, right=0.985, top=0.87, bottom=0.2); a, b, c = ax
    yrs = list(range(2015, 2027)); cls = ["same / close", "supporting", "context"]; cc = [C["red"], C["gold"], C["sky"]]; cnt = {(y, k): 0 for y in yrs for k in cls}
    for x in inc:
        if not (x["year"] and str(x["year"]).isdigit()): continue
        cnt[(max(int(x["year"]), 2015), F4.level_class(x["level"]))] += 1
    for y in yrs:
        base = 0
        for k, col in zip(cls, cc):
            v = cnt[(y, k)]
            if v: a.bar(y, v, 0.72, bottom=base, color=col, zorder=3, label=k if y == 2015 else None); base += v
    a.set_xticks([2015, 2018, 2021, 2024, 2026]); a.set_xticklabels([r"$\leq$2015", "2018", "2021", "2024", "2026"]); a.set_ylabel("included records"); grid(a, "y"); a.legend(loc="upper left", fontsize=6.3); panel(a, "a", "records by year")
    claims = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"]; cn = {(k, l): 0 for k in claims for l in cls[:2]}
    for x in inc:
        lv = F4.level_class(x["level"])
        if lv not in cls[:2]: continue
        for k in str(x["claims"]).split(","):
            if k.strip() in claims: cn[(k.strip(), lv)] += 1
    for i, k in enumerate(claims):
        base = 0
        for l, col in zip(cls[:2], cc[:2]):
            v = cn[(k, l)]
            if v: b.bar(i, v, 0.66, bottom=base, color=col, zorder=3); base += v
        b.text(i, base + 0.4, str(base), ha="center", fontsize=6)
    b.set_xticks(range(8)); b.set_xticklabels(claims, fontsize=6.5); b.set_ylim(0, 40); b.set_ylabel("records"); grid(b, "y"); from matplotlib.patches import Patch
    b.legend(handles=[Patch(color=c_, label=l) for c_, l in zip(cc[:2], cls[:2])], loc="upper left", ncol=2, fontsize=6.3, columnspacing=1.0); panel(b, "b", "prior work per claim")
    stages = [("S1$-$14", 27), ("S1$-$17", 29), ("keyword", 23), ("title", 21), ("final", 25)]
    for i, (lab, v) in enumerate(stages): c.bar(i, v, 0.66, color=C["teal"] if i != 4 else C["green"], zorder=3); c.text(i, v + 0.4, str(v), ha="center", fontsize=6.3)
    c.axhline(33, color=C["gray"], lw=0.5, ls=(0, (2, 2))); c.set_xticks(range(5)); c.set_xticklabels([s[0] for s in stages], rotation=40, ha="right", fontsize=6.3); c.set_ylim(0, 36); c.set_ylabel("known papers found"); grid(c, "y"); panel(c, "c", "recall (of 33)")
    save(fig, "fig_review")


def fig_prisma():
    P = D.REV["prisma"]; N = lambda x: f"{x:,}"; fig = plt.figure(figsize=(TEXTW, 4.6)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 10); ax.set_ylim(0, 9.2); ax.axis("off")
    def box(x, y, w, h, title, body, fc="#E3F1EF", ec_="#2A8C8C"):
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.0,rounding_size=0.08", fc=fc, ec=ec_, lw=0.8))
        ax.text(x, y + h / 2 - 0.2, title, ha="center", va="top", fontproperties=BOLD, fontsize=7.3); ax.text(x, y - 0.2, body, ha="center", va="center", fontsize=6.8)
    def arrow(x0, y0, x1, y1): ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7, lw=0.7, color=C["gray"], shrinkA=0, shrinkB=0))
    cx, w, h = 2.95, 5.6, 0.95; ys = [8.4, 6.95, 5.5, 4.2, 2.9, 1.6]
    box(cx, ys[0], w, h, "Records retrieved", f"S1$-$S14: {N(P['raw_main'])} (arXiv {N(P['raw_main_arxiv'])}; OpenAlex {N(P['raw_main_oa'])})\nS15$-$S17: {N(P['raw_supp'])}; citation chasing: {P['snow_fwd']} + {P['snow_bwd']}")
    box(cx, ys[1], w, h, "After de-duplication (new records)", f"{N(P['main_unique'])} + {N(P['supp_new_unique'])} + {P['snow_new']}")
    box(cx, ys[2], w, h, "Stage 1: keyword screen, passed", f"{N(P['main_stage1'])} + {P['supp_stage1']} + {P['snow_stage1']}")
    box(cx, ys[3], w, h, "Amendment 2 (post hoc)", f"S1$-$S14 rejects re-screened by title: {P['amend2_screened']} screened, {P['amend2_kept']} kept", fc="#FBEFD5", ec_="#A9822A")
    box(cx, ys[4], w, h, "Stage 2: title screening, kept", f"{P['main_title_kept']} + {P['supp_title_kept']} + {P['snow_title_kept']} + {P['amend2_kept']} (re-screened)")
    box(cx, ys[5], w, h, "Stage 3: abstract screening, included", f"{P['included']} records")
    for i in range(5): arrow(cx, ys[i] - h / 2, cx, ys[i + 1] + h / 2)
    sx, sw = 8.15, 3.3
    box(sx, ys[2], sw, h, "auto-excluded", f"{N(P['main_auto_excluded'])} (S1$-$S14)\n{P['supp_auto_excluded']} (S15$-$S17), {P['snow_new'] - P['snow_stage1']} (chasing)", fc="#EFEFEF", ec_="#9A9A9A")
    box(sx, ys[4], sw, h, "dropped at title", f"{N(P['main_title_dropped'])} (S1$-$S14)\n{P['supp_title_dropped']} (S15$-$S17), {P['snow_title_dropped']} (chasing)", fc="#EFEFEF", ec_="#9A9A9A")
    box(sx, ys[5], sw, h, "excluded at abstract", f"{P['abstract_excluded']}: E1 2, E2 7, E4 8, none 6", fc="#EFEFEF", ec_="#9A9A9A")
    for i in (2, 4, 5): arrow(cx + w / 2, ys[i], sx - sw / 2, ys[i])
    L = P["levels"]; same = sum(v for k, v in L.items() if k.startswith("SAME") or k.startswith("CLOSE")); npr = sum(1 for x in D.REV["coded"] if x["level"] and x["peer"].startswith("non-peer"))
    ax.text(5.0, 0.55, f"coded: {same} same or close to a claim, {L['SUPPORT']} supporting, {L['CONTEXT']} context; {npr} non-peer-reviewed (Zenodo)", ha="center", fontsize=7)
    save(fig, "fig_prisma")
