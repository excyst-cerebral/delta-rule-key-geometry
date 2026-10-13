"""Matplotlib figures: overview, theory checks, measurement study, certificate."""
import math
import numpy as np
import matplotlib.pyplot as plt
from mp import *
import data as D
import figs1 as F1
import figs4 as F4

NAMES = [k for k, _ in D.CERT_MODELS]
SHORT = F1.SHORT
FAMILY = F1.FAMILY
FC = {"delta": C["blue"], "rwkv": C["green"], "tf": C["orange"]}
MC = [C["blue"], C["sky"], C["teal"], C["green"], C["orange"], C["red"]]
LONG = ["GDN 340M", "GDN 360M", "Qwen3.5", "RWKV-7", "Qwen2.5", "SmolLM2"]


def fig_overview():
    fig, ax = new(2.5, 1, 3, widths=[1.0, 1.0, 1.0], wspace=0.95, left=0.075, right=0.985, top=0.87, bottom=0.2)
    a, b, c = ax
    N = 60; g = (np.arange(N) + 0.5) / N; Z = np.outer(g, g)
    m = a.pcolormesh(np.linspace(0, 1, N + 1), np.linspace(0, 1, N + 1), Z, cmap=CMAP, vmin=0, vmax=1, shading="flat", rasterized=True)
    xs = np.linspace(0.01, 1, 200)
    for lev in (0.1, 0.25, 0.5):
        a.plot(xs, np.minimum(lev / xs, 1.2), color="white", lw=0.7, ls=(0, (3, 2))); a.text(0.96, lev / 0.96 + 0.03, f"{lev:g}", color="white", fontsize=6, ha="right", va="bottom")
    a.set_xlim(0, 1); a.set_ylim(0, 1); a.set_xlabel(r"key overlap $|\langle k_t,p\rangle|$"); a.set_ylabel(r"surprise $\|e_t\|$"); a.set_xticks([0, 0.5, 1]); a.set_yticks([0, 0.5, 1])
    for s in a.spines.values(): s.set_visible(True)
    cbar(fig, m, a, label=r"interference$/\beta_t$", ticks=[0, 0.5, 1])
    panel(a, "a", "interference law")
    ys = np.arange(6)[::-1]
    b.barh(ys, F1.R90[32], color=[FC[f] for f in FAMILY], height=0.62, zorder=3)
    for y, v in zip(ys, F1.R90[32]): b.text(v + 0.05, y, f"{v:.2f}", va="center", fontsize=6.5)
    b.set_yticks(ys); b.set_yticklabels(LONG); b.set_xlim(0, 4.9); b.set_xlabel(r"$r_{90}$ in $\mathrm{RMS}\leq r_{90}\psi$ ($n=32$)"); grid(b, "x"); b.tick_params(axis="y", length=0)
    from matplotlib.patches import Patch
    b.legend(handles=[Patch(color=C["blue"], label="delta rule"), Patch(color=C["green"], label="RWKV-7"), Patch(color=C["orange"], label="softmax")], loc="upper right", fontsize=6.3, handlelength=1.0)
    panel(b, "b", "calibrated bound")
    rows = [(nm, *D.TASK_LOG[nm]["lin"], "lin") for nm in ("GDN-340M", "GDN-360M")] + [(nm, *D.TASK_LOG[nm]["ex"], "ex") for nm in ("GDN-340M", "GDN-360M")]
    ypos = [3.2, 2.6, 1.2, 0.6]
    for (nm, r, lo, hi, kind), y in zip(rows, ypos):
        col = C["orange"] if kind == "lin" else C["blue"]; c.errorbar(r, y, xerr=[[r - lo], [hi - r]], fmt="o" if "340" in nm else "s", color=col, ms=4, lw=1.0, capsize=2, zorder=3)
    c.axvline(0, color=C["gray"], lw=0.6, ls=(0, (3, 2)), zorder=1); c.set_yticks(ypos); c.set_yticklabels(["340M", "360M", "340M", "360M"]); c.set_ylim(0, 4.3); c.set_xlim(-0.6, 0.2)
    c.text(0.02, 3.95, "age as a linear term", transform=c.get_yaxis_transform(), fontsize=6.8, color=C["orange"], va="center"); c.text(0.02, 1.85, "age matched exactly", transform=c.get_yaxis_transform(), fontsize=6.8, color=C["blue"], va="center")
    c.set_xlabel(r"rank corr. with log-prob."); grid(c, "x"); c.tick_params(axis="y", length=0); panel(c, "c", "recall of a stored fact")
    save(fig, "fig_overview")


def fig_theory():
    t = F1.theory_data(); fig, ax = new(2.5, 1, 3, wspace=0.55, left=0.08, right=0.985, top=0.87, bottom=0.2); a, b, c = ax
    xs, ys = t["xs"], t["ys"]; lo = 10 ** math.floor(math.log10(min(xs.min(), ys.min()))); hi = 10 ** math.ceil(math.log10(max(xs.max(), ys.max())))
    a.loglog([lo, hi], [lo, hi], color=C["gray"], lw=0.6, ls=(0, (3, 2)), zorder=1); a.scatter(xs, ys, s=5, color=C["blue"], alpha=0.5, lw=0, zorder=3)
    a.set_xlim(lo, hi); a.set_ylim(lo, hi); a.set_xlabel(r"$\mathrm{tr}(MB)$"); a.set_ylabel(r"$n\,\mathrm{RMS}^2$ (simulated pass)"); a.set_xticks([1e-2, 1, 1e2]); a.set_yticks([1e-2, 1, 1e2])
    a.text(0.05, 0.95, f"300 random instances\nmax. rel. error ${t['relerr'] * 1e15:.1f}\\times10^{{-15}}$", transform=a.transAxes, va="top", fontsize=6.5); panel(a, "a", "gain-weighted identity")
    Bs = np.asarray(t["Bs"]); b.plot(Bs, t["floor"], color=C["red"], lw=1.6, label="capacity floor", zorder=3)
    b.plot(Bs[::7], np.asarray(t["att"])[::7], "o", color=C["blue"], ms=3.2, mfc="white", mew=0.9, label="attaining memory", zorder=4)
    b.plot([t["Bop"]], [t["err_delta"]], "D", color=C["gold"], ms=5, label="one delta-rule pass", zorder=5)
    b.set_xlim(0, 6); b.set_ylim(0, 12.5); b.set_xlabel(r"operator-norm budget $B$"); b.set_ylabel("squared recall error"); grid(b, "y"); b.legend(loc="upper right", fontsize=6.5)
    panel(b, "b", "capacity floor")
    cols = (C["blue"], C["green"], C["orange"]); ii = np.arange(1, 31)
    for mu, col in zip((0.01, 0.03, 0.06), cols):
        Fi, form, dev = t["coh"][mu]; c.plot(ii, Fi, color=col, lw=1.5, label=rf"$\mu={mu}$", zorder=3); c.plot(ii[::2], np.asarray(form)[::2], "o", color=C["ink"], ms=2.2, zorder=4)
    c.plot([], [], "o", color=C["ink"], ms=2.5, ls="", label="closed form")
    c.set_xlim(1, 30); c.set_ylim(0, 0.9); c.set_xlabel(r"item index $i$ ($n=30$)"); c.set_ylabel(r"error $F_i$"); grid(c, "y"); c.legend(loc="upper right", fontsize=6.5, title="simulated pass", title_fontsize=6.5); panel(c, "c", r"coherence: $\Theta(\mu n)$")
    save(fig, "fig_theory"); return dict(relerr=t["relerr"], floor_gap=t["err_delta"] - t["floor_at_Bop"], coh_dev=max(v[2] for v in t["coh"].values()))


def fig_measure():
    fig, ax = new(2.6, 1, 2, widths=[1.1, 1.0], wspace=0.28, left=0.07, right=0.985, top=0.88, bottom=0.27); a, b = ax
    x = np.arange(6)
    for i in range(6):
        col = FC[FAMILY[i]]; a.plot([i, i], [F1.R90[16][i], F1.R90[32][i]], color=col, lw=1.2, zorder=2); a.plot(i, F1.R90[16][i], "o", mfc="white", mec=col, ms=4.5, zorder=3); a.plot(i, F1.R90[32][i], "o", color=col, ms=4.5, zorder=3)
    a.set_xticks(x); a.set_xticklabels(SHORT); a.set_ylim(1, 4); a.set_ylabel(r"pooled $r_{90}$"); grid(a, "y"); a.plot([], [], "o", mfc="white", mec=C["gray"], ls="", label=r"$n=16$"); a.plot([], [], "o", color=C["gray"], ls="", label=r"$n=32$"); a.legend(loc="upper left", fontsize=6.5)
    panel(a, "a", "calibration constant")
    off = {"G1": (0, -9, "center"), "G2": (6, -5, "left"), "Q3.5": (-4, 4, "right"), "R7": (6, 0, "left"), "Q2.5": (-6, 0, "right"), "S2": (6, -4, "left")}
    for i in range(6):
        b.plot(F1.RANKRATIO[i], F1.R90[32][i], "o", color=FC[FAMILY[i]], ms=4.5, zorder=3); dx, dy, ha = off[SHORT[i]]; b.annotate(SHORT[i], (F1.RANKRATIO[i], F1.R90[32][i]), xytext=(dx, dy), textcoords="offset points", ha=ha, va="center", fontsize=6.8, color=FC[FAMILY[i]])
    b.set_xscale("log"); b.set_xlim(0.3, 9); b.set_ylim(1.2, 3.9); b.axvline(1, color=C["gray"], lw=0.5, ls=(0, (2, 2))); b.set_xticks([0.5, 1, 2, 5]); b.set_xticklabels(["0.5", "1", "2", "5"]); b.set_xlabel("values/keys effective-rank ratio"); b.set_ylabel(r"pooled $r_{90}$ ($n=32$)"); grid(b, "y")
    panel(b, "b", "what sets the level")
    fig.text(0.07, 0.015, "G1, G2: Gated DeltaNet 340M, 360M;  Q3.5: Qwen3.5-0.8B;  R7: RWKV-7 0.1B;  Q2.5: Qwen2.5-1.5B;  S2: SmolLM2-135M", fontsize=6.3, color=C["gray"])
    save(fig, "fig_measure")


def fig_decay():
    fig, ax = new(2.3, 1, 1, w=3.3, left=0.2, right=0.97, top=0.86, bottom=0.2); c = ax[0]
    xs = np.arange(3); w = 0.36; c.bar(xs - w / 2, F1.INTERF, w, color=C["red"], label="later writes", zorder=3); c.bar(xs + w / 2, F1.DECAY, w, color=C["blue"], label="designed decay", zorder=3)
    for xx, v in zip(xs - w / 2, F1.INTERF): c.text(xx, v + 0.01, f"{v:g}", ha="center", fontsize=6.5)
    for xx, v in zip(xs + w / 2, F1.DECAY): c.text(xx, v + 0.01, f"{v:g}", ha="center", fontsize=6.5)
    c.set_xticks(xs); c.set_xticklabels(["16", "64", "256"]); c.set_xlabel("tokens after the write"); c.set_ylabel("relative loss of a stored association", fontsize=7.5); c.set_ylim(0, 0.52); grid(c, "y"); c.legend(loc="upper left", fontsize=6.8); panel(c, "a", "interference vs decay")
    save(fig, "fig_decay")


def fig_cert():
    n = 32; fig, ax = new(2.6, 1, 3, widths=[1.0, 1.05, 0.95], wspace=0.7, left=0.07, right=0.985, top=0.87, bottom=0.2); a, b, c = ax
    for k, col, nm in zip(NAMES, MC, LONG):
        r = np.sort(F4.ratio(k, n)); a.plot(r, np.arange(1, len(r) + 1) / len(r), color=col, lw=1.1, label=nm); a.plot(D.r90(k, n), 0.9, "o", color=col, ms=3.5, zorder=4)
    a.axhline(0.9, color=C["gray"], lw=0.5, ls=(0, (2, 2))); a.set_xscale("log"); a.set_xlim(0.3, 60); a.set_xticks([0.5, 1, 2, 5, 10, 30]); a.set_xticklabels(["0.5", "1", "2", "5", "10", "30"]); a.set_ylim(0, 1.0); a.set_xlabel(r"window ratio $\mathrm{RMS}/\psi$ ($n=32$)"); a.set_ylabel("cumulative fraction"); grid(a, "y"); a.legend(loc="lower right", fontsize=6)
    panel(a, "a", "ratio distribution")
    M = F4.coverage_matrix(n); m = heat(b, M[::-1], np.arange(7), np.arange(7), 0, 1, annotate=lambda v: (f"{v:.2f}".lstrip("0") if v < 0.995 else "1.0"), fs=6)
    b.set_xticks(np.arange(6) + 0.5); b.set_xticklabels(SHORT); b.set_yticks(np.arange(6) + 0.5); b.set_yticklabels(SHORT[::-1]); b.set_xlabel("tested on"); b.set_ylabel(r"$r_{90}$ calibrated on"); b.tick_params(length=0)
    cb = cbar(fig, m, b, ticks=[0, 0.5, 1]); panel(b, "b", "borrowed constant")
    for i, k in enumerate(NAMES):
        d = D.CERT[k]; mm = d["n"] == n; rms = float(np.median(d["rms"][mm])); bd = float(np.median(D.r90(k, n) * d["psi"][mm]))
        c.bar(i - 0.19, rms, 0.36, color=C["blue"], zorder=3, label="measured RMS" if i == 0 else None); c.bar(i + 0.19, bd, 0.36, color=C["orange"], zorder=3, label=r"calibrated bound $r_{90}\psi$" if i == 0 else None); c.text(i + 0.19, bd + 0.03, f"{bd:.2f}", ha="center", fontsize=5.8)
    c.axhline(1, color=C["gray"], lw=0.5, ls=(0, (2, 2))); c.set_xticks(range(6)); c.set_xticklabels(SHORT); c.set_ylim(0, 2.45); c.set_ylabel("median error / value norm"); grid(c, "y"); c.legend(loc="upper left", fontsize=6.2); panel(c, "c", "bound vs data")
    save(fig, "fig_cert")


def fig_cert2():
    n = 32; fig, ax = new(2.6, 1, 3, wspace=0.5, left=0.075, right=0.985, top=0.87, bottom=0.22); a, b, c = ax
    for i, k in enumerate(NAMES):
        d = D.CERT[k]; mm = d["n"] == n
        vals = [float(np.quantile(d["rms"][mm & (d["dom"] == j)] / d["psi"][mm & (d["dom"] == j)], 0.9)) for j in range(5)]
        a.vlines(i, min(vals), max(vals), color=MC[i], lw=2.4, alpha=0.5, zorder=2); a.plot(i, D.r90(k, n), "o", color=MC[i], ms=5, zorder=3)
        cv = []
        for j in range(5):
            tr = mm & (d["dom"] != j); te = mm & (d["dom"] == j); rr = d["rms"][tr] / d["psi"][tr]; cv.append(float(np.mean(d["rms"][te] / d["psi"][te] <= np.quantile(rr, 0.9))))
        b.vlines(i, min(cv), max(cv), color=MC[i], lw=2.4, alpha=0.5, zorder=2); b.plot(i, np.mean(cv), "o", color=MC[i], ms=5, zorder=3)
    for ax_ in (a, b): ax_.set_xticks(range(6)); ax_.set_xticklabels(SHORT); grid(ax_, "y")
    a.set_ylim(1, 4.1); a.set_ylabel(r"$r_{90}$ ($n=32$)"); panel(a, "a", "across text domains")
    b.axhline(0.9, color=C["gray"], lw=0.5, ls=(0, (2, 2))); b.set_ylim(0.8, 1.0); b.set_ylabel("coverage (nominal 0.90)"); panel(b, "b", "leave one domain out")
    for i, k in enumerate(NAMES):
        d = D.CERT[k]; mm = d["n"] == n; v = []
        for t in range(3):
            lo, hi = t / 3, (t + 1) / 3; m2 = mm & (d["layfrac"] >= lo - 1e-9) & ((d["layfrac"] < hi) if t < 2 else (d["layfrac"] <= 1)); v.append(float(np.quantile(d["rms"][m2] / d["psi"][m2], 0.9)))
        vv = [min(y, 6.4) for y in v]; c.plot([0, 1, 2], vv, "-", color=MC[i], lw=1.2, ls="-" if v[0] <= 6.4 else (0, (3, 2)), label=LONG[i])
        for x, y, y0 in zip([0, 1, 2], vv, v):
            c.plot(x, y, "o" if y0 <= 6.4 else "^", color=MC[i], ms=3.5)
            if y0 > 6.4: c.annotate(f"{y0:.0f}", (x, y), xytext=(4, -2), textcoords="offset points", fontsize=6, color=MC[i])
    c.set_xticks([0, 1, 2]); c.set_xticklabels(["early", "middle", "late"]); c.set_xlim(-0.25, 2.25); c.set_ylim(1, 7.4); c.set_ylabel(r"$r_{90}$ by depth"); grid(c, "y"); c.legend(loc="upper right", ncol=2, fontsize=5.8, columnspacing=0.8, handlelength=1.1); panel(c, "c", "across depth")
    fig.text(0.075, 0.02, "(a, b) dot: pooled value; bar: range over the five text domains. (c) depth in thirds", fontsize=6.3, color=C["gray"])
    save(fig, "fig_cert2")
