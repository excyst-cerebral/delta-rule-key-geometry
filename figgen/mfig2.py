"""Matplotlib figures: task-level test and training-time lever."""
import math
import numpy as np
import matplotlib.pyplot as plt
from mp import *
import data as D
import figs2 as F2

M2 = ["GDN-340M", "GDN-360M"]; MCL = {"GDN-340M": C["blue"], "GDN-360M": C["orange"]}; MSY = {"GDN-340M": "o", "GDN-360M": "s"}; LOADS = [4, 8, 16, 24]


def fig_task():
    S = F2.TS(); fig, ax = new(2.8, 1, 3, widths=[1.0, 1.05, 1.0], wspace=0.5, left=0.075, right=0.985, top=0.88, bottom=0.2); a, b, c = ax
    x = np.arange(4); a.plot(x, [1 / n for n in LOADS], color=C["gray"], lw=0.9, ls=(0, (3, 2)), label=r"chance $1/n$")
    for nm in M2:
        d = D.TASK[nm]; ks = [(int(d["fc"][d["n"] == n].sum()), int((d["n"] == n).sum())) for n in LOADS]; p = [k / m for k, m in ks]; ci = [F2.wilson(k, m) for k, m in ks]
        a.fill_between(x, [q[0] for q in ci], [q[1] for q in ci], color=MCL[nm], alpha=0.15, lw=0); a.plot(x, p, marker=MSY[nm], color=MCL[nm], ms=4, label=nm.replace("GDN-", "GDN "))
    a.set_xticks(x); a.set_xticklabels([str(n) for n in LOADS]); a.set_xlabel("facts in the prompt $n$"); a.set_ylabel("forced-choice accuracy"); a.set_ylim(0, 0.62); grid(a, "y"); a.legend(loc="upper right", fontsize=6.3); panel(a, "a", "recall falls with load")
    keys = [("age + load", C["gray"], "age + load"), ("age + load + P1", C["blue"], r"$+P_1$"), ("age + load + shuffled P1", C["sky"], r"$+$ shuffled $P_1$")]
    for gi, nm in enumerate(M2):
        for ki, (k, col, lab) in enumerate(keys):
            xx = gi * 3.6 + ki * 0.95; v = S[nm]["auc"][k]; lo, hi = S[nm]["aucci"][k]
            b.bar(xx, v - 0.35, 0.85, bottom=0.35, color=col, zorder=3, label=lab if gi == 0 else None); b.errorbar(xx, v, yerr=[[v - lo], [hi - v]], color=C["ink"], lw=0.7, capsize=1.5, zorder=4)
    b.axhline(0.5, color=C["gray"], lw=0.5, ls=(0, (2, 2)), zorder=1)
    def dl(nm):
        v = D.TASK_LOG[nm]["H1"][0]; sg = "$-$" if v <= -0.0005 else ("+" if v >= 0.0005 else r"$\pm$"); return f"{nm.replace('GDN-', 'GDN ')}\n$\\Delta$AUC {sg}{abs(v):.3f}"
    b.set_xticks([0.95, 4.55]); b.set_xticklabels([dl(n) for n in M2]); b.set_ylim(0.35, 0.98); b.set_xlim(-0.7, 6.1); b.set_ylabel("test AUC"); grid(b, "y"); b.legend(loc="upper right", fontsize=6.2, ncol=1); panel(b, "b", r"adding $P_1$ to age and load")
    rows = [("linear age", "lin", C["orange"]), ("shuffled null", "lin_s", C["gray"]), ("age matched exactly", "ex", C["blue"]), ("shuffled null", "ex_s", C["gray"])]
    y = 0; yt = []; yl = []
    for lab, key, col in rows:
        for nm in M2:
            r, lo, hi = D.TASK_LOG[nm][key]; c.errorbar(r, y, xerr=[[r - lo], [hi - r]], fmt=MSY[nm], color=col, ms=3.8, lw=0.9, capsize=1.8, zorder=3); yt.append(y); yl.append(nm.replace("GDN-", "")); y -= 1
        c.text(-0.605, yt[-2] + 0.85, lab, fontsize=6.8, color=col, va="center"); y -= 0.9
    c.axvline(0, color=C["gray"], lw=0.6, ls=(0, (3, 2)), zorder=1); c.set_yticks(yt); c.set_yticklabels(yl); c.set_xlim(-0.62, 0.22); c.set_ylim(y + 0.5, 1.45); c.tick_params(axis="y", length=0)
    c.set_xlabel(r"rank corr. of $P_1$ with log-prob."); grid(c, "x"); panel(c, "c", "age matched exactly")
    save(fig, "fig_task")


def cell_matrix(d, v):
    Mx = np.full((4, 24), np.nan)
    for i, n in enumerate(LOADS):
        for q in range(n):
            m = (d["n"] == n) & (d["q"] == q)
            if m.sum() >= 1: Mx[i, q] = np.mean(v[m])
    return Mx


def fig_task_app():
    S = F2.TS(); d = D.TASK["GDN-340M"]; fig, ax = new(4.6, 2, 3, wspace=0.62, hspace=0.78, left=0.075, right=0.93, top=0.92, bottom=0.09, heights=[0.8, 1.0])
    specs = [("mean age (tokens)", d["age"]), (r"mean $P_1$", d["P1"]), ("mean log-prob. of truth", d["clp"])]
    for i, (ttl, v) in enumerate(specs):
        a = ax[0, i]; Mx = cell_matrix(d, v)[::-1]; m = heat(a, Mx, np.arange(25), np.arange(5), edge="none", lw=0)
        a.set_xticks([0.5, 4.5, 9.5, 14.5, 19.5, 23.5]); a.set_xticklabels(["0", "4", "9", "14", "19", "23"]); a.set_yticks([0.5, 1.5, 2.5, 3.5]); a.set_yticklabels(["24", "16", "8", "4"]); a.set_xlabel("queried fact index"); a.set_ylabel("facts $n$") if i == 0 else None
        a.tick_params(length=2); cbar(fig, m, a, ticks=None); panel(a, "abc"[i], ttl)
    a = ax[1, 0]; w = 0.38
    for gi, nm in enumerate(M2):
        dd = D.TASK[nm]; vals = [float(np.corrcoef(D.rk(dd["P1"][dd["n"] == n]), D.rk(dd["age"][dd["n"] == n]))[0, 1]) for n in LOADS]
        xs = np.arange(4) + (gi - 0.5) * w; a.bar(xs, np.array(vals) - 0.5, w * 0.92, bottom=0.5, color=MCL[nm], zorder=3, label=nm.replace("GDN-", "GDN "))
        for x_, v_ in zip(xs, vals): a.text(x_, v_ - 0.02, f"{v_:.2f}", ha="center", va="top", fontsize=5.8, rotation=90, color="white")
    a.set_xticks(range(4)); a.set_xticklabels([str(n) for n in LOADS]); a.set_ylim(0.5, 1.12); a.set_xlabel("facts $n$"); a.set_ylabel(r"Spearman$(P_1,\mathrm{age})$"); grid(a, "y"); a.legend(loc="upper left", fontsize=6, ncol=2); panel(a, "d", r"$P_1$ tracks age")
    for i, nm in enumerate(M2):
        a = ax[1, i + 1]; r, ci, xs_, ys_, keep = S[nm]["ex"]; a.scatter(xs_[keep], ys_[keep], s=4, color=MCL[nm], alpha=0.35, lw=0, zorder=3)
        sl = float(np.polyfit(xs_[keep], ys_[keep], 1)[0]); a.plot([-0.5, 0.5], [-0.5 * sl, 0.5 * sl], color=C["ink"], lw=1.0, zorder=4)
        a.set_xlim(-0.5, 0.5); a.set_ylim(-0.5, 0.5); a.set_xlabel(r"$P_1$ rank, centred in cell"); a.set_ylabel("log-prob. rank, centred") if i == 0 else None; grid(a, "xy")
        a.text(0.04, 0.95, f"$r={r:.3f}$, {int(keep.sum())} prompts", transform=a.transAxes, fontsize=6.3, va="top", bbox=dict(fc="white", ec="none", pad=1.0, alpha=0.9)); panel(a, "ef"[i], f"exact control, {nm.replace('GDN-', 'GDN ')}")
    save(fig, "fig_task_app")


def fig_task_roc():
    fig, ax = new(2.6, 1, 1, w=3.6, left=0.15, right=0.97, top=0.88, bottom=0.21); b = ax[0]; b.axhline(0.5, color=C["gray"], lw=0.5, ls=(0, (2, 2)))
    for nm in M2:
        dd = D.TASK[nm]; ti = ~dd["cal"]
        for lab, v, ls in ((r"$P_1$", dd["P1"], "-"), ("age", dd["age"], (0, (4, 2)))):
            ys = [D.auc(dd["fc"][ti & (dd["n"] == n)], v[ti & (dd["n"] == n)]) for n in LOADS]; b.plot(range(4), ys, ls=ls, color=MCL[nm], lw=1.2, marker=MSY[nm] if ls == "-" else None, ms=4, label=f"{nm.replace('GDN-', '')} {lab}")
    b.set_xticks(range(4)); b.set_xticklabels([str(n) for n in LOADS]); b.set_xlabel("facts $n$"); b.set_ylabel("test AUC"); b.set_ylim(0.15, 0.8); grid(b, "y"); b.legend(loc="upper left", fontsize=6.3, ncol=2, columnspacing=0.8)
    save(fig, "fig_task_roc")


# ---------------------------------------------------------------------------------------------------- lever
LC = {"none": C["gray"], "dec": C["red"], "match": C["orange"], "rank": C["green"]}; LM = {"none": "o", "dec": "s", "match": "D", "rank": "^"}


def fig_lever():
    fig, ax = new(2.5, 1, 3, widths=[1.0, 0.9, 1.0], wspace=0.55, left=0.075, right=0.985, top=0.87, bottom=0.21); a, b, c = ax
    ns = np.array([8, 16, 24, 32])
    for reg, lam, lab, *_ in F2.LCFG:
        byn = D.mq_by_n(reg, lam); m = np.array([np.mean(byn[n]) for n in ns]); se = np.array([np.std(byn[n], ddof=1) / math.sqrt(len(byn[n])) if len(byn[n]) > 1 else 0 for n in ns])
        a.plot(ns, m, marker=LM[reg], color=LC[reg], ms=3.8, label=f"{lab} ({len(D.mq_seeds(reg, lam))})")
    a.set_xticks(ns); a.set_xlabel("stored pairs $n$"); a.set_ylabel("mean recall accuracy"); a.set_ylim(0.2, 1.03); grid(a, "y"); a.legend(loc="lower left", fontsize=6); panel(a, "a", "recall against load")
    sb = D.mq_seeds("none", 0.0); sd = D.mq_seeds("dec", 1.0); pa = []
    for s in sorted(set(sb) & set(sd)):
        x_ = D.mq_acc(D.MQ[("none", 0.0, s)][1]); y_ = D.mq_acc(D.MQ[("dec", 1.0, s)][1]); pa.append((x_, y_)); fail = x_ < 0.9
        b.plot([0, 1], [x_, y_], color=C["red"] if fail else C["light"], lw=0.9 if fail else 0.8, zorder=2 if fail else 1); b.plot([0], [x_], "o", color=C["gray"], ms=2.8, zorder=3); b.plot([1], [y_], "o", color=C["red"], ms=2.8, zorder=3)
    mb, md = np.mean([p[0] for p in pa]), np.mean([p[1] for p in pa]); b.plot([0, 1], [mb, md], color=C["ink"], lw=1.8, zorder=4); b.plot([0, 1], [mb, md], "s", color=C["ink"], ms=4.5, zorder=5)
    from matplotlib.lines import Line2D
    b.legend(handles=[Line2D([], [], color=C["ink"], lw=1.8, marker="s", ms=3.5, label=f"mean {mb:.3f} $\\to$ {md:.3f}"), Line2D([], [], color=C["red"], lw=0.9, label="baseline below 0.9")], loc="lower right", fontsize=6)
    b.set_xlim(-0.3, 1.3); b.set_ylim(0, 1.05); b.set_xticks([0, 1]); b.set_xticklabels(["baseline", "key decorr."]); b.set_ylabel("mean recall (seed)"); grid(b, "y"); panel(b, "b", "12 paired seeds")
    for reg, lam, lab, *_ in F2.LCFG[:2]:
        esc = [D.mq_escape(D.MQ[(reg, lam, s)][1]) for s in D.mq_seeds(reg, lam)]; N = len(esc); xs = sorted(e for e in esc if e is not None); ys = np.arange(1, len(xs) + 1) / N
        c.step([0] + xs + [3000], [0] + list(ys) + [ys[-1]], where="post", color=LC[reg], lw=1.4, label=lab)
    c.set_xlim(0, 3050); c.set_ylim(0, 1.03); c.set_xlabel("training step"); c.set_ylabel("runs with CE $<2$"); grid(c, "xy"); c.legend(loc="lower right", fontsize=6.2); panel(c, "c", "leaving the plateau")
    save(fig, "fig_lever")


def fig_lever_app():
    fig, ax = new(4.4, 2, 2, wspace=0.28, hspace=0.62, left=0.075, right=0.97, top=0.93, bottom=0.09); R = F2.lever_run_table()
    a = ax[0, 0]; Mx = np.full((4, 12), np.nan)
    for i, (reg, *_r) in enumerate(F2.LCFG):
        for s_, (v, e) in R[reg].items(): Mx[3 - i, s_] = v
    m = heat(a, Mx, np.arange(13), np.arange(5), 0, 1, annotate=lambda v: f"{v * 100:.0f}", fs=6)
    a.set_xticks(np.arange(12) + 0.5); a.set_xticklabels([str(i) for i in range(12)], fontsize=6.5); a.set_yticks(np.arange(4) + 0.5); a.set_yticklabels(["rank", "Gram", "decorr.", "baseline"]); a.set_xlabel("seed"); a.tick_params(length=0); cbar(fig, m, a, ticks=[0, 0.5, 1]); panel(a, "a", "accuracy per seed (%)")
    b = ax[0, 1]
    for reg, lam, lab, *_ in F2.LCFG[:2]:
        for k, s_ in enumerate(D.mq_seeds(reg, lam)):
            h = np.array(D.MQ[(reg, lam, s_)][1]["hist"]); b.plot(h[:, 0], h[:, 1], color=LC[reg], lw=0.5, alpha=0.45, label=lab if k == 0 else None)
    b.set_xlim(0, 3000); b.set_ylim(0, 5.4); b.set_xlabel("step"); b.set_ylabel("training CE"); grid(b, "y"); b.legend(loc="upper right", fontsize=6.5, handlelength=1.2); panel(b, "b", "training cross-entropy")
    G = D.mq_geometry(); xs, ys, cs = [], [], []
    for (reg, lam, s_), arr in G.items(): xs.append(float(np.median(arr[1][:, 0]))); ys.append(D.mq_acc(D.MQ[(reg, lam, s_)][1])); cs.append(LC[reg])
    c = ax[1, 0]; c.scatter(xs, ys, s=14, c=cs, alpha=0.85, lw=0, zorder=3); c.set_xscale("log"); c.set_xticks([1, 2, 5, 10]); c.set_xticklabels(["1", "2", "5", "10"]); c.set_xlabel(r"layer-1 key effective rank $r_{\mathrm{eff}}$"); c.set_ylabel("mean recall"); grid(c, "xy")
    c.text(0.19, 0.6, rf"Spearman $\rho_s={F2.spear(xs, ys):.2f}$" + "\n" + f"({len(xs)} runs)", transform=c.transAxes, ha="left", fontsize=6.8)
    from matplotlib.lines import Line2D
    c.legend(handles=[Line2D([], [], marker="o", color=LC[r_], ls="", ms=4, label=l_) for r_, l_ in (("none", "baseline"), ("dec", "decorrelation"), ("match", "Gram"), ("rank", "rank"))], loc="lower center", ncol=2, bbox_to_anchor=(0.62, 0.0), fontsize=6.2, labelspacing=0.2, handletextpad=0.1, columnspacing=0.8); panel(c, "c", "accuracy against key rank")
    f_ = ax[1, 1]; f_.plot([0, 1], [0, 1], color=C["gray"], lw=0.6, ls=(0, (3, 2)))
    for key, (fn, r) in D.MQ.items():
        for n in (8, 16, 24, 32): f_.plot(r["grid"][f"{n}_0"], r["grid"][f"{n}_96"], "o", color=LC[key[0]], ms=2.4, alpha=0.5, mew=0)
    learned = [r for k, (fn, r) in D.MQ.items() if D.mq_acc(r) > 0.5]; dmax = max(abs(np.mean([r["grid"][f"{n}_0"] for n in (8, 16, 24, 32)]) - np.mean([r["grid"][f"{n}_96"] for n in (8, 16, 24, 32)])) for r in learned)
    f_.set_xlim(0, 1.02); f_.set_ylim(0, 1.02); f_.set_xlabel("accuracy at $L=0$"); f_.set_ylabel("accuracy at $L=96$"); grid(f_, "xy"); f_.text(0.96, 0.08, f"largest per-run gap {dmax:.3f}", transform=f_.transAxes, ha="right", fontsize=6.8); panel(f_, "d", "no distractor effect")
    save(fig, "fig_lever_app")
