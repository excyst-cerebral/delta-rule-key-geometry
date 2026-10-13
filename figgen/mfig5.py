"""Matplotlib figure: gate-aware key-only analysis in the actual recurrence (post hoc)."""
import numpy as np
from mp import *
import data as D

M2 = [("GDN-340M", "model_gdn340m", "340M"), ("GDN-360M", "model_gdn360m", "360M")]; DELTAS = [16, 64, 256, 512]
PRED = [("Dgate", r"gate-aware $D_{\rm gate}$", C["blue"], "o", "-", -0.12), ("Dshuf", "keys permuted", C["gray"], "s", (0, (3, 2)), 0.0), ("Dfree", r"gate-free $D_{\rm free}$", C["orange"], "^", "-", 0.12)]


def fig_gate():
    fig, ax = new(2.6, 1, 3, widths=[1.0, 1.0, 1.05], wspace=0.5, left=0.075, right=0.985, top=0.87, bottom=0.22); xs = np.arange(4)
    for i, (nm, tag, short) in enumerate(M2):
        a = ax[i]; r = D.GATE_ACT[nm]["subsets"]["long"]; a.axhline(0, color=C["gray"], lw=0.5, zorder=1)
        for k, lab, col, mk, ls, off in PRED:
            e = np.array([[r[str(d)][k]["partial"][q] for d in DELTAS] for q in ("est", "lo", "hi")])
            a.errorbar(xs + off, e[0], yerr=[e[0] - e[1], e[2] - e[0]], color=col, marker=mk, ls=ls, ms=3.6, lw=1.0, capsize=1.5, elinewidth=0.7, label=lab, zorder=3)
        a.set_xticks(xs); a.set_xticklabels([str(d) for d in DELTAS]); a.set_xlabel(r"delay $\Delta$ (tokens)"); a.set_ylim(-0.2, 0.95); grid(a, "y")
        if i == 0: a.set_ylabel(r"rank corr. with $D_{\rm act}$, decay removed", fontsize=7.5); a.legend(loc="lower left", fontsize=6.2)
        panel(a, "ab"[i], f"{short}, long-memory heads")
    c = ax[2]; x, y, n = D.gate_scatter("model_gdn340m", 64); c.scatter(x, y, s=2.2, color=C["blue"], alpha=0.3, lw=0, rasterized=True); lo, hi = 3e-3, 3.0
    c.plot([lo, hi], [lo, hi], color=C["gray"], lw=0.6, ls=(0, (3, 2))); c.set_xscale("log"); c.set_yscale("log"); c.set_xlim(lo, hi); c.set_ylim(lo, hi)
    c.set_xlabel(r"gate-aware $D_{\rm gate}$ (keys only)"); c.set_ylabel(r"measured $D_{\rm act}$", fontsize=7.5); panel(c, "c", r"340M, $\Delta=64$")
    rho = D.GATE_ACT["GDN-340M"]["subsets"]["long"]["64"]["Dgate"]["within"]["est"]; c.text(0.05, 0.93, rf"within-head $\rho={rho:.2f}$", transform=c.transAxes, fontsize=6.5, va="top")
    save(fig, "fig_gate")


SC = [("GDN-340M", "340M", "o"), ("GDN-360M", "360M", "s"), ("SCALE", "1.3B", "D")]; SP = [("P1", r"gate-free $P_1$", C["orange"]), ("P1g", r"gate-aware $P_{1g}$", C["blue"]), ("P1r", r"retention $P_{1r}$", C["teal"])]


def _scale_vals(nm, k, kind):
    """(estimate, lo, hi) of predictor k: 'flex' = held-out AUC gain over the flexible age baseline, 'cell' = exact within-cell rank correlation with the candidate log-probability."""
    if nm == "SCALE": r = D.SCALE
    else: r = D.GATE_TASK[nm]
    if kind == "flex":
        if k == "P1" and nm != "SCALE": _, _, dl, ci = D.flex_auc(D.TASK[nm]); return dl, ci[0], ci[1]
        v = r["flexible_age"][k]; return v["delta"], v["ci"][0], v["ci"][1]
    if k == "P1" and nm != "SCALE": return D.TASK_LOG[nm]["ex"]
    v = r["within_cell_logp"][k]; return v["r"], v["ci"][0], v["ci"][1]


def fig_scale():
    fig, ax = new(2.9, 1, 3, widths=[0.9, 1.1, 1.1], wspace=0.55, left=0.075, right=0.985, top=0.87, bottom=0.27); a, b, c = ax; loads = [4, 8, 16, 24]
    for nm, lab, mk in SC:
        acc = [float(np.mean(D.TASK[nm]["fc"][D.TASK[nm]["n"] == v])) for v in loads] if nm != "SCALE" else [D.SCALE["accuracy_by_load"][str(v)]["acc"] for v in loads]
        col = {"GDN-340M": C["blue"], "GDN-360M": C["orange"], "SCALE": C["green"]}[nm]; a.plot(range(4), acc, marker=mk, color=col, ms=4, lw=1.1, label=lab)
    a.plot(range(4), [1 / v for v in loads], color=C["gray"], lw=0.8, ls=(0, (3, 2)), label=r"chance $1/n$"); a.set_xticks(range(4)); a.set_xticklabels([str(v) for v in loads]); a.set_xlabel("facts in the prompt $n$")
    a.set_ylabel("forced-choice accuracy"); a.set_ylim(0, 0.62); grid(a, "y"); a.legend(loc="upper right", fontsize=6.2); panel(a, "a", "recall by load")
    for ax_, kind, ttl, lab, xlim in ((b, "flex", "gain over flexible age", r"held-out $\Delta$AUC", (-0.075, 0.075)), (c, "cell", "within exact cells", r"within-cell rank corr. $\rho$", (-0.3, 0.3))):
        y = 0; yt, yl = [], []
        for nm, mlab, mk in SC:
            for k, plab, col in SP:
                e, lo, hi = _scale_vals(nm, k, kind); ax_.errorbar(e, y, xerr=[[e - lo], [hi - e]], fmt=mk, color=col, ms=3.6, lw=0.9, capsize=1.6, zorder=3, label=None); y -= 1
            yt.append(y + 2); yl.append(mlab); y -= 0.6
        ax_.axvline(0, color=C["gray"], lw=0.5, zorder=1); ax_.set_yticks(yt); ax_.set_yticklabels(yl); ax_.set_xlim(*xlim); ax_.set_xlabel(lab, fontsize=7.4); grid(ax_, "x"); panel(ax_, "bc"[kind == "cell"], ttl)
    b.axvline(0.03, color=C["gray"], lw=0.6, ls=(0, (3, 2)), zorder=1)
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([0], [0], marker="o", color=col, lw=0.9, ms=3.6, label=plab) for _, plab, col in SP] + [Line2D([0], [0], color=C["gray"], lw=0.6, ls=(0, (3, 2)), label=r"criterion $+0.03$")], loc="lower center", ncol=4, fontsize=6.6, bbox_to_anchor=(0.62, -0.005), frameon=False, columnspacing=1.4)
    save(fig, "fig_scale")


def fig_lever_rep():
    """Replication of the key-decorrelation effect with new seeds (post hoc; exp_mqar/PROTOCOL_replication.md)."""
    import json as _j, os as _o
    LR = _j.load(open(_o.path.join(D.ROOT, "exp_mqar", "result_replication.json")))
    fig, ax = new(2.6, 1, 3, widths=[1.0, 1.0, 1.05], wspace=0.62, left=0.075, right=0.985, top=0.87, bottom=0.21); a, b, c = ax
    base = {s_: D.mq_acc(r[1]) for (rg, lm, s_), r in D.MQ_REP.items() if rg == "none"}; dec = {s_: D.mq_acc(r[1]) for (rg, lm, s_), r in D.MQ_REP.items() if rg == "dec"}; ss = sorted(set(base) & set(dec))
    for s_ in ss:
        fail = base[s_] < 0.9; a.plot([0, 1], [base[s_], dec[s_]], color=C["red"] if fail else C["light"], lw=0.9 if fail else 0.8, zorder=2 if fail else 1); a.plot([0], [base[s_]], "o", color=C["gray"], ms=2.8, zorder=3); a.plot([1], [dec[s_]], "o", color=C["red"], ms=2.8, zorder=3)
    if ss:
        mb, md = np.mean([base[s_] for s_ in ss]), np.mean([dec[s_] for s_ in ss]); a.plot([0, 1], [mb, md], color=C["ink"], lw=1.8, marker="s", ms=3.5, zorder=4)
        from matplotlib.lines import Line2D
        a.legend(handles=[Line2D([], [], color=C["ink"], lw=1.8, marker="s", ms=3.5, label=f"mean {mb:.3f} $\\to$ {md:.3f}"), Line2D([], [], color=C["red"], lw=0.9, label="baseline below 0.9")], loc="lower right", fontsize=6)
    a.set_xticks([0, 1]); a.set_xticklabels(["baseline", "key decorr."]); a.set_xlim(-0.3, 1.3); a.set_ylim(0, 1.03); a.set_ylabel("mean recall (seed)"); grid(a, "y"); panel(a, "a", f"{len(ss)} new paired seeds")
    rows = [("seeds 0-11", LR["original"]), ("seeds 12-23", LR["replication"]), ("all 24", LR["pooled"])]
    for i, (lab, r) in enumerate(rows):
        if not r["n"]: continue
        b.errorbar(r["mean"], -i, xerr=[[r["mean"] - r["ci"][0]], [r["ci"][1] - r["mean"]]], fmt="o", color=[C["gray"], C["blue"], C["ink"]][i], ms=4, lw=1.0, capsize=1.8, zorder=3); b.text(1.0, -i + 0.28, f"p={r['signflip_p']:.3f}", fontsize=6, ha="right", transform=b.get_yaxis_transform() if False else b.transData)
    b.axvline(0, color=C["gray"], lw=0.5, zorder=1); b.set_yticks([0, -1, -2]); b.set_yticklabels([r[0] for r in rows]); b.set_ylim(-2.6, 0.7); b.set_xlim(-0.2, 1.05); b.set_xlabel(r"paired difference in recall"); grid(b, "x"); panel(b, "b", "effect with new seeds")
    regs = [("key decorrelation", "dec", C["blue"]), ("Gram matching", "match", C["orange"]), ("rank matching", "rank", C["green"])]
    for i, (lab, k, col) in enumerate(regs):
        r = LR["regularisers_seeds0_11"][k]; c.errorbar(r["mean"], -i, xerr=[[r["mean"] - r["ci"][0]], [r["ci"][1] - r["mean"]]], fmt="o", color=col, ms=4, lw=1.0, capsize=1.8, zorder=3); c.text(0.97, -i + 0.3, f"Holm p={r['holm_p']:.2f}", fontsize=6, ha="right", transform=c.transData)
    c.axvline(0, color=C["gray"], lw=0.5, zorder=1); c.set_yticks([0, -1, -2]); c.set_yticklabels([r[0] for r in regs]); c.set_ylim(-2.6, 0.7); c.set_xlim(-0.8, 1.0); c.set_xlabel("paired difference to baseline"); grid(c, "x"); panel(c, "c", "three regularisers")
    save(fig, "fig_lever_rep")

