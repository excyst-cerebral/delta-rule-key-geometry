"""Matplotlib figures: weight-space theory, MNIST benchmarks, deep SSM, adapters, toy testbed."""
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mp import *
import data as D
import figs3 as F3


def fam_handles(keys):
    return [Line2D([], [], marker=MARK[k], color=FAMC[k], ls="", ms=4.5, label=FAMN[k]) for k in keys]


def fig_frontier():
    F = F3.frontier_numbers(); pts = np.array(F["pts"]); mus = F["mus"]; fig, ax = new(2.6, 1, 3, wspace=0.55, left=0.08, right=0.985, top=0.87, bottom=0.2); a, b, c = ax
    a.axhline(F["c"], color=C["red"], lw=0.8, ls=(0, (4, 2))); a.text(1.55, F["c"] + 0.02, rf"unavoidable error $c={F['c']:.3f}$", ha="right", va="bottom", fontsize=6.5, color=C["red"])
    a.plot(pts[:, 0], pts[:, 1], color=C["blue"], lw=1.5, zorder=3)
    for mu, col, off in ((0.3, C["green"], (0.08, -0.05)), (2.0, C["orange"], (0.09, -0.05)), (15.0, C["purple"], (0.09, -0.06))):
        e_, E_ = F3.frontier_point(F["Phi"], F["P"], F["e"], mu); a.plot(e_, E_, "o", color=col, ms=5, zorder=5); a.text(e_ + off[0], E_ + off[1], rf"$\mu={mu:g}$", fontsize=6.5, color=col, va="center")
    a.set_xlim(0, 1.6); a.set_ylim(0, 0.8); a.set_xlabel(r"allowed interference $\varepsilon=\|P\Delta\|$"); a.set_ylabel(r"best fit error $E^\ast(\varepsilon)$"); grid(a, "xy"); panel(a, "a", "the frontier")
    rng = np.random.default_rng(2); p = 12; ns = np.arange(1, 26)
    for r_, col in ((2, C["green"]), (5, C["blue"]), (8, C["red"])):
        th = [max(n - min(n, p - r_), 0) / n for n in ns]; b.plot(ns, th, color=col, lw=1.2, label=f"$p-r={p - r_}$")
        Q = np.linalg.qr(rng.normal(size=(p, p)))[0]; P = Q[:, :r_] @ Q[:, :r_].T; mc = []
        for n in ns:
            v = []
            for _ in range(60):
                Phi = rng.normal(size=(n, p)); G = Phi @ (np.eye(p) - P); Pi = G @ np.linalg.pinv(G); e = rng.normal(size=n); v.append(np.sum(((np.eye(n) - Pi) @ e) ** 2) / np.sum(e ** 2))
            mc.append(np.mean(v))
        b.plot(ns, mc, "o", color=col, ms=2.2, mew=0)
    b.plot([], [], "o", color=C["ink"], ms=2.5, label="Monte Carlo"); b.set_xlim(0, 25); b.set_ylim(0, 1.02); b.set_xlabel("new examples $n$"); b.set_ylabel(r"unavoidable error$^2/\|e\|^2$"); grid(b, "xy"); b.legend(loc="upper left", fontsize=6.3); panel(b, "b", "cost of zero interference")
    for rho, col in ((0.3, C["green"]), (0.6, C["blue"]), (0.9, C["red"])):
        kap = math.sqrt(1 - rho ** 2); es = np.linspace(0, 1, 300); ys = [rho * e_ + kap * math.sqrt(max(1 - e_ * e_, 0)) if e_ <= rho else 1.0 for e_ in es]
        c.plot(es, ys, color=col, lw=1.3, label=rf"$\rho={rho}$"); c.plot(0, kap, "o", color=col, ms=3.5, clip_on=False); c.axvline(rho, color=col, lw=0.5, ls=(0, (2, 2)))
    c.set_xlim(0, 1); c.set_ylim(0.3, 1.04); c.set_xlabel(r"allowed interference $\varepsilon/B$"); c.set_ylabel("largest output change $/B$"); grid(c, "xy"); c.legend(loc="lower right", fontsize=6.3); panel(c, "c", "one example: reach")
    save(fig, "fig_frontier"); return F


def fig_minmem():
    M = F3.minmem_data(); fig, ax = new(2.7, 1, 2, wspace=0.3, left=0.075, right=0.985, top=0.88, bottom=0.2); a, b = ax; cols = [C["blue"], C["green"], C["orange"], C["purple"], C["teal"], C["red"]]
    for d, col in zip(M, cols):
        a.plot(d["ms"], d["med"], "o-", color=col, ms=3, lw=1.1); a.annotate(f"$({d['r']},{d['p']})$", (d["ms"][-1], d["med"][-1]), xytext=(4, 0), textcoords="offset points", fontsize=6.3, color=col, va="center", annotation_clip=False)
        b.plot(d["pred"], d["slope"], "o", color=col, ms=5)
    a.set_yscale("log"); a.set_xlim(1.5, 11.8); a.set_ylim(0.03, 2); a.set_xticks([2, 4, 6, 8, 10]); a.set_xlabel("stored bits $m$"); a.set_ylabel(r"median interference $\delta'$"); a.set_yticks([0.05, 0.1, 0.2, 0.5, 1]); a.set_yticklabels(["0.05", "0.1", "0.2", "0.5", "1"]); grid(a, "y"); panel(a, "a", "interference against stored bits")
    b.plot([-0.55, 0], [-0.55, 0], color=C["gray"], lw=0.6, ls=(0, (3, 2))); b.set_xlim(-0.55, 0.02); b.set_ylim(-0.55, 0.02); b.set_xlabel(r"predicted exponent $-1/(r(p-r))$"); b.set_ylabel(r"fitted slope of $\log_2\delta'$"); grid(b, "xy"); panel(b, "b", "decay exponent per bit")
    save(fig, "fig_minmem"); return M


def fig_nomem():
    fig, ax = new(2.5, 1, 1, w=3.5, left=0.15, right=0.97, top=0.87, bottom=0.2); c = ax[0]; ts = np.linspace(-2, 2, 300)
    c.plot(ts, (1 + ts) ** 2 / 2, color=C["blue"], lw=1, ls=(0, (4, 2)), label=r"loss on $A_+$"); c.plot(ts, (1 - ts) ** 2 / 2, color=C["green"], lw=1, ls=(0, (4, 2)), label=r"loss on $A_-$"); c.plot(ts, np.maximum((1 + ts) ** 2 / 2, (1 - ts) ** 2 / 2), color=C["red"], lw=1.7, label="worst case")
    c.plot(0, 0.5, "D", color=C["red"], ms=4.5, zorder=5); c.plot([-1], [0], "o", color=C["blue"], ms=4, zorder=5); c.plot([1], [0], "o", color=C["green"], ms=4, zorder=5)
    c.set_xlim(-2, 2); c.set_ylim(0, 2.1); c.set_xlabel(r"$\theta_2$ after fitting $B$ ($\theta_1=1$)"); c.set_ylabel("old-task loss"); grid(c, "xy"); c.legend(loc="upper center", fontsize=6.5); panel(c, "a", "no memoryless guarantee")
    save(fig, "fig_nomem", tight=True)


# ---------------------------------------------------------------------------------------------------- MNIST
def fig_mnist_pareto():
    fig, ax = new(2.75, 1, 3, wspace=0.42, left=0.075, right=0.985, top=0.87, bottom=0.27)
    for i, (bench, view, ttl) in enumerate(F3.BENCH):
        a = ax[i]; pts = F3.mnist_points(bench, view); jm, js = F3.joint_ref(bench, view); ft = pts[0][2]["final_avg"]
        a.axhline(jm, color=C["ink"], lw=0.7, ls=(0, (4, 2))); a.axhline(ft[0], color=C["gray"], lw=0.7, ls=(0, (1, 1.5)))
        for fam, lab, v in pts[1:]:
            if v["bits"] <= 0: continue
            a.plot(v["bits"], v["final_avg"][0], MARK[fam], color=FAMC[fam], ms=5.5, alpha=0.9, zorder=3)
        a.set_xscale("log"); a.set_xlim(1e4, 3e8); a.set_ylim(0, 1.02); a.set_xlabel("stored bits"); a.set_ylabel("final average accuracy") if i == 0 else None; a.set_xticks([1e4, 1e5, 1e6, 1e7, 1e8]); grid(a, "xy")
        panel(a, "abc"[i], ttl)
    fig.legend(handles=fam_handles(["ewc", "replay", "lw", "lwf", "ogd", "ad"]) + [Line2D([], [], color=C["ink"], lw=0.8, ls=(0, (4, 2)), label="joint training"), Line2D([], [], color=C["gray"], lw=0.8, ls=(0, (1, 1.5)), label="fine-tuning")], loc="lower center", ncol=8, fontsize=6.2, bbox_to_anchor=(0.5, 0.0), columnspacing=1.0, handletextpad=0.3)
    save(fig, "fig_mnist_pareto")


def fig_mnist_bars():
    sel = [("ft", "fine-tune", lambda b, v: D.mnist(b, "ft", v)), ("ewc", "EWC 10", lambda b, v: D.mnist(b, "ewc", v, ewc=10.0)), ("replay", "replay 20", lambda b, v: D.mnist(b, "replay", v, k=20)), ("replay", "replay 200", lambda b, v: D.mnist(b, "replay", v, k=200)),
           ("lw", "layerwise", lambda b, v: D.mnist(b, "gpm", v, rho=0.99, ssm="freeze")), ("ogd", "OGD", lambda b, v: D.mnist(b, "ogd", v, rho=0.99)), ("ad", "adapter 4", lambda b, v: D.adapter(b, 4, v))]
    fig, ax = new(2.8, 1, 3, wspace=0.32, left=0.075, right=0.985, top=0.87, bottom=0.27)
    for i, (bench, view, ttl) in enumerate(F3.BENCH):
        a = ax[i]; jm, _ = F3.joint_ref(bench, view); a.axhline(jm, color=C["ink"], lw=0.6, ls=(0, (4, 2)), zorder=1)
        for j, (fam, lab, g) in enumerate(sel):
            if fam == "ad" and bench == "split" and view == "full": continue
            v = g(bench, view); m, s = v["final_avg"]; a.bar(j, m, 0.7, color=FAMC[fam], zorder=3); a.errorbar(j, m, yerr=s, color=C["ink"], lw=0.6, capsize=1.2, zorder=4); a.plot(j, v["forgetting"][0], "D", mfc="white", mec=C["ink"], ms=3.5, mew=0.7, zorder=5)
        a.set_xticks(range(len(sel))); a.set_xticklabels([s[1] for s in sel], rotation=45, ha="right", fontsize=6.3); a.set_ylim(0, 1.05); grid(a, "y"); a.set_ylabel("final avg. accuracy (bars), forgetting ($\\diamond$)", fontsize=6.6) if i == 0 else None; panel(a, "abc"[i], ttl)
    save(fig, "fig_mnist_bars")


def fig_mnist_heat():
    rows = [("fine-tuning", lambda b, v: D.mnist(b, "ft", v)), (r"EWC $\lambda=10$", lambda b, v: D.mnist(b, "ewc", v, ewc=10.0)), (r"EWC $\lambda=1000$", lambda b, v: D.mnist(b, "ewc", v, ewc=1000.0)), ("replay 20", lambda b, v: D.mnist(b, "replay", v, k=20)), ("replay 200", lambda b, v: D.mnist(b, "replay", v, k=200)),
            (r"layerwise frozen $\rho{=}0.99$", lambda b, v: D.mnist(b, "gpm", v, rho=0.99, ssm="freeze")), (r"layerwise free $\rho{=}0.99$", lambda b, v: D.mnist(b, "gpm", v, rho=0.99, ssm="free")), (r"gradient subspace $\rho{=}0.99$", lambda b, v: D.mnist(b, "ogd", v, rho=0.99)), (r"adapter $r{=}4$", lambda b, v: D.adapter(b, 4, v)), ("joint training", None)]
    fig, ax = new(3.1, 1, 3, widths=[1.5, 1, 1], wspace=0.1, left=0.2, right=0.985, top=0.88, bottom=0.14)
    metrics = [("final_avg", "final average accuracy", CMAP, 0, 1), ("forgetting", "forgetting", SEQ, 0, 1), ("last_task", "last-task accuracy", CMAP, 0, 1)]
    for mi, (key, ttl, cm, lo, hi) in enumerate(metrics):
        a = ax[mi]; Z = np.full((len(rows), 3), np.nan)
        for ri, (lab, g) in enumerate(rows):
            for bi, (bench, view, _t) in enumerate(F3.BENCH):
                if "adapter" in lab and bench == "split" and view == "full": continue
                if g is None:
                    j = D.joint(bench, view); Z[ri, bi] = j["final_avg"][0] if key == "final_avg" else (0.0 if key == "forgetting" else j["last_task"][0]); continue
                Z[ri, bi] = g(bench, view)[key][0]
        m = heat(a, Z[::-1], np.arange(4), np.arange(len(rows) + 1), lo, hi, cmap=cm, annotate=lambda v: f"{v:.2f}".lstrip("0") if v < 1 else "1.0", fs=6)
        a.set_xticks([0.5, 1.5, 2.5]); a.set_xticklabels(["Split\nclass", "Split\ntask", "Permuted"], fontsize=6.5); a.set_yticks(np.arange(len(rows)) + 0.5); a.set_yticklabels([r_[0] for r_ in rows][::-1] if mi == 0 else [], fontsize=6.8); a.tick_params(length=0); panel(a, "abc"[mi], ttl)
    save(fig, "fig_mnist_heat")


def fig_mnist_plane():
    fig, ax = new(2.75, 1, 3, wspace=0.42, left=0.075, right=0.985, top=0.87, bottom=0.27)
    for i, (bench, view, ttl) in enumerate(F3.BENCH):
        a = ax[i]
        for fam, lab, v in F3.mnist_points(bench, view):
            fm, fs = v["forgetting"]; lm, ls = v["last_task"]; a.plot(fm, lm, MARK[fam], color=FAMC[fam], ms=5.5, alpha=0.9, zorder=3)
        a.set_xlim(-0.03, 1.03); a.set_ylim(-0.03, 1.03); a.set_xlabel("forgetting"); a.set_ylabel("last-task accuracy") if i == 0 else None; grid(a, "xy")

        panel(a, "abc"[i], ttl)
    fig.legend(handles=fam_handles(["ft", "ewc", "replay", "lw", "lwf", "ogd", "ad"]), loc="lower center", ncol=7, fontsize=6.3, bbox_to_anchor=(0.5, 0.0), columnspacing=1.0)
    save(fig, "fig_mnist_plane")


# ---------------------------------------------------------------------------------------------------- deep SSM
def fig_deep1():
    L = F3.deep_learners(); fig, ax = new(2.9, 1, 2, widths=[1, 1.15], wspace=0.7, left=0.075, right=0.985, top=0.88, bottom=0.2); a, c = ax
    pick = {0: ("fine-tuning", (2, 10, "center")), 2: ("EWC", (7, 0, "left")), 5: ("replay", (0, 9, "center")), 7: ("gradient subspace", (-8, 0, "right")), 9: ("layerwise, frozen", (7, 0, "left")), 11: ("layerwise, free", (0, -11, "center"))}
    for idx, (lab, (dx, dy, ha)) in pick.items():
        _, fam, v = L[idx]; x_, y_ = v["forgetting"][0], v["last_task"][0]
        a.plot(x_, y_, MARK[fam], color=FAMC[fam], ms=7, zorder=3); a.annotate(lab, (x_, y_), xytext=(dx, dy), textcoords="offset points", ha=ha, va="center", fontsize=6.8, color=FAMC[fam])
    a.set_xlim(-0.03, 0.9); a.set_ylim(0, 1.1); a.set_xlabel("forgetting"); a.set_ylabel("last-task accuracy"); grid(a, "xy"); panel(a, "a", "protection against plasticity")
    base = L[0][2]["per_seed"]["forgetting"]; yl = []
    for k, (lab, fam, v) in enumerate(L[1:]):
        y = len(L) - 2 - k; diffs = [x - z for x, z in zip(v["per_seed"]["forgetting"], base)]; c.plot([min(diffs), max(diffs)], [y, y], color=FAMC[fam], lw=0.6); c.plot(diffs, [y] * len(diffs), "o", color=FAMC[fam], ms=2.4, mew=0); c.plot(np.mean(diffs), y, "D", color=FAMC[fam], mec=C["ink"], mew=0.5, ms=4.2); yl.append(lab)
    c.axvline(0, color=C["gray"], lw=0.6); c.set_yticks(range(len(L) - 1)); c.set_yticklabels(yl[::-1], fontsize=6.5); c.tick_params(axis="y", length=0); c.set_xlim(-0.8, 0.25); c.set_xlabel(r"$\Delta$ forgetting vs fine-tuning (per seed)"); grid(c, "x"); panel(c, "b", "paired change in forgetting")
    save(fig, "fig_deep1")


def fig_deep2():
    items = [("fine-tuning", D.D1[("ft", None, None, None, None)]), (r"EWC $\lambda=10$", D.D1[("ewc", None, 10.0, None, None)]), ("replay 32", D.D1[("replay", None, None, None, 32)]), ("layerwise frozen", D.D2[("gpm", "freeze", 0.99, 32, 12, 0.03, None)])]
    fig, ax = new(2.3, 1, 4, wspace=0.35, left=0.06, right=0.985, top=0.84, bottom=0.2)
    for i, (lab, runs) in enumerate(items):
        a = ax[i]; A = np.array(D.dagg(runs)["acc"]); m = heat(a, A[::-1], np.arange(6), np.arange(6), 0.4, 1.0, annotate=lambda z: f"{z * 100:.0f}", fs=6)
        a.set_xticks(np.arange(5) + 0.5); a.set_xticklabels([str(j + 1) for j in range(5)]); a.set_yticks(np.arange(5) + 0.5); a.set_yticklabels([str(5 - j) for j in range(5)] if i == 0 else []); a.set_xlabel("evaluated task"); a.set_ylabel("after learning task") if i == 0 else None; a.tick_params(length=0); a.set_aspect("equal"); panel(a, "abcd"[i], lab)
    save(fig, "fig_deep2")


def fig_deep3():
    fig, ax = new(2.6, 1, 3, wspace=0.55, left=0.08, right=0.985, top=0.87, bottom=0.21); a, b, c = ax
    vals = np.array([[s[t]["mean_cos2"] for t in range(4)] for s in D.DRIFT]); a.bar(range(4), vals.mean(0), 0.6, color=C["purple"], zorder=3)
    for s in range(len(vals)): a.plot(np.arange(4) + (s - 2) * 0.08, vals[s], "o", color=C["ink"], ms=2.2, mew=0, zorder=4)
    for t in range(4): a.text(t, vals[:, t].max() + 0.03, f"{vals[:, t].mean():.2f}", ha="center", fontsize=6)
    a.axhline(0.005, color=C["red"], lw=0.8, ls=(0, (3, 2))); a.text(-0.4, 0.97, "dashed: random subspace (0.005)", color=C["red"], fontsize=6, ha="left", va="top")
    a.set_xticks(range(4)); a.set_xticklabels(["4", "3", "2", "1"]); a.set_ylim(0, 1); a.set_xlabel("tasks learned before the end"); a.set_ylabel(r"mean $\cos^2$ of principal angles"); grid(a, "y"); panel(a, "a", "subspace rotation")
    rhos = sorted(D.PROBE); xs = [1 - r for r in rhos]; mean_ch, max_ch = [], []
    for r in rhos:
        mean_ch.append([np.mean([p["mean"] for p in run["probe"]]) for run in D.PROBE[r].values()]); max_ch.append([np.max([p["max"] for p in run["probe"]]) for run in D.PROBE[r].values()])
    for x, ch, mx in zip(xs, mean_ch, max_ch): b.plot([x] * len(ch), ch, "o", color=C["blue"], ms=2.2, alpha=0.5, mew=0)
    b.plot(xs, [np.mean(c_) for c_ in mean_ch], "o-", color=C["blue"], ms=4, lw=1.1, label="mean over probes"); b.plot(xs, [np.mean(m_) for m_ in max_ch], "^--", color=C["red"], ms=4, lw=0.9, label="maximum")
    b.set_xscale("log"); b.set_yscale("log"); b.set_xlim(5e-6, 0.3); b.set_ylim(1e-3, 40); b.set_xticks([1e-5, 1e-4, 1e-3, 1e-2, 1e-1]); b.set_xlabel(r"energy outside the subspace $1-\rho$"); b.set_ylabel("change of old-task logits"); grid(b, "xy"); b.legend(loc="upper left", fontsize=6.2); panel(b, "b", "layerwise exactness")
    for i, r in enumerate(rhos):
        fg = [run["forgetting"] for run in D.PROBE[r].values()]; nd = np.mean([sum(sum(x) for x in run["lranks"]) for run in D.PROBE[r].values()])
        c.bar(i, np.mean(fg), 0.6, color=C["blue"], zorder=3); c.errorbar(i, np.mean(fg), yerr=[[np.mean(fg) - min(fg)], [max(fg) - np.mean(fg)]], color=C["ink"], lw=0.6, capsize=1.5, zorder=4); c.text(i, 0.66, f"{nd:.0f} dir.", ha="center", fontsize=6, color=C["gray"])
    c.set_xticks(range(4)); c.set_xticklabels(["0.9", "0.99", "0.999", "0.99999"], fontsize=6.5); c.set_ylim(0, 0.75); c.set_xlabel(r"energy threshold $\rho$"); c.set_ylabel("forgetting"); grid(c, "y"); panel(c, "c", "stored directions")
    save(fig, "fig_deep3")


def fig_adapters():
    ranks = [0, 1, 4, 16]; fig, ax = new(2.6, 1, 2, wspace=0.32, left=0.1, right=0.985, top=0.88, bottom=0.2); a, c = ax
    for bench, view, col, lab in (("split", "task", C["blue"], "Split task-inc."), ("perm", "full", C["orange"], "Permuted (task id)")):
        v = [D.adapter(bench, r, view)["final_avg"] for r in ranks]; a.errorbar(range(4), [x[0] for x in v], yerr=[x[1] for x in v], color=col, marker="o", ms=4, lw=1.2, capsize=1.5, label=lab); a.axhline(D.joint(bench, view)["final_avg"][0], color=col, lw=0.6, ls=(0, (4, 2)))
    a.plot([], [], color=C["gray"], ls=(0, (4, 2)), lw=0.8, label="joint training"); a.set_xticks(range(4)); a.set_xticklabels([str(r) for r in ranks]); a.set_xlabel("adapter rank $r$ (0: head only)"); a.set_ylabel("accuracy (task id given)"); a.set_ylim(0.45, 1.02); grid(a, "y"); a.legend(loc="lower right", fontsize=6.5); panel(a, "a", "accuracy against rank")
    ad = D.adapter("split", 4, "task")["bits"] / 4; r20 = D.mnist("split", "replay", "task", k=20)["bits"] / 5; r200 = D.mnist("split", "replay", "task", k=200)["bits"] / 5; Ts = np.arange(1, 11)
    c.plot(Ts[1:], [ad * (t - 1) for t in Ts[1:]], color=C["purple"], lw=1.3, label="adapters, rank 4"); c.plot(Ts, [r20 * t for t in Ts], color=C["red"], lw=1.2, ls=(0, (4, 2)), label="replay, 20 / task"); c.plot(Ts, [r200 * t for t in Ts], color=C["red"], lw=1.3, label="replay, 200 / task")
    c.set_yscale("log"); c.set_xlim(1, 10); c.set_ylim(1e4, 3e7); c.set_xlabel("tasks $T$"); c.set_ylabel("stored bits"); grid(c, "xy"); c.legend(loc="lower right", fontsize=6.5); panel(c, "b", "memory against stream length")
    save(fig, "fig_adapters")


def fig_toy():
    items = [("fine-tune", ("lti", "ft", "none", 0.0), C["gray"]), ("EWC", ("lti", "ewc", "none", 0.0), C["orange"]), ("replay", ("lti", "replay", "none", 0.0), C["red"]), None,
             ("fine-tune", ("sel", "ft", "none", 0.0), C["gray"]), ("gate sparsity", ("sel", "ft", "sparse", 0.03), C["sky"]), ("gate decorr.", ("sel", "ft", "decor", 2.0), C["sky"]), ("EWC", ("sel", "ewc", "none", 0.0), C["orange"]), ("replay", ("sel", "replay", "none", 0.0), C["red"])]
    fig, ax = new(2.5, 1, 1, w=4.6, left=0.1, right=0.985, top=0.88, bottom=0.3); a = ax[0]
    for i, it in enumerate(items):
        if it is None: continue
        lab, key, col = it; t = D.toy(*key); fs = [r["forgetting"] for r in t["seed"].values()]; m, s_ = np.mean(fs), np.std(fs, ddof=1)
        a.bar(i, m, 0.7, color=col, zorder=3); a.errorbar(i, m, yerr=s_, color=C["ink"], lw=0.7, capsize=1.5, zorder=4)
    a.set_xticks([i for i, it in enumerate(items) if it]); a.set_xticklabels([it[0] for it in items if it], rotation=35, ha="right", fontsize=6.8); a.set_ylim(0, 0.66); a.set_ylabel("forgetting"); grid(a, "y")
    a.text(1, 0.6, "diagonal LTI SSM", ha="center", fontsize=7.5); a.text(6, 0.6, "selective SSM", ha="center", fontsize=7.5)
    save(fig, "fig_toy")


