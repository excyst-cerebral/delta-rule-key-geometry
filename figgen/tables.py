"""LaTeX table fragments generated from data.py (booktabs, no float wrapper: the caption and label live in the document)."""
import math
import numpy as np
import data as D
import figs2

TOP, MID, BOT = r"\toprule", r"\midrule", r"\bottomrule"


def p3(x, nd=3):
    s = f"{x:.{nd}f}"
    if abs(x) < 1: s = s.replace("0.", ".", 1) if x >= 0 else s.replace("-0.", "-.", 1)
    return s.replace("-", "$-$")


def sgn(x, nd=3):
    s = f"{abs(x):.{nd}f}".replace("0.", ".", 1) if abs(x) < 1 else f"{abs(x):.{nd}f}"
    return ("$-$" if x < 0 and round(abs(x), nd) > 0 else "$+$" if x > 0 and round(x, nd) > 0 else "$\\pm$") + s


def sd(m, s, nd=3): return f"{p3(m, nd)}$\\pm${p3(s, nd)}"


def bits(b):
    if b == 0: return "0"
    if b < 1e5: return f"{int(round(b)):,}".replace(",", "\\,")
    if b < 1e6: return f"{int(round(b)):,}".replace(",", "\\,")
    return f"{b / 1e6:.2f}\\,M" if b < 1e7 else f"{b / 1e6:.1f}\\,M"


def tab(rows, colspec, header, extra=""):
    out = [f"\\begin{{tabular}}{{{colspec}}}", TOP, header + r" \\", MID]
    for r in rows:
        out.append(MID if r == "MID" else r + r" \\")
    out += [BOT, "\\end{tabular}"]; return "\n".join(out) + "\n"


# --------------------------------------------------------------------------------------------- task level
def tab_task():
    rows = []
    for nm, lab in (("GDN-340M", "GDN 340M (primary)"), ("GDN-360M", "GDN 360M (replic.)")):
        d = D.TASK[nm]; S = figs2.TS()[nm]; acc = "/".join(p3(d["fc"][d["n"] == n].mean(), 2) for n in figs2.LOADS)
        dl, lo, hi = D.TASK_LOG[nm]["H1"]
        rows.append(f"{lab} & {acc} & {p3(S['auc']['age + load'])} & {p3(S['auc']['age + load + P1'])} & {sgn(dl)} ({sgn(lo)}, {sgn(hi)})")
    return tab(rows, "lcccc", r"Model & accuracy at $n=4/8/16/24$ & AUC age$+$load & AUC $+P_1$ & $\Delta$ (95\% CI)")


def tab_task_full():
    rows = []; S = figs2.TS()
    def row(lab, f): return f"{lab} & {f('GDN-340M')} & {f('GDN-360M')}"
    A = lambda nm, key: S[nm]["auc"][key]
    def a_raw(nm, v, flip=False):
        d = D.TASK[nm]; ti = ~d["cal"]; x = D.auc(d["fc"][ti], d[v][ti]); return p3(1 - x if flip else x)
    rows += [row("age", lambda nm: a_raw(nm, "age", True)), row("load $n$", lambda nm: a_raw(nm, "n", True)), row("age + one-hot load (B3)", lambda nm: p3(A(nm, "age + load"))),
             row(r"$P_1$ (long-memory heads)", lambda nm: a_raw(nm, "P1", True)), row(r"$P_{1b}$ (all heads)", lambda nm: a_raw(nm, "P1b", True)), row(r"$P_1$ shuffled within load", lambda nm: a_raw(nm, "P1s", True)),
             row(r"B3 + $P_1$", lambda nm: p3(A(nm, "age + load + P1")))]
    def b3b(nm):
        d = D.TASK[nm]; N = len(d["n"]); onehot = np.stack([(d["n"] == v).astype(float) for v in sorted(set(d["n"]))[1:]], 1); X = np.column_stack([np.ones(N), d["age"], onehot, d["P1b"]])
        return p3(D.auc(d["fc"][~d["cal"]], D.logistic(X, d["fc"], d["cal"])[~d["cal"]]))
    rows += [row(r"B3 + $P_{1b}$", b3b), row(r"B3 + shuffled $P_1$", lambda nm: p3(A(nm, "age + load + shuffled P1"))), "MID",
             row(r"H1: AUC(B3+$P_1$) $-$ AUC(B3), 95\% CI", lambda nm: f"{sgn(D.TASK_LOG[nm]['H1'][0])} ({sgn(D.TASK_LOG[nm]['H1'][1])}, {sgn(D.TASK_LOG[nm]['H1'][2])})"),
             row(r"AUC(B3+$P_1$) $-$ AUC(B3+shuffled), 95\% CI", lambda nm: f"{sgn(D.TASK_LOG[nm]['H1s'][0])} ({sgn(D.TASK_LOG[nm]['H1s'][1])}, {sgn(D.TASK_LOG[nm]['H1s'][2])})")]
    return tab(rows, "lcc", r"predictor & GDN 340M & GDN 360M")


# --------------------------------------------------------------------------------------------- training-time lever
def tab_lever():
    rows = []
    for reg, lam, lab, *_ in [x[:4] for x in figs2.LCFG]:
        byn = D.mq_by_n(reg, lam); seeds = D.mq_seeds(reg, lam); allm = [D.mq_acc(D.MQ[(reg, lam, s)][1]) for s in seeds]
        cells = [p3(np.mean(byn[n])) for n in (8, 16, 24, 32)]
        if reg == "dec": cells = [f"\\textbf{{{c}}}" for c in cells]; m, s_ = f"\\textbf{{{p3(np.mean(allm))}}}", f"\\textbf{{{p3(np.std(allm, ddof=1))}}}"; lab = f"\\textbf{{{lab}}}"
        else: m, s_ = p3(np.mean(allm)), p3(np.std(allm, ddof=1))
        rows.append(f"{lab} & {len(seeds)} & " + " & ".join(cells) + f" & {m} & {s_}")
    return tab(rows, "lccccccc", r"learner & seeds & $n{=}8$ & 16 & 24 & 32 & mean & SD across seeds")


def tab_lever_seeds():
    R = figs2.lever_run_table(); out = [r"\begin{tabular}{lcccccccccccc}", TOP, "seed & " + " & ".join(str(i) for i in range(12)) + r" \\", MID]
    for reg, lab in (("none", "baseline"), ("dec", "key decorrelation"), ("match", "Gram matching"), ("rank", "rank matching")):
        accs = [(p3(R[reg][s][0], 2) if s in R[reg] else "--") for s in range(12)]
        if reg in ("none", "dec"): out.append(f"{lab} & " + " & ".join(accs) + r" \\")
        else: out.append(f"{lab} & " + " & ".join(accs) + r" \\")
        if reg in ("none", "dec"): out.append(r"\quad escape step & " + " & ".join((str(R[reg][s][1]) if (s in R[reg] and R[reg][s][1] is not None) else "--") for s in range(12)) + r" \\")
    out += [BOT, r"\end{tabular}"]; return "\n".join(out).replace("1.00", "1.00") + "\n"


def tab_geometry():
    rows = []
    for reg, lam, lab, *_ in [x[:4] for x in figs2.LCFG]:
        g0, g1 = D.mq_geom_pooled(reg, lam, 0), D.mq_geom_pooled(reg, lam, 1); f = lambda k, nd=2: f"{g0[k]:.{nd}f} / {g1[k]:.{nd}f}"
        hl = (lambda s: s) if reg != "dec" else (lambda s: s)
        rows.append(f"{lab} & {g0['kr']:.2f} / {g1['kr']:.2f} & {g0['vr']:.2f} / {g1['vr']:.2f} & {g0['ratio']:.2f} / {g1['ratio']:.2f} & {g0['q']:.2f} / {g1['q']:.2f} & {g0['rms_psi']:.2f} / {g1['rms_psi']:.2f}")
    return tab(rows, "lccccc", r"learner & key $r_{\mathrm{eff}}$ & value $r_{\mathrm{eff}}$ & values/keys & $q$ & $\mathrm{RMS}/\psi$")


# --------------------------------------------------------------------------------------------- MNIST
def tab_mnist_main():
    L = [("Fine-tuning", lambda b, v: D.mnist(b, "ft", v)), (r"EWC, $\lambda=10$", lambda b, v: D.mnist(b, "ewc", v, ewc=10.0)), ("Replay, 20 / task", lambda b, v: D.mnist(b, "replay", v, k=20)), ("Replay, 200 / task", lambda b, v: D.mnist(b, "replay", v, k=200)),
         ("Layerwise projection", lambda b, v: D.mnist(b, "gpm", v, rho=0.99, ssm="freeze")), ("Gradient subspace (OGD)", lambda b, v: D.mnist(b, "ogd", v, rho=0.99)), (r"Task adapters (rank 4)$^{\dagger}$", lambda b, v: D.adapter(b, 4, v))]
    best = {}
    for bi, (b, v) in enumerate((("split", "full"), ("split", "task"), ("perm", "full"))):
        best[bi] = max(f(b, v)["final_avg"][0] for lab, f in L if "adapters" not in lab)
    rows = []
    for lab, f in L:
        cells = []
        for bi, (b, v) in enumerate((("split", "full"), ("split", "task"), ("perm", "full"))):
            if "adapters" in lab and bi == 0: cells.append("--"); continue
            x = f(b, v); s = f"{p3(x['final_avg'][0])} ({p3(x['forgetting'][0])})"
            if abs(x["final_avg"][0] - best[bi]) < 1e-9 and "adapters" not in lab: s = f"\\textbf{{{p3(x['final_avg'][0])}}} ({p3(x['forgetting'][0])})"
            cells.append(s)
        bs = [f("split", "task")["bits"], f("perm", "full")["bits"]]
        rows.append(f"{lab} & {bits(bs[0])} / {bits(bs[1])} & " + " & ".join(cells))
    return tab(rows, "lcccc", r"Learner & bits (Split / Permuted) & Split, class-inc. & Split, task-inc. & Permuted")


def _mnist_full(bench, view, with_joint=True):
    rows = []; g = lambda x: f"{p3(x['final_avg'][0])}$\\pm${p3(x['final_avg'][1])} & {p3(x['forgetting'][0])}$\\pm${p3(x['forgetting'][1])} & {p3(x['last_task'][0])}$\\pm${p3(x['last_task'][1])}"
    def grp(lab, items, fmt=None):
        xs = [i for i in items]; m = lambda k: "/".join(p3(x[k][0]) for x in xs)
        return f"{lab} & {m('final_avg')} & {m('forgetting')} & {m('last_task')} & " + "/".join(bits(x["bits"]) for x in xs)
    x = D.mnist(bench, "ft", view); rows.append(f"fine-tuning & {g(x)} & 0")
    rows.append(grp(r"EWC $\lambda=10/100/1000$", [D.mnist(bench, "ewc", view, ewc=l) for l in (10.0, 100.0, 1000.0)]).rsplit(" & ", 1)[0] + f" & {bits(D.mnist(bench, 'ewc', view, ewc=10.0)['bits'])}")
    rows.append(grp(r"layerwise, free, $\rho=0.95/0.99$", [D.mnist(bench, "gpm", view, rho=r, ssm="free") for r in (0.95, 0.99)]))
    rows.append(grp(r"layerwise, frozen, $\rho=0.95/0.99/0.999$", [D.mnist(bench, "gpm", view, rho=r, ssm="freeze") for r in (0.95, 0.99, 0.999)]))
    rows.append(grp(r"OGD $\rho=0.9/0.99$", [D.mnist(bench, "ogd", view, rho=r) for r in (0.9, 0.99)]))
    rows.append(grp(r"replay $k=20/200$", [D.mnist(bench, "replay", view, k=k) for k in (20, 200)]))
    j = D.joint(bench, view); rows.append(f"joint training & \\multicolumn{{4}}{{c}}{{{p3(j['final_avg'][0])}$\\pm${p3(j['final_avg'][1])}}}")
    return tab(rows, "lcccc", r"learner & final avg & forgetting & last task & bits")


def tab_mnist_split_ci(): return _mnist_full("split", "full")
def tab_mnist_split_ti(): return _mnist_full("split", "task")
def tab_mnist_perm(): return _mnist_full("perm", "full")


def tab_adapters():
    rows = []; labs = {0: "head only ($r=0$)", 1: "rank 1 + head", 4: "rank 4 + head", 16: "rank 16 + head"}
    for r in (0, 1, 4, 16):
        s, p = D.adapter("split", r, "task"), D.adapter("perm", r, "full")
        rows.append(f"{labs[r]} & {bits(s['bits'])} / {bits(p['bits'])} & {p3(s['final_avg'][0])} & {p3(p['final_avg'][0])}")
    return tab(rows, "lccc", r"learner & bits (Split / Permuted) & Split task-inc.\ acc & Permuted (task id) acc")


# --------------------------------------------------------------------------------------------- deep SSM, toy testbed
def tab_deep1():
    G = D.dagg; d1 = D.D1; base = d1[("ft", None, None, None, None)]; bf = [base[s]["forgetting"] for s in sorted(base)]
    def pd(runs):
        return float(np.mean([runs[s]["forgetting"] - base[s]["forgetting"] for s in sorted(runs)])), int(sum(runs[s]["forgetting"] < base[s]["forgetting"] for s in sorted(runs)))
    def line(lab, keys, fmtk=lambda k: k):
        xs = [G(d1[k]) for k in keys]; pds = [pd(d1[k]) for k in keys]
        m = lambda key: "/".join(p3(x[key][0], 3 if key == "forgetting" else (2 if key == "final_avg" and False else 3)) for x in xs)
        wins = f" ({'/'.join(str(p[1]) for p in pds)}/5)" if len(pds) > 0 else ""
        return f"{lab} & {'/'.join(bits(x['bits']) for x in xs)} & {'/'.join(p3(x['final_avg'][0], 2) for x in xs)} & {m('forgetting')} & {m('last_task')} & " + "/".join(sgn(p[0]) for p in pds) + wins
    x0 = G(base)
    rows = [f"fine-tuning & 0 & {p3(x0['final_avg'][0], 2)} & {p3(x0['forgetting'][0])} & {p3(x0['last_task'][0])} & --",
            line(r"projection, $\rho=0.5/0.9/0.99/0.9999$, 32 bits", [("ogd", r, None, 32, None) for r in (0.5, 0.9, 0.99, 0.9999)]),
            line(r"projection, $\rho=0.9$, 2/4/8 bits", [("ogd", 0.9, None, q, None) for q in (2, 4, 8)]),
            line(r"EWC, $\lambda=1/10/100$", [("ewc", None, l, None, None) for l in (1.0, 10.0, 100.0)]),
            line("replay, 4/32 per task", [("replay", None, None, None, k) for k in (4, 32)])]
    return tab(rows, r">{\raggedright\arraybackslash}p{3.6cm}rcccc", r"learner & bits & final avg & forgetting & last task & $\Delta$forgetting (wins)")


def tab_toy():
    rows = []
    def r_(arch, lab, key, joint=False):
        t = D.toy(*key)
        if joint: return f"{arch} & {lab} & {sd(*t['final_avg'])} & -- & --"
        fa = sd(*t["final_avg"]); fg = sd(*t["forgetting"])
        if key[1] == "ft" and key[2] == "none": fg = f"\\textbf{{{fg}}}"; pdl = "--"
        else:
            m, w, n = D.toy_paired(key[0], key[1], key[2], key[3]); pdl = f"{sgn(m)} ({w}/{n})"
        return f"{arch} & {lab} & {fa} & {fg} & {pdl}"
    rows += [r_("LTI", "joint (reference)", ("lti", "joint", "none", 0.0), True), r_("LTI", "fine-tuning (pure)", ("lti", "ft", "none", 0.0)), r_("LTI", "EWC (anchor)", ("lti", "ewc", "none", 0.0)), r_("LTI", "replay", ("lti", "replay", "none", 0.0)),
             r_("selective", "joint (reference)", ("sel", "joint", "none", 0.0), True), r_("selective", "fine-tuning (pure)", ("sel", "ft", "none", 0.0))]
    for lab, reg, lams in (("gate sparsity", "sparse", (0.01, 0.03)), ("gate decorrelation", "decor", (0.5, 2.0))):
        ts = [D.toy("sel", "ft", reg, l) for l in lams]; ps = [D.toy_paired("sel", "ft", reg, l) for l in lams]
        rows.append(f"selective & + {lab}, $\\lambda={'/'.join(f'{l:g}' for l in lams)}$ & {'/'.join(p3(t['final_avg'][0]) for t in ts)} & {'/'.join(p3(t['forgetting'][0]) for t in ts)} & " + " / ".join(f"{sgn(p[0])} ({p[1]}/{p[2]})" for p in ps))
    rows += [r_("selective", "EWC (anchor)", ("sel", "ewc", "none", 0.0)), r_("selective", "replay", ("sel", "replay", "none", 0.0))]
    return tab(rows, "llccc", r"arch & learner & final avg.\ accuracy & forgetting & paired $\Delta$ forgetting (wins)")


def tab_r90():
    rank = [0.55, 0.85, 0.47, 2.2, 5.5, 6.4]; trained = ["1.07", "1.21", "0.82--1.02", "1.93", "1.95", "2.45"]; names = ["Gated DeltaNet 1", "Gated DeltaNet 2", "Qwen3.5-0.8B", "RWKV-7 0.1B", "Qwen2.5-1.5B", "SmolLM2-135M"]; rows = []
    for i, (k, _) in enumerate(D.CERT_MODELS):
        cells = []
        for n in (16, 32):
            r, (lo, hi), ns = D.r90_ci(k, n); cells.append(f"{r:.2f} [{lo:.2f}, {hi:.2f}]")
        body = " / ".join(cells); rr = f"{rank[i]:g}"; tr = trained[i]
        if k == "R7": body, rr, tr = f"\\textbf{{{body}}}", f"\\textbf{{{rr}}}", f"\\textbf{{{tr}}}"
        rows.append(f"{names[i]} & {D.r90_ci(k, 32)[2]} & {body} & {rr} & {tr}")
    return tab(rows, "lcccc", r"model & sequences & pooled $r_{90}$ ($n=16$ / $n=32$) & rank ratio & trained $\mathrm{RMS}/\psi$")


def tab_validation():
    import json
    rows = []
    for tag, nm in (("model_gdn340m", "Gated DeltaNet 340M"), ("model_gdn360m", "Gated DeltaNet 360M"), ("model_gdn13b", "Gated DeltaNet 1.3B")):
        r = json.load(open(f"{D.ROOT}/validate_gdn_{tag}.json")); ml = r["model_level"]; lay = max(r["layers"].values())
        fmt = lambda x: f"{x:.1e}".replace("e-0", "e-").replace("e-", "\\times10^{-") + "}"
        sci = lambda x: "$" + fmt(x) + "$"
        rows.append(f"{nm} & {sci(lay)} & {sci(max(m['max_abs_logit_diff'] for m in ml))} & {ml[0]['logit_scale']:.1f} / {ml[1]['logit_scale']:.1f} & {sci(max(m['max_kl'] for m in ml))} & {100 * min(m['top1_agree'] for m in ml):.0f}\\% & {ml[0]['ppl_mine']:.4f} / {ml[0]['ppl_ref']:.4f} & {ml[1]['ppl_mine']:.4f} / {ml[1]['ppl_ref']:.4f}")
    rw = json.load(open(f"{D.ROOT}/validate_rwkv7_reference.json")); p, c = rw["prose"]["float32"], rw["code"]["float32"]
    rows.append(f"RWKV-7 0.1B & --- & {sci(max(p['max_abs_dlogit'], c['max_abs_dlogit']))} & {p['mean_abs_logit']:.1f} / {c['mean_abs_logit']:.1f} & {sci(max(p['max_kl'], c['max_kl']))} & {100 * min(p['top1_agree'], c['top1_agree']):.0f}\\% & {p['ppl_ours']:.4f} / {p['ppl_ref']:.4f} & {c['ppl_ours']:.4f} / {c['ppl_ref']:.4f}")
    return tab(rows, "lccccccc", r"model & layer & max $|\Delta$logit$|$ & mean $|$logit$|$ & max KL & top-1 & prose ppl (ours / ref.) & code ppl (ours / ref.)")


def tab_flexauc():
    rows = []
    for nm, lab in (("GDN-340M", "GDN 340M"), ("GDN-360M", "GDN 360M")):
        for lam in (0.3, 1.0, 3.0):
            a0, a1, dl, ci = D.flex_auc(D.TASK[nm], lam=lam)
            rows.append(f"{lab} & {lam:g}" + f" & {p3(a0)} & {p3(a1)} & {sgn(dl)} ({sgn(ci[0])}, {sgn(ci[1])})")
    return tab(rows, "lcccc", r"model & ridge $\lambda$ & AUC cell effects & AUC $+P_1$ & $\Delta$ (95\% CI)")


def tab_taskresults():
    rows = []
    for nm, lab in (("GDN-340M", "GDN 340M"), ("GDN-360M", "GDN 360M")):
        L = D.TASK_LOG[nm]; a0, a1, dl, ci = D.flex_auc(D.TASK[nm])
        rows.append(f"{lab} & {sgn(L['H1'][0])} ({sgn(L['H1'][1])}, {sgn(L['H1'][2])}) & {sgn(dl)} ({sgn(ci[0])}, {sgn(ci[1])}) & {sgn(L['ex'][0])} ({sgn(L['ex'][1])}, {sgn(L['ex'][2])})")
    return tab(rows, "lccc", r"model & pre-registered $\Delta$AUC & categorical-age $\Delta$AUC & within-cell $\rho$")


def numbers():
    """Macros for numbers quoted in the prose."""
    S = figs2.TS(); N = {}
    for k, nm in (("a", "GDN-340M"), ("b", "GDN-360M")):
        N[f"autoLin{k}"] = f"{D.TASK_LOG[nm]['lin'][0]:.3f}"
    byn = D.mq_by_n("none", 0.0); sb = D.mq_seeds("none", 0.0); sd_ = D.mq_seeds("dec", 1.0)
    mb = np.mean([D.mq_acc(D.MQ[("none", 0.0, s)][1]) for s in sb]); md = np.mean([D.mq_acc(D.MQ[("dec", 1.0, s)][1]) for s in sd_])
    N["mqBase"] = f"{mb:.2f}"; N["mqDec"] = f"{md:.2f}"
    return N


def tab_gate_actual():
    f = lambda e: f"{p3(e['est'], 2)} ({p3(e['lo'], 2)}, {p3(e['hi'], 2)})"; rows = []
    for nm, lab in (("GDN-340M", "GDN 340M"), ("GDN-360M", "GDN 360M")):
        r = D.GATE_ACT[nm]["subsets"]["long"]
        if rows: rows.append("MID")
        for i, dl in enumerate((16, 64, 256, 512)):
            q = r[str(dl)]; rows.append(f"{lab if i == 0 else ''} & {dl} & {p3(q['Dgate']['within']['est'], 2)} & {f(q['Dgate']['partial'])} & {f(q['Dshuf']['partial'])} & {f(q['Dfree']['partial'])} & {f(q['Bkey']['partial'])}")
    return tab(rows, "llccccc", r"model & delay & $D_{\rm gate}$, within & $D_{\rm gate}$, decay removed & $D_{\rm shuf}$ & $D_{\rm free}$ & $B_{\rm key}$")


def tab_gate_task():
    lab = {"P1g": r"$P_{1g}$, gate-aware (long-memory heads)", "P1r": r"$P_{1r}$, retention", "P1A": r"decay only, $-\log A$", "P1g_shuffled": r"$P_{1g}$ shuffled within load", "P1gb": r"$P_{1g}$, all heads"}; rows = []
    for nm, mn in (("GDN-340M", "GDN 340M"), ("GDN-360M", "GDN 360M")):
        r = D.GATE_TASK[nm]
        if rows: rows.append("MID")
        for i, k in enumerate(lab):
            a, b, c = r["pre_registered_style"][k], r["flexible_age"][k], r["within_cell_logp"][k]
            rows.append(f"{mn if i == 0 else ''} & {lab[k]} & {sgn(a['delta'])} ({sgn(a['ci'][0])}, {sgn(a['ci'][1])}) & {sgn(b['delta'])} ({sgn(b['ci'][0])}, {sgn(b['ci'][1])}) & {sgn(c['r'])} ({sgn(c['ci'][0])}, {sgn(c['ci'][1])})")
    return tab(rows, "llccc", r"model & predictor & $\Delta$AUC, age + load & $\Delta$AUC, flexible age & within-cell $\rho$")


def tab_scale_task():
    lab = {"P1": r"$P_1$, gate-free $\psi$ (long-memory heads)", "P1g": r"$P_{1g}$, gate-aware (long-memory heads)", "P1r": r"$P_{1r}$, retention", "P1A": r"decay only, $-\log A$", "P1_shuffled": r"$P_1$ shuffled within load",
           "P1g_shuffled": r"$P_{1g}$ shuffled within load", "P1b": r"$P_{1b}$, gate-free, all heads", "P1gb": r"$P_{1g}$, all heads"}; r = D.SCALE; rows = []
    for k in lab:
        a, b, c = r["pre_registered_style"][k], r["flexible_age"][k], r["within_cell_logp"][k]
        rows.append(f"{lab[k]} & {sgn(a['delta'])} ({sgn(a['ci'][0])}, {sgn(a['ci'][1])}) & {sgn(b['delta'])} ({sgn(b['ci'][0])}, {sgn(b['ci'][1])}) & {sgn(c['r'])} ({sgn(c['ci'][0])}, {sgn(c['ci'][1])})")
    return tab(rows, "lccc", r"predictor & $\Delta$AUC, age + load & $\Delta$AUC, flexible age & within-cell $\rho$")


def tab_heldout():
    rows = []
    for k, nm in D.CERT_MODELS:
        for i, n in enumerate(("16", "32")):
            v = D.HELDOUT[k][n]; rows.append(f"{nm if i == 0 else ''} & {n} & {v['r90_published']:.2f} & {v['r90_heldout']:.2f} & {p3(v['coverage'])} ({p3(v['ci'][0])}, {p3(v['ci'][1])}) & {v['sequences']}")
        if k != D.CERT_MODELS[-1][0]: rows.append("MID")
    return tab(rows, "llcccc", r"model & $n$ & published $r_{90}$ & held-out $r_{90}$ & coverage (95\% CI) & sequences")


def tab_lever_rep():
    import json as _j
    LR = _j.load(open(f"{D.ROOT}/exp_mqar/result_replication.json")); rows = []
    for key, lab in (("original", "0--11"), ("replication", "12--23"), ("pooled", "all 24")):
        r = LR[key]; n = r["n"]
        rows.append(f"{lab} & {p3(r['base_mean'])} & {p3(r['dec_mean'])} & {sgn(r['mean'])} ({sgn(r['ci'][0])}, {sgn(r['ci'][1])}) & {r['higher']}/{n} & {p3(r['signflip_p'])} & {p3(r['wilcoxon_p'])} & {r['base_failed']}/{r['dec_failed']}")
    return tab(rows, "lcccccc c".replace(" ", ""), r"seeds & baseline & decorr. & paired difference & higher & sign-flip $p$ & Wilcoxon $p$ & below 0.9")

