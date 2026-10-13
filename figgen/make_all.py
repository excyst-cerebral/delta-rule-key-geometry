"""Writes every figure (matplotlib, vector PDF at the paper's text width) and table fragment into overleaf/build/.  Run with figgen/.venv/bin/python."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mp, mfig1, mfig2, mfig3, mfig4, mfig5, tables

OUT = mp.OUT; os.makedirs(OUT, exist_ok=True); INFO = {}
for mod, names in ((mfig1, ["fig_overview", "fig_theory", "fig_measure", "fig_decay", "fig_cert", "fig_cert2"]), (mfig2, ["fig_task", "fig_task_app", "fig_task_roc", "fig_lever", "fig_lever_app"]),
                   (mfig3, ["fig_frontier", "fig_minmem", "fig_nomem", "fig_mnist_pareto", "fig_mnist_bars", "fig_mnist_heat", "fig_mnist_plane", "fig_deep1", "fig_deep2", "fig_deep3", "fig_adapters", "fig_toy"]), (mfig4, ["fig_review", "fig_prisma"]), (mfig5, ["fig_gate", "fig_scale", "fig_lever_rep"])):
    for n in names: INFO[n] = getattr(mod, n)(); print("figure", n)
for n in ("tab_task", "tab_task_full", "tab_lever", "tab_lever_seeds", "tab_geometry", "tab_mnist_main", "tab_mnist_split_ci", "tab_mnist_split_ti", "tab_mnist_perm", "tab_adapters", "tab_deep1", "tab_toy", "tab_r90", "tab_validation", "tab_flexauc", "tab_taskresults", "tab_gate_actual", "tab_gate_task", "tab_scale_task", "tab_heldout", "tab_lever_rep"):
    open(os.path.join(OUT, n + ".tex"), "w").write(getattr(tables, n)()); print("table", n)
F = INFO["fig_frontier"]; M = INFO["fig_minmem"]
import data as Dd
def sg(x): return "0.000" if abs(x) < 0.0005 else f"{x:+.3f}"
mac = f"\\newcommand{{\\frontierC}}{{{F['c']:.3f}}}\n\\newcommand{{\\frontierPct}}{{{100 * F['c']:.0f}}}\n"
for tag, nm in (("a", "GDN-340M"), ("b", "GDN-360M")):
    a0, a1, dl, ci = Dd.flex_auc(Dd.TASK[nm]); mac += f"\\newcommand{{\\flexD{tag}}}{{{sg(dl)}}}\n\\newcommand{{\\flexLo{tag}}}{{{sg(ci[0])}}}\n\\newcommand{{\\flexHi{tag}}}{{{sg(ci[1])}}}\n"
def r2(x): return f"{x:.2f}".replace("-", "$-$") if x < 0 else f"{x:.2f}"
def rng2(vals): return r2(min(vals)), r2(max(vals))
Ga = Dd.GATE_ACT; part = lambda k, sub="long": [Ga[nm]["subsets"][sub][str(d)][k]["partial"]["est"] for nm in Ga for d in (16, 64, 256, 512)]
for tag, k, sub in (("gate", "Dgate", "long"), ("shuf", "Dshuf", "long"), ("free", "Dfree", "long"), ("gateAll", "Dgate", "all")):
    lo, hi = rng2(part(k, sub)); mac += f"\\newcommand{{\\{tag}PartLo}}{{{lo}}}\n\\newcommand{{\\{tag}PartHi}}{{{hi}}}\n"
mac += f"\\newcommand{{\\gateFreeWithinHi}}{{{r2(max(Ga[nm]['subsets']['long'][str(d)]['Dfree']['within']['est'] for nm in Ga for d in (16, 64, 256, 512)))}}}\n"
for tag, nm in (("a", "GDN-340M"), ("b", "GDN-360M")):
    f_ = Dd.GATE_TASK[nm]["flexible_age"]["P1g"]; w_ = Dd.GATE_TASK[nm]["within_cell_logp"]["P1g"]
    mac += f"\\newcommand{{\\gFlexD{tag}}}{{{sg(f_['delta'])}}}\n\\newcommand{{\\gFlexLo{tag}}}{{{sg(f_['ci'][0])}}}\n\\newcommand{{\\gFlexHi{tag}}}{{{sg(f_['ci'][1])}}}\n"
    mac += f"\\newcommand{{\\gCorrAge{tag.upper()}}}{{{Dd.GATE_TASK[nm]['corr_with_age']['P1g']:.2f}}}\n"
    mac += f"\\newcommand{{\\gWithinR{tag}}}{{{sg(w_['r'])}}}\n\\newcommand{{\\gWithinLo{tag}}}{{{sg(w_['ci'][0])}}}\n\\newcommand{{\\gWithinHi{tag}}}{{{sg(w_['ci'][1])}}}\n"
# ---- training-time lever numbers (exp_mqar/analyze_replication.py output + run files); macros for every number quoted in the text
import numpy as _np, json as _json
LR = _json.load(open(os.path.join(os.path.dirname(OUT), "..", "exp_mqar", "result_replication.json"))); f2 = lambda x: f"{x:.2f}"; f3 = lambda x: f"{x:.3f}"
def mc(name, val): return f"\\newcommand{{\\{name}}}{{{val}}}\n"
for tag, key in (("mq", "original"), ("rep", "replication"), ("pool", "pooled")):
    r = LR[key]
    if not r["n"]: continue
    mac += mc(tag + "Diff", sg(r["mean"])) + mc(tag + "CiLo", f2(r["ci"][0])) + mc(tag + "CiHi", f2(r["ci"][1])) + mc(tag + "Higher", r["higher"]) + mc(tag + "N", r["n"]) + mc(tag + "SF", f3(r["signflip_p"])) + mc(tag + "Wil", f3(r["wilcoxon_p"]))
    mac += mc(tag + "BaseMean", f3(r["base_mean"])) + mc(tag + "DecMean", f3(r["dec_mean"])) + mc(tag + "FailBase", r["base_failed"]) + mc(tag + "FailDec", r["dec_failed"]) + mc(tag + "Fisher", f3(r["fisher_p"]))
    if r["gain_nonfailed_baseline"] is not None: mac += mc(tag + "GainRest", sg(r["gain_nonfailed_baseline"])) + mc(tag + "NRest", r["n"] - r["base_failed"])
mac += mc("mqHolm", f3(LR["regularisers_seeds0_11"]["dec"]["holm_p"]))
for k, tag in (("match", "match"), ("rank", "rank")):
    r = LR["regularisers_seeds0_11"][k]; mac += mc(tag + "Diff", sg(r["mean"])) + mc(tag + "N", r["n"]) + mc(tag + "SF", f3(r["signflip_p"])) + mc(tag + "Holm", f3(r["holm_p"]))
_b = {s: Dd.mq_acc(Dd.MQ[("none", 0.0, s)][1]) for s in Dd.mq_seeds("none", 0.0)}; _d = {s: Dd.mq_acc(Dd.MQ[("dec", 1.0, s)][1]) for s in Dd.mq_seeds("dec", 1.0)}
mac += mc("mqFailGains", ", ".join(f"${_d[s] - _b[s]:+.2f}$" for s in sorted(_b, key=lambda s: _b[s]) if s in _d and _b[s] < 0.9 and s < 12))
_learned = [(k, r) for k, (f_, r) in Dd.MQ.items() if Dd.mq_acc(r) > 0.5]
mac += mc("mqNRuns", len(Dd.MQ)) + mc("mqNLearned", len(_learned)) + mc("mqMaxGap", f3(max(abs(_np.mean([r["grid"][f"{n_}_0"] for n_ in (8, 16, 24, 32)]) - _np.mean([r["grid"][f"{n_}_96"] for n_ in (8, 16, 24, 32)])) for k, r in _learned)))
_G = Dd.mq_geometry(); _ys, _k1, _v1, _mn, _v0 = [], [], [], [], []
for k, arr in _G.items():
    if k not in Dd.MQ: continue
    _ys.append(Dd.mq_acc(Dd.MQ[k][1])); _k1.append(_np.median(arr[1][:, 0])); _v1.append(_np.median(arr[1][:, 1])); _mn.append(min(_np.median(arr[0][:, 0]), _np.median(arr[1][:, 0]))); _v0.append(_np.median(arr[0][:, 1]))
_sp = lambda x, y: _np.corrcoef(Dd.rk(_np.array(x)), Dd.rk(_np.array(y)))[0, 1]
mac += mc("mqSpKey", f"{_sp(_k1, _ys):+.2f}") + mc("mqSpMin", f"{_sp(_mn, _ys):+.2f}") + mc("mqSpValZero", f"{_sp(_v0, _ys):+.2f}") + mc("mqSpVal", f"{_sp(_v1, _ys):+.2f}") + mc("mqSpValMin", f"{_sp([min(a, b) for a, b in zip(_v0, _v1)], _ys):+.2f}")
_g2 = LR["geometry_new_runs"]; mac += mc("mqNewSp", f"{_g2['spearman_acc_key_rank_layer1']:+.2f}") + mc("mqNewKeyBase", f"{_g2['key_rank_layer1_median']['baseline']:.1f}") + mc("mqNewKeyDec", f"{_g2['key_rank_layer1_median']['decorrelation']:.1f}") + mc("mqNewFailRank", f"{_g2['key_rank_layer1_failed_vs_learned']['failed']:.1f}") + mc("mqNewLearnRank", f"{_g2['key_rank_layer1_failed_vs_learned']['learned']:.1f}")
_gm = [Dd.mq_geom_pooled("match", 1.0, l) for l in (0, 1)]
mac += mc("gramRmsLo", f2(min(g["rms_psi"] for g in _gm))) + mc("gramRmsHi", f2(max(g["rms_psi"] for g in _gm))) + mc("gramRankLo", f"{min(min(g['kr'], g['vr']) for g in _gm):.0f}") + mc("gramRankHi", f"{max(max(g['kr'], g['vr']) for g in _gm):.0f}")
Hd = Dd.HELDOUT; cv = [Hd[k][n]["coverage"] for k in Hd for n in Hd[k]]; seqs = [Hd[k][n]["sequences"] for k in Hd for n in Hd[k]]
mac += f"\\newcommand{{\\heldCovLo}}{{{min(cv):.2f}}}\n\\newcommand{{\\heldCovHi}}{{{max(cv):.2f}}}\n\\newcommand{{\\heldSeqLo}}{{{min(seqs)}}}\n\\newcommand{{\\heldSeqHi}}{{{max(seqs)}}}\n"
Sc = Dd.SCALE
for k, tag in (("P1", "Gfree"), ("P1g", "Gaware"), ("P1r", "Gret")):
    for an, an_tag, key in (("pre_registered_style", "H", "delta"), ("flexible_age", "Flex", "delta"), ("within_cell_logp", "In", "r")):
        v = Sc[an][k]; mac += f"\\newcommand{{\\sc{tag}{an_tag}D}}{{{sg(v[key])}}}\n\\newcommand{{\\sc{tag}{an_tag}Lo}}{{{sg(v['ci'][0])}}}\n\\newcommand{{\\sc{tag}{an_tag}Hi}}{{{sg(v['ci'][1])}}}\n"
for k, tag in (("P1_shuffled", "Gfreesh"), ("P1g_shuffled", "Gawaresh")):
    v = Sc["within_cell_logp"][k]; mac += f"\\newcommand{{\\sc{tag}In}}{{{sg(v['r'])}}}\n"
for v_, w in ((4, "Four"), (8, "Eight"), (16, "Sixteen"), (24, "TwentyFour")): mac += f"\\newcommand{{\\scAcc{w}}}{{{Sc['accuracy_by_load'][str(v_)]['acc']:.2f}}}\n"
mac += f"\\newcommand{{\\scLong}}{{{Sc['n_long_heads']}}}\n\\newcommand{{\\scHeads}}{{{Sc['n_heads']}}}\n\\newcommand{{\\scAccAll}}{{{Sc['accuracy_overall']:.2f}}}\n"
open(os.path.join(OUT, "numbers.tex"), "w").write(mac)
print("frontier c", round(F["c"], 3), "| minmem slopes", [round(d["slope"], 3) for d in M], "| theory", INFO["fig_theory"])
