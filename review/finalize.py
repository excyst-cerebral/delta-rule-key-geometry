"""Code the records surviving title screening, compute PRISMA counts, and write review/coded_records.json and review/literature_review.md."""
import json, glob, re, collections, os
HERE = os.path.dirname(os.path.abspath(__file__)); L = lambda f: json.load(open(f"{HERE}/{f}"))
K = L("stage1_kept.json"); dec = L("stage2_decisions.json"); queue = L("stage3_queue.json"); S1 = L("stage_counts.json"); supp = {d["sidx"]: d for d in L("supp_stage1.json")}; snow = L("snowball_stage1.json"); snowraw = L("snowball_raw.json"); recall = L("recall_check.json")
raw = {os.path.basename(f)[:-5]: len(json.load(open(f))) for f in glob.glob(f"{HERE}/raw/*.json")}
ax = lambda qs: sum(raw[f"arxiv_{q}"] for q in qs); oa = lambda qs: sum(raw[f"openalex_{q}"] for q in qs)
MAIN_Q = [f"S{i}" for i in (1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14)] + ["S4a", "S4b"]; SUPP_Q = ["S15", "S16", "S17"]
EX = {1077: "E2", 1112: "E2", 1431: "E2", 1324: "E1", 1588: "E2", 1411: "E1", 815: "E4", 76: "E4", 80: "E4", 36: "E2", 107: "E2", 1569: "E4", 1584: "E4", 1421: "E4", 1117: "E4", 1393: "E2", 1086: "E5", 1087: "E5", 1302: "E5", 1303: "E5", 1391: "E5", 1491: "E5", 1688: "E4"}
C = lambda claims, level, note: dict(claims=claims, level=level, note=note)
CLOSE = {
 612: C("C5,C6,C7", "SAME (construction)", "ORFit Lemma 3.1: in a linear model, the step along the gradient projected orthogonal to previous gradients, with step size set to interpolate the new example, leaves all previous predictions unchanged. This is the zero-interference exact-fit update of Theorem 15(b) / Theorem 16(a). It also truncates the stored basis by PCA and calls this minimax-optimal for worst-case forgetting. (Full text read, Sections 2-3.)"),
 693: C("C5", "CLOSE (technique)", "Dagan, Kur, Shamir: Omega(d^2) memory for streaming linear regression via a Grassmannian packing of exp(Omega(d^2)) subspaces that the memory must tell apart; the same counting argument as Theorem 15(a), for a different problem (no forgetting, no precision dependence). (Full text read, pp.1-3.)"),
 1243: C("C5", "CLOSE (technique)", "Duplicate record of the preceding paper."),
 1497: C("C4,C5", "CLOSE (non-peer-reviewed)", "Zenodo v1 release (Nowicki): lower bound r(d-r)+r+O(1) bits for exact lifelong retention of consequences, matching constructive learner, over the binary field F_2, counting rank-r subspaces (Gaussian binomial). No real-valued covering or precision result; record page read."),
 1012: C("C4,C5", "CLOSE", "Chen, Papadimitriou, Peng: memory linear in the number of tasks for general continual learners."),
 1014: C("C4", "CLOSE", "Knoblauch et al.: optimal continual learning needs perfect memory and is NP-hard."),
 845: C("C5,C6", "CLOSE", "Memory-statistics trade-off in two-task continual linear regression: memory counted as the number of vectors defining a structural regularisation; upper and lower bounds on excess risk (abstract read)."),
 722: C("C5", "CLOSE (covering part)", "Dai, Liu, Rider: quantisation / rate-distortion bounds and exact ball volumes on Grassmann manifolds (MIMO feedback); the covering-number step of Theorem 15 is standard in this literature."),
 719: C("C5", "CLOSE (covering part)", "Version of the preceding result for arbitrary dimensions."), 1198: C("C5", "CLOSE (covering part)", "Conference version of the preceding results."),
 1045: C("C1", "CLOSE", "Attention as Conditioning: DeltaNet's update is term-for-term Rescorla-Wagner error correction, with an exact closed form for blocking (cue competition), verified to 1e-7; this is the delta-rule interference law in animal-learning form."),
 1053: C("C1,C2", "CLOSE (framing)", "Wang, Shi, Fox: test-time regression as a unifying framework for sequence models with associative memory."),
 1034: C("C1", "CLOSE (architecture)", "Kaczmarz Linear Attention: key-norm-normalised step size for Gated DeltaNet derived from the online-regression objective (Zou, Ren, Liu)."),
 1033: C("C1", "CLOSE (architecture)", "OSDN: provable online preconditioning of the delta rule, convergence and residual-contraction bounds."),
 1370: C("C1", "CLOSE (architecture)", "Fast Weight Attention for Continual Learning: normalised (NLMS-type) first-order updates for fast-weight memories (Falcon family)."),
 1344: C("C1,C7", "CLOSE (non-peer-reviewed)", "Zenodo: decay rate as one knob for context length and stability/plasticity; delta form is online Widrow-Hoff; freezing slow weights confines forgetting to the bounded memory."),
 826: C("C1,C7", "CLOSE", "Palimpsa: in-context learning in gated linear attention as continual learning with a stability-plasticity dilemma; fixed capacity and interference."),
 1021: C("C3", "CLOSE (empirical)", "Boesch, Wee: at a fixed state budget, interference under sparse supervision rather than capacity limits associative recall in fixed-state recurrences."),
 1376: C("C3,C4", "CLOSE (non-peer-reviewed)", "Zenodo: impossibility of safe incremental concept addition in linear attention via the key Gram matrix (transfer of a Hopfield result)."),
 1024: C("C1", "CLOSE (architecture)", "HOLA: keeps the delta-rule state and caches the tokens with the largest beta*||e||, i.e. the committed prediction residual (our 'surprise')."),
 806: C("C5", "CLOSE", "SketchOGD: compresses the stored OGD gradient subspace by matrix sketching (Min, Wright, Bernstein, Azizan)."),
 814: C("C7", "CLOSE", "Generalisation guarantees for OGD in the NTK (lazy) regime."), 811: C("C7", "CLOSE", "Heckel: EWC provably forgets with a poor diagonal Hessian approximation; sketched-Jacobian regularisation is provable."),
 558: C("C7,C8", "CLOSE", "Lanzillotta et al.: guarantees against forgetting for parameter-isolation methods."), 956: C("Weights-law", "CLOSE", "NTK overlap matrix and forgetting (Doan et al.)."),
 1126: C("C6,C7", "CLOSE", "Zhao et al.: stability/plasticity feature-space decoupling in gradient projection."), 1412: C("C6", "CLOSE", "AdaBOP: strict orthogonal projection sacrifices plasticity; adaptive trade-off."),
 916: C("C6,C8", "CLOSE", "Mahdaviyeh et al.: in continual linear regression with tasks as subspaces, replay can provably increase forgetting."),
 853: C("Weights-law", "CLOSE", "Evron et al. 2022."), 844: C("Weights-law", "CLOSE", "Evron et al. 2026."), 799: C("Weights-law", "CLOSE", "Levinstein et al. 2025."), 848: C("Weights-law", "CLOSE", "Ding et al. 2024."), 851: C("Weights-law", "CLOSE", "Lin et al. 2023."), 791: C("Weights-law", "CLOSE", "Exact operator identity for forgetting under i.i.d. task sampling in linear regression (2026)."),
 3: C("C7,C8", "CLOSE", "Inf-SSM."), 7: C("C7,C8", "CLOSE", "Mamba-CL."), 6: C("C7,C8", "CLOSE", "MambaCL (meta-learned)."), 10: C("C7,C8", "CLOSE", "Regularisation-based continual learning in deep state-space models (system identification)."),
}
SUPPORT = {731, 732, 1167, 1169, 1176, 723, 726, 1168, 1197, 1202, 1203, 705, 1209, 1211, 714}
SCLOSE = {581: C("C1", "CLOSE (non-peer-reviewed)", "Zenodo (Escobar): closed-form residual laws for selective forgetting in matrix-valued associative memory; lineage theorem relating a complex re-read erase to the delta rule at beta=1; LMS stability condition."),
          424: C("C1,C3", "CLOSE (empirical)", "How Linear Attention Remembers (Lee et al.): decomposition of writes/reads in pretrained GLA and GDN; cross-fact coupling and degradation with memory load."),
          569: C("C2,C3", "CLOSE", "On State Reduction in Linear Attention (Nazari, Rusch): analysis of how keys and values shape the rank of the recurrent state; low-rank states in trained models."),
          694: C("C2,C3", "CLOSE (empirical)", "State Rank Dynamics in Linear Attention LLMs: head-level rank stratification of the recurrent state."),
          380: C("Weights-law", "CLOSE", "Theoretical analysis of fine-tuning forgetting of in-context learning in linear attention models."),
          556: C("C3", "CLOSE (prior art)", "Prados, Kak 1989: delta-rule storage capacity exceeds N patterns in an N-neuron network; capacity of the delta rule studied analytically and experimentally."),
          726: C("C3", "CLOSE (prior art)", "1989 dissertation on the information capacity of networks trained with a modified delta rule (about 50% separability at m = 2N).")}
SK = [134, 136, 162, 218, 219, 273, 303, 380, 406, 424, 427, 498, 556, 726, 496, 569, 581, 694, 718, 483, 404, 307, 261, 299, 617, 319, 374, 422, 759, 678, 26, 30, 78, 95, 18, 100, 106, 385, 630, 742]
SN = [1, 7, 11, 18]
sorted_snow = sorted(snow, key=lambda w: w["title"].lower())
rows = []
def flag(venue, doi): return "non-peer-reviewed (Zenodo)" if "Zenodo" in (venue or "") or "zenodo" in (doi or "") else ("preprint (arXiv)" if (venue or "").startswith("arXiv") else "published (venue as recorded)")
for i in queue:
    r = K[i]; code = EX.get(i)
    if code: status, level, claims, note = "EXCLUDED", None, "", f"abstract-level exclusion {code}"
    elif i in CLOSE: c = CLOSE[i]; status, level, claims, note = "INCLUDED", c["level"], c["claims"], c["note"]
    elif i in SUPPORT: status, level, claims, note = "INCLUDED", "SUPPORT", "C5", "covering / packing / volume result for Grassmannians or subspace codes"
    else: status, level, claims, note = "INCLUDED", "CONTEXT", "", ""
    rows.append(dict(set="main", idx=i, title=r["title"], year=r["year"], venue=r.get("venue"), arxiv=r.get("arxiv"), doi=r.get("doi"), rules=r["rules"], status=status, level=level, claims=claims, note=note, evidence=("full text (partial)" if i in (612, 693) else "abstract"), peer=flag(r.get("venue"), r.get("doi")), found_by=r["queries"]))
for i in SK:
    r = supp[i]; c = SCLOSE.get(i)
    rows.append(dict(set="supplementary", idx=i, title=r["title"], year=r["year"], venue=r.get("venue"), arxiv=r.get("arxiv"), doi=r.get("doi"), rules=r["rules"], status="INCLUDED", level=(c["level"] if c else "CONTEXT"), claims=(c["claims"] if c else ""), note=(c["note"] if c else ""), evidence="abstract", peer=flag(r.get("venue"), r.get("doi")), found_by=r["queries"]))
for i in SN:
    w = sorted_snow[i]; rows.append(dict(set="snowball", idx=i, title=w["title"], year=w["year"], venue=w.get("venue"), arxiv=None, doi=w.get("doi"), rules=w["rules"], status="INCLUDED", level="CONTEXT", claims="", note="", evidence="abstract", peer=flag(w.get("venue"), w.get("doi")), found_by=w["kinds"]))

REJ = {r["ridx"]: r for r in json.load(open(f"{HERE}/stage1_rejected.json"))}
RCLOSE = {890: C("C7", "CLOSE (non-peer-reviewed)", "Zenodo: measured damage of sequential weight edits; a frozen protected-subspace fence failed pre-registered bars; the small in-span component of an update is ~4000x more damaging per unit energy (cf. our negative result for gradient-subspace projection)."),
          789: C("C5", "CLOSE", "Continual Pick-to-Learn: sample-compression theory gives computable generalisation bounds after each task (retains representative samples per task)."),
          256: C("C7", "CLOSE", "Continual learning with guarantees: interval constraints on the parameter space bound forgetting (Hyperrectangle Training)."),
          566: C("C5", "CLOSE", "Sketched structural regularisation: compressing the importance matrix (EWC) by sketching, with guarantees."),
          303: C("C2", "CLOSE", "Friedman, Meir: data-dependent upper bounds on forgetting for exemplar-free methods (certificate-like objects)."),
          689: C("C7", "CLOSE", "Orthogonal Gradient Descent (Farajtabar et al. 2020)."),
          708: C("C5,C6", "CLOSE", "PMF-CL (Nair, Eryilmaz, Liu): states that what information must be stored to minimise forgetting is not understood; derives Pareto-minimal-forgetting learners for linear regression with O(d^2) static memory and an exact forgetting expression; abstract and snippets read, not the full text."),
          883: C("C4,C5", "CLOSE", "Peng, Giampouras, Vidal: the ideal continual learner avoids forgetting by construction and unifies memory-, regularisation- and expansion-based methods; stores information about past tasks."),
          572: C("C3", "CLOSE (prior art)", "1989: linear and logarithmic storage capacities in associative nets, spectral storage strategy."),
          6: C("C2", "CLOSE (framing)", "Conformal-prediction-based monitoring of forgetting (CPCF).")}
RSUP = {198}
RCTX = [726, 686, 725, 523, 537, 29, 40, 583, 186, 437, 790, 845, 961, 764, 56, 663, 660, 877, 857, 247, 236, 255, 238]
for i in sorted(set(RCLOSE) | RSUP | set(RCTX)):
    r = REJ[i]; c = RCLOSE.get(i)
    rows.append(dict(set="amendment2", idx=i, title=r["title"], year=r["year"], venue=r.get("venue"), arxiv=r.get("arxiv"), doi=r.get("doi"), rules=[], status="INCLUDED", level=(c["level"] if c else ("SUPPORT" if i in RSUP else "CONTEXT")), claims=(c["claims"] if c else ("C5" if i in RSUP else "")), note=(c["note"] if c else ""), evidence="abstract", peer=flag(r.get("venue"), r.get("doi")), found_by=r.get("queries", [])))

json.dump(rows, open(f"{HERE}/coded_records.json", "w"), indent=1)
inc = [r for r in rows if r["status"] == "INCLUDED"]; cnt = collections.Counter(r["level"] for r in inc); exc = [r for r in rows if r["status"] == "EXCLUDED"]
N = dict(
 raw_main=sum(raw[f"{s}_{q}"] for q in MAIN_Q for s in ("arxiv", "openalex")), raw_main_arxiv=ax(MAIN_Q), raw_main_oa=oa(MAIN_Q), raw_supp=sum(raw[f"{s}_{q}"] for q in SUPP_Q for s in ("arxiv", "openalex")), raw_supp_arxiv=ax(SUPP_Q), raw_supp_oa=oa(SUPP_Q),
 snow_fwd=len(snowraw["forward"]), snow_bwd=len(snowraw["backward"]), main_unique=S1["unique_after_dedupe"], main_stage1=S1["stage1_kept"], main_auto_excluded=S1["auto_excluded"], main_tierA=S1["tierA"], main_tierB=S1["tierB"],
 supp_new_unique=1484, supp_stage1=len(supp), supp_auto_excluded=1484 - len(supp), snow_new=266, snow_stage1=len(snow),
 main_title_kept=len(queue), main_title_dropped=S1["stage1_kept"] - len(queue), supp_title_kept=len(SK), supp_title_dropped=len(supp) - len(SK), snow_title_kept=len(SN), snow_title_dropped=len(snow) - len(SN),
 abstract_excluded=len(exc), exc_codes=dict(collections.Counter(r["note"] for r in exc)), included=len(inc), levels=dict(cnt), unflagged_nonpeer=sum(1 for r in inc if r["peer"].startswith("non-peer")), amend2_screened=len(REJ), amend2_kept=len(set(RCLOSE) | RSUP | set(RCTX)), by_source=S1["by_source"], recall=recall)
json.dump(N, open(f"{HERE}/prisma_counts.json", "w"), indent=1); print(json.dumps(N, indent=1))
