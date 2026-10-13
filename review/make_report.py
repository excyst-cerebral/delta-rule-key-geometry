"""Generate review/literature_review.md from the coded records and counts."""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__)); L = lambda f: json.load(open(f"{HERE}/{f}"))
N = L("prisma_counts.json"); R = L("coded_records.json"); A = L("authors.json"); rec1 = L("recall_check.json"); rec2 = L("recall_end_to_end.json"); rec3 = L("recall_end_to_end_final.json")
auth = lambda r: next((", ".join(v["authors"][:3]) + (" et al." if len(v["authors"]) > 3 else "") for t, v in A.items() if t[:40].lower() in r["title"].lower()), "")
tbl = lambda rows: "\n".join(f"| {r['title'][:92]} | {auth(r)} | {r['year']} | {r['peer']} | {r['claims']} | {r['level']} | {r['note'][:330]} |" for r in rows)
inc = [r for r in R if r["status"] == "INCLUDED"]; close = sorted([r for r in inc if r["level"] not in ("CONTEXT", "SUPPORT")], key=lambda r: (r["claims"], r["title"])); support = [r for r in inc if r["level"] == "SUPPORT"]; ctx = [r for r in inc if r["level"] == "CONTEXT"]
zen = [r for r in inc if r["peer"].startswith("non-peer")]
md = f"""# Systematic literature review: what in `paper.md` is and is not new

*Protocol written before searching: [`protocol.md`](protocol.md), with two post-hoc amendments, flagged as such. Single automated reviewer; no second human screener. A documented, repeatable search of two databases plus citation chasing, **not** an exhaustive review (see Limitations; the end-to-end recall on a 33-paper reference set was {rec3['included']}/{rec3['n']} = {rec3['included']/rec3['n']:.0%}).*

## 1. Method in brief
Two structured sources: the **arXiv API** and the **OpenAlex API** (title+abstract search). Fourteen pre-specified search strings (S1-S14) covered continual learning in SSMs and linear-attention/fast-weight models, continual linear-regression theory, projection/null-space methods, stability-plasticity theory, memory lower bounds, delta-rule/associative-memory capacity, task-specific adapters, Grassmannian covering numbers and loss of plasticity. **Amendment 1** (after a recall check) added three strings (S15-S17) aimed at delta-rule theory and task-ordering results. Backward and forward citation chasing used 13 pre-specified seed papers and OpenAlex only. Records were de-duplicated; screened by a fixed keyword rule (Stage 1), by title (Stage 2) and by abstract (Stage 3); the closest records were read beyond the abstract where the full text could be retrieved (Stage 4). **Amendment 2** (after an end-to-end recall check showed the Stage-1 keyword rule dropped known-relevant papers) added a manual title screen of all {N['amend2_screened']} records that Stage 1 had auto-excluded in the main set.

## 2. PRISMA-style flow

| step | S1-S14 | S15-S17 (amend. 1) | citation chasing | Stage-1 rejects re-screened (amend. 2) |
|---|---|---|---|---|
| records retrieved | {N['raw_main']:,} (arXiv {N['raw_main_arxiv']:,}; OpenAlex {N['raw_main_oa']:,}) | {N['raw_supp']:,} (arXiv {N['raw_supp_arxiv']:,}; OpenAlex {N['raw_supp_oa']:,}) | {N['snow_fwd']} forward + {N['snow_bwd']} backward | - |
| unique after de-duplication (new relative to the previous set) | {N['main_unique']:,} | {N['supp_new_unique']:,} | {N['snow_new']} | - |
| passed automatic keyword screen (Stage 1) | {N['main_stage1']:,} (tier A {N['main_tierA']}, tier B {N['main_tierB']}) | {N['supp_stage1']} | {N['snow_stage1']} | {N['amend2_screened']} screened manually |
| kept after title screening (Stage 2) | {N['main_title_kept']} | {N['supp_title_kept']} | {N['snow_title_kept']} | {N['amend2_kept']} |
| excluded at abstract (Stage 3) | {N['abstract_excluded']} (E1 {N['exc_codes'].get('abstract-level exclusion E1',0)}, E2 {N['exc_codes'].get('abstract-level exclusion E2',0)}, E4 {N['exc_codes'].get('abstract-level exclusion E4',0)}, E5 no abstract {N['exc_codes'].get('abstract-level exclusion E5',0)}) | 0 | 0 | 0 |
| **included** | {N['main_title_kept']-N['abstract_excluded']} | {N['supp_title_kept']} | {N['snow_title_kept']} | {N['amend2_kept']} |

**Total included: {len(inc)}**: {len(ctx)} context, {len(support)} supporting (Grassmannian covering/packing), {len(close)} classified SAME or CLOSE to a claim of the paper. {len(zen)} included records are non-peer-reviewed Zenodo releases (flagged below). In the main set, {N['by_source']['arxiv']} records were found by arXiv only, {N['by_source']['openalex']} by OpenAlex only and {N['by_source']['arxiv+openalex']} by both, so using both sources mattered.

**Recall checks** (against 33 papers previously found by ad hoc search):
* retrieval, S1-S14: {rec1['found']}/{rec1['n']} ({rec1['found']/rec1['n']:.0%}); retrieval, S1-S17: {rec2['retrieved']}/{rec2['n']} ({rec2['retrieved']/rec2['n']:.0%});
* after the Stage-1 keyword rule: {rec2['passed_stage1']}/{rec2['n']} ({rec2['passed_stage1']/rec2['n']:.0%}); after title screening, before amendment 2: {rec2['included']}/{rec2['n']};
* **end to end after amendment 2: {rec3['included']}/{rec3['n']} ({rec3['included']/rec3['n']:.0%})**. Not included: Jeong et al. 2023 (OpenAlex holds no abstract and the title does not match the rules); InfLoRA and O-LoRA (retrieved and kept past Stage 1, then classed as empirical adapter-family context at title screening, a deliberate choice); and five delta-rule / test-time-training architecture papers (Titans, Longhorn, DeltaNet, Gated DeltaNet, TTT), which are framed as sequence modelling, not continual learning, and which the search strings reach only partly (S17 retrieved Longhorn; the keyword rules then discarded it).
In other words, recall for continual-learning theory is good; for delta-rule/fast-weight *architecture* theory it is lower and unquantified, which matters for claim C1.

## 3. Records classified SAME or CLOSE to a claim of the paper
Claims: C1 delta-rule interference law; C2 gain-weighted error identity, key-only expectation and calibrated certificate; C3 capacity floor / key-only impossibility; C4 no memoryless protector; C5 minimal memory (Grassmannian); C6 protection-plasticity frontier; C7 layerwise exactness and SSM weight-space results; C8 empirical comparisons; "Weights-law" the weight-space interference identity. Evidence is the abstract unless the note says the full text was read.

| record | authors (OpenAlex) | year | status | claim | overlap | note |
|---|---|---|---|---|---|---|
{tbl(close)}

Supporting results for C5 (Grassmannian covering, packing, volumes; {len(support)} records): """ + "; ".join(f"{r['title'][:70]} ({r['year']})" for r in support) + f""".

## 4. Assessment by claim (after the review)
* **C1 (interference law).** Not new as a mathematical fact. The delta rule's error-correcting update is Widrow-Hoff/LMS and Rescorla-Wagner (a closed form for cue blocking already exists, record 'Attention as Conditioning'); several 2025-26 papers derive delta-rule steps from online regression or Kaczmarz projection (Kaczmarz Linear Attention, OSDN, Fast Weight Attention, test-time regression), one uses the committed residual beta*||e|| as its write criterion (HOLA), and a non-peer-reviewed Zenodo release gives closed-form forgetting laws for matrix-valued associative memory. Lemma 1 and Proposition 2 should be presented as restatements. Caveat: recall for this literature is the weakest part of the search.
* **C2 (gain-weighted identity, key-only expectation, calibrated certificate).** No record states these at abstract level. The nearest are rank analyses of recurrent states (On State Reduction in Linear Attention; State Rank Dynamics), data-dependent forgetting bounds (Friedman and Meir), a conformal-prediction monitor of forgetting, and empirical studies of recall load in trained GLA/GDN (How Linear Attention Remembers). None gives an identity of the form n*RMS^2 = tr(MB) or a V-free certificate calibrated per model family. Retained as new, with the caveat that abstract-level screening cannot rule out the identity inside a longer proof.
* **C3 (capacity).** Classical: delta-rule storage capacity was studied in 1989 (Prados and Kak; a 1989 dissertation), Schlag et al. (2021) give the key-orthogonality argument, and several 2026 papers study interference versus capacity in fixed-state recurrences. Theorem 7 is a restatement with a budgeted, attained floor.
* **C4 (no memoryless protector).** Weak: close to the memory lower bound of Chen et al., to Knoblauch et al., to the 'ideal continual learner' (which stores information about past tasks) and to non-peer-reviewed impossibility claims. The two-line construction is elementary.
* **C5 (minimal memory).** The *construction* of Theorem 15(b) (project the new example's feature off the stored old subspace and set the step to interpolate) **is the ORFit/OGD update in the linear case** (Min, Cho, Azizan, Lemma 3.1; full text read), which also truncates the stored basis by PCA and calls it minimax-optimal for worst-case forgetting. The *counting argument* of Theorem 15(a) (a Grassmannian packing forces about r(p-r)log(1/eps) bits) is the technique of Dagan, Kur and Shamir (2019) for streaming regression (full text read), and the covering numbers and quantisation bounds are classical (Szarek, Pajor, Dai-Liu-Rider). A finite-field analogue with a matching construction appeared in a non-peer-reviewed 2026 release; PMF-CL (2026 preprint) addresses 'what information needs to be stored to minimise forgetting' for linear regression with O(d^2) memory; Sketched OGD and sketched structural regularisation compress the stored object. What we did not find, at abstract level: the specific statement 'protecting an unknown subspace to interference delta' needs r(p-r)log2(1/delta) bits up to constants'. The contribution is the packaging for continual learning with an interference criterion, not the technique.
* **C6 (frontier).** Elementary convex analysis; the same trade-off underlies the stability/plasticity decoupling literature, adaptive orthogonal projection and PMF-CL's Pareto analysis. A restatement with explicit closed forms, not a new result.
* **C7 (layerwise exactness; SSM results).** The layerwise argument is GPM/Adam-NSCL/Mamba-CL with a frozen recurrence. The negative result for gradient-subspace projection in a small deep SSM and the fixed-pole control have no counterpart among the retrieved abstracts; a non-peer-reviewed Zenodo study of sequential weight edits reports a related failure of a protected-subspace fence.
* **C8 (empirical comparisons).** Standard-benchmark comparisons are routine; no retrieved abstract reports the same learners on a diagonal SSM, so the numbers are not duplicated, but they are not novel in kind.

## 5. Limitations of this review
1. **Sources.** arXiv and OpenAlex only. Not searched: Web of Science, Scopus, IEEE Xplore, ACM DL, Google Scholar, Semantic Scholar (rate-limited without a key), books, theses, non-English venues.
2. **Citation chasing is weak.** OpenAlex stores many arXiv papers without reference lists or with near-zero citation counts (the Evron et al. 2022 record showed 4 citations), so chasing from the seeds returned only {N['snow_fwd']}+{N['snow_bwd']} records and found no new close work. This is a coverage limit, not evidence of completeness.
3. **Search-term and abstract sensitivity.** Papers whose abstracts do not use our terms, or that have no abstract in the database, are missed or discarded (Jeong et al.). The keyword rule of Stage 1 was too narrow and was repaired by amendment 2 for the main set only; the supplementary set (S15-S17) was screened only through the keyword rules.
4. **Screening depth.** One automated reviewer. Title screening of about {N['main_stage1']+N['supp_stage1']+N['snow_stage1']+N['amend2_screened']:,} records; abstract screening of {N['main_title_kept']+N['supp_title_kept']+N['snow_title_kept']+N['amend2_kept']} records with abstracts read to roughly 150-1,000 characters depending on relevance; only two papers read in full for the relevant sections and one record page. A closer match to a claim could hide in an unread proof.
5. **Post-hoc amendments.** Both amendments were made after seeing results; they widen the search and cannot be treated as pre-registered.
6. **OpenAlex noise.** The supplementary set contained many irrelevant records (retracted supplement advertisements, other fields). They were screened out; they inflate the raw counts, not the included set.
7. **Non-peer-reviewed records.** {len(zen)} included records are Zenodo releases or comparable; they are flagged and are not evidence of established results.
8. **Date.** Searches were run once; later work is not covered.

## 6. Files
`protocol.md`, `search.py`, `screen.py`, `snowball.py`, `finalize.py`, `make_report.py`, `raw/` (raw results per query and source), `search_log.json`, `stage*_*.json`, `stage2_decisions.json`, `coded_records.json` (every included/excluded record with level, claims, evidence and peer-review flag), `prisma_counts.json`, `recall_check.json`, `recall_end_to_end*.json`.
"""
open(f"{HERE}/literature_review.md", "w").write(md); print(len(md.split()), "words;", len(close), "close/same;", len(inc), "included")
