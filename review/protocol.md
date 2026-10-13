# Systematic literature review protocol (written before any search was run)

Reviewer: a single automated reviewer (a language model) under the author's direction; no second human screener (limitation).
Purpose: establish what is and is not new in `paper.md` (claims C1–C8 below) and locate every closely related result, in a documented, repeatable way.
This is a *systematic* review with stated limits. It is **not** claimed to be exhaustive: coverage is bounded by the sources below.

## Claims of the paper whose novelty is being checked
- C1 Exact interference law for a delta-rule / fast-weight memory (change in a stored association = key overlap × surprise), and bounds/capacity for it.
- C2 Gain-weighted error identity n·RMS² = tr(MB), key-only expectation of mean gain, calibrated V-free certificate.
- C3 Capacity floor / key-only impossibility for delta-rule states.
- C4 No memoryless learner can protect arbitrary old tasks while fitting arbitrary new tasks (2-D counterexample).
- C5 Minimal memory for protection: r(p−r)log2(1/δ') bits (Grassmannian metric entropy), achieved by quantised projection basis.
- C6 Exact protection–plasticity frontier for linear-in-features learners (zero-interference fit space min(n,p−r); price of protection = Lagrange multiplier).
- C7 Layerwise exact protection for frozen-recurrence SSMs; its empirical protection/plasticity trade-off; negative result for gradient-subspace projection in a small deep SSM.
- C8 Empirical comparison of fine-tuning, EWC, replay, projection, adapters on small SSMs (toy streams, Split-/Permuted-MNIST).

## Research questions (RQ) → topics
- RQ1 Continual learning / forgetting in SSMs, linear attention, fast-weight, RWKV, test-time-training models.
- RQ2 Theory of forgetting in continual linear models (linear regression, Kaczmarz, alternating projections, NTK, rates).
- RQ3 Projection / null-space / gradient-subspace / orthogonal methods and stability–plasticity theory.
- RQ4 Memory, information-theoretic, communication-complexity or hardness results for continual learning.
- RQ5 Delta-rule / associative-memory capacity and interference theory; sequence models as online regression.
- RQ6 Task-specific parameters: adapters, LoRA, parameter isolation for continual learning.
- RQ7 Covering numbers / metric entropy of Grassmannians (supporting result for C5).
- RQ8 Loss of plasticity (supporting context for C6–C7).

## Sources (databases) and why
1. arXiv API (all categories): primary source for ML theory and recent work.
2. OpenAlex API (title+abstract search): covers journals and proceedings (NeurIPS/ICML/ICLR/CVPR/COLT/ALT/...) and gives reference and citation links for snowballing.
3. Backward and forward citation chasing (OpenAlex referenced_works / cites) from pre-specified seed papers.
4. Previously run web searches (ad hoc, recorded in the conversation, ~35 queries) are treated as a third, unstructured source.
NOT searched (no access): Web of Science, Scopus, IEEE Xplore, ACM DL, Google Scholar, Semantic Scholar (rate-limited), books/theses. Non-English papers not screened. Dates: all years up to the search date.

## Search strings (fixed in advance; see `search.py` for the exact encoding)
S1  (Mamba | "state space" | "state-space" | SSM) in title  AND  ("continual learning" | "catastrophic forgetting" | lifelong | "class-incremental" | "incremental learning") in abstract.
S2  ("linear attention" | "fast weight" | "fast weights" | "delta rule" | DeltaNet | RWKV | "test-time training" | "linear transformer") AND ("catastrophic forgetting" | "continual learning" | "lifelong").
S3  ("continual learning" | "catastrophic forgetting" | "sequential learning") AND ("linear regression" | "linear models" | "linear model") AND (Kaczmarz | "alternating projection" | theory | "upper bound" | "lower bound" | overparameterized | "rates").
S4  "continual" AND Kaczmarz;  "forgetting" AND "neural tangent kernel" AND continual.
S5  ("gradient projection" | "orthogonal gradient" | "null space" | "null-space" | "orthogonal projection" | "subspace") AND ("continual learning" | "catastrophic forgetting").
S6  ("stability-plasticity" | "stability plasticity") AND (theory | theoretical | bound | trade-off | tradeoff | "Pareto").
S7  "continual learning" AND (memory | "memory requirement") AND ("lower bound" | "communication complexity" | "information-theoretic" | "NP-hard" | "perfect memory" | impossibility).
S8  ("delta rule" | "associative memory" | "fast weight") AND ("linear attention" | "linear transformer" | recurrent | "state space") AND (capacity | interference | orthogonal | "key").
S9  ("online least squares" | "online regression" | "test-time regression") AND ("sequence model" | "state space" | "linear attention" | "recurrent").
S10 (LoRA | "low-rank adaptation" | adapter | adapters | "parameter isolation" | "task-specific") AND "continual learning" AND (orthogonal | forgetting | interference | "zero forgetting").
S11 (Grassmann | Grassmannian) AND ("covering number" | "covering numbers" | "metric entropy" | "packing").
S12 "loss of plasticity" | "plasticity loss" AND (continual | "deep learning").
S13 ("exemplar-free" | "rehearsal-free" | "replay-free" | "memory-free") AND ("continual learning") AND (theory | guarantee | bound | provable).
S14 "continual learning" AND ("provable" | "provably" | "guarantee" | "guarantees") AND forgetting.

## Screening (fixed in advance)
Stage 0: de-duplicate (DOI, arXiv id, normalised title).
Stage 1 (automatic, transparent keyword rule): keep a record only if its title+abstract contains a *domain term* (continual, lifelong, catastrophic forgetting, incremental, sequential learning, forgetting; or for RQ5/RQ7: associative memory, delta rule, fast weight, Grassmann...) AND at least one *topic term* of the RQ it was retrieved for (see `screen.py`). Records failing are logged as "auto-excluded" with the failing rule; counts reported.
Stage 2 (manual, title): the reviewer reads titles and marks keep/drop with a reason code from E1–E6.
Stage 3 (manual, abstract): the reviewer reads abstracts of kept titles and applies the eligibility criteria.
Stage 4: for items judged "same result" or "closely related" to C1–C8, the reviewer reads the available full-text excerpt via the arXiv abstract page (or states that only the abstract was read).

## Eligibility criteria
Include (any of):
 I1 theory (theorem, bound, exact characterisation) of forgetting/interference/stability–plasticity in sequential or continual learning of linear or linearised models;
 I2 continual learning/forgetting studied specifically for SSM, linear-attention, fast-weight, recurrent-state or TTT models;
 I3 memory/information-theoretic/complexity results for continual learning, or capacity/interference theory for delta-rule/associative memories;
 I4 projection/null-space/subspace/adapter methods *with* a formal guarantee, a stability–plasticity analysis, or a compression of the stored object;
 I5 covering numbers / metric entropy of Grassmannians or closely related (Stiefel) manifolds;
 I6 analyses of loss of plasticity.
Context (recorded, not analysed in depth): surveys and benchmarks of continual learning; empirical projection/adapter methods without guarantee (counted by family).
Exclude: E1 application-only (domain-specific, no method or theory relevant to C1–C8); E2 empirical continual-learning method with no theory and no SSM/linear-attention focus, no projection/adapter family relevance; E3 duplicate/version; E4 not about forgetting/memory (e.g. long-context SSM without continual aspect); E5 not accessible (no abstract); E6 non-English.

## Coding of included records
For each included record: year, venue/arXiv id, eligibility reason(s), RQ, which claim(s) C1–C8 it bears on, and an overlap level: SAME (states essentially the same result), CLOSE (a special case, a stronger/weaker form, or the same technique for a neighbouring setting), CONTEXT (background). Overlap SAME/CLOSE items are read beyond the abstract where accessible; otherwise the coding is marked "abstract only".

## Snowballing
Seeds (fixed now): Evron et al. 2022; Evron et al. 2026; Lin et al. 2023; Chen–Papadimitriou–Peng 2022; Knoblauch et al. 2020; Saha et al. 2021 (GPM); Wang et al. 2021 (Adam-NSCL); Cheng et al. 2024 (Mamba-CL); Lee et al. 2025 (Inf-SSM); Liu et al. 2025 (Longhorn); Wang–Shi–Fox 2025; Schlag et al. 2021; Szarek 1998; Nair et al. 2026 (PMF-CL).
Backward: all referenced works of the seeds that are in OpenAlex. Forward: all works citing a seed. Candidates are screened with the same Stage 1–4 rules.

## Stopping rule and reporting
A search is "complete" when every query has been run to its natural end (all results fetched) or truncation is explicitly logged. Final report: PRISMA-style counts per stage and per source, tables of included works by RQ and overlap level, a list of limitations, and the effect on the paper's novelty claims.

## Amendment 1 (after recall check; made after seeing results, so flagged as post-hoc)
A recall check against 33 papers previously found by ad hoc search showed 27/33 (82%) were retrieved by S1–S14. The 6 missed were five delta-rule / test-time-training architecture papers (Titans, Longhorn, DeltaNet, Gated DeltaNet, TTT) and Jeong et al. 2023 (cyclic forgetting). Because delta-rule memory theory is where overlap with claims C1–C3 could hide, three supplementary queries were added:
S15  ("delta rule" | DeltaNet | "gated delta" | "fast weight" | "fast weights" | "test-time training" | "test-time regression" | "linear attention") AND (theory | theoretical | convergence | capacity | interference | forgetting | retention | Kaczmarz | "least squares" | regret | bound) in title/abstract.
S16  (cyclic | "task ordering" | "random ordering" | "random orderings" | "task order") AND ("continual learning" | "catastrophic forgetting" | forgetting).
S17  ("online learner" | "online learners" | "online optimization" | "online regression") AND ("state space" | "sequence model" | "sequence models" | "recurrent" | "associative memory") in title/abstract.
Screening of S15–S17 records uses supplementary Stage-1 rule R9: title+abstract matches (delta rule|deltanet|gated delta|fast weight|test-time (training|regression)|online learner|linear attention|state space) AND (theor|converge|capacity|interference|forget|retention|kaczmarz|least[- ]squares|regret|bound|stability|plasticity) — then manual title and abstract screening as before.

## Amendment 2 (after the end-to-end recall check; post-hoc)
End-to-end recall on the 33-paper reference set was 29/33 at retrieval but only 23/33 after the Stage-1 keyword rule and 21/33 after title screening (two of the losses were deliberate: empirical adapter papers). The Stage-1 rule is therefore too narrow (it also discards records with no abstract unless the title matches: 189 of 214 no-abstract records in the main pool). Amendment: all records auto-excluded by Stage 1 in the main set (S1-S14; n = 962) are screened by title manually (Stage 2) and carried forward through Stages 3-4 under the same criteria. The auto-excluded supplementary records (S15-S17; n = 668) are dominated by irrelevant OpenAlex matches and are screened only after a documented junk filter (publication year >= 2015 and presence of an ML term in title+abstract).
