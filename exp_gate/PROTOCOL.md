# Gate-aware key-only analysis in the actual recurrence (post hoc; protocol fixed before the scripts are run)

## Question
The key-only quantity psi of the paper is computed for an idealised memory (alpha = beta = 1, one pass). The trained models write with their own
gates, so the question is whether a key-only quantity that uses the model's real decay and write-strength gates (but still no values) tracks the
interference that actually happens in the real recurrence, and whether it carries information beyond decay and distance. A positive answer would
mean the task-level null (Section 4) is a property of the gate-free bound; a negative one that it is not.

## Models and data
Gated DeltaNet 340M and 360M (validated re-implementations, `gdn_extract.py`). Part A: sixteen 1024-token sequences per model (eight prose, eight code; the first four of each are the sequences of Figure 4), every layer and head. Part B: the 600 recall prompts per model of the pre-registered task test (same prompts, same
calibration/test split, same long-memory head definition: median decay half-life >= 256 tokens on calibration prompts).

## Quantities (fixed now)
For a stored position s and a delay Delta, with k_u the unit-norm keys, alpha_u and beta_u the real gates, A = prod_{u in (s, s+Delta]} alpha_u:
* D_act   = || S_{s+Delta} k_s - A S_s k_s || / || S_s k_s ||   in the actual recurrence (real values).        [the measured interference, as in Figure 4]
* D_gate  = the same quantity in the key-only recurrence run with the REAL gates and the values replaced by the keys (v_u = k_u).   [primary gate-aware predictor]
* D_free  = || S_{s+Delta} k_s - S_s k_s || / || S_s k_s || for the gate-free memory: alpha = beta = 1, v = k, started empty 32 tokens before s.   [gate-free predictor]
* D_shuf  = D_gate with the keys permuted over positions (real gates kept at their positions): the key-specificity control, as the shuffled P1 of the task test.
* Bkey    = sum_{u in (s, s+Delta]} beta_u |<k_u, k_s>|   (gate-weighted key overlap, no recurrence).
* Baselines that use gates or distance only: A, sum beta_u, Delta.
Part A grid: s in {64, 128, ..., 512}, Delta in {16, 64, 256, 512} (cells with s + Delta > 1023 are dropped).
Part B: for the queried fact the three span tokens (key, "is", value) play the role of s, Delta runs to the query position; P1g = mean over long-memory
heads and over the span of D_gate (primary); P1r = same with retention || S_T k_s || / || S_s k_s || (secondary).

## Analyses and success criteria (fixed now)
Part A: (1) pooled Spearman correlation of D_act with each predictor over all cells; (2) the within-group correlation (rank-transform within each
(layer, head, Delta) group, then pool) with a cluster bootstrap over the sixteen sequences, and the same after removing log A by rank regression
(partial correlation). The gate-aware predictor "tracks which positions suffer interference" if its partial within-group correlation is >= 0.3 with
a bootstrap interval excluding 0 and exceeds that of D_free by more than the interval width. Reported separately for long-memory heads.
Part B: (1) AUC(B3 + P1g) - AUC(B3) on test prompts (B3 = age + one-hot load) with the same bootstrap as H1; (2) the same gain with the flexible
categorical-age baseline (a fixed effect for every (load, queried position) cell, ridge weight 1); (3) the rank correlation of P1g with the
candidate log-probability within exact (load, queried position) cells, pooled as in Appendix D. The gate-aware predictor "adds information at the
task level" only if (2) gives a gain >= 0.03 with an interval excluding 0 in the primary model and the same sign in the replication.

## Reporting
All of this is post hoc relative to the pre-registered test; every number is reported whatever its sign. No predictor, threshold or head definition
is changed after seeing results. Deviations, if any, are logged below.

## Deviation log
D0 (before any statistic was computed): the sequence count was raised from eight to sixteen per model after a timing test (one prose and one code sequence, used only to check that the script runs and that the key-only interference identity holds, residual 5e-7) to tighten the cluster bootstrap.

D1 (before the full runs; after running the analysis script once on the two-sequence timing file, whose numbers were not used): added the key-permutation control D_shuf, so that a gate-aware advantage can be attributed to the keys and not only to the gates; its success criterion is that D_gate exceeds D_shuf by more than the interval width.
