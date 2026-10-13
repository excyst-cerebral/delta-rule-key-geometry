# Scale replication of the task-level test on a 1.2B Gated DeltaNet (extension of the pre-registered test; protocol fixed before any recall prompt is run)

## Question
Section 4 of the paper tests, on two small Gated DeltaNets (340M, 360M; weak recall, prompts of at most about 100 tokens), whether the key-only quantity psi
predicts recall of a stored fact beyond age and load, and finds no detectable incremental association (also for the gate-aware version, Appendix D.7).
Does the null survive in a larger model of the same family, where recall is stronger?

## Model
`linear-moe-hub/Gated-Deltanet-1.3B` (1.21B parameters, 24 layers, hidden size 2048, 4 heads of dimension 256, trained on SlimPajama; the same hub and family as the
340M primary model). Differences from the 340M model: hidden size 2048 instead of 1024 and no short convolution (the re-implementation `gdn_extract.py` uses
SiLU of the projection in that case). The re-implementation is validated against the HuggingFace gated-delta-rule layer (`validate_gdn_reference.py`, with an identity
convolution kernel in the reference): layer outputs agree to at most 4.3e-6 (relative), logits to 1.3e-3 at logit scale 16-17, identical top-1 predictions and perplexity.

## Task, prompts, split, outcome (unchanged)
Exactly as `exp_task/PREREGISTRATION.md` after its Amendment A: prompts of n facts "<Key> is <value>." with n in {4, 8, 16, 24}, 150 prompts per load (600), prompt
determined by (n, j), calibration prompts j even, test prompts j odd; outcome = forced-choice recall (true value ranked first among the n stated values; chance 1/n);
secondary outcome = log-softmax of the true value's logit over the n candidates.

## Pilot (accuracy only, before any predictor is computed)
20 prompts per load on the new model, no certificate quantity. If forced-choice accuracy is at ceiling (>= 0.95) at every load n <= 24 the test cannot discriminate;
in that case the loads are extended to the largest values the single-token vocabulary allows (this would be logged as Amendment A of this protocol). Otherwise the
design above is run unchanged.

## Predictors (fixed now; computed per layer and head on the prefix up to the query, with the model's own unit-norm keys)
* P1, P1b: the gate-free quantity psi of the paper (alpha = beta = 1, one pass over the whole prefix), item score = mean over the three span tokens; P1 = mean over
  long-memory heads, P1b = mean over all heads (exactly as in the pre-registration).
* P1g, P1r, P1A, P1g shuffled, P1gb: the gate-aware key-only quantities of Appendix B.7 / D.7 (`exp_gate/`), same head sets.
* Long-memory heads: median per-token decay half-life >= 256 tokens, computed on the calibration prompts only. Fallback fixed now: if fewer than 8 heads qualify, the
  20% of heads with the longest median half-life are used instead (this would be reported as a deviation).
* Baselines: age, load n, age + one-hot load (B3), and the flexible categorical-age baseline (a fixed effect for every (load, queried position) cell, ridge weight 1).

## Analyses and success criteria (fixed now)
Identical to the pre-registration and to Appendix D: held-out AUC with a logistic model fitted on the calibration prompts, evaluated on the test prompts, 2000
bootstrap resamples of test prompts stratified by load.
1. H1 (as pre-registered): AUC(B3 + P1) - AUC(B3) >= 0.03 with a 95% interval excluding 0.
2. The same gain for P1 and for P1g over the flexible categorical-age baseline: a signal requires >= 0.03 with an interval excluding 0.
3. Within exact (load, queried position) cells, the rank correlation of P1 and P1g with the true value's log-probability, and the shuffled controls.
Verdict rule: the larger model "confirms the null" if neither P1 nor P1g reaches the criterion in (2) and the intervals in (2) and (3) include 0; it "shows a signal" if
(2) is met for P1 or P1g with the expected sign. Anything else is reported as mixed, with every number.

## Reporting
Every number is reported whatever its sign. No predictor, threshold or head definition is changed after seeing results. Deviations are logged below.

## Deviation log
Pilot result (20 prompts per load, accuracy only, before any predictor was computed): forced-choice accuracy 0.55 / 0.45 / 0.35 / 0.05 / 0.00 at n = 2 / 4 / 8 / 16 / 32 (chance 0.50 / 0.25 / 0.125 / 0.06 / 0.03). Not at ceiling at any load, so the design above was run unchanged. No other deviation.
