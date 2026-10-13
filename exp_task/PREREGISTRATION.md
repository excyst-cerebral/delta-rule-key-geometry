# Pre-registration: does the key-only certificate predict task-level recall failure in trained delta-rule models?
Written BEFORE any recall prompt was run through the models (a pilot of accuracy-versus-load only, with no certificate computation, is run first to choose a non-degenerate difficulty range; it is reported).

## Question
Section 7 of the paper shows that, for the idealised delta-rule memory built from a layer's keys and values, a *value-free* quantity computed from the keys alone (psi) predicts recall error (calibrated bound RMS <= r90*psi). Does the same key-only quantity, computed from the model's own keys on a prompt, predict whether the *model* (a trained Gated-DeltaNet LM) recalls a stored fact on a downstream key-value recall task, beyond what simple baselines predict?

## Models
`linear-moe-hub/Gated-Deltanet-340M` (primary) and `puigde/gated-deltanet-360M-15B-slimpajama` (replication). Both pure delta-rule models, using the validated re-implementations in `gdn_extract.py`.

## Task
Prompts of n facts "<Key> is <value>." (one per line; keys are single-token capitalised names, values single-token common words, all distinct within a prompt) followed by a query "<Key_q> is"; success = the true value is the top-1 next token. Load n in a range chosen by the pilot so that accuracy spans roughly 0.2-0.9. Query fact uniform over the n facts. Distractor facts are the other n-1 facts; no other filler. 600 prompts per model (a few load levels), fixed seeds. Train/test split of prompts 50/50 by prompt (calibration set used only to define head groups and any thresholds; test set used for all reported numbers).

## Predictors (fixed now)
For every layer and head, on the prefix up to (not including) the query, with the model's own unit-norm keys: the idealised one-pass error operator T = L^T (I+L)^-1 over the whole prefix, and
 psi_i = || (T K)_i ||  (key-only self-association error of token i, alpha = beta = 1, as in Section 4).
Item score for the queried fact = mean of psi over the tokens of that fact's span (key token, "is", value token).
 * **P1 (primary):** mean item score over *long-memory heads* (median half-life >= 256 tokens, computed on the calibration set from the model's own decay gates), over all layers.
 * **P1b:** mean over all heads.
 * Baselines: B1 = number of tokens between the end of the fact and the query (age); B2 = n; B3 = age + n.
 * Specificity control: P1 computed from the keys of a *different* prompt with the same n (shuffled).

## Analysis and success criteria (primary)
Outcome y = recall success on the TEST prompts. Logistic regression fitted on the CALIBRATION prompts only; evaluated on TEST prompts by AUC.
 * Primary hypothesis H1: AUC(P1 + B3) - AUC(B3) >= 0.03 with a 95% bootstrap interval (over test prompts) that excludes 0, on the primary model, replicated in the second model (same sign and interval excluding 0).
 * Secondary: AUC(P1) alone vs the shuffled control (P1 must beat the control with interval excluding 0); a decision-style test: among test prompts, recall accuracy in the lowest-psi tercile minus highest-psi tercile within each load level (certified-safe prompts should do better).
 * Mechanism-level check (exploratory): correlation between psi and the actual recurrence's self-recall error with the model's real gates.
## What would count as failure
If H1 is not met in either model, we report that the key-only certificate adds no predictive value over age and load for task-level recall in these models. No change to predictors, thresholds, head definition or analysis after seeing test outcomes; any additional analysis is labelled exploratory.

## Amendment A (made after the accuracy-only pilot on GDN-340M, before any certificate was computed on any prompt)
Pilot (40 prompts per load, no certificate): strict top-1 over the whole vocabulary was 0.05-0.15 at every load (the format prior "<Key> is" -> generic word dominates), while **forced-choice recall** (true value ranked first among the n values stated in the prompt) was 0.60 / 0.40 / 0.30 / 0.20 / 0.07 at n = 2 / 4 / 8 / 16 / 32 against chances 0.50 / 0.25 / 0.125 / 0.06 / 0.03. Therefore:
 * **Primary outcome = forced-choice correct** (replaces strict top-1). Secondary continuous outcome = log-softmax of the true value's logit over the n candidate values.
 * **Load levels n in {4, 8, 16, 24}**, 150 prompts each (600 per model); n = 2 is near chance and n = 32 is at the floor, both dropped. Because chance depends on n, every model includes n as a baseline and AUCs are reported pooled and within each load.
 * Everything else (predictors, split, head definition, success criteria) is unchanged.

## Deviation log
**D1 (implementation, not outcome-driven).** The first analysis run on GDN-340M found 0 long-memory heads because my extraction stored each head's *mean log-decay per prompt* (dominated by strongly decaying tokens) while this pre-registration defines long-memory heads by the *median per-token half-life*. P1 was therefore undefined and the primary criterion could not be evaluated. That run's printed secondary numbers (including P1b, which does not use the head group) were seen. Fix: compute the median per-token half-life per head from decay gates on **calibration prompts only** (`exp_task/halflife_cal.py`, 10 calibration prompts per load) and use it for the head group, exactly as written above. No outcome information enters the head definition. Everything else is unchanged.
