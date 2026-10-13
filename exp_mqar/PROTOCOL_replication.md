# Replication of the key-decorrelation effect with new seeds, balanced seeds for the matching regularisers, and a multiplicity correction (post hoc extension; protocol fixed before the runs)

## Question
Section 5 reports that key decorrelation raised mean recall of a small delta-rule model from 0.804 to 0.969 over 12 paired seeds (paired difference +0.165, exact sign-flip p = 0.042,
uncorrected for three regularisers), predominantly by removing three training failures of the baseline (seeds that did not learn); Gram matching (6 seeds) and rank matching (3 seeds) were
worse. Limitations: one 12-seed comparison, fewer seeds for the matching regularisers, no multiplicity correction.

## Additions (identical configuration to the original runs: 3000 steps, d = 64, two heads, d_k = 32, batch 64, lr 0.002, n_max = 32, L_max = 96)
1. Replication: baseline and key decorrelation (weight 1) with new seeds 12-23 (12 new pairs). Nothing about the original 12 seeds changes.
2. Balance: Gram matching (weight 1) seeds 6-11 and rank matching (weight 0.1) seeds 3-11, so that all four learners have seeds 0-11.
3. Multiplicity: Holm correction of the three regularisers' paired comparisons with the baseline on seeds 0-11.

## Analyses and success criteria (fixed now)
Outcome per run: mean recall accuracy over n in {8, 16, 24, 32} and L in {0, 32, 96} (as in the paper); a run "failed to learn" if its mean recall is below 0.9 (as in the paper).
* Primary (replication): paired difference (key decorrelation minus baseline) over the 12 new pairs, 95% paired bootstrap interval, exact two-sided sign-flip p. The effect is replicated if the difference is positive, the interval excludes 0 and p < 0.05.
* Secondary: failure counts (Fisher exact), pooled analysis of all 24 pairs, and whether the gain is again concentrated in baseline failures (gain among pairs where the baseline did not fail).
* Matching regularisers: paired differences with the baseline on seeds 0-11, exact sign-flip p, and Holm-adjusted p over the three regularisers.
Every number is reported whatever its sign; no analysis is changed after seeing results.

## Deviation log
Result: the effect was not replicated (new seeds 12-23: paired difference +0.058, 95% interval [-0.136, +0.282], 5 of 12 higher, sign-flip p = 0.638; see exp_mqar/result_replication.json and Appendix C.5 of the paper). Exploratory addition, made after the runs: the relation between accuracy and layer-1 key effective rank in the 24 new runs. No other deviation.
