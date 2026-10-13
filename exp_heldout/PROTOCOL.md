# Held-out check of the calibrated key-only bound on new text (post hoc extension; protocol fixed before any held-out sequence is run)

## Question
The calibrated constant r90 of Section 3 (the 90th percentile of RMS/psi_rms over windows, heads and layers) rests on 10-20 sequences per model from five small local text collections.
Do the constants, fixed at their published values, still cover about 90% of the windows of sequences that were not used to calibrate them?

## Data (new sequences, disjoint from the calibration sequences)
Same six models and pipelines as Section 3 (`gdn_transfer2.py`, `rwkv7_flat.py`, `qwen35_flat.py`, `tf_flat.py`; switch `HELDOUT=1`), same windows (n = 16, 32 and the
windows ends of the original pipelines), same idealised-memory definition of RMS and psi_rms. Held-out sequences: prose and code from APPS problems 600-1199 (the calibration
sequences used problems 0-599), and docs and legal sequences taken at start positions interleaved between the calibration positions (different text, no overlap); the
project-notes collection is not used. Sequences per domain are those of the original pipelines (4 for the Gated DeltaNets, 3 for the Transformers, 2 for RWKV-7 and Qwen3.5).

## Quantities and success criteria (fixed now)
* Coverage: the fraction of held-out windows with RMS <= r90 * psi_rms, using the published pooled r90 of the model and window size n (nominal 0.90), with a 95% cluster bootstrap over held-out sequences (all windows, heads and layers of a sequence resampled together).
* Also reported: the held-out 90th percentile of RMS/psi_rms next to the published r90.
* The calibration "transfers to new text" for a model if the interval of its coverage contains 0.90 or lies within [0.85, 0.95], at both window sizes. Anything else is reported as it is.
Every number is reported whatever its sign; no constant is re-fitted and no threshold changed after seeing results.

## Deviation log
(none yet)
