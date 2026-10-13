# Key geometry in delta-rule memory

Code, protocols, raw results and validation scripts for the paper *Key Geometry in Delta-Rule Memory: Exact Interference Laws, a Calibrated Diagnostic, and Pre-Registered Negative Results*.

The paper offers (i) exact laws that split the recall error of a delta-rule memory into a key-only factor and a value-dependent gain, (ii) a calibrated, value-free diagnostic (`RMS <= r90 * psi`) measured on six trained models, with constants that cover 0.87-0.92 of the windows of new text but do not transfer between model families, and a gate-aware version that tracks the interference in the real recurrence, and (iii) pre-registered negative results: the diagnostic does not predict recall of a stored fact beyond age and load in two Gated DeltaNets, with gate-aware predictors, or in a 1.3B model, and a training-time effect of key decorrelation seen in twelve seeds did not replicate in twelve new ones. The paper source is in `overleaf/build/` (`template.tex`, `appendix.tex`).

**Everything is run from the repository root**, and several scripts read paths relative to it. Python 3.13 with `torch`, `transformers`, `numpy`, `safetensors`, `matplotlib`, `scipy` and `huggingface_hub`; scripts are run as `python -I <script>` (some take a model directory argument or the `GDN_MD` environment variable).

## What is here and what is not
Included: all scripts, the small result files they produce (`results_*.json`, `exp_*/runs/*.json`, `exp_task/`, `exp_gate/`, `review/`), the figure and table generator `figgen/` with its two caches, and the paper sources.
Not included: model checkpoints, MNIST (public; see `data_mnist/README.md`), large intermediate pickles and per-run activation caches (the figure generator uses `figgen/cache_cert.npz` and `figgen/cache_mq_geom.json` instead), run logs, and two internal documents (the author's working log and an older draft of the paper). One of the five text collections used in the measurement study was the author's own working notes; it is not released, so that single domain cannot be reproduced exactly (the other four can).

## Map: paper section to code
| paper | what | code |
|---|---|---|
| 2 Exact laws; Appendix A | identities, capacity floor, rank and comparator bounds (random-instance checks) | `verify.py`, `verify_tight.py`, `verify_rank.py`, `verify_kv.py`, `verify_q.py` |
| 3 Measurements; Appendix B | Gated DeltaNet 340M and 360M, interference in the actual recurrence | `gdn_extract.py` (forward pass and measurements), `gdn_keydiag.py`, `gdn_rank.py`, `gdn_kv*.py`, `gdn_q.py`, `gdn_latent*.py`, `gdn_flat*.py`, `gdn_transfer*.py`, `analyze_transfer.py`, `analyze_cross_model.py`, `summarize_gdn.py` |
| 3 (other families) | Qwen3.5, RWKV-7, two softmax Transformers | `qwen35_flat.py` + `analyze_q35.py`; `rwkv7_flat.py` + `analyze_rwkv7.py`; `tf_flat.py` + `analyze_tf.py` |
| Appendix B.6 (held-out check) | the published r90 constants applied to new sequences (post hoc) | `exp_heldout/` (`PROTOCOL.md`, `run_heldout.sh`, `analyze_heldout.py`, `heldout_windows.npz`); the pipelines `gdn_transfer2.py`, `rwkv7_flat.py`, `qwen35_flat.py`, `tf_flat.py` read new text when run with `HELDOUT=1` (switch in `heldout.py`, off by default) |
| Appendix D.8 | scale replication of the task test on a 1.3B Gated DeltaNet (post hoc extension) | `exp_scale/` (`PROTOCOL.md`, `extract_scale.py`, `analyze_scale.py`; the pilot is `exp_task/pilot.py`) |
| Appendix B.7, D.7 | gate-aware key-only analysis (post hoc) | `exp_gate/` (`PROTOCOL.md`, `gate_actual.py`, `analyze_actual.py`, `extract_gate_task.py`, `analyze_task_gate.py`, `run_after_actual.sh`) |
| 4 Pre-registered task test; Appendix D | key-value recall prompts on both models | `exp_task/` (`PREREGISTRATION.md`, `extract_task.py`, `analyze_task.py`, `exploratory_task.py`) |
| 5 Training-time experiment; Appendix C | multi-query associative recall with a small delta-rule model and three regularisers; replication with new seeds | `exp_mqar/` (`mqar_train.py`, `run_grid.sh`, `run_stage2.sh`, `run_stage3.sh`, `run_stage4.sh`, `analyze_mqar*.py`, `analyze_replication.py`, `PROTOCOL_replication.md`) |
| 6 Related work; Appendix E | systematic literature review with a protocol written before searching | `review/` (`protocol.md`, `search.py`, `screen.py`, `snowball.py`, `finalize.py`, `make_report.py`, `literature_review.md`) |
| Appendix F | validation of the re-implemented forward passes | `validate_gdn_reference.py`, `validate_rwkv7_reference.py` (+ `.json` results) |
| Supplement G-I | continual learning in the weights of small SSMs | `exp_cl/`, `exp_mem/`, `exp_bench/` |
| figures and tables | every figure (matplotlib, vector PDF at the text width) and table fragment | `figgen/` (`make_all.py` writes into `overleaf/build/`) |

## Validation of the re-implementations
* **Gated DeltaNet.** `GDN_MD=model_gdn340m python -I validate_gdn_reference.py` (or `model_gdn360m`) maps the checkpoint into the gated-delta-rule layer of HuggingFace `transformers` and compares layer outputs and full-model logits (max |logit difference| 4.6e-5 and 1.7e-4, max KL below 7e-7, 100% top-1 agreement, identical perplexity). Results: `validate_gdn_model_gdn340m.json`, `validate_gdn_model_gdn360m.json`, `validate_gdn_model_gdn13b.json` (the 1.3B checkpoint has no short convolution; the reference layer's convolution is set to an identity kernel).
* **RWKV-7.** `python -I validate_rwkv7_reference.py <path>/rwkv_v7_numpy.py 512` runs the unmodified functions of the NumPy reference in the authors' repository (`BlinkDL/RWKV-LM`, `RWKV-v7/rwkv_v7_numpy.py`; download it yourself, it is not included) on the converted checkpoint and compares logits with `rwkv7_flat.py` (float64: max difference 2.3e-13; float32: 4.5e-5, identical top-1 and perplexity). Result: `validate_rwkv7_reference.json`. Not covered: the authors' GPU kernels.

## Models and data
Checkpoints are downloaded from the Hugging Face Hub into local directories named `model_gdn340m`, `model_gdn360m`, `model_qwen35_08b`, `model_rwkv7_01b`, `model_gdn13b` (`linear-moe-hub/Gated-Deltanet-340M`, `puigde/gated-deltanet-360M-15B-slimpajama`, `Qwen/Qwen3.5-0.8B-Base`, `fla-hub/rwkv7-0.1B-g1`, data files only, and `linear-moe-hub/Gated-Deltanet-1.3B`), plus `HuggingFaceTB/SmolLM2-135M-Instruct` and `Qwen/Qwen2.5-1.5B-Instruct` in the Hub cache. No repository code from a checkpoint is executed. Text for the measurement study: the cached APPS dataset (`codeparrot/apps`), `pydoc_data`, licence files.

## Notes
* **Process isolation matters.** On the machine used, mixing torch and NumPy linear algebra in one process caused intermittent crashes and NaNs, so the pipelines run torch (extraction, training) and NumPy (analysis) in separate processes.
* Several scripts write large intermediates to a `scratch/` directory next to the repository root; change `SCRATCH` if you prefer elsewhere.
* The task-level, gate-aware and review protocols (`exp_task/PREREGISTRATION.md`, `exp_gate/PROTOCOL.md`, `review/protocol.md`) were written before the corresponding runs; deviations and amendments are logged in them.

## Status
A preprint-stage release (the paper reports its nulls and one non-replication: the training-time effect of key decorrelation seen in the first twelve seeds did not replicate in twelve new seeds; see `exp_mqar/result_replication.json`): the theorems are proved in the paper and checked numerically; the empirical claims are for six checkpoints, a small downstream test and small synthetic studies, and several are underpowered (see the paper's Limitations). Nothing here has been peer reviewed.

## Licence
Released under the MIT licence (see `LICENSE`). The bibliographic records in `review/` were retrieved from the arXiv and OpenAlex APIs (e-mail addresses of third parties that appeared inside retrieved abstracts were removed); check their terms before reusing them. Model checkpoints and datasets referred to here are not part of this repository and keep their own licences.
