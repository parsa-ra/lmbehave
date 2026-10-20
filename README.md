# Do Knowledge- and Reasoning-Oriented Benchmarks Induce Stable Representation Signatures in LLMs?

**Parsa Rahimi Noshanagh** (EPFL, Idiap Research Institute) · **Azim Dehghani Amirabad** (Independent Researcher)

NeurIPS 2026 Workshop on *Interpretability as a Science*

[**Paper**](https://arxiv.org/abs/2610.24029) · [**Project page**](https://parsa-ra.github.io/lmbehave/) · [**Artifacts**](https://drive.google.com/drive/folders/1eS3xhP3qB_QBep2mUcOfbvddVSj4qHy2?usp=drive_link) · [**Website source**](https://github.com/parsa-ra/lmbehave-page)

<p align="center"><img src="https://parsa-ra.github.io/lmbehave/static/images/overview.png" width="92%" alt="Study overview: layerwise measurement, task-level inference, and evidence across tests."></p>

## The study in one paragraph

Benchmark suites routinely split tasks into *knowledge* (recall) and *reasoning* (multi-step inference). If a language model processed the two groups differently, that internal difference could drive monitors, adaptive computation, or causal interventions. We test whether such a signature exists in five current 7B–31B models (OLMo 3 7B, Apertus 1.5 8B, Qwen 3.5 9B/27B, Gemma 4 31B) using **the task as the unit of inference**: probes are evaluated only on held-out subjects, labels are permuted across whole subjects (all 12,870 balanced labelings of 16 MMLU subjects), and the best layer and stream are reselected for every labeling. Activations at the last prompt token recover the label in all five models, but a probe on the **prompt text alone, run through the same estimator, does nearly as well**: the activation gain is +0.017–0.070 macro-F1 and is significant in no model. The best layer also moves between task panels and site selectors, and two released training trajectories (OLMo 3, Apertus 1.0) move in opposite directions. The contribution is the evaluation protocol: task-level holdout and permutation, a matched prompt comparison, transfer tests, and detection calibration.

| Model | Raw probe macro-F1 (exact *p*) | Prompt-text probe | Activation gain Δ (exact *p*) | Mean-shift site (exact *p*) |
| --- | ---: | ---: | ---: | --- |
| OLMo 3 7B | 0.781 (0.0016) | 0.755 | +0.048 (0.290) | layer 3, FFN (0.072) |
| Apertus 1.5 8B | 0.804 (0.0019) | 0.755 | +0.070 (0.177) | layer 0, FFN (0.0044) |
| Qwen 3.5 9B | 0.805 (0.0019) | 0.755 | +0.047 (0.291) | layer 2, FFN (0.0011) |
| Qwen 3.5 27B | 0.796 (0.0042) | 0.755 | +0.017 (0.440) | layer 1, token mixing (0.00078) |
| Gemma 4 31B | 0.805 (0.0040) | 0.755 | +0.051 (0.297) | layer 23, FFN (0.0145) |

Primary MMLU panel (8 knowledge-oriented vs. 8 reasoning-oriented subjects, 100 questions each). The prompt-text probe uses character n-gram TF–IDF features under the same folds, feature width, classifier, and permutation null. Full tables: `results/`.

## What is in this repository

| Path | Content |
| --- | --- |
| `experiments/` | Frozen experiment manifests: models with immutable revisions, task panels, sample freezes, batch sizes, analysis settings, decision thresholds. |
| `results/` | Sample-free summary tables (CSV/JSON) behind every number in the paper; each `report.json` records the SHA-256 of the raw files it was computed from. |
| `scripts/` | Orchestration (`run_representation_experiment.py`), study launchers (`run_*.sh`), validators (`validate_*.py`, `audit_*.py`), and the summarizers (`summarize_*.py`) that produce `results/`. |
| `lmms-eval/` | Bundled research snapshot of [`lmms-eval`](https://github.com/EvolvingLMMs-Lab/lmms-eval) with the read-only activation tracker, the task-level inference code, the control analyses, and the task adapters (branch `research/representation-dynamics`). |
| [Project page source](https://github.com/parsa-ra/lmbehave-page) | Independent static website repository. |
| `docs/` | [`ARTIFACTS.md`](docs/ARTIFACTS.md) (provenance of every result and how to restore the raw artifacts), [`REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md), [`SCIENTIFIC_STATUS.md`](docs/SCIENTIFIC_STATUS.md) (what the evidence does and does not support), [`OPERATIONS.md`](docs/OPERATIONS.md) (machine setup and the complete command reference), [`WORKLOG.md`](docs/WORKLOG.md) (dated execution audit), [`EXTERNAL_REVIEW_RESPONSE.md`](docs/EXTERNAL_REVIEW_RESPONSE.md). |
| `tests/` | Unit tests for the orchestration, validators, and summarizers (`python -m pytest -q tests`). |
| `AGENTS.md` | Handoff contract for anyone (or any coding assistant) continuing the research: scientific constraints, instrumentation semantics, panel definitions, open threats. |

## How the measurements are made

1. **Measure.** `--representation_dynamics` attaches three forward hooks per decoder layer and records, at the final valid prompt token, the token-mixing update, the feed-forward update, and the residual state (one float32 vector per example, layer, and stream). The hooks only read activations; evaluation without the flag is unchanged.
2. **Infer at the task level.** `lmms_eval.representations.analysis` computes the PCA-space mean-shift ratio per layer and stream, task-held-out linear and RBF probes, and exact permutation tests that reassign labels over whole tasks and reselect the maximizing site for every labeling (Benjamini–Hochberg across sites; selection-aware maximum).
3. **Compare with the prompt and control.** `selected_site_probe`, `prompt_baseline`, `matched_excess`, `lexical_diagnostics`, `probe_calibration`, `whitened_selector`, and the width/split-unit sensitivities in `lmms_eval/representations/` implement the controls reported in the paper.

## Setup

The evaluation implementation is bundled at revision `2f121d64e804b83fe8724981fd180d1d5a4c3047`; no private submodule access is required. Its upstream MIT license is retained in `lmms-eval/LICENSE`.

Requirements: Linux, Python 3.12, [`uv`](https://github.com/astral-sh/uv), one NVIDIA GPU with ≥80 GB for the 27B–31B models (the study ran on one GH200), and RAPIDS cuML 26.06 for the GPU analysis backend (the scikit-learn backend runs on CPU).

```bash
git clone https://github.com/parsa-ra/lmbehave.git
cd lmbehave
python3 scripts/bootstrap_workspace.py   # pinned Transformers checkout (repos.lock.json)
./install.sh                             # .venv: torch, transformers, lmms-eval (editable)
```

`docs/OPERATIONS.md` has the full machine bootstrap (`scripts/setup_machine.sh`), the isolated Apertus runtime, and the cuML analysis environment.

## Reproducing the results

### A. From the released artifacts (no GPU needed)

The raw outputs of every run are released as a verified bundle (`persistent-data.tar.zst`, 77 GB; SHA-256 `160ab6cb…`) together with a text-only archive (`analysis-outputs-text.tar.gz`, 0.56 GB) that contains all reports, CSVs, logs, and permutation nulls. The committed `results/` tables regenerate byte for byte from the text archive:

```bash
sha256sum -c SHA256SUMS
mkdir -p /data/lmbehave && tar -xzf analysis-outputs-text.tar.gz -C /data/lmbehave
# the artifacts' own reports record absolute paths of the original compute volume;
# the helper redirects them to the extraction directory without modifying any file
export PYTHONPATH=scripts/artifact_path_remap LMB_REMAP_TARGET=/data/lmbehave/
python3 scripts/summarize_main_track_instrument_validation.py \
  /data/lmbehave/vlmbehave-artifacts/2026-08-16/representation-dynamics-main-track-instrument-validation-v1 /tmp/regen
diff /tmp/regen/matched_excess_summary.csv results/representation_dynamics_main_track_instrument_validation_v1/
```

`docs/ARTIFACTS.md` maps every `results/` directory to its bundle directory and summarizer, and records the verification performed before release.

### B. The full pipeline (models and GPU)

Every study is driven by a frozen manifest. The primary 8+8 MMLU panel:

```bash
MANIFEST=experiments/representation_dynamics_main_track_v1.json
RUN=logs/representation-dynamics-main-track-v1
.venv/bin/python scripts/run_representation_experiment.py --manifest $MANIFEST plan
.venv/bin/python scripts/run_representation_experiment.py --manifest $MANIFEST check-tasks
.venv/bin/python scripts/run_representation_experiment.py --manifest $MANIFEST freeze-samples --run-dir $RUN
.venv/bin/python scripts/run_representation_experiment.py --manifest $MANIFEST run --run-dir $RUN
.venv/bin/python scripts/run_representation_experiment.py --manifest $MANIFEST validate --run-dir $RUN
.venv/bin/python scripts/run_representation_experiment.py --manifest $MANIFEST analyze --run-dir $RUN \
  --analysis-backend cuml --analysis-name primary-fisher-exhaustive --jobs 1 --parallel-backend process
```

| Study (paper section) | Manifest | Launcher | Summarizer → `results/` |
| --- | --- | --- | --- |
| Primary 8+8 MMLU panel: mean-shift maxima (§3.3) | `representation_dynamics_main_track_v1.json` | `run_representation_experiment.py` | `summarize_main_track_confirmation.py` → `representation_dynamics_main_track_v1` |
| Selected-site probe (§3.3) | same | `run_selected_site_probe_panel.sh` | `representation_dynamics_main_track_selected_probe_v1` |
| Prompt baselines, prompt-feature adjustment, validity (§3.3–3.4) | `..._main_track_controls_v1.json` | `run_main_track_controls.sh` | `summarize_main_track_controls.py` → `..._controls_v1` |
| Matched activation-vs-prompt test, lexical diagnostics, calibration (§3.3–3.4) | `..._main_track_instrument_validation_v1.json` | `run_main_track_instrument_validation.sh` | `summarize_main_track_instrument_validation.py` → `..._instrument_validation_v1` |
| Null summaries, PCA-95 width, unit-of-inference contrast, Holm (§3.3) | `..._main_track_review_response_v1.json` | `run_main_track_review_response.sh` | `summarize_main_track_review_response.py` → `..._review_response_v1` |
| Lexical-32/64 adjustment (§3.4) | `..._main_track_lexical_control_v1.json` | `run_main_track_lexical_control.sh` | `summarize_main_track_lexical_control.py` → `..._lexical_control_v1` |
| Covariance-whitened selector (§3.3, App. D) | `..._selector_sensitivity_v1.json` | `run_whitened_selector_sensitivity.sh` | `summarize_whitened_selector_sensitivity.py` → `..._selector_sensitivity_v1` |
| MMLU-Pro 5+5 and mixed-benchmark panels, cross-panel transfer (§3.3) | `representation_dynamics_text_expansion_v1.json` | `run_full_representation_experiment.sh` | `run_calibrated_matched_reanalysis.py`, `summarize_representation_experiment.py` → `..._main_track_fisher_v1`, `..._text_expansion_corrected_v1` |
| OLMo 3 training trajectory (§3.1) | `olmo3_longitudinal_v1.json` | `run_olmo_longitudinal_experiment.py` | `olmo3_longitudinal_v1` |
| Apertus 1.0 training trajectory (§3.1) | `apertus1_longitudinal_v1.json` | `run_full_apertus1_longitudinal.sh` | `apertus1_longitudinal_v1` |
| Response-position pilot (§3.4) | `..._relative_response_pilot_v2.json` | `run_relative_response_pilot.sh` | `summarize_relative_response_pilot.py` → `..._relative_response_pilot_v2` |
| Multimodal appendix panel (App. B.5) | `..._multimodal_side_v1.json` | `run_multimodal_side_analysis.sh` | `summarize_multimodal_side_analysis.py` → `..._multimodal_side_v1` |

Model revisions, batch sizes, and sample freezes are fixed in the manifests; changing them changes the study.

### C. Article and project page

Read the [article on arXiv](https://arxiv.org/abs/2610.24029). The independent [website repository](https://github.com/parsa-ra/lmbehave-page) includes the displayed figures and PDF; open its `index.html` directly to preview it.

### Tests

```bash
python -m pytest -q tests
(cd lmms-eval && python -m pytest -q test/eval/test_representation_dynamics.py test/eval/test_cli_parse_args.py)
```

## Scope

The subject labels are operational proxies that entangle domain, format, and difficulty; 16 subjects are few independent units; each training trajectory is a single run; and a synthetic-shift control detects none of ten injected directions at up to two standard deviations. The paper therefore reports task-label recovery without a transferable knowledge/reasoning signature, and moderate effects remain possible. `docs/SCIENTIFIC_STATUS.md` states the claim boundary in detail.

## Citation

```bibtex
@inproceedings{rahiminoshanagh2026knowledge,
  title     = {Do Knowledge- and Reasoning-Oriented Benchmarks Induce Stable
               Representation Signatures in {LLMs}?},
  author    = {Rahimi Noshanagh, Parsa and Dehghani Amirabad, Azim},
  booktitle = {NeurIPS 2026 Workshop on Interpretability as a Science},
  year      = {2026},
  eprint    = {2610.24029},
  archivePrefix = {arXiv},
  url       = {https://arxiv.org/abs/2610.24029}
}
```

## License

MIT (see `LICENSE`). The `lmms-eval` fork keeps its upstream MIT license; model weights, benchmark text, and the released activations remain subject to their own licenses and terms.
