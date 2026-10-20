# Operations and command reference

This is the complete operational reference that preceded the release README: machine bootstrap,
environment, every collection and analysis command, and the development workflow. The
front-page overview is `../README.md`; artifact provenance is `ARTIFACTS.md`.


This workspace tests whether knowledge- and reasoning-oriented benchmark labels induce a
stable internal representation signature in Transformer language models. All reported
primary evidence is text-only. A separately identified appendix analysis covers a
heterogeneous multimodal panel without changing or pooling with the primary results. The
paper is accepted at the NeurIPS 2026 Workshop *Interpretability as a Science* (project page:
`https://parsa-ra.github.io/lmbehave/`, source in `parsa-ra/lmbehave-page`); pruning or routing is a later
question that requires a replicated site and controlled intervention.

The active implementation is deliberately confined to the bundled `lmms-eval/` source.
It is opt-in and uses forward hooks that observe, pool, and save activations without
replacing module outputs or changing model weights. The clean Transformers checkout remains
the default model implementation.

## Current status

- Root repository: public `parsa-ra/lmbehave` on branch `main`.
- Active code: bundled `lmms-eval/` source at research revision `2f121d64e804b83fe8724981fd180d1d5a4c3047`.
- Project page: https://parsa-ra.github.io/lmbehave/, with source in
  https://github.com/parsa-ra/lmbehave-page.
- Artifact provenance and release: `docs/ARTIFACTS.md` maps every `results/` directory to the
  verified raw-artifact bundle (`persistent-data.tar.zst`, SHA-256 `160ab6cb…`, released in the
  Google Drive folder `lmbehave-release`) and gives restore and recomputation commands.
- Implemented streams at every selected decoder layer:
  `attention_delta`, `mlp_delta`, and `residual_state`.
- Implemented quantitative analysis: a PCA-space mean-shift trace ratio (retained as
  `FDR_trace` in artifact schemas), effective and stable rank, separate mean-shift and
  within-group terms, task-held-out linear and RBF probes,
  task-label permutation tests, Benjamini–Hochberg q-values, paired linear CKA, unpaired
  covariance alignment, and cosine silhouette.
- The current inference revision adds task-label permutation calibration for mean-shift ratios,
  selection-aware maximum statistics, null-calibrated rejection counts, and persisted
  sample-free null score vectors. For a balanced five-versus-five task design, all
  class-symmetric statistics have 126 complementary equivalence classes and minimum
  attainable exact p-value `1/126`, not `1/252`.
- The primary five-model panel uses 16 MMLU subjects and 100 examples per subject. Exact
  mean-shift-maximum tests reject at `p<0.01` in Apertus 1.5 and both Qwen sizes, while the
  selected-site task-held-out representation probe rejects in all five models. These are
  tests of the named subject partition, not evidence of a shared computation type.
- Prompt text predicts the same primary partition under the identical task holdout and all
  12,870 balanced assignments: word 1--2-gram TF--IDF gives macro-F1 `0.701`
  (`p=0.00870`), and character-within-word 3--5-grams give `0.768`
  (`p=0.000777`). Representation rejection therefore does not by itself establish signal
  beyond prompt surface, and the paper does not attribute it to knowledge or reasoning.
- A matched 64-feature analysis now puts representation and character-TF--IDF prompt probes
  under identical folds, scaling, cuML logistic regression, regularization, and all 12,870
  task assignments, with the activation site reselected for every assignment. Representation
  macro-F1 exceeds prompt macro-F1 by `.017--.070`, but none of the five paired differences
  is unusual under the complete task-label null (`p=.177--.440`). This is a failure to detect
  excess under the named estimator, not an equivalence result.
- The effective matched null has mean `.006--.017`, SD `.057--.067`, and 95th percentile
  `.108--.129`. Replacing activation PCA-64 by fold-local 95%-variance PCA gives
  representation-minus-prompt differences `.025--.050` with `p=.257--.466`. Thus the
  matched result is not created by forcing activations to 64 dimensions; the unequal-width
  comparison is a compression sensitivity, not capacity matching.
- On an arbitrary OLMo 8+8 split, deliberately invalid random-example validation gives
  macro-F1 `.802` with sample-label permutation `p=.0001`, whereas complete-task holdout
  gives `.574` with exact task-label `p=.228`. This demonstrates task-identity leakage and
  is not a reproduction of another paper's model or datasets.
- A post-hoc regularized-whitened selector grid is complete: all 15 model-by-ridge maxima
  reject (`p=.00062--.00233`, all Holm-adjusted `p<=.00933` across the grid), but none
  selects the unwhitened trace-ratio site. This
  supports raw recoverability while showing that localization is selector-dependent; it is
  not a causal or cross-panel result.
- Exact fixed-16, fixed-256, and raw activation-width sensitivities and a selection-aware
  matched-delta injection calibration are running under a frozen post-review manifest on
  persistent storage. They are not paper results until their complete reports,
  complement-symmetry gates, and hashes pass.
- A separately frozen precision extension adds ten new injected directions per signal
  level. It was specified after the $d=.5$ parent result but before any $d=1/2/4$ result;
  these are simulation replicates, not model or training-seed replicates.
- Its summarizer validates every report and distribution hash, recomputes each exact
  p-value, and requires complement symmetry before reporting 20-direction detection
  fractions with Wilson intervals. The first precision-extension execution was cancelled
  with zero completed reports during GPU/CPU rebalancing; it contributes no evidence and
  can be resumed only as a complete rerun. Raw distributions remain on persistent storage.
- The OLMo primary-panel `mean_valid` pooling sensitivity is frozen in
  `experiments/representation_dynamics_main_track_mean_pooling_v1.json`. Its 16-task,
  1,600-example extraction passed exact sample-identity and behavior parity against the
  `last_valid` parent and produced 153,600 complete exact vectors. Exhaustive analysis is
  in progress; this is not yet a manuscript result.
- Fold-local residualization against 64 character TF--IDF components lowers selected-site
  macro-F1 from `.781--.805` to `.613--.698`. Exact p-values are `.01197--.06605`, so no
  model meets the declared `.01` criterion, although four remain below a nominal per-model
  `.05` threshold. The 32-component sensitivity also has no p-value below `.01`. This is
  non-robustness to a finite low-rank lexical control, not evidence of causal mediation or
  removal of all lexical information.
- The lexical transform retains `31--54%` of between-task variance, `57--76%` of total
  variance, and high 16-way task-identity accuracy at the selected sites. Its effect therefore
  cannot be summarized as deleting all task identity. Every one of the 12,870 permuted
  assignments also has balanced train and test task folds.
- A synthetic activation-only calibration at one fixed OLMo site detects 0/10 seeded
  directions at standardized mean differences `.5`, `1`, and `2`, and 5/10 at difference
  `4`, after the complete lexical-adjusted exact probe. This demonstrates that the
  instrument can return a positive result, but only at a large, direction-dependent scale.
  It is not model evidence and does not calibrate site selection. A post-hoc extension at
  differences `5/6/8` was terminated with zero completed reports when near-separable fits
  made the fixed solver prohibitively slow; it is excluded from every scientific table.
- Localization does not replicate. Only two of five models retain the same maximum site
  between the MMLU-Pro 5+5 and MMLU 8+8 panels, and none retains one across MMLU-Pro,
  MMLU, and the corrected cross-benchmark stress test.
- Historical MMLU 8+8 logs have model-dependent parse validity: `.992` for OLMo, `.573`
  for Apertus 1.5, `1.000` for both Qwen sizes, and `.761` for Gemma. Prefill states remain
  valid, but cross-model accuracy and correctness-stratum claims are not used. Future sample
  logs retain the model output before task filters under a separate `raw_resps` field while
  preserving the established filtered fields.
- The primary cross-model comparison uses only current open-weight families: OLMo 3 7B
  Instruct, Apertus 1.5 8B, Qwen 3.5 9B/27B, and Gemma 4 31B. Older Gemma, Qwen, and
  OLMo generations are not evidence in the main manuscript.
- An appendix-only multimodal side panel uses Qwen 3.5 9B/27B and Gemma 4 31B on ten
  heterogeneous tasks. Raw task-held-out F1 is `.858--.899`, but neither the raw probes nor
  selection-aware Fisher maxima meet the declared `.01` criterion. Surface/modality and
  lexical controls sharply reduce the scores. Because image presence, domain, task family,
  response format, and prompt content are confounded, this is evidence only about those ten
  named task labels and remains separate from all primary tables. The supplement also
  reports RBF F1, cosine silhouette, covariance alignment, and rank at the selected sites.
  Paired CKA remains missing because the task groups have no declared example pairs.
- Apertus 1.5 8B is a completed endpoint anchor and a prospective second longitudinal
  family. Three disjoint 32-example/task blocks produced 92,160 validated vectors and 288
  failure-free analysis rows. Corrected linear-probe evidence repeated at 87/96
  layer--stream sites. The blocks measure example-sampling sensitivity, not training-seed
  variation, and the public repository still has no intermediate checkpoint refs. This is
  repeated endpoint recoverability, not evidence about change during training.
- Apertus 1.0 8B has a separate, completed pretraining trajectory: the official repository
  exposes token-indexed checkpoints from 210B through 15T tokens. An 11-checkpoint protocol
  samples within and around the four documented 8B data stages. Exact observer parity now
  passes at 210B, 7.047T, and 15T tokens: monitoring leaves logits and greedy tokens
  bit-identical, captures 384/384 finite exact vectors per checkpoint, and removes all 96
  hooks. The released-model ARC anchor gives 72.29% pooled raw accuracy versus the report's
  72.7%; because the report does not state the pooling rule and the values are not identical,
  this is a loading/scoring check rather than an exact reproduction. Across 11 checkpoints
  and three paired blocks, accuracy rises and endpoint Fisher ratios rise in all streams,
  opposite to OLMo's endpoint direction; neither path is monotonic. Long-context,
  SFT/QRPO, and Apertus 1.5 are not appended to this pretraining curve.
- A separate text-only parity check passed all 32 layers and three exact streams (384
  finite vectors, zero skips), observer-off/on response and score parity, and bit-identical
  forward logits under the recorded Python xIELU fallback.
- The 100-example/task extraction produced 100 artifacts and 1,320,000 float32 vectors.
  A later prompt-protocol audit found that all 500 MuSR prompts across five models had
  split the dataset's string-encoded two-choice list into characters. The 50 matched
  MMLU-Pro artifacts remain valid. Corrected MuSR extraction passed for all five models,
  and the complete cross-benchmark panel was reanalyzed. The validator now runs a
  task-aware prompt audit so this failure cannot pass silently again.
- In the MMLU-Pro 5+5 panel, corrected linear/RBF probe-F1 evidence appears at 81/88 of
  96 OLMo rows, 70/88 of 96 Apertus 1.5 rows, 0/77 of 96 Qwen 9B rows, 92/154 of 192
  Qwen 27B rows, and 88/0 of 180 Gemma 4 rows. After a label-blind Ridge map removes the
  variance predicted from six prompt-surface features inside each held-out-task fold, all
  five conditions have 0/0 corrected rows. The matched subject-set result is therefore not
  robust to the recorded prompt variables; residualization does not identify their causal
  role or remove unmeasured confounds.
- Those historical row counts are structurally coarse: at the corrected `1/126` p-value
  floor, BH at 0.05 cannot reject fewer than 16/96, 29/180, or 31/192 rows. The completed
  selection-aware Fisher reanalysis gives maximum-statistic p-values `0.0159`--`0.0317`
  across all five models, so none meets the prespecified `0.01` threshold. The alphabetical
  negative-control split is unremarkable (`p=0.397`--`0.571`, zero rejected rows).
- In the distinct MMLU-Pro 5+5 panel, the corrected scalar prompt-length diagnostic has
  task-level macro-F1 permutation p-values from `0.167` to `0.270`. Those values do not
  describe MMLU 8+8, whose knowledge/reasoning prompts differ by roughly 43--45 model
  tokens on average. Primary-panel length-only adjustment retains probe rejection in all
  five models. Six-feature adjustment retains it in four; Gemma gives macro-F1 `0.732`,
  `p=0.0137`. Scalar residualization does not resolve the character-level prompt baseline.
- Repeated cuML RBF fits are not bitwise reproducible. The strict replay audit records the
  differing fields; the paper treats threshold-adjacent RBF decisions as numerically
  fragile rather than as a qualitative linear--nonlinear result.
- The corrected cross-benchmark panel has zero unadjusted linear/RBF rejections for every
  current model. Prompt adjustment creates 66/96 linear rejections only for Apertus 1.5;
  this demonstrates that residualization can introduce a contrast and is not a construct
  identification procedure.
- Gemma and post-fix Qwen tracker-off/on output parity each passed a two-example
  deterministic check. Logit-level parity, broader output parity, and observer overhead
  measurement remain required before publication.
- RAPIDS/cuML 26.06 is an explicit, separately versioned analysis backend. It uses
  float64 GPU analysis of float32 stored vectors, full PCA followed by an explicit 95%
  cumulative-variance cutoff, and PCA-space cuML QN logistic/RBF probes. Covariance and
  silhouette remain in the original activation space on GPU. It is
  not numerically interchangeable with the earlier scikit-learn/liblinear analysis. The
  ten current model--panel conditions were therefore analyzed uniformly with fold-local
  PCA; historical estimator families and older model tables are not pooled.
- The complete current launcher telemetry recorded 364.9 W mean power, 612.6 W peak power,
  100% peak utilization, and 65,082 MiB peak memory. The frozen batches therefore fit the
  intended 80 GB H100 envelope on this run.
- A local-only engineering run completed the matched text MMLU control for both models:
  five task groups per working class, two examples/task, matching document IDs, 20/20
  valid artifacts, exact hooks, finite float32 vectors, and zero analysis failures. Its
  adjusted probe results are not significant and its sample size is deliberately too
  small for scientific interpretation. No part of this debug run was sent to W&B.
- A second local two-example engineering run covered all ten heterogeneous tasks for both
  models after standardizing MedQA, GPQA, and MMMU direct-answer generation. All 20 paired
  artifacts passed and both analyses had zero failed rows. The observed maxima changed
  with this protocol change and no q-value was below 0.05, reinforcing that the tiny run
  is a pipeline check rather than evidence. It was not sent to W&B.
- Debug and parity runs remained local. The current-model study uploaded one aggregate-only
  W&B report ([`byibgfcd`](https://wandb.ai/parano/vlmbehave-representation-dynamics/runs/byibgfcd))
  with 0 media and 0 artifact files; raw activations, prompts, responses, and per-sample
  tables were not uploaded.
  Earlier validated scalar-only summaries are on W&B with
  zero media or artifact uploads: the 32-example diagnostic
  ([`1ywp6db3`](https://wandb.ai/implicit-pg/vlmbehave-representation-dynamics/runs/1ywp6db3)),
  4B/9B full run
  ([`96g5fep0`](https://wandb.ai/implicit-pg/vlmbehave-representation-dynamics/runs/96g5fep0)),
  and 12B/27B scaling extension
  ([`5ax24h0s`](https://wandb.ai/implicit-pg/vlmbehave-representation-dynamics/runs/5ax24h0s)).

No current result establishes a general knowledge-versus-reasoning distinction or a causal
pruning rule. The strongest result is instead that substantial passive separability can
co-occur with prompt-only predictability and remain model-specific and unstable across
panels or training checkpoints.

## Scientific position and roadmap

The current system is a strong measurement and exploratory task-set-separation pipeline.
It is not yet a construct-valid test of a universal knowledge-versus-reasoning mechanism:
the effective label diversity is ten tasks, the heterogeneous panel is confounded by
modality/domain/format, and the matched MMLU panel remains confounded by subject and
vocabulary. Final-prompt prefill vectors can also encode prompt identity or difficulty
rather than reasoning dynamics.

The claim sequence is intentionally gated: validate passive measurement; establish
held-out-task predictability; survive construct and nuisance controls; replicate a
layer/branch profile; intervene causally; then measure quality--efficiency trade-offs.
Stages beyond held-out-task predictability are not complete. The maintained assessment,
suitability table, inference status, and staged follow-up plan are in
[`docs/SCIENTIFIC_STATUS.md`](docs/SCIENTIFIC_STATUS.md).

Effective rank is a diagnostic, not a replacement score. Dividing the Fisher ratio by
effective rank would make a low-rank layer's score larger, so the analysis reports rank,
the distance between group means, and within-group spread separately. Sparse
autoencoders are also outside the current primary study: using a token-level encoder for
one group and a step-level encoder for the other would apply different measurements to the
two labels. A later study would need one label-free encoder trained on separate data and
applied identically, or a separate process-trace design with explicit reasoning steps.

## Repository map

```text
.
├── README.md                         Project entry point and operational commands
├── AGENTS.md                         Detailed scientific and development handoff
├── lmms-eval/                        Bundled evaluation source; tracker and analysis code
│   ├── lmms_eval/tasks/representation_dynamics_mmlu/
│                                      Concise, template-matched MMLU control tasks
│   ├── lmms_eval/tasks/representation_dynamics_text/
│   │                                  Direct-answer text expansion tasks
│   └── lmms_eval/tasks/representation_dynamics_controls/
│                                      Direct-answer MedQA, GPQA, and MMMU study adapters
├── bench_gemma.sh                    Gemma 3 pilot Slurm launcher
├── bench_qwen.sh                     Qwen 3.5 pilot Slurm launcher
├── experiments/
│   ├── representation_dynamics_v1.json
│   │                                  Frozen two-panel experiment definition
│   ├── representation_dynamics_text_expansion_v1.json
│   │                                  Frozen, completed current-model text study
│   ├── representation_dynamics_main_track_v1.json
│   │                                  Frozen 8+8-task selection-aware confirmation
│   ├── representation_dynamics_main_track_controls_v1.json
│   │                                  Post-review prompt baseline and adjustment sensitivities
│   ├── representation_dynamics_main_track_lexical_control_v1.json
│   │                                  Fold-local character-TF--IDF activation sensitivity
│   ├── representation_dynamics_main_track_instrument_validation_v1.json
│   │                                  Matched prompt comparison and detection calibration
│   ├── representation_dynamics_main_track_review_response_v1.json
│   │                                  Post-review width, null, and inferential-unit checks
│   ├── representation_dynamics_selector_sensitivity_v1.json
│   │                                  Post-review regularized-whitened selector grid
│   ├── representation_dynamics_multimodal_side_v1.json
│   │                                  Appendix-only heterogeneous multimodal reanalysis
│   ├── representation_dynamics_pooling_sensitivity_v1.json
│   │                                  Direct-answer prefill/decode sensitivity; smoke-bounded
│   ├── olmo3_longitudinal_v1.json     Frozen seven-checkpoint repeated-block trajectory
│   ├── apertus1_longitudinal_v1.json  Released 11-checkpoint pretraining protocol
│   ├── apertus15_longitudinal_v1.json Prospective trajectory; blocked on checkpoint release
│   ├── apertus15_endpoint_anchor_v1.json
│   │                                  Executed three-block final-endpoint protocol
│   └── representation_dynamics_text_expansion_apertus15_v1.json
│                                      Executed Apertus 1.5 current-panel extension
├── results/olmo3_longitudinal_v1/     Sample-free longitudinal aggregate CSVs and hashes
├── results/apertus15_endpoint_anchor_v1/
│                                      Sample-free endpoint means/std and repeatability
├── results/apertus1_preflight_v1/     Generated preflight summary path; absent from Git
├── results/response_trajectory_smoke_v1/
│                                      Sample-free parity and response-step coverage
├── results/representation_dynamics_text_expansion_v1/
│                                      Sample-free current-model linear/nonlinear summaries
├── results/representation_dynamics_main_track_v1/
│                                      Hash-bound 8+8-task Fisher confirmation
├── results/representation_dynamics_main_track_correctness_support_v1/
│                                      Sample-free correct/incorrect support audit
├── results/representation_dynamics_main_track_controls_v1/
│                                      Prompt, parser, additivity, adjustment, and GPU summaries
├── results/representation_dynamics_main_track_lexical_control_v1/
│                                      Exact lexical-adjusted probe summaries and hashes
├── results/representation_dynamics_main_track_instrument_validation_v1/
│                                      Matched, diagnostic, and calibration summaries
├── results/representation_dynamics_main_track_review_response_v1/
│                                      Hash-bound null, PCA-width, and inference-unit checks
├── results/representation_dynamics_multimodal_side_v1/
│                                      Appendix-only multimodal controls and exact inference
├── docs/
│   ├── REPOSITORY_AUDIT.md           Local-state and threat audit
│   ├── REPRODUCIBILITY.md            Migration and publication run contract
│   └── SCIENTIFIC_STATUS.md          Claim boundary, current stage, risks, and roadmap
├── scripts/
│   ├── bootstrap_workspace.py        Recreate pinned external repositories
│   ├── check_private_ready.py        Secret, size, and packaging checks
│   ├── validate_representation_artifact.py
│   │                                    Pre-analysis completeness/finite-value gate
│   ├── run_representation_experiment.py
│   │                                    Plan, run, pair-validate, and analyze the manifest
│   ├── run_olmo_longitudinal_experiment.py
│   │                                    OLMo checkpoint/repeated-block runner and summary
│   ├── check_apertus15_release.py      Read-only checkpoint-availability audit
│   ├── check_apertus1_release.py       Frozen Apertus 1.0 ref/config audit
│   ├── run_apertus1_observer_parity.py Exact off/on/removed observer gate
│   ├── summarize_apertus1_preflight.py Sample-free preflight validator
│   ├── run_full_apertus1_longitudinal.sh
│   │                                    Persistent-volume extraction/analysis launcher
│   ├── run_selected_site_probe_panel.sh  Exact five-model Fisher-selected-site probe
│   ├── run_pending_main_track_experiments.sh
│   │                                    Gate selected-site results, then trajectory/Apertus analyses
│   ├── run_main_track_controls.sh        Exact 8+8 prompt/control launcher under GPU MPS
│   ├── summarize_main_track_controls.py  Hash-bound sample-free control summaries
│   ├── run_main_track_lexical_control.sh Exact five-model lexical-adjusted probe launcher
│   ├── summarize_main_track_lexical_control.py
│   │                                    Hash-bound lexical-control summary
│   ├── run_main_track_instrument_validation.sh
│   │                                    Exact matched/diagnostic/calibration launcher
│   ├── summarize_main_track_instrument_validation.py
│   │                                    Hash-bound instrument-validation summary
│   ├── run_main_track_review_response.sh Exact activation-width sensitivity launcher
│   ├── run_matched_excess_reselection_calibration.sh
│   │                                    Selection-aware matched-delta positive control
│   ├── run_whitened_selector_sensitivity.sh
│   │                                    Regularized-whitened localization sensitivity
│   ├── summarize_main_track_review_response.py
│   │                                    Hash-gated post-review aggregate summary
│   ├── summarize_matched_delta_calibration_precision.py
│   │                                    Hash-gated 20-direction calibration summary
│   ├── run_full_representation_experiment.sh
│   │                                    Resumable one-GH200 full-run launcher and gates
│   ├── setup_machine.sh              SSH, tools, shell, repositories, Python environment
│   │                                Commit reviewed changes, then ordered push
│   ├── snapshot_overlays.sh          Refresh non-submodule prototype patches
│   └── plot_capability_knowledge.py  Legacy aggregate-monitor plots; exploratory only
├── repos.lock.json                   Exact external repository revisions and overlays
└── overlays/                         Archived or non-primary deterministic patches
```

`transformers/`, `transformers_qwenvl/`, `vllm/`, `vLLM-Hook/`, `IVC-Prune/`, `logs/`, and
`wandb/` are intentionally not tracked by the root repository. Their roles and pinned
revisions are documented in `AGENTS.md` and `repos.lock.json`.

## Recreate the workspace on a compute server

Copy an SSH private key that is already registered with GitHub onto the new machine, keep
it outside the repository, verify GitHub's host fingerprint at the first connection, and
clone once with that key:

```bash
chmod 600 "$HOME/.ssh/id_ed25519"
GIT_SSH_COMMAND="ssh -i $HOME/.ssh/id_ed25519 -o IdentitiesOnly=yes" \
  git clone https://github.com/parsa-ra/lmbehave.git
cd lmbehave

# Configure SSH, Homebrew, Git/GitHub CLI, uv, dust, fzf, rg, and fd;
# persist the shell environment; update repositories; and install Python packages.
./scripts/setup_machine.sh "$HOME/.ssh/id_ed25519"
```

The setup command writes a small shell fragment under `~/.config/vlmbehave/` and sources it
from `~/.zshrc` or `~/.bashrc`. New shells receive Homebrew, fzf integration,
`GIT_SSH_COMMAND`, `VLM_WORKSPACE`, and the `vlm_activate` helper. The private key itself is
never copied into the repository.

Linux Homebrew requires the distribution's compiler/bootstrap packages first. On
Debian/Ubuntu this normally means `build-essential`, `procps`, `curl`, `file`, and `git`;
the script fails with a diagnostic instead of invoking `sudo`. On managed clusters, install
on the persistent login/development node or use `--skip-homebrew` when equivalent tools are
already provided.

Git push/pull uses SSH and requires no PAT. `install.sh` creates a Python 3.12 environment
and uses the intentionally simple command
`uv pip install torch torchvision --torch-backend auto`; `uv` selects the compatible
CPU/CUDA build. The script prints the resolved PyTorch version, CUDA runtime, CUDA
availability, and device count. The PyTorch wheel does not install the NVIDIA kernel
driver or a full CUDA compiler toolkit.

Useful partial/repeated setup commands are:

```bash
# Recheck/reconstruct pinned repositories without overwriting mismatched worktrees.
python3 scripts/bootstrap_workspace.py

# Reinstall or verify the environment separately; uv reuses .venv.
./install.sh
```

The evaluation implementation is already included in the public checkout.

Do not install `transformers_qwenvl` for the primary representation study. It is an
unvalidated pruning prototype and must be selected explicitly with
`TRANSFORMERS_REPO=transformers_qwenvl ./install.sh` only for a separately designed
intervention.

## Collect representations

The manifest runner is the publication path. The main-track manifest defines a 16-subject
MMLU comparison matched on dataset, prompt template, answer format, and generation path;
the earlier heterogeneous panel remains a stress test. The runner launches one task per process, forces W&B off,
locks source/environment/package state, refuses dirty final trees, verifies that both
evaluation and representation outputs exist after every task, and records exact artifact
and result hashes. Validation also audits logged multiple-choice prompts against task-aware
choice-count and target rules. The cross-benchmark panel uses project-specific direct-answer adapters
for MedQA, GPQA, and MMMU; these retain source datasets and scoring but are not standard
leaderboard protocols. The MMLU control dataset is pinned at revision
`b2e1ec9aa795adafe68e8e983248dbd4b52a1c60`; its tasks are adapted for representation collection and emit
only A/B/C/D; their scores must not be described as standard MMLU protocol results.

```bash
# Confirmatory 8+8 design. Its primary analysis enumerates all 12,870 balanced
# assignments for Fisher inference without refitting a probe at every layer.
MAIN_TRACK=experiments/representation_dynamics_main_track_v1.json
MAIN_RUN=logs/representation-dynamics-main-track-v1
.venv/bin/python scripts/run_representation_experiment.py \
  --manifest "$MAIN_TRACK" plan
.venv/bin/python scripts/run_representation_experiment.py \
  --manifest "$MAIN_TRACK" check-tasks
.venv/bin/python scripts/run_representation_experiment.py \
  --manifest "$MAIN_TRACK" freeze-samples --run-dir "$MAIN_RUN"
.venv/bin/python scripts/run_representation_experiment.py \
  --manifest "$MAIN_TRACK" run --run-dir "$MAIN_RUN"
.venv/bin/python scripts/run_representation_experiment.py \
  --manifest "$MAIN_TRACK" validate --run-dir "$MAIN_RUN"
.venv/bin/python scripts/run_representation_experiment.py \
  --manifest "$MAIN_TRACK" analyze --run-dir "$MAIN_RUN" \
  --analysis-backend cuml --analysis-name primary-fisher-exhaustive \
  --jobs 1 --parallel-backend process

# Inspect the design and verify every task is registered.
.venv/bin/python scripts/run_representation_experiment.py plan
.venv/bin/python scripts/run_representation_experiment.py check-tasks
.venv/bin/python scripts/run_representation_experiment.py check-models

# Inspect the completed current-model text-only study.
.venv/bin/python scripts/run_representation_experiment.py \
  --manifest experiments/representation_dynamics_text_expansion_v1.json plan
.venv/bin/python scripts/run_representation_experiment.py \
  --manifest experiments/representation_dynamics_text_expansion_v1.json check-tasks
.venv/bin/python scripts/run_representation_experiment.py \
  --manifest experiments/representation_dynamics_text_expansion_v1.json check-models

# A clean publication run freezes all 18 unique task selections before loading a model.
.venv/bin/python scripts/run_representation_experiment.py \
  --manifest experiments/representation_dynamics_text_expansion_v1.json freeze-samples \
  --run-dir logs/representation-dynamics-text-expansion-v1

# Local engineering check. --allow-dirty and --limit are debug-only.
bash -ic '.venv/bin/python scripts/run_representation_experiment.py run \
  --panel matched_text_mmlu_control --run-dir logs/repdyn-debug \
  --limit 2 --allow-dirty'
.venv/bin/python scripts/run_representation_experiment.py validate \
  --panel matched_text_mmlu_control --run-dir logs/repdyn-debug
.venv/bin/python scripts/run_representation_experiment.py \
  --python .venv/bin/python analyze \
  --panel matched_text_mmlu_control --run-dir logs/repdyn-debug \
  --analysis-backend cuml --analysis-name analysis-cuml-26.06 \
  --jobs 1 --parallel-backend process

# Historical six-condition run; retained for repository history, not the main paper.
bash -ic './scripts/run_full_representation_experiment.sh --dry-run'

# Launch the historical manifest, validation, analyses, and a local aggregate preview.
# Rerunning this command safely skips complete extraction task pairs.
bash -ic './scripts/run_full_representation_experiment.sh'

# Reproduce the current-generation text study in a new run directory.
bash -ic './scripts/run_full_representation_experiment.sh \
  --manifest experiments/representation_dynamics_text_expansion_v1.json \
  --run-dir logs/representation-dynamics-text-expansion-v1-v2'

# Exact nonlinear inference is a separately named sensitivity analysis. This
# preserves the frozen extraction manifest while recording the effective 251
# whole-task label permutations in every analysis manifest.
bash -ic './scripts/run_full_representation_experiment.sh \
  --manifest experiments/representation_dynamics_text_expansion_v1.json \
  --run-dir logs/representation-dynamics-text-expansion-v1-v2 \
  --phase analyze --analysis-name analysis-cuml-26.06-nonlinear-251 \
  --nonlinear-permutations 251'

# Regenerate the committed sample-free tables from both hash-verified analyses.
# Do this only after the task-aware audit passes; the archived v1 cross panel does not.
.venv/bin/python scripts/audit_text_protocol_samples.py \
  logs/representation-dynamics-text-expansion-v1-v2 \
  --output logs/representation-dynamics-text-expansion-v1-v2/protocol_audit.json
.venv/bin/python scripts/summarize_representation_experiment.py \
  logs/representation-dynamics-text-expansion-v1-v2 \
  --analysis-name analysis-cuml-26.06 \
  --nonlinear-analysis-name analysis-cuml-26.06-nonlinear-251 \
  --output-dir results/representation_dynamics_text_expansion_v1

# Reconstruct the prompt-only nuisance table from an extracted current-study run.
# The CSV contains numeric surface features only; it excludes prompts and responses.
./scripts/build_representation_nuisance_table.py \
  logs/representation-dynamics-text-expansion-v1-v2 \
  --output logs/representation-dynamics-text-expansion-v1-v2/nuisance_features.csv

# Run the hash-bound fold-local Ridge sensitivity analysis. Until corrected MuSR
# extraction is complete, the runner and summarizer default to the valid matched panel.
# The launcher owns a
# private NVIDIA MPS service, uses two workers per model, records GPU telemetry,
# and resumes only when all input hashes and estimator settings match.
uv venv --python 3.12 .venv-analysis
uv pip install --python .venv-analysis/bin/python \
  --extra-index-url https://pypi.nvidia.com 'cuml-cu13==26.6.0' 'torch==2.9.1'
./scripts/run_full_nuisance_control.sh \
  logs/representation-dynamics-text-expansion-v1-v2 \
  logs/representation-dynamics-text-expansion-v1-v2/nuisance_features.csv \
  logs/nuisance-control-v1

# Before interpreting adjusted probes, require exact agreement with the archived
# non-adjusted analysis (apart from the separately validated silhouette replacement).
./scripts/verify_nuisance_control_baseline.py \
  logs/nuisance-control-v1 logs/restored-current-study-reference \
  --output logs/nuisance-control-v1/baseline_verification.json

# Emit only aggregate, sample-free tables for the repository and manuscript.
./scripts/summarize_nuisance_control.py logs/nuisance-control-v1 \
  --nuisance-table logs/representation-dynamics-text-expansion-v1-v2/nuisance_features.csv \
  --output-dir results/representation_dynamics_nuisance_control_v1

# Reproduce the appendix-only heterogeneous multimodal reanalysis from the
# SHA-verified persistent archive. This does not run or modify the primary panel.
./scripts/run_multimodal_side_analysis.sh
.venv/bin/python scripts/summarize_multimodal_side_analysis.py \
  /home/ubuntu/data-imu-us-east3/vlmbehave-artifacts/2026-08-16/representation-dynamics-multimodal-side-v1/analysis \
  results/representation_dynamics_multimodal_side_v1 \
  --expected-head 0b69f69e455a00b748d799a517939a033338bdd8

# Optional only after reviewing logs/.../wandb_aggregate_preview.json. There is no
# implicit W&B project: pass the intended existing project explicitly.
bash -ic './scripts/run_full_representation_experiment.sh \
  --phase report --upload-wandb --wandb-project YOUR_PRIVATE_PROJECT'
```

The validated scaling extraction is locked under
`logs/representation-dynamics-v1-scaling-extension`. Its run manifest predates the current
documentation commit, so do not attempt to resume extraction after changing source state.
Analyze its already validated artifacts with explicit selectors:

```bash
.venv/bin/python scripts/run_representation_experiment.py \
  --manifest experiments/representation_dynamics_v1.json \
  --python .venv/bin/python analyze \
  --run-dir logs/representation-dynamics-v1-scaling-extension \
  --model gemma3-12b --model gemma3-27b --model qwen3_5-27b \
  --analysis-backend cuml --analysis-name analysis-cuml-26.06 \
  --jobs 1 --parallel-backend process
```

The completed v1 manifest uses 100 examples/task, not 512. This is the largest exact common count
supported by every pinned matched-MMLU test split: `abstract_algebra` and
`college_mathematics` each contain exactly 100 examples. Requesting 512 would make the
nominally matched panel impossible and fail exact-count validation. The launcher rechecks
all selected task capacities and freezes their document identities before starting.

The separate OLMo 3 longitudinal study is frozen in
`experiments/olmo3_longitudinal_v1.json`. Its default tier follows seven 7B base-training
checkpoints. Three disjoint blocks of 32 examples per task (offsets 0, 32, and 64) are
paired across checkpoints. Their sample standard deviation measures example-sampling
sensitivity; it is not variance across independently trained models. All ten tasks run
under one model load per checkpoint/block.

```bash
# Inspect the default seven-checkpoint, three-block plan.
.venv/bin/python scripts/run_olmo_longitudinal_experiment.py plan

# Quality smoke: one real checkpoint and one 32-example/task block, W&B disabled.
.venv/bin/python scripts/run_olmo_longitudinal_experiment.py run \
  --run-dir logs/olmo3-longitudinal-v1 \
  --checkpoint base-stage1-step10000 --replicate block-0

# Full default base trajectory. Reruns skip only complete conditions.
.venv/bin/python scripts/run_olmo_longitudinal_experiment.py run \
  --run-dir logs/olmo3-longitudinal-v1 --skip-existing
.venv/bin/python scripts/run_olmo_longitudinal_experiment.py validate \
  --run-dir logs/olmo3-longitudinal-v1
.venv/bin/python scripts/run_olmo_longitudinal_experiment.py analyze \
  --run-dir logs/olmo3-longitudinal-v1 \
  --analysis-backend cuml --jobs 1
.venv/bin/python scripts/run_olmo_longitudinal_experiment.py summarize \
  --run-dir logs/olmo3-longitudinal-v1 --analysis-backend cuml

# Secondary tiers are opt-in and are not interpreted as one continuous trajectory.
.venv/bin/python scripts/run_olmo_longitudinal_experiment.py plan --tier mixture_control
.venv/bin/python scripts/run_olmo_longitudinal_experiment.py plan --tier posttraining
```

Final runs require clean root and `lmms-eval` trees. `--allow-dirty` is only for debugging.
The runner keeps W&B disabled and writes raw activations only below the selected local run
directory. Its two summary CSVs report layer-wise and checkpoint-wise mean/std, accuracy
by task group, every representation stream, rank diagnostics, and FDR velocity only within
a stage whose checkpoint counter has a shared unit. Stage counters reset, so cross-stage
velocity is left missing until exact cumulative token positions are available.

The primary trajectory is complete: 21/21 extraction conditions, 645,120 vectors, and
2,016/2,016 cuML analysis rows passed with zero failures. Sample-free aggregate outputs
are committed under `results/olmo3_longitudinal_v1/`; raw activations remain ignored under
`logs/olmo3-longitudinal-v1/`. Accuracy improves across the released trajectory, but the
metrics do not support one monotonic-divergence claim. Depth-mean FDR, cosine silhouette,
and RBF-probe F1 decrease overall, while linear-probe F1 and its repeatable task-level
significance strengthen after Stage 1. Effective-rank fractions rise overall, so the FDR
decrease is not a rank-collapse artifact. See `docs/SCIENTIFIC_STATUS.md` for exact values
and the claim boundary.

The Apertus 1.0 pretraining replication is frozen separately at 11 token-indexed
checkpoints. Verify the live refs and inspect the plan before downloading weights:

```bash
.venv/bin/python scripts/check_apertus1_release.py --require-ready
.venv/bin/python scripts/run_olmo_longitudinal_experiment.py \
  --manifest experiments/apertus1_longitudinal_v1.json plan
```

The same runner is used because the extraction and summary contract is model-agnostic;
the historical filename predates the Apertus protocol. The full plan is 33 model/block
loads and 1,013,760 vectors. Exact observer parity passes at the earliest, first Stage-3,
and final checkpoints. A full released-model `ai2_arc` run gives 72.29% when raw accuracy
is pooled over ARC-Easy and ARC-Challenge, compared with 72.7% in the technical report.
The task YAML files match the report-time official harness, but the paper does not specify
the ARC aggregation and the scores are not identical; this is therefore a near anchor, not
an exact reproduction. Run the final persistent-volume study only from clean source:

```bash
./scripts/run_full_apertus1_longitudinal.sh
```

The launcher revalidates released refs and the preflight, keeps W&B disabled, logs GPU
power/utilization, and writes extraction and analysis directly below
`/home/ubuntu/data-imu-us-east3/vlmbehave-artifacts/2026-08-13/apertus1-longitudinal-v1/`.
All 33 checkpoint/block extractions and analyses are complete and validated. Sample-free
summaries are committed under `results/apertus1_longitudinal_v1/`; full artifacts remain on
the persistent volume.

The Apertus 1.5 endpoint anchor is complete under
`experiments/apertus15_endpoint_anchor_v1.json`; its sample-free three-block aggregates are
under `results/apertus15_endpoint_anchor_v1/`. This establishes repeated recoverability at
the final checkpoint only. The longitudinal extension remains release-gated. Recheck the
official repository without downloading model weights:

```bash
python scripts/check_apertus15_release.py
# Exit nonzero until enough candidate intermediate refs exist:
python scripts/check_apertus15_release.py --require-ready
```

Candidate names are not accepted automatically. Their training relationship, immutable
revisions, architecture, model code, and position unit must be audited before the manifest
is promoted to a runnable trajectory.

The validated text-only endpoint smoke is reproducible with:

```bash
./scripts/setup_apertus15_environment.sh
./scripts/run_apertus15_endpoint_smoke.sh logs/apertus15-endpoint-smoke-new
```

It runs the same four examples with monitoring enabled and disabled, validates all expected
layers/streams/counts/exactness, compares sample IDs, answers, targets, and scores, and keeps
W&B disabled. The isolated environment must use the pinned Swiss-AI Transformers commit;
do not install an incompatible TorchAudio wheel for this text-only path.

The final OLMo 3 7B base checkpoint also passed a protocol-matched standard-task check
using OLMES `arc:mc::xlarge` and its pinned Transformers 4.57.6: ARC-Easy was 93.573%
on 5,197 examples and ARC-Challenge was 84.749% on 2,590. Their macro average was
89.161%, which reproduces the model card's 89.2% after rounding. This validates
checkpoint loading and answer-choice likelihood scoring, not the representation
hypothesis. Exact protocol, ARM64 compatibility, and runtime-sensitivity details are in
`docs/SCIENTIFIC_STATUS.md`; the zero-shot `lmms-eval` ARC scores are a separate broad
sanity check and must not be compared directly with the model card.

Measured tracker-enabled DocVQA calibration on this 94.5-GiB GH200 froze batches Gemma 4B
=100, Qwen 9B=20, Gemma 12B=16, Gemma 27B=8, and Qwen 27B=8. The original Qwen 9B batch
24 passed at 85.4 GiB but was rejected as too close to the limit; batch 32 OOMed. Against
the batch-4 quality smoke, generated 4B/9B responses were identical for all shared DocVQA
examples, but pooled vectors were not bit-identical. Full-run activations and each batch
policy are therefore distinct frozen conditions and must not be silently merged.

The launcher requires clean root and submodule trees, the intended editable imports, one
visible GPU with at least 80 GiB free, at least 100 GiB free disk, an exported `HF_TOKEN`,
focused tracker/runner tests, registered tasks, and the common-support capacity check. Task
processes force W&B off. It
records `launcher.log` and five-second `gpu_telemetry.csv`, refuses concurrent launchers,
supports `--phase extract|validate|analyze|report`, and writes a scalar-only W&B preview
before any optional upload. Telemetry covers both extraction and GPU analysis and reports
mean/peak board power for the current invocation alongside peak SM utilization and memory.
The analysis launcher starts a private NVIDIA MPS service, runs up to 12 model/panel
conditions concurrently with two workers each, and removes only the MPS service it owns.

`run_manifest.json` binds the manifest, source fingerprints, full Python package inventory
and its SHA-256, import paths, hardware/runtime environment, and effective
sample count. `completed_manifest.json` binds validated artifact/result hashes and sample
ID sets. Each model/panel analysis writes `analysis_manifest.json`, which binds the
analysis source state, settings, inputs, and output hashes. Resuming after any source or
environment change is rejected rather than silently mixing conditions.

The two root launchers remain convenient legacy pilot Slurm entry points. Their account,
partition, GPU, email, and task settings must be reviewed on a new cluster; use the
manifest runner for the frozen study.

```bash
# Debug runs stay local; neither launcher sends task runs to W&B.
NUM_SAMPLES=2 sbatch bench_gemma.sh
NUM_SAMPLES=2 sbatch bench_qwen.sh

# Frozen pilot defaults: 512 documents/task, batch size 4, seed 0,
# deterministic decoding, pinned revisions, and float32 storage.
sbatch bench_gemma.sh
sbatch bench_qwen.sh
```

The essential opt-in flags are:

```text
--representation_dynamics
--representation_dynamics_output ./logs/representations-<model>/<category>-<task>/
--representation_dynamics_pooling last_valid
--representation_dynamics_phases prefill
--representation_dynamics_storage_dtype float32
```

Each run produces:

- `representations.pt`: tensor-bearing per-example records;
- `records.csv`: vector-free record manifest;
- `metadata.json`: extraction and provenance metadata.

Tracking is disabled when `--representation_dynamics` is absent. A successful process is
not sufficient validation: inspect selected decoder stack, layer specifications, exactness
flags, record counts, skipped counts, sample IDs, and revisions in every artifact.
Float32 is the default because real Gemma visual-task residuals exceeded float16's finite
range during validation. The tracker now fails on non-finite source values or storage
conversion instead of silently saving corrupt vectors.

Run the reusable artifact gate before analysis:

```bash
python scripts/validate_representation_artifact.py \
  /path/to/gemma/representations.pt /path/to/qwen/representations.pt \
  --expected-task repdyn_medqa=32 \
  --expected-task ok_vqa_val2014_lite=32 \
  --expected-task pope_random=32 \
  --expected-task textvqa_val_lite=32 \
  --require-exact
```

## Analyze representations

Run from `lmms-eval/` and pass exact task names:

```bash
python -m lmms_eval.representations.analysis \
  /path/to/run-a/representations.pt /path/to/run-b/representations.pt \
  --knowledge-tasks ok_vqa_val2014_lite,repdyn_medqa,repdyn_mmmu_val,ai2d_lite,repdyn_gpqa_main \
  --reasoning-tasks pope_random,mme,textvqa_val_lite,ocrbench_v2,docvqa_val_lite \
  --output-dir /path/to/separability-report \
  --cv-folds 5 --permutations 251 --seed 0 \
  --jobs 16 --parallel-backend process
```

Outputs are `layer_separability.csv`, `report.json`, and compressed
`permutation_nulls.json.gz`. The preregistered discovery criterion is the maximum per-model
PCA-space mean-shift trace ratio (stored as `FDR_trace` for compatibility). Its p-value recomputes the maximum over all declared rows for every balanced
task partition; the report also calibrates row-rejection counts against the same null.
Linear-probe significance is computed by permuting labels at the whole-task level. CKA is valid only for
explicit `pair_id` matches and is otherwise `NaN`; covariance alignment is the available
unpaired geometry descriptor. No t-SNE or UMAP result is accepted as primary evidence.
For exhaustive balanced linear probes, class-label complements share one deterministic fit
while both orientations remain in the reported null. RBF fits are not deduplicated because
complement symmetry is a numerical validity check for that estimator.

The corrected five-model matched-panel Fisher calibration reuses the hash-verified raw
activations and writes results directly to persistent storage:

```bash
.venv/bin/python scripts/run_calibrated_matched_reanalysis.py \
  --output-root /home/ubuntu/data-imu-us-east3/vlmbehave-artifacts/2026-08-13/calibrated-matched-panel-v1 \
  --resume
.venv/bin/python scripts/summarize_calibrated_matched_reanalysis.py \
  /home/ubuntu/data-imu-us-east3/vlmbehave-artifacts/2026-08-13/calibrated-matched-panel-v1
```

The runner verifies every input against its completion-manifest SHA-256 and excludes the
superseded Apertus 1.0 condition. It runs all 251 non-observed Fisher assignments but does
not rerun the historical row-wise linear/RBF permutation screens.

The main-track instrument validation compares representations with prompt text under one
64-feature cuML probe, audits what lexical adjustment removes, verifies grouped-fold balance
for every assignment, and calibrates the adjusted probe with injected directions. Raw and
null distributions stay on persistent storage; the summarizer writes only sample-free,
hash-bound tables:

```bash
PHASE=all PARALLEL_MODELS=5 CALIBRATION_PARALLEL=12 \
  ./scripts/run_main_track_instrument_validation.sh
.venv/bin/python scripts/summarize_main_track_instrument_validation.py \
  /home/ubuntu/data-imu-us-east3/vlmbehave-artifacts/2026-08-16/representation-dynamics-main-track-instrument-validation-v1 \
  results/representation_dynamics_main_track_instrument_validation_v1
```

The matched statistic is the exact paired macro-F1 difference, representation minus prompt,
with site reselection under every balanced assignment. Synthetic calibration is conditional
on one fixed OLMo site and does not establish a model mechanism. Calibration workers share
one digest-bound, atomically written cache of the 229 fold-local lexical transforms; set
`LEXICAL_CACHE_WORKERS` to control cache construction and `CALIBRATION_PARALLEL` to control
concurrent exact probes. Rebuilding the same lexical folds independently is intentionally
avoided because it leaves the GPU waiting on redundant CPU work.

The post-review analyses are frozen separately because the primary and matched-64 outcomes
were already known. They are resumable and write raw distributions only to the persistent
volume:

```bash
MAX_PARALLEL=10 ./scripts/run_main_track_review_response.sh
MAX_PARALLEL=10 ./scripts/run_matched_excess_reselection_calibration.sh
MAX_PARALLEL=5 ./scripts/run_whitened_selector_sensitivity.sh
.venv/bin/python scripts/summarize_whitened_selector_sensitivity.py \
  /home/ubuntu/data-imu-us-east3/vlmbehave-artifacts/2026-08-16/representation-dynamics-selector-sensitivity-v1 \
  results/representation_dynamics_selector_sensitivity_v1
.venv/bin/python scripts/summarize_main_track_review_response.py \
  --selected-probe-root /home/ubuntu/data-imu-us-east3/vlmbehave-artifacts/2026-08-15/selected-site-probe-main-track-v1 \
  --matched-root /home/ubuntu/data-imu-us-east3/vlmbehave-artifacts/2026-08-16/representation-dynamics-main-track-instrument-validation-v1/matched-excess \
  --review-artifact-root /home/ubuntu/data-imu-us-east3/vlmbehave-artifacts/2026-08-16/representation-dynamics-main-track-review-response-v1 \
  --output-dir results/representation_dynamics_main_track_review_response_v1
```

The width and whitened-selector arms intentionally do not replace the primary estimator.
They test whether its conclusions depend on activation compression or the unwhitened
localization rule; the completed selector grid changes every reported site. The injected-site
calibration reselects over all real sites plus the synthetic candidate for every assignment;
it is the positive control for the matched statistic, not evidence about the benchmark labels.

The response-trajectory acquisition uses `--representation_dynamics_phases both` and
`--representation_dynamics_decode_aggregation all_updates`. It emits an explicit
zero-based `decode_step`; capped absolute-position runs remain engineering controls.
Absolute step numbers are not comparable response stages when output lengths differ.

A real OLMo 3 smoke over two MMLU-Pro subjects and eight examples per subject passed
observer-off/on response and score parity, exact hook semantics, and artifact validation
for 3,072 vectors. Coverage was complete for prefill and `decode_step=0`, but no response
reached `decode_step=1` under the frozen direct-answer stop rules. Here `decode_step=0` is
the first cached one-token forward after the first answer token has been emitted; it predicts
the second token. The direct-answer protocol therefore supports a two-timepoint sensitivity,
not an eight-step reasoning trajectory. A longer explanation protocol must be frozen and
reported as a separate experiment rather than changing the primary prompts.

The separate long-response protocol and its 16 matched MMLU task adapters are frozen. It
requests an explicit final-answer marker followed by two short explanatory sentences
(at most 60 words), uses model-EOS stopping, and can retain generated updates. The
answer-first order makes this a post-answer response trajectory, not computation leading
to the choice. Run the two-task
observer-off/on gate before any panel extraction:

```bash
./scripts/run_response_trajectory_smoke.sh \
  /home/ubuntu/data-imu-us-east3/vlmbehave-artifacts/$(date -u +%F)/response-trajectory-longform-smoke-v1
```

The corrected OLMo smoke passed on 2026-08-15: all eight raw responses parsed, monitored
and unmonitored outputs matched, all 25,344 vectors were finite and exact, and both tasks
covered all 32 stored positions. The sample-free record is
`results/response_trajectory_long_response_smoke_v1/report.json`. Earlier failed attempts
remain on persistent storage: an implicit paragraph stop removed the answer marker, then
two algebra responses reached the earlier length cap. Neither is evidence.

After the smoke passes, run the full 8+8-task, 16-example-per-task extraction:

```bash
./scripts/run_full_response_trajectory_experiment.sh \
  /home/ubuntu/data-imu-us-east3/vlmbehave-artifacts/$(date -u +%F)/response-trajectory-olmo3-v1
```

The completed OLMo run passed parsing, observer parity, and exact finite capture for all
256 examples through its 32-position storage cap. That artifact is a valid absolute-prefix
control, but it cannot represent the final fraction of longer responses. Passing either
gate does not make token positions semantic reasoning steps.

After a full extraction passes, the trajectory analysis computes all layer--stream--position
Fisher rows, then reselects the maximum decode-minus-prefill change under every balanced
8+8 task assignment:

```bash
TRAJECTORY_ANALYSIS_JOBS=16 ./scripts/run_response_trajectory_analysis.sh \
  /home/ubuntu/data-imu-us-east3/vlmbehave-artifacts/2026-08-15/response-trajectory-olmo3-v3
```

The analysis launcher owns an isolated NVIDIA MPS service and therefore refuses to overlap
another MPS owner. `wait_and_run_response_trajectory_analysis.sh` queues it behind an active
extraction and MPS-backed study.

The normalized-position pilot instead uses uncapped, batch-one acquisition. For each
response with $T$ valid cached updates, it averages states in eight equal-width bins using
`min(7, floor(8*t/T))`. Batch size one is deliberate: in a batch, examples that reach EOS
early may continue through padded forward calls until the longest response stops. The
pilot uses 4+4 tasks and four examples per task, so its class-symmetric p-value floor is
`2/70=.0286`; it is descriptive and cannot satisfy the paper's `p<.01` rule.

```bash
RUN_RELATIVE_PILOT_ANALYSIS=0 ./scripts/run_relative_response_pilot.sh \
  /home/ubuntu/data-imu-us-east3/vlmbehave-artifacts/$(date -u +%F)/relative-response-pilot-olmo3-eager-v1
```

Raw and binned activations remain on persistent storage. After cuML analysis completes,
`scripts/summarize_relative_response_pilot.py` creates only hash-bound, sample-free
aggregates for Git and the manuscript. The first SDPA run failed strict raw-response
parity; the revised manifest freezes eager attention only after two independent
observer-off eager runs matched exactly. See `docs/WORKLOG.md` for the execution audit.

The eager pilot passed all gates. Its selected maximum occurs at layer 31's token-mixing
update in relative bin 5/8 (Fisher `.193 → .479`, change `.286`, exact `p=.0286`). This is
the attainable p-value floor for 4+4 tasks and does not meet the paper's `.01` threshold;
the result is descriptive. Sample-free aggregates are in
`results/representation_dynamics_relative_response_pilot_v2/`.

Debug and parity jobs stay local. After a quality smoke passes artifact validation and
analysis completes without failed rows, create one W&B run containing only aggregate task
metrics, layer/stream maxima, p/q-value bounds, source fingerprints, and artifact counts.
Do not upload raw activations, prompts, responses, sample tables, or credentials.

The 8+8-task primary extraction is complete for OLMo 3 7B, Apertus 1.5 8B, Qwen 3.5
9B/27B, and Gemma 4 31B: 80/80 model--task artifacts validated and all 660 Fisher rows
completed without failure. Exact selection-aware maximum p-values are `0.0721`, `0.00435`,
`0.00109`, `0.000777`, and `0.01445`, respectively. At the frozen `0.01` threshold,
Apertus and both Qwen models pass; OLMo and Gemma do not. Maxima occur at different sites,
and all arbitrary reference splits are non-significant, so this supports a model-dependent
property of the named MMLU subject partition rather than one shared knowledge--reasoning
location. See `results/representation_dynamics_main_track_v1/` for the sample-free table and
hash bindings.

The manifest runner never contacts W&B. An aggregate upload is a separate, explicit final
quality-smoke action; do not use it for the `--limit 2` engineering workflow above.

```bash
python scripts/log_quality_smoke_wandb.py /path/to/validated-smoke \
  --project vlmbehave-representation-dynamics \
  --name quality-smoke-aggregate
```

Use `--dry-run` first to inspect the exact JSON payload without contacting W&B. The helper
now requires clean extraction/analysis sources, at least 32 samples/task by default, bound
manifest/output hashes, the manifest runner's validation pass, one complete analysis for
every panel/model group, and zero analysis failures. It sets `save_code=False` and sends
scalar aggregates only.

For the common-support full run, add `--kind full-experiment`; this additionally requires
the run sample count to equal the frozen manifest count. Never rely on W&B's implicit
project selection: pass `--project` or use the full launcher's `--wandb-project` option.

## Extending to another 4B–31B model

Keep extraction in `lmms-eval`. A compatible model needs one discoverable text-Transformer
stack with indexed layers and direct attention/token-mixer and MLP children, plus a wrapper
that opens a per-batch representation context around `model.generate()`. Start with two
examples, require a non-empty artifact, inspect the selected stack and exactness flags, and
run tracker-off/on parity before a sweep. Unknown structural matches must remain
`residual_delta_exact=false`; do not claim branch-level results until residual-addition
semantics are verified from the installed implementation. Compare metrics within each
model, never raw vectors across hidden widths.

## Scientific constraints

- Compare representations within a model; do not pool raw hidden vectors across models or
  widths.
- Hold out whole tasks in probe validation. Random example splits primarily learn task
  identity and are not acceptable confirmatory evidence.
- Freeze exact document IDs, prompts, preprocessing, batch construction, generation
  settings, seeds, revisions, dtype, and backend.
- Match or explicitly model modality, domain, prompt/output length, correctness, visual
  token count, and decoding policy.
- Treat passive separability as descriptive. Pruning claims require causal interventions,
  matched budgets, random/established baselines, task quality, latency, and memory results.
- A p-value below 0.01 is evidence against the declared task-label null, not proof of a
  universal distinction. Report adjusted q-values across screened layers and streams.

The detailed working contract, file ownership, known confounds, validation gates, and
handoff checklist are in [`AGENTS.md`](AGENTS.md). The article is
available at https://arxiv.org/abs/2610.24029.

## Development workflow

All study code, including `lmms-eval/`, is tracked in this repository. Review `git status` and `git diff`, run focused tests and `python3 scripts/check_private_ready.py`, and commit reviewed changes. The private-workspace submodule synchronization helpers are not part of the public release.

The project page source lives in https://github.com/parsa-ra/lmbehave-page. After updating that repository, deploy it with:

```bash
gh workflow run pages.yml --repo parsa-ra/lmbehave
```

The Pages workflow also refreshes the site daily, preserving https://parsa-ra.github.io/lmbehave/.
