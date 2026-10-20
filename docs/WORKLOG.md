# Research Worklog

This append-only log records material protocol, code, execution, and manuscript changes.
It complements Git history: raw artifacts stay on persistent storage, while each entry
records paths, hashes, validation outcomes, and claim limits. Failed runs are retained and
identified rather than overwritten.

## 2026-08-24 — Formal manuscript and study-overview revision

- Consolidated the main manuscript into four top-level sections: Introduction, Problem
  Formulation and Method, Experiments, and Discussion and Conclusion.
- Recast the method with task-indexed notation for the grouped data, three observation
  streams, final-valid-token pooling, PCA-space mean-shift selector, grouped macro-F1,
  matched activation-minus-prompt statistic, exhaustive task-label null, localization
  transfer, and longitudinal depth summary. No implementation or experiment behavior changed.
- Stated that probe folds hold out tasks but panel-level site selection is transductive,
  distinguished the example-fold 16-way task-identity diagnostic, and limited the
  heterogeneous-panel probe results to descriptive evidence because their site selection
  was not repeated inside the test.
- Promoted the per-model matched-null and PCA-width results to the main body, added the
  exact retained task-identity range after lexical adjustment, and disclosed the completed
  selection-aware calibration in which the appended site was never selected.
- Rendered and inspected multiple compact, triptych, and two-row Figure 1 alternatives at
  final manuscript width. The dashboard-style revisions were not selected because they
  fragmented the measurement--inference--evidence flow; they remain under the candidate
  directory for audit.
- Restored the pre-existing native TikZ three-card design as the active figure. After
  comparing a lower-branch residual rail with an inline residual-block layout, widened the
  measurement card and selected the inline layout: every computation arrow begins at a
  state or module and terminates at a module or residual addition, while separate colored
  leaders connect the three observation sites to vector glyphs. Removed setup and estimator
  prose from the card, retained the task counts, same-model comparison, every-layer streams,
  and final-valid-token pooling, and preserved the protocol, reported numbers, result order,
  and noncausal scope. The active PDF, PNG, and SVG are generated from the same TikZ source.
- Replaced the inference card's unexplained color rows with a four-stage diagram: complete
  tasks are held out, category labels are reassigned over fixed task identities, the
  layer--stream maximum is reselected and the probe refit for each of 12,870 balanced
  assignments, and the observed maximum is ranked in the exact task-label null. Replaced
  the evidence card's prose rows with filled/hollow five-model glyphs, across-model score
  ranges, a schematic site-transfer rail, the CSV-backed training trajectories, and the
  paper's bounded takeaway. Vector-component tiles in the measurement card now use
  non-monotonic illustrative shading so they cannot be read as a depth or effect-size trend.
- Revised the abstract, page-one box, motivation, applications, and conclusion to state the
  question, empirical answer, and use earlier, while retaining the moderate-effect caveat
  and avoiding any claim that the task sets induce identical computation.
- Compiled the anonymous NeurIPS manuscript with Tectonic 0.16.9. The discussion and
  conclusion remain on page 9, and references begin on page 10; no undefined references,
  citations, or overfull boxes were reported.
- Corrected the Apertus 1.5 endpoint summary's stale class-symmetric permutation floor from
  $1/252$ to $2/252=1/126$ and recorded the 126 effective assignments. This metadata change
  does not alter any stored score, p-value, or manuscript conclusion.
- Removed the Apertus 1.0 ARC-anchor number from the manuscript because its generated
  sample-free preflight report is absent from the current Git checkout. The persistent
  provisional source and its checksum record remain documented for later regeneration.

## 2026-08-19 — Commit-and-sync convenience workflow

- Added `scripts/commit_and_sync_repositories.sh` for the explicitly authorized workflow
  in which all reviewed root and submodule changes should be committed and pushed together.
- The wrapper scans prospective tracked and non-ignored untracked files, rejects secret-like
  content and files above 95 MiB, checks conflicts and branch state, and preflights every
  push before creating commits.
- Dirty submodules are committed first; the root commit then records ordinary changes and
  updated gitlinks. Ordered pushing and exact remote-SHA verification remain delegated to
  `scripts/sync_repositories.sh`.
- Before its first commit, the mutating workflow invokes the established change-detected
  gzip tool-state backup, which retains the two newest archives on persistent storage. Dry-run
  mode reports this step without writing an archive.
- A dirty detached submodule is attached only when its configured local branch already
  points to the same commit. The dry-run on the current workspace identified the paper
  change, the three root documentation corrections, and this new script without modifying
  the index or remotes.

## 2026-08-17 — Scientific-language and coherence revision

- Revised the manuscript from the abstract through the statistical appendix to use
  transitive hypothesis-test language, leading-zero decimals, zero-indexed layers, and the
  consistent stream names `token-mixing update`, `feed-forward update`, and `residual state`.
- Renamed the primary descriptive quantity to the PCA-space mean-shift ratio
  $R_{\ell,s}$, while retaining `FDR_trace` once as a legacy artifact field. The text
  explicitly distinguishes this ratio from a whitened Fisher discriminant and from false
  discovery rate terminology.
- Removed project chronology and run-log vocabulary from the scientific narrative while
  preserving implementation, hook-invariance, parser-validity, and estimator limitations in
  the appendix. Named causal studies are used as motivation, not described as studies that
  the passive protocol reproduces or refutes.
- Reorganized transitions so the argument proceeds from measurement definitions to released
  training trajectories, current-checkpoint recoverability and prompt controls, sensitivity
  analyses, limitations, and conclusion. The conclusion states that the tested passive
  measures do not establish a transferable knowledge--reasoning computation and do not
  exclude moderate effects.
- Regenerated the study overview and all four empirical/synthetic paper figures so their
  labels use the same terminology as the text. Both deterministic figure checks pass.
- The anonymous NeurIPS build has 32 pages; all main-text content ends on page 9 and
  references begin on page 10. Tectonic reports no undefined citations/references or
  overfull boxes, and the paper passes `git diff --check`.

## 2026-08-17 — GPU/CPU rebalance and primary-panel mean-pooling extraction

- Diagnosed the 70-worker configuration as CPU submission-bound: 65 live compute clients
  occupied 64 logical CPUs while thousands of small synchronous cuML fits held reported SM
  utilization near 99% at only about 138--170 W. A reversible 25-client pause left power
  essentially unchanged, showing that more independent fits did not improve the workload
  shape.
- Suspended the five raw-width workers at their hash-bound checkpoints (50--250 of 12,870
  oriented assignments) and later suspended the five fixed-256 workers at 4,100--6,700
  assignments. No completed or checkpointed result was deleted; both arms remain resumable
  with unchanged configuration digests.
- Cancelled the optional 20-direction precision extension after it produced zero completed
  reports. Its 25 paused clients had no internal checkpoints and retained about 25 GB of
  CUDA contexts. The frozen extension manifest remains as provenance, but no result from
  that incomplete execution may be reported. The original ten-direction calibration
  remains active.
- Froze `experiments/representation_dynamics_main_track_mean_pooling_v1.json` before its
  outputs. Relative to the OLMo primary 8+8 MMLU condition, the only intended change is
  prompt pooling from `last_valid` to `mean_valid`; revision, tasks, prompts, batch size,
  generation, 100 examples/task, prefill-only capture, and float32 storage are unchanged.
- The new sample freeze matches all 16 parent task-wise sample-ID lists exactly. An
  eight-example tracker-off/on smoke matches document IDs, hashes, targets, arguments, raw
  and filtered responses, parsed correctness, and produces 768/768 valid vectors from 96
  exact hooks.
- Sharded the full extraction over two disjoint OLMo processes without changing per-process
  batch size. It completed all 1,600 examples and 153,600 vectors; the publication validator
  passes all 16 artifacts. Every recorded behavior field, including token counts, matches
  the `last_valid` parent for all examples. Two shards raised peak power to 348.9 W; after
  suspending fixed-256 jobs, live memory remained within the intended 80 GB envelope.
- Exhaustive mean-pooling layer/stream analysis is running with 12 cuML workers alongside
  30 original matched-delta calibration workers. A live sample reported 99% utilization,
  52,347 MiB VRAM, 19% memory-controller utilization, and 249.82 W. The analysis is pending
  and is not yet paper evidence.

## 2026-08-17 — Review-response analyses fully parallelized

- Preserved the ten active fixed-16/fixed-256 width workers and ten active $d=1$
  calibration workers, then stopped only their launcher parents so they could not enqueue
  duplicate future conditions.
- Started all twenty untouched $d=2/4$ selection-aware calibration conditions and all five
  raw-width conditions in separate process groups. The resulting 45 independent workers
  use non-overlapping output directories; existing width progress remains checkpointed and
  no estimator, fold, task assignment, or solver setting changed.
- After startup, the GH200 reported 99% SM utilization, 29% memory-controller utilization,
  `47,464 MiB` allocated VRAM, and `345.95 W`, compared with approximately `21.9 GiB` and
  `236 W` before the expansion. Host memory remained unconstrained (`336 GiB` available).
- Cleanup watchers retire the two stopped original launcher groups only after their complete
  40-calibration and 15-width report gates pass. The existing completion watcher then
  regenerates the hash-bound summary and reruns its test.
- Froze `experiments/representation_dynamics_main_track_calibration_precision_v1.json`
  before any parent $d=1/2/4$ report was available. The extension adds direction seeds
  10--19 at each injected scale; it is a post-hoc simulation-precision analysis, not new
  benchmark evidence. Twenty-five extension workers run concurrently while the remainder
  queue, bringing the live total to 70 workers.
- After the precision extension loaded, the GH200 reported 99% SM utilization, 40% memory-
  controller utilization, `77,086 MiB` allocated VRAM, `321.14 W`, P0 state, and a
  `1,980 MHz` SM clock. This deliberately stays near, but below, the 80 GB deployment target
  so the analysis remains portable to an 80 GB H100 and retains memory headroom.
- Added `scripts/summarize_matched_delta_calibration_precision.py`. It refuses incomplete or
  non-passing conditions, validates each distribution hash, recomputes every exact p-value,
  requires complement symmetry, and reports 20-direction detection fractions with Wilson
  intervals. A completion watcher will run it only after all 80 parent-plus-extension
  condition reports exist.
- A later 15-second telemetry window over the 70 live workers measured `98.9%` mean SM
  utilization, `72,561 MiB` mean and `75,944 MiB` peak allocated VRAM, and `255.9 W` mean
  with `343.9 W` peak power at P0/`1,980 MHz`. No further workers were added because the
  observed peak leaves only about 4 GB below the intended 80 GB H100 portability ceiling.

## 2026-08-16 — Main-track sensitivity response started

- Froze `experiments/representation_dynamics_main_track_review_response_v1.json` after the
  matched-64 results were known and before the new width, inferential-unit, and
  selection-aware calibration outputs. These analyses are explicitly post-review
  sensitivities, not preregistered confirmation.
- Added a deliberately naive random-example comparison on OLMo for the alphabetical 8+8
  split. Random example folds give macro-F1 `.8022` with sample-label permutation
  `p=.0001`; holding out complete tasks gives `.5737` with exact task-label `p=.2277`.
  This demonstrates task-identity leakage under the wrong inferential unit; it is not a
  reproduction of Sahoo et al.'s model or datasets.
- Recombined the existing hash-gated PCA-95 activation and prompt-64 distributions without
  new fitting. Representation-minus-prompt differences are `.0251--.0500` with exact
  `p=.2567--.4656` in all five models. The fixed 64-dimensional activation projection is
  therefore not the reason the matched statistic is unremarkable. Because the arms have
  unequal widths, this is a compression sensitivity rather than a capacity-matched test.
- Added an exact width-sensitivity runner for activation widths 16, 256, and raw hidden
  states against the fixed prompt-64 arm. Fifteen resumable model-by-width jobs are running
  concurrently on the persistent volume, with GPU power, utilization, and memory telemetry.
- Added a second calibration that appends a synthetic activation site, recomputes the
  mean-shift selector under every balanced task assignment, and tests the matched
  representation-minus-prompt statistic. The frozen design uses ten directions at
  standardized mean differences `.5/1/2/4`; it runs concurrently with the width tests. This closes
  the mismatch between the paper's fixed-site positive control and its selection-aware
  matched arbiter.
- Completed the post-hoc regularized-whitened selector grid for all five models and ridge
  multipliers `.01/.1/1`. All 15 selection-aware maxima reject with exact
  `p=.00062--.00233` and Holm-adjusted `p<=.00933` across the grid, but none selects the
  unwhitened trace-ratio site. On sampled OLMo
  assignments, the vectorized calculation agrees with direct pooled-covariance inversion
  within `2.35e-13`. This establishes selector dependence of localization, not a causal
  site or cross-panel transfer.
- The manuscript now names causal layerwise knowledge/reasoning studies as motivation
  without claiming to reproduce or refute them, demonstrates the inferential-unit effect,
  reports the effective matched-null spread, qualifies localization by its trace-ratio
  selector, and states that moderate prompt-independent effects are not excluded.
- Verification after these changes: 56/56 root tests and 44/44 focused representation
  tests pass; new Python modules compile; all three launchers pass shell syntax; root and
  both submodules pass whitespace checks. The anonymous NeurIPS build has 32 pages,
  Conclusion ends on main-text page 9, and References begin on page 10. It has no undefined
  references/citations or overfull boxes. All four empirical figure checks and the native
  overview PDF/PNG/SVG regeneration check pass.

## 2026-08-16 — Final verification after instrument-validation integration

- The active representation-analysis suite passes 40/40 tests and the root suite passes
  54/54. Python compilation, shell syntax, both new manifests, and whitespace checks pass.
  A broader adapter check passes 22/24 tests; the two failures are pre-existing Qwen video
  helper expectations (`decord` and `_subsample_video_inputs`) outside the active text-only
  path. The broad suite is therefore recorded as partially failing, not silently counted as
  passing.
- The official anonymous NeurIPS 2026 build succeeds at 31 pages. The conclusion ends on
  main-text page 9 and references begin there; no undefined citations/references or overfull
  boxes are reported. Remaining TeX diagnostics are underfull boxes and the official
  `lineno.sty` invalid-UTF-8 warning.
- The native study overview, all four empirical figures, and all six deterministic metric
  illustration files pass their regeneration checks. No instrument-validation or telemetry
  process remains active.

## 2026-08-16 — Matched prompt comparison and instrument calibration completed

- Completed five exact matched comparisons under one 64-feature estimator family. The
  representation-minus-prompt macro-F1 differences are `.0172--.0695`; exact paired
  p-values are `.1770--.4396`. Representation and prompt probes separately reject the
  named partition, but no observed difference is unusually large among the 12,870 balanced
  task assignments. This is not an equivalence result.
- The alphabetical 8+8 reference split is unremarkable: representation p-values are
  `.2277--.3125` and paired-difference p-values are `.7902--.8614`.
- Verified all assignment-specific folds. The maximum train- or test-task class imbalance
  is zero across all 12,870 assignments. Lexical adjustment retains `30.97--54.34%` of
  between-task variance, `56.81--75.92%` of total variance, and `84.85--95.15%` of
  within-task variance. The separate 16-way example-split task diagnostic remains high
  before and after adjustment; it is not construct inference.
- Completed 41 calibration conditions at one fixed OLMo site. At standardized injected
  mean differences `.5`, `1`, and `2`, 0/10 seeded directions meet exact `p<.01`; at
  difference `4`, 5/10 do. The positive control shows that the adjusted probe can detect a
  known activation-only signal, but sensitivity is direction-dependent and the calibration
  does not include site selection.
- Replaced ten redundant lexical-cache builds with one atomically written, digest-bound
  229-fold cache (`00038a7ad61137a0e712a2b174b42c1d896f4dc388b1ea05abaf9e05c83e0293`).
  A parallel-versus-sequential unit test gives identical arrays. Peak telemetry for this
  three-hour trace is `57,201 MiB`, `513.74 W`, and 100% peak utilization.
- A declared post-hoc calibration extension at mean differences `5/6/8` was terminated
  with zero completed reports. Near-separable conditions made the unchanged 3,000-iteration
  cuML solver prohibitively slow (about 47/46/73 CPU-minutes per direction before the stop).
  The failed extension is retained in the manifest and telemetry but excluded from all
  scientific tables.
- Raw reports and exact null vectors remain under
  `/home/ubuntu/data-imu-us-east3/vlmbehave-artifacts/2026-08-16/representation-dynamics-main-track-instrument-validation-v1/`.
  Independently recomputed p-values, complement symmetry, source hashes, cache hash, and all
  41 calibration distributions pass. Sample-free summaries are under
  `results/representation_dynamics_main_track_instrument_validation_v1/`.

## 2026-08-16 — Matched prompt comparison and instrument calibration started

- Froze `experiments/representation_dynamics_main_track_instrument_validation_v1.json`
  before inspecting any completed matched-comparison report. The primary added statistic is
  the exact paired macro-F1 difference between representation and prompt probes. Both arms
  use the same 64-feature width, assignment-specific grouped folds, scaling, cuML logistic
  estimator, regularization, and all 12,870 balanced task assignments; the mean-shift site is
  reselected for every assignment.
- Implemented an alphabetical 8+8 negative control, exhaustive assignment-specific fold
  audit, lexical-adjustment variance decomposition, and 16-way task-identity diagnostic.
  Every one of the 12,870 assignments produces balanced train and test folds. Across five
  models, lexical adjustment retains `30.97--54.34%` of between-task variance,
  `56.81--75.92%` of total variance, and `84.85--95.15%` of within-task variance at the
  observed selected site. The diagnostic task-identity F1 is `.728--.929` before adjustment
  and `.765--.961` after adjustment; this is an example-split diagnostic, not primary
  construct inference.
- Strengthened the synthetic calibration to exercise the complete fold-local lexical-64
  Ridge adjustment, 95%-variance PCA, and task-held-out probe. It injects seeded directions
  aligned with an arbitrary balanced partition at standardized mean differences
  `0/.5/1/2/4`, evaluates ten directions at every nonzero level, and uses the complete exact
  task-label null. It calibrates one fixed OLMo site, not layer selection or a model
  mechanism.
- Full outputs are written under
  `/home/ubuntu/data-imu-us-east3/vlmbehave-artifacts/2026-08-16/representation-dynamics-main-track-instrument-validation-v1/`.
  Five matched tests are active; no numerical conclusion from them is recorded before their
  reports, complement symmetry, and distribution hashes pass. Calibration follows the
  matched run in 41 independently resumable conditions.
- Added direct related-work positioning against Sahoo et al., task/function vectors,
  concept-removal limits, MDL probing, and concurrent excess-separability work. The paper
  now distinguishes recoverability, excess over prompt text, site transfer, and training
  association rather than defining all four as necessary conditions for existence. The
  localization statistic is named accurately as a PCA-space mean-shift trace ratio, with the
  classical whitened Fisher distinction stated explicitly.

## 2026-08-16 — Heterogeneous multimodal side analysis completed

- Restored and SHA-256-verified the frozen six-model archive, then restricted the new
  analysis to current Qwen 3.5 9B/27B and Gemma 4 31B conditions. All 30 task artifacts
  pass count, finiteness, semantic-exactness, prefill-only, float32, and prompt-protocol
  checks. Older Gemma 3 artifacts remain excluded.
- Enumerated all 252 balanced five-versus-five task assignments. Fisher maxima were
  reselected for every assignment, and 126 label-complement probe refits were performed for
  each of four conditions per model: raw, surface/modality adjusted, 64-component character
  TF--IDF adjusted, and combined. Every exact score distribution and p-value was retained.
- Raw held-out-task F1 is `.879/.858/.899` for Qwen 9B, Qwen 27B, and Gemma, with exact
  p-values `.0159/.0317/.0159`; Fisher-maximum p-values are `.0238/.0159/.0476`. No test
  meets the declared `.01` criterion. All adjusted representation p-values are at least
  `.0635`; prompt-only word and character baselines each give about `.80` F1 with `p=.0159`.
- Added the predefined descriptive measurements at each observed Fisher-selected site:
  RBF F1, cosine silhouette, covariance alignment, within-group effective-rank fraction,
  and stable rank. Paired CKA is explicitly missing in all three models because no
  cross-group `pair_id` exists; it is not replaced by an arbitrary row alignment or zero.
- The initial appendix-only insertion preserved pre-appendix SHA-256
  `5a6b77579e1a81f9234d0238bd3c69ba8ee34b094c579f0b8a9dd3c507a446fc`. Subsequent
  manuscript edits add bounded references to the side analysis in the sensitivity and
  limitation paragraphs; the current pre-appendix SHA-256 is
  `a3035b74ff2a966a0ae40095bc1611dfe3061fa5a9a0da10ae276183e64553e7`. The primary
  numerical table, abstract, and conclusion remain unchanged.
- Full outputs, source audits, null vectors, and telemetry are under
  `/home/ubuntu/data-imu-us-east3/vlmbehave-artifacts/2026-08-16/representation-dynamics-multimodal-side-v1/`.
  Sample-free summaries are under `results/representation_dynamics_multimodal_side_v1/`.
  A no-op idempotence replay overwrote the first-run GPU telemetry with one idle sample; the
  invalid trace is excluded from the summary, while all inferential reports and null vectors
  remain hash-bound and unchanged. The launcher now starts and appends telemetry only when
  analysis work is pending.
- Claim boundary: this panel confounds task label with modality, domain, task family,
  response format, and prompt content. The controls establish sensitivity, not causal
  mediation or a modality-independent knowledge--reasoning distinction.

## 2026-08-16 — Scientific narrative and native study overview revised

- Reorganized the manuscript around the supported result: the held-out-task probe is
  positive before prompt controls, but the declared `.01` decision is not robust to the
  fold-local low-rank lexical sensitivity, localization does not replicate across panels,
  and separation does not change coherently along the two released training trajectories.
- Replaced causal or exhaustive wording with the measured claim. The paper states that the
  adjustment removes only activation variance linearly predicted by a finite character
  TF--IDF basis, acknowledges that four models retain nominal `p<.05`, and does not equate
  the sensitivity result with lexical mediation or absence of all partition information.
- Rebuilt Figure 1 as one native TikZ/pgfplots source. It shows the 8+8 task contrast, the
  three per-layer streams, task-level randomization and site reselection, prompt controls,
  site replication, and the two checkpoint trends. PDF, 300-dpi PNG, and editable-text SVG
  exports now regenerate and byte-verify through one deterministic wrapper.
- Verification: 53/53 root tests and 51/51 focused representation/CLI/text-adapter tests
  pass. The two known Qwen video-adapter tests remain outside this change and still fail for
  the previously absent `decord` attribute and video-subsampling helper. The anonymous paper
  compiles to 28 pages; the conclusion and references begin on main-text page 9, no author
  identifiers are present, and there are no unresolved citations/references or overfull
  boxes. All deterministic paper-figure gates pass. Paper revision: `499b793`.

## 2026-08-16 — Fold-local lexical activation control completed

- Implemented character-within-word TF--IDF activation controls in the selection-aware
  probe at `lmms-eval` revision `0b69f69`. For every grouped fold, the vocabulary, inverse
  document frequencies, 32/64-component truncated SVD, activation Ridge map, PCA, scaling,
  and classifier exclude held-out tasks. The Fisher maximum is reselected from the
  unadjusted profile for every balanced assignment.
- Completed five models at both component counts: 12,870 oriented assignments and 6,435
  complement-pair probe refits per condition. All ten reports pass clean-revision, exact
  support, 229-fold inductive-fit, complement-symmetry, and distribution-hash gates.
- At 64 components, macro-F1 falls from `.781--.805` unadjusted to `.613--.698`; exact
  p-values are `.01197`, `.06605`, `.01476`, `.04165`, and `.02020` for OLMo 3, Apertus
  1.5, Qwen 9B, Qwen 27B, and Gemma 4. None meets the declared `.01` criterion, although
  four remain below a nominal per-model `.05`. The 32-component p-values are
  `.01880--.06402`, with the same `.01` decision.
- Raw reports and score distributions are under
  `/home/ubuntu/data-imu-us-east3/vlmbehave-artifacts/2026-08-16/representation-dynamics-main-track-lexical-control-v1`.
  The compact summary hashes every report and telemetry file under
  `results/representation_dynamics_main_track_lexical_control_v1/`.
- The 397 telemetry samples report mean/peak GPU utilization `92.9%/100%`, mean/peak
  power `485.2/543.9 W`, and peak allocation `22,279 MiB`. Concurrent MPS kept the GPU
  occupied, but thousands of small PCA/logistic fits do not approach dense-model power.
- Revised the main table, abstract, sensitivity result, conclusion, external-review
  response, repository status, and figure legend. The claim is sensitivity to a finite
  low-rank lexical control, not causal mediation or absence of all task-partition signal.
- Verification: 51/51 root tests and 47/47 focused representation/CLI tests pass; the ten
  randomization p-values, distribution hashes, and label-complement symmetries were also
  recomputed independently. The manuscript and deterministic figures pass regeneration;
  LaTeX has no unresolved citations/references or overfull boxes, and the conclusion remains
  on page 9. The paper revision is `21ddf57`.

## 2026-08-15 — Review revision finalized and independently checked

- Reframed the manuscript around the supported negative measurement result: the MMLU 8+8
  label is predictable from both prompts and representations, but no result establishes a
  prompt-independent, cross-panel, model-shared, or training-monotonic computation-type
  signature. The paper no longer treats task-dependent activations as a test of whether a
  model is “purely statistical,” and all primary claims are explicitly text-only.
- Moved the prompt-only baselines, primary-panel residualization controls, response-validity
  limitation, and three-panel localization comparison into the main results. The distinct
  MMLU-Pro 5+5 and MMLU 8+8 panels are named separately throughout.
- Committed the revised manuscript and reproducible figures at paper revision `9ac49c2`.
  The anonymous NeurIPS build has 29 total pages including appendix and checklist; the
  conclusion is on main-text page 9. It has no unresolved references, citations, or
  overfull boxes, and the generated PDF contains no author names or affiliations.
- Final verification: 49/49 root tests pass. The focused `lmms-eval` suite passes 79/81;
  the two failures are Qwen video-adapter tests for a `decord` module attribute and a
  video-subsampling helper that were already absent at the parent revision. This review
  changed generated-token metadata in the same adapter but did not add or remove either
  missing video capability. Both deterministic paper figure checks pass, as do JSON
  parsing, Python compilation, shell syntax, and whitespace checks.

## 2026-08-15 — Main-track review audit and primary prompt controls

- Reproduced the review's central concern on the primary MMLU 8+8 panel. A word 1--2-gram
  TF--IDF classifier, refit with its vocabulary and inverse-document frequencies inside
  every held-out-task fold and all 12,870 balanced assignments, obtains macro-F1 `0.701`
  (`p=0.00870`; null median `0.488`, 95th percentile `0.644`). Representation-probe
  rejection therefore does not by itself establish information beyond prompt text.
- Rebuilt primary-panel prompt covariates for 8,000 model--example rows. Mean
  knowledge/reasoning token counts are `179.0/135.3` for OLMo, `172.0/127.1` for both Qwen
  sizes, `172.2/129.5` for Gemma, and `206.6/163.5` for Apertus 1.5. The earlier
  `274/229--308/267` range belongs only to MMLU-Pro 5+5.
- Audited direct-answer parse validity: OLMo `.992`, Apertus 1.5 `.573`, Qwen 9B/27B
  `1.000/1.000`, and Gemma `.761`. Historical failures cannot be inspected because raw
  generations were discarded; new sample logs retain outputs before task filters under
  `raw_resps` without changing the established response fields.
  Primary prefill states remain usable, but cross-model accuracy and correctness-stratum
  claims are not.
- Audited the longitudinal response records. OLMo validity is `1.000` at every checkpoint
  and block; Apertus 1.0 is `.997` at 210B and 3.15T and `1.000` thereafter. The reported
  longitudinal accuracy trends are therefore not driven by the severe endpoint-panel parser
  failure.
- Added exact character TF--IDF and fold-local length-only/six-feature selected-site controls
  under `experiments/representation_dynamics_main_track_controls_v1.json`. The character
  baseline completed at macro-F1 `0.768` (`p=0.000777`; null 95th percentile `0.657`). All
  ten GPU controls completed under
  `/home/ubuntu/data-imu-us-east3/vlmbehave-artifacts/2026-08-16/representation-dynamics-main-track-controls-v1/`.
  Length-only adjusted probe p-values are `.00357`, `.00342`, `.00761`, `.00420`, and
  `.00761`; all five reject at `.01`. Six-feature adjusted p-values are `.00326`, `.00839`,
  `.00171`, `.00544`, and `.0137`; Gemma is the only condition above `.01`. The order is
  OLMo, Apertus 1.5, Qwen 9B, Qwen 27B, and Gemma 4. These are post-hoc sensitivities, not
  mediation estimates.
- The control launcher recorded 392 ten-second GPU samples: mean utilization `99.23%`,
  mean board power `486.18 W`, peak power `529.58 W`, and peak allocated memory
  `19,804 MiB`. All ten reports bind to clean `lmms-eval` commit `d828722` and preserve
  12,870 assignment scores with 6,435 complement-pair refits.
- Hardened inference and collection: missing rows retain the declared BH family size;
  linear complement equality is described as imposed reuse rather than independent replay;
  ambiguous decoder stacks are rejected; decode capture excludes per-example EOS/padding
  updates; future raw generations are retained; and cosine silhouette uses explicit batched
  distances. Seventy-one focused tests passed before the run was launched.
- The manuscript now leads with the negative result, names the two distinct MMLU and
  MMLU-Pro panels correctly, reports parse validity and cross-panel site instability, removes
  untraceable implementation numerics, and demotes the post-answer 4+4 response result to a
  descriptive pilot. The exact prompt and adjustment controls are now in the main results
  table; compact hash-bound summaries are under
  `results/representation_dynamics_main_track_controls_v1/`.

## 2026-08-15 — Completed analyses incorporated into the manuscript

- The five-model 8+8 selected-site probe completed all 12,870 oriented assignments per
  model. All reports pass exact complement symmetry. Macro-F1 p-values are `0.00155`,
  `0.00186`, `0.00186`, `0.00420`, and `0.00404` for OLMo, Apertus 1.5, Qwen 9B,
  Qwen 27B, and Gemma 4. Sample-free hash bindings are under
  `results/representation_dynamics_main_track_selected_probe_v1/`.
- Apertus 1.0 completed all 33 checkpoint--block analyses and 1,056 layer--stream rows.
  Accuracy rises over the released 210B--15T trajectory and endpoint Fisher ratios rise,
  opposite to OLMo's endpoint direction; both trajectories remain non-monotonic. Compact
  aggregates and persistent-source hashes are under `results/apertus1_longitudinal_v1/`.
- The 16-task capped OLMo response acquisition produced 811,008 finite records and complete
  32-position coverage, but failed observer-off/on raw-response parity. Its queue exited
  before analysis. The artifact remains on persistent storage with
  `status=FAIL_VALIDATION`; it is not scientific evidence. This corrects the provisional
  statement in the earlier 17:18 UTC workload snapshot.
- The manuscript abstract, contributions, longitudinal figure/table, cross-model table,
  appendix validation record, limitations, and conclusion were revised to reflect these
  outcomes. The valid 4+4 normalized-position pilot remains a separate descriptive result.
- Compact Apertus values and selected-site probe values were checked against the persistent
  source files and their recorded SHA-256 hashes. All four generated paper figures pass the
  pixel-level regeneration check. The NeurIPS build completes with no unresolved references,
  citations, or overfull boxes; the conclusion remains on content page 9.

## 2026-08-15 — Development implications and live workload audit

- The manuscript now states the practical role of the measurements: checkpoint/release
  diagnostics, benchmark-taxonomy audit, and selection of candidate sites for controlled
  interventions. Conditional compute, retrieval/tool routing, regression alarms, adaptation,
  and compression are described only as potential downstream uses after held-out replication
  and matched interventions; no passive metric is presented as a deployment rule.
- At 17:18 UTC, `vlm-selected-site-probe-panel.service` was actively evaluating the exact
  8+8-task selected-site linear probe for five models. Completed oriented assignments were
  7,120/12,870 for Apertus 1.5 8B, 11,840/12,870 for Gemma 4 31B, 10,100/12,870 for OLMo 3
  7B, 9,620/12,870 for Qwen 3.5 27B, and 9,700/12,870 for Qwen 3.5 9B. No model report was
  complete at this snapshot.
- `vlm-apertus1-analysis-resume-4.service` had completed 16/33 checkpoint--block analyses
  in the 11-checkpoint Apertus 1.0 pretraining trajectory. Four conditions were active and
  13 remained queued after them.
- `vlm-response-trajectory-analysis-queue-v2.service` was waiting for the Apertus MPS owner
  to finish before analyzing the already validated 16-task OLMo absolute-position response
  artifact. The completed 4+4-task normalized-position pilot is not an active job.
- The GH200 snapshot showed 99% reported utilization, 358 W board power, and 19,186 MiB
  allocated. These concurrent permutation/SVD workloads contain many small matrix operations;
  instantaneous utilization is not equivalent to dense full-chip tensor throughput.
- All active raw and intermediate outputs are under
  `/home/ubuntu/data-imu-us-east3/vlmbehave-artifacts/2026-08-15/` or the dated Apertus root
  `/home/ubuntu/data-imu-us-east3/vlmbehave-artifacts/2026-08-13/`.

## 2026-08-15 — Normalized response-position pilot started

- Question: compare internal states at matched fractional progress through responses of
  different lengths, instead of treating absolute token indices as comparable stages.
- Frozen pilot: OLMo 3 7B Instruct at revision `6e5971d9...`, four knowledge-oriented and
  four reasoning-oriented MMLU subjects, four examples per task, greedy generation, eight
  equal-width response bins, and three streams at all 32 decoder layers.
- Estimator: deterministically average every captured update within each example-specific
  relative bin. One randomly sampled update per bin is deferred as a sensitivity analysis
  because a single draw adds avoidable sampling variance.
- Safety decision: use batch size one and uncapped decode capture. In batched generation,
  finished examples may continue through padded forward calls while other examples remain
  active; those calls must not define response length. Batch one makes every captured
  decode update response-valid for this pilot.
- Scope: the first bin includes the fixed answer marker. Relative bins align response
  progress, not semantic reasoning steps, and the answer-first prompt makes the trajectory
  post-answer. With only 4+4 tasks, the class-symmetric minimum p-value is `2/70=0.0286`,
  so this pilot cannot meet the paper's `p<.01` decision threshold.
- Code changes under review: explicit relative-position labels and joint prefill/decode
  analysis in `lmms-eval`; deterministic binning, generalized maximum inference, a frozen
  manifest, and a persistent-volume launcher in the root repository.
- Tests at protocol freeze: 32 focused `lmms-eval` representation tests passed; two root
  binning tests passed. Execution results and artifact hashes will be appended after the
  real pilot completes.

### First real pass: rejected before analysis

- Persistent run: `/home/ubuntu/data-imu-us-east3/vlmbehave-artifacts/2026-08-15/relative-response-pilot-olmo3-v1`.
- The unmonitored parser gate passed 32/32 responses. The monitored artifact passed exact,
  finite, uncapped capture validation: 175,392 records over 32 layers and three streams,
  with no skipped or truncated records.
- Strict observer parity failed for two of 32 responses. Both retained the same parsed
  answer and score, but their generated explanation text differed; one monitored response
  reached the 256-token generation cap. Because the frozen gate requires exact raw-response
  equality, these activations were rejected and were not binned or analyzed.
- The prompt requested at most 60 words, but the unmonitored responses ranged from 25 to
  207 re-tokenized tokens and one response contained 147 whitespace-delimited words. A
  request for a fixed format is therefore not evidence that the model followed it.
- Next diagnostic: repeat the observer-off generation under the same immutable settings.
  If observer-off replay is not exact, generation stability must be addressed before
  attributing the difference to monitoring or reporting trajectory metrics.

### Replay diagnosis and revised runtime freeze

- A second observer-off run under the original backend changed the same two explanations
  while preserving their parsed answers and scores. The failed off/on comparison therefore
  cannot be attributed specifically to the hooks; long greedy BF16 generation was not
  bitwise text-stable under that runtime path.
- Two independent observer-off runs with eager attention then matched exactly for all 32
  examples on document IDs and hashes, prompts, targets, complete response strings, parsed
  answers, scores, and token counts. Their persistent paths are
  `relative-response-pilot-olmo3-eager-v1/unmonitored-a` and `unmonitored-b` under the dated
  artifact root.
- The rejected protocol remains `...pilot-v1.json`. The revised
  `...pilot-v2.json` freezes eager attention before monitored extraction and adds the
  independent observer-off replay as an explicit gate. No activation result is accepted
  unless the subsequent eager observer-off/on comparison is also exact.
- Planned next steps: collect one monitored eager run; validate exactness, completeness,
  and observer parity; bin response-valid updates; run selection-aware Fisher inference;
  then add only sample-free descriptive aggregates and the attainable p-value to the paper.

### Eager pilot acquisition and binning

- The monitored eager run passed the artifact gate: 178,752 vectors, 32 layers, three
  streams, 32 examples, float32 storage, exact branch semantics, no non-finite values, and
  no skipped or truncated captures.
- Observer-off/on parity passed exactly for document identity, prompts, targets, complete
  response strings, parsed answers, scores, and token counts. This is the first accepted
  activation artifact for the normalized-position pilot.
- Response-valid cached-update counts range from 25 to 205 (median 47.5). Deterministic
  within-example averaging produced all eight bins for every sample; the binned artifact
  is 444 MB and the raw artifact is 2.8 GB. Both remain under
  `/home/ubuntu/data-imu-us-east3/vlmbehave-artifacts/2026-08-15/relative-response-pilot-olmo3-eager-v1`.
- Selection-aware cuML analysis is in progress. It computes 32 layers × three streams ×
  nine positions (prefill plus eight bins) and reselects the decode-minus-prefill maximum
  for all 70 balanced 4+4 task assignments. No numerical result is recorded here until
  that run and its inference report pass.

### Normalized-position pilot result

- All 864 layer--stream--position rows completed with cuML. The selection-aware inference
  report passed and verified exact label-complement symmetry.
- The largest decode-minus-prefill increase occurs at layer 31's token-mixing update in
  relative bin 5/8: Fisher `0.193350 → 0.479463` (`Δ=0.286113`). Reselecting layer, stream,
  and bin under all 70 assignments gives `p=2/70=0.028571`; the null median and 95th
  percentile for the maximum change are `0.059584` and `0.178176`.
- At that selected site, the Fisher numerator increases `4.254 → 18.540`, the within-group
  term also increases `22.002 → 38.668`, and the effective-rank fraction is nearly stable
  (`0.886 → 0.889`). The local increase is therefore not produced by shrinking
  within-group variance or rank collapse in this condition.
- The depth-mean result points in the other direction: from prefill to the final response
  bin, Fisher falls `0.226 → 0.146` for token mixing, `0.289 → 0.114` for feed-forward,
  and `0.289 → 0.133` for residual states, while effective-rank fractions rise. This
  distinction between a selected late-layer effect and the depth-wide profile is retained
  in the manuscript.
- Decision: report the pilot descriptively. Its exact p-value is the attainable floor but
  cannot meet the prespecified `p<.01` threshold; 4+4 tasks, post-answer generation, and
  fractional rather than semantic positions preclude a main-track claim.
- Sample-free, hash-bound outputs are committed under
  `results/representation_dynamics_relative_response_pilot_v2/`; raw activations remain
  only on persistent storage.
- Manuscript status: the normalized result paragraph and compact wrap figure are in the
  main text, the figures reproduce exactly, and LaTeX compiles. The auxiliary file places
  Sensitivity on page 9 and Limitations/Conclusion on page 10. This main-text length risk
  remains a submission task: confirm the final venue limit and compress the
  manuscript without removing primary results before submission.

## 2026-10-15 — Camera-ready verification and artifact release

- Verified the migration bundle `vlmbehave-migration-20260827T000543Z` end to end without a
  full unpack: bundle SHA-256 `160ab6cb…` equals the pack-time checksum and the Drive copy;
  a streaming pass matched all 2,769 members to `persistent-files.sha256`, scanned 2,498 text
  members plus 1,588 nested-archive members for credentials (zero hits), and extracted the
  2,498 text members (1.76 GB) for inspection.
- Re-hashed the 426 extracted files referenced by `results/` (all match), confirmed the 39
  self-hashes and 12 workspace-file hashes, and regenerated all 25 committed CSV summaries of
  nine results directories byte for byte from the extracted reports. Details and the
  results-to-bundle map are in `docs/ARTIFACTS.md`; `scripts/audit_migration_bundle.py` is the
  verifier.
- Created the Google Drive release folder `lmbehave-release` holding only the artifacts and
  their hashes: the server-side-copied `persistent-data.tar.zst` and `persistent-files.sha256`
  (hashes re-read server-side), a text-only `analysis-outputs-text.tar.gz` (563,834,133 bytes,
  SHA-256 `f889e7da…`, 2,498 members), `SHA256SUMS`, and a short README. The folder is private
  until the authors share it.
- Camera-ready: merged the Overleaf edits (title "… in LLMs?", `knowledge/reasoning`
  spelling, inline two-author block, no checklist), corrected the grammar and meaning slips they
  introduced (abstract "selected benchmark" → "selected site", "macroF1", the fourth-question
  and contribution sentences, the set definition, a leftover `% REVIEW` comment), restored the
  title hyphens, and reorganized entry points (`main.tex` camera-ready, `arxiv.tex`,
  `camera_ready.tex` with checklist, `review.tex` anonymous). All builds compile with the main
  text ending on page 9; `check_text_fidelity.py` reports no number changes relative to the
  submission other than the abstract's deliberately removed p-range.

## 2026-10-20 — Public release preparation

- Prepared a separate release branch without the manuscript submodule or its configuration and build references. The original research checkout is preserved.
- Moved project-page ownership to `parsa-ra/lmbehave-page` and added the arXiv and verified-artifact folder links.
- Research protocols, numerical result tables, analysis code, and evaluation defaults are unchanged.
