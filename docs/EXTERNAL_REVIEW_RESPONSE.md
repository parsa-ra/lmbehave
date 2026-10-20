# External review assessment and revision decisions

This document records how the August 2026 review in `/home/ubuntu/review.tex` was checked
against the artifacts, code, and cited literature. The review is advice, not an empirical
source. Claims below are bound to the completed OLMo and five-checkpoint studies.

## Second main-track review: accepted criticisms and bounded response

The second review correctly identifies three remaining threats: the fixed-site positive
control does not calibrate the matched statistic used in the abstract; equal feature count
does not guarantee equal information retention; and localization depends on an unwhitened
mean-shift selector that is not a linear-separability criterion. It is also right that the
paper needed an empirical demonstration, rather than only an argument, for treating the
task as the inferential unit.

The following corrections are complete:

- On the same arbitrary OLMo 8+8 split and activation site, random-example validation gives
  macro-F1 `.8022` (`p=.0001` under 9,999 sample-label permutations), whereas complete-task
  holdout gives `.5737` (`p=.2277` under all 12,870 task assignments). The example-level
  test is deliberately invalid for the construct; it demonstrates the leakage channel and
  does not claim to reproduce another paper.
- The effective matched-64 null has mean `.0057--.0168`, SD `.0566--.0670`, and 95th
  percentile `.1084--.1292` across models. The observed `.0172--.0695` deltas lie inside
  this broad conditional distribution. The data support a small positive null location,
  not the stronger claim that the observed surplus occurs for most task partitions.
- Replacing the fixed 64-dimensional activation arm with each model's existing fold-local
  95%-variance PCA arm gives delta `.0251--.0500` and exact `p=.2567--.4656`. Thus the
  matched non-rejection is not created by forcing every activation to 64 components. This
  unequal-width comparison remains a sensitivity analysis, not matched capacity.
- The manuscript now says explicitly that the matched delta is the arbiter only for the
  incremental-over-prompt claim, that it was added after the initial localization study,
  and that trace-ratio site-transfer results are conditional on that selector. It does not
  present the fixed `.01` rule or a non-rejection as equivalence.
- Verified published causal layer studies are now cited as concrete motivation. They test
  pruning, patching, or ablation on different tasks; the present passive benchmark study
  neither reproduces nor refutes those causal results.

The regularized-whitened selector sensitivity is complete. Across five models and ridge
multipliers `.01/.1/1`, all 15 selection-aware maxima reject (`p=.00062--.00233`), but
all also retain `p_{Holm}<=.00933` across the grid and none selects the unwhitened
trace-ratio site. A direct-inverse check on sampled real
assignments agrees with the vectorized calculation within `2.35e-13`. This strengthens the
raw recoverability result while showing that layer--branch localization is conditional on
the selector and ridge. It does not test causal importance or cross-panel transfer.

Exact fixed-16, fixed-256, and raw-width analyses and selection-aware matched-delta
calibration remain active under the frozen post-review manifest. Mean-valid-token
re-extraction, parser repair, MDL/nonlinear probes, and a scientifically justified
equivalence margin remain separate work. They will not be written as completed until their
artifacts and numerical gates pass.

## Matched prompt comparison and instrument-validity review

The later review correctly identified that a representation score and a prompt score from
different estimator families cannot support a quantitative claim about their difference.
It also correctly requested a positive control, a check of what lexical residualization
removes, and an audit of the folds under every permuted assignment. These checks are now
implemented under one frozen follow-up protocol.

- Representation and character-TF--IDF prompt features are each mapped to 64 fold-local
  components and enter the same scaler, cuML logistic estimator, regularization, task folds,
  and 12,870-assignment support. The activation site is reselected under every assignment.
  Representation-minus-prompt macro-F1 is `.017--.070`; exact paired p-values are
  `.177--.440` in the five models. This is not an equivalence result. It shows that the
  observed advantage is not unusually large among the balanced task partitions under this
  matched estimator.
- The alphabetical 8+8 reference partition is unremarkable: representation p-values are
  `.228--.313` and representation-minus-prompt p-values are `.790--.861`.
- Lexical residualization retains `30.97--54.34%` of between-task variance,
  `56.81--75.92%` of total variance, and `84.85--95.15%` of within-task variance. A
  separate example-split 16-way task diagnostic remains strong after adjustment. The
  adjustment therefore does not simply erase all task identity; it remains a sensitivity
  analysis rather than evidence of concept removal.
- Grouped folds are regenerated for every assignment. Across the complete support, all
  test folds contain equal task counts per class and no training fold is imbalanced. The
  review's proposed fixed-fold inflation does not occur in this implementation.
- The paper now calls the localization statistic a PCA-space mean-shift trace ratio. It is
  not the covariance-whitened Fisher linear-discriminant criterion. The legacy artifact key
  `FDR_trace` is retained for schema compatibility.

The synthetic activation-only calibration is deliberately narrower than a model claim: it
tests one OLMo site, an arbitrary balanced partition, and the complete lexical-adjusted
fixed-site probe. Detection is 0/10 directions at standardized mean differences `.5`, `1`,
and `2`, and 5/10 at difference `4`. It therefore demonstrates that the instrument can say
yes to a known prompt-independent activation shift, while also showing that its sensitivity
is direction-dependent and weak below a large scale. It does not calibrate layer selection.
A post-hoc `5/6/8` extension was terminated with no completed report when the unchanged
solver became prohibitively slow near separability; it is excluded from the paper.
Regularized-whitening and PCA-retention ablations,
mean-valid-token pooling, a reliable nonlinear primary-panel test, and interval/equivalence
analysis remain open and are named as limitations rather than presented as completed work.

## Primary-panel audit and revision

The later main-track review identified three material reporting problems and one central
missing control. All were confirmed.

- The manuscript had conflated the MMLU-Pro 5+5 pilot with the distinct MMLU 8+8 primary
  panel. The data, subject sets, prompts, and answer spaces are now named separately.
- Fisher maxima do not localize to a stable site: only two of five models select the same
  layer and branch in the two matched panels, and no model retains one site across both
  matched panels and the corrected heterogeneous panel. This is now a main result.
- Historical MMLU 8+8 answer parsing is unreliable for Apertus 1.5 (`.573` valid) and Gemma
  4 (`.761` valid), with substantial task variation. The current-model accuracy values and
  correctness-stratified comparison are therefore not used as evidence. The longitudinal
  logs were audited separately and have near-complete parsing.
- Prompt text predicts the primary label under the same task holdout and exact assignment
  support. Word TF--IDF obtains macro-F1 `.701` (`p=.00870`); character TF--IDF obtains
  `.768` (`p=.000777`). This does not prove mediation, but it prevents attributing the
  representation-probe rejection to a prompt-independent computation type.

The paper now leads with the negative result: the named task partition is recoverable from
both prompts and representations, while representation localization is model-specific,
unstable across panels, and unrelated to a common training-time direction. Fold-local
length-only and six-feature controls on the
primary panel are reported as post-hoc sensitivities, not confirmatory tests or causal
adjustments.

## Fold-local lexical activation control

The follow-up review correctly identified the missing combination of the prompt baseline
and activation residualizer. We froze a separate post-hoc sensitivity before inspecting
its adjusted scores. For every balanced task-label assignment, the analysis reselects the
maximum from that assignment's unadjusted Fisher profile. Within each grouped probe fold,
character 3--5-gram TF--IDF, truncated SVD, the activation Ridge map, PCA, scaling, and the
classifier are fit without held-out tasks. We ran 32- and 64-component conditions for all
five models, covering all 12,870 oriented assignments and 6,435 class-label-complement
refits per condition.

At 64 components, macro-F1 falls from `.781--.805` to `.613--.698`; exact p-values are
`.01197`, `.06605`, `.01476`, `.04165`, and `.02020`. No model meets the paper's declared
`.01` criterion, while four remain below a nominal per-model `.05` threshold. The
32-component sensitivity gives p-values `.01880--.06402` and the same `.01` decision. Thus the
unadjusted result is materially sensitive to a richer lexical control, but predictability
is not uniformly absent. The manuscript does not interpret residualization as causal
mediation or as removal of all lexical information.

Implementation changes made during the audit preserve the existing estimand. Failed probe
rows remain missing while retaining the declared BH family size; linear label-complement
scores are described as shared-fit identities rather than replay checks; generated-token
capture excludes per-example EOS and padding updates; decoder-stack ambiguity is an error;
and trajectory p-values are withheld when complement symmetry fails. Future sample logs
retain raw model outputs separately from filtered parser outputs. A post-hoc adjacent-state
audit reports reconstruction error directly and records Gemma 4's post-residual scalar; it
does not redefine the three measured streams or the earlier artifacts.

## Decisions

| Review proposal | Decision | Scientific reason and action |
| --- | --- | --- |
| Multivariable format deconfounding | Accept as a high-priority sensitivity analysis | The current scalar prompt-token probe does not remove prompt form, option count, lexical distribution, response length, domain, or difficulty. Fit nuisance models inside each task-held-out fold, apply them to held-out tasks, and retain the unadjusted analysis. Do not describe residual signal as proof of reasoning. |
| Increase from 10 to at least 30 tasks | Accept with a construct-validity condition | More independent tasks improve task-level resolution; more examples from the same tasks do not. Tasks must be selected under a frozen taxonomy and balanced for format/domain. For 15 versus 15 tasks there are `choose(30,15)=155,117,520` directed balanced assignments. Exact enumeration is impractical; a Monte Carlo test has resolution `1/(B+1)`, determined by the number of sampled assignments, not the combinatorial space alone. |
| Add mean-prompt and first-decode-token pooling | Accept as prespecified sensitivity analyses | The current result is a final-prompt-token prefill result. Mean-valid pooling is implemented. Decode capture now has separately named `mean_updates` and `first_update` modes; the latter is unit-tested but needs real-model artifact and observer-parity validation before an empirical claim. |
| Mean-difference activation steering | Modify and defer | Steering can test output sensitivity or sufficiency, not causal necessity. Estimate the direction on training tasks, test unseen tasks, sweep symmetric magnitudes, and compare with norm-matched random and principal-direction controls while reporting off-target task quality. The present localization maxima do not replicate, so intervention is not yet the primary next result. |
| Add SFT/RL checkpoints | Accept as a separate branch study | OLMo SFT, RL-Math, and RL-Code checkpoints are already frozen as secondary tiers. They are branches, not one continuation of the base trajectory, and the concise four-token protocol must first be adapted for reasoning traces. |
| Explain the OLMo trend as lower-dimensional compression | Reject | Within-group effective-rank fractions increased from `.669/.654/.700` to `.694/.683/.716` for attention/MLP/residual streams. The endpoint Fisher decrease is therefore not explained by rank collapse. Fisher numerator and denominator both changed and remain separately reported. |

## Citation audit

The review's `sahoo2026format` entry has an incorrect title, author list, and venue. The
relevant paper is Sahoo, Jain, Chadha, and Chaudhary, “Linear Probes Detect Task Format,
Not Reasoning Mode in Language Model Hidden States,” TrustNLP 2026, ACL Anthology ID
`2026.trustnlp-main.12`. The manuscript now cites the verified record.

The review's `skean2025layerwise` entry also has incorrect metadata. The manuscript already
cites the verified ICML 2025 paper “Layer by Layer: Uncovering Hidden Representations in
Language Models” as `skean2025layer`; no second invented record is added.

The current target is a NeurIPS 2026 main-track submission. The paper uses the unmodified
official anonymous main-track style. Recheck the final call, page limit, anonymity policy,
checklist, and supplemental-material rules immediately before submission.

## Revised priority order

1. Completed: validate the Apertus 1.5 endpoint adapter and observer parity without
   claiming a trajectory.
2. Completed: correct and re-extract MuSR for all five current models. The malformed
   artifacts remain excluded and identified for audit.
3. Completed on MMLU-Pro 5+5: fold-local six-variable prompt adjustment removes all probe
   rejections. On the primary MMLU 8+8 panel, length-only adjustment retains rejection in
   all five models and six-variable adjustment retains it in four; Gemma gives `p=.0137`.
   These remain post-hoc sensitivities.
4. Completed on MMLU 8+8: exact task-held-out word and character TF--IDF baselines predict
   the task-set label (macro-F1 `0.701`, `p=0.00870`, and `0.768`, `p=0.000777`).
5. Completed: fold-local 32/64-component character TF--IDF activation residualization.
   No model meets the declared `.01` criterion; four remain below a nominal per-model `.05`.
6. Completed: selection-aware Fisher inference on the 8+8 panel, an arbitrary-split
   negative control, and a cross-panel localization table. No site replicates across all
   three panels.
7. Completed as a descriptive pilot: response states pooled over eight relative response
   bins. It is post-answer, has only 4+4 tasks, and cannot meet the primary `p<0.01`
   threshold.
8. Next: use the primary prompt controls to decide whether any site is credible enough for
   a held-out-task intervention. Do not proceed from an unadjusted or non-replicating site.
