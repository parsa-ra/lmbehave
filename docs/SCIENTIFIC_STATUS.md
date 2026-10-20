# Scientific Status and Research Roadmap

Last verified: 2026-08-16 UTC. This document states what the current experiment measures,
which claims it can support, and which gates remain before submission. Operational commands
belong in `README.md`; implementation constraints belong in `AGENTS.md`; immutable run
settings belong in `experiments/representation_dynamics_v1.json`.

## Research question

The immediate question is whether operationally defined knowledge-oriented and
reasoning-oriented task sets induce objectively separable internal representations in
frozen language models, and whether any separation profile replicates across task panels,
model families, and released training checkpoints. The primary and longitudinal evidence
is text-only. A separate appendix-only analysis uses historical multimodal tasks and is not
pooled with those claims.

This is not a test of whether models are “purely statistical.” Statistical next-token
models are expected to form input-dependent internal states, so task-dependent activations
alone cannot distinguish pattern learning from reasoning. The defensible estimand is whether
the predeclared task partition explains reproducible structure that transfers to unseen
tasks and survives controls for task identity and nuisance variables.

This decomposes into a claim ladder:

1. **Measurement:** the selected model streams can be observed completely and without a
   meaningful change in model behavior.
2. **Task-set predictability:** intermediate representations predict the declared task-set
   label when complete tasks are excluded from supervised fitting.
3. **Construct validity:** the prediction survives controls for task identity, modality,
   domain, prompt/response format, length, difficulty, and correctness.
4. **Localization:** the direction and strongest layer/branch replicate across panels,
   seeds, pooling rules, and independent models.
5. **Causality:** intervening on the replicated site changes the relevant behavior more
   than matched random or established baselines.
6. **Efficiency:** a causal intervention produces a measured quality--latency,
   throughput, or memory improvement on the intended inference backend.

The current project has strong evidence for Gate 1. The unadjusted primary probe establishes
Gate 2 for the named 8+8 MMLU partition, but prompt text alone predicts that partition under
the same task-held-out null. Gate 3 is therefore not established; the primary length-only
adjustment rejects in all five models and the six-feature adjustment rejects in four, but
fold-local 64-component character TF--IDF residualization moves every exact p-value above
the declared `.01` criterion. Four remain below a nominal per-model `.05` threshold, so
this is sensitivity to a finite low-rank lexical control rather than complete removal of
task-set predictability. Selected sites do not replicate across all panels, so Gates 4--6
are not established.

The post-review matched analysis narrows the interpretation further. In all five models,
the observed representation-minus-prompt macro-F1 difference (`.017--.070`) lies inside
the exact balanced-partition reference (`p=.177--.440`). The effective null has SD
`.057--.067`, so this is a broad finite-task reference rather than a precise equivalence
result. Replacing the fixed 64-dimensional activation projection with fold-local
95%-variance PCA leaves every exact p-value above `.25`; fixed-16, fixed-256, and raw-width
sensitivities are running. On an arbitrary OLMo partition, invalid random-example
validation rejects (`F1=.802`, `p=.0001`) while valid task holdout does not (`F1=.574`,
`p=.228`). This directly demonstrates why examples cannot define the null for a claim about
benchmark categories.

Localization is also selector-dependent on the primary panel. A post-hoc regularized-
whitened mean-displacement selector rejects in every model at all three ridge settings
(`p=.00062--.00233`, all Holm-adjusted `p<=.00933` across the grid), yet all 15 selected
sites differ from the unwhitened trace-ratio
site. Raw recoverability is therefore not unique to the trace ratio, but the reported layer
and branch are not stable to a reasonable change in localization statistic. This grid does
not establish a causal site or cross-panel transfer.

## What is measured

For each example and selected text-Transformer layer, the study saves one vector for each
of three operational streams:

| Stream | Scientific meaning |
| --- | --- |
| Token-mixing update | The attention, linear-attention, or other sequence-mixer contribution before residual addition |
| Feed-forward update | The MLP/feed-forward contribution before residual addition |
| Residual state | The complete layer output after residual updates |

The primary estimator is the final valid prompt token during prefill. It is therefore a
depth-wise representation profile, not a complete token-by-token account of reasoning.
Mean-token pooling and decode-time trajectories are sensitivity conditions. Raw features
are analyzed within model and never combined across incompatible hidden widths.

The first real decode-time smoke passed exact capture and observer-off/on response parity
for OLMo 3 on two tasks. It yielded complete prefill and `decode_step=0` coverage for all
16 examples and no `decode_step=1` records. The zero-based decode step is the cached forward
conditioned on the first emitted answer token, not the state that produced that token.
Consequently, the frozen direct-answer protocol supports only a prompt-versus-post-answer-
token sensitivity. A multi-token explanation trajectory is a separate future protocol;
absolute token positions must not be described as semantic reasoning steps.

The separate long-response protocol is frozen in
`experiments/representation_dynamics_response_trajectory_v1.json`. It requests a final-answer
marker followed by exactly two short explanatory sentences totaling at most 60 words for all
16 MMLU subjects, records at most 32 cached decode updates, and requires at least 95%
coverage in every task for a common prefix of eight or more updates. Its primary statistic
is the selection-corrected maximum increase in Fisher ratio from prefill over layer,
stream, and response position. A corrected OLMo 3 smoke over two tasks and four examples
per task passed parsing, exact observer-off/on response parity, exact capture, and all 32
stored positions (25,344 vectors). This validates the sampling protocol only. The full
8+8-task, 16-example-per-task extraction produced 811,008 finite records with complete
32-position coverage, but failed exact observer-off/on response parity and was rejected
before analysis.
The answer-first order makes this a post-answer response trajectory; it cannot identify
computation leading to the answer.

The completed normalized pilot uses eager attention after two independent observer-off
replays matched exactly. Its 4+4 tasks and four examples per task contribute uncapped valid
responses to eight equal-width bins. The selection-aware maximum Fisher increase is
`0.2861` at layer 31's token-mixing update in bin 5/8 (`0.1934` at prefill to `0.4795`),
with exact `p=2/70=.0286`. Depth-mean Fisher instead falls in every stream, and the selected
site's within-group variance rises while its effective-rank fraction is stable. The result
is descriptive: the design cannot attain the paper's `.01` threshold, bins are not semantic
steps, and only post-answer dynamics are observed.

The quantitative outcomes are PCA-space Fisher separation; effective and stable rank;
the separate between-group and within-group Fisher terms; task-held-out linear and RBF
prediction; task-level label randomization; multiplicity-adjusted probe values; cosine
silhouette; explicitly paired linear CKA where valid; and order-invariant covariance
alignment otherwise. Low-dimensional visual inspection is not primary evidence.

Layerwise activations, probes, and representation-similarity statistics are established
observational analysis methods. They do not exhaust current model-monitoring or
interpretability practice. Layerwise predictive decoders track the output distribution,
sparse feature models address superposition, and activation patching or ablation tests
causal relevance. The current final-prompt-token prefill estimator is intentionally a
model-agnostic depth profile; it is not a complete token-by-token account of inference.

## Frozen current-model experiment

The cross-model study deliberately uses current open-weight generations: OLMo 3 7B
Instruct, Apertus 1.5 8B, Qwen 3.5 9B/27B, and Gemma 4 31B. Older OLMo, Qwen, and
Gemma generations are excluded from the main comparison. The initial study used two
five-task-per-category panels with 100 examples per task:

- a within-family MMLU-Pro panel with five knowledge-oriented and five
  reasoning-oriented subject proxies under one prompt and answer format; and
- an exploratory cross-benchmark panel spanning closed-book Natural Questions,
  no-context TriviaQA, MedQA, SuperGPQA, MMLU-Pro, GSM8K, MuSR, ARC-Challenge, and
  HellaSwag.

The later primary panel is a separate MMLU 8+8 design with 100 examples per subject. It is
not an expansion of the MMLU-Pro pilot: the dataset, subject composition, prompt, and answer
space differ. Its larger task set supplies 6,435 label-complement pairs for exact
task-level inference. Conclusions are therefore compared across panels, not pooled across
them.

The labels follow source task objectives only as comparative working labels. GSM8K and
MuSR were constructed around multi-step reasoning. Natural Questions and no-context
TriviaQA provide the clearest closed-book factual-retrieval conditions. ARC-Challenge
explicitly requires both science knowledge and reasoning; SuperGPQA explicitly measures
both; HellaSwag measures commonsense continuation inference rather than formal reasoning.
The heterogeneous panel therefore remains a stress test, not a construct-valid
knowledge--reasoning contrast.

Exact document IDs and processed-dataset fingerprints were frozen before model loading.
The extraction produced 100 model--task artifacts and 1,320,000 exact layer/stream vectors.
They passed the original count, finiteness, semantic-exactness, stack, source, and
cross-model sample-ID checks, and uniform cuML 26.06 analysis completed 1,320/1,320 rows.
A later prompt-protocol audit found a material MuSR adapter defect: the pinned dataset stores
its two choices as a string-encoded list, which the adapter iterated character by character.
All 500 original MuSR examples across five models are invalid. Corrected extraction renders
exactly two choices for every replacement record and passes the protocol audit. The full
cross-benchmark panel has been reanalyzed. A protocol audit is now a mandatory validation gate.

Direct-answer scores are study-protocol values, not standard leaderboard scores. Apertus
used the Transformers Python xIELU fallback because the optional CUDA kernel was absent;
that runtime condition is frozen. Model paths with incompatible managed reasoning traces
were excluded rather than receiving model-specific answer-generation rules.

## Current evidence

The table below records the corrected current-model comparison.

| Model | MMLU-Pro 5+5 | MMLU 8+8 | Cross-benchmark stress test |
| --- | --- | --- | --- |
| OLMo 3 7B | Layer 3, feed-forward | Layer 3, feed-forward | Layer 5, token-mixing |
| Apertus 1.5 8B | Layer 0, feed-forward | Layer 0, feed-forward | Layer 8, token-mixing |
| Qwen 3.5 9B | Layer 1, token-mixing | Layer 2, feed-forward | Layer 11, token-mixing |
| Qwen 3.5 27B | Layer 11, feed-forward | Layer 1, token-mixing | Layer 5, feed-forward |
| Gemma 4 31B | Layer 12, token-mixing | Layer 23, feed-forward | Layer 14, feed-forward |

For linear-probe F1, BH-adjusted $q<0.05$ occurred in 81/96 OLMo, 70/96 Apertus 1.5,
0/96 Qwen 9B, 92/192 Qwen 27B, and 88/180 Gemma 4 rows in the valid matched panel. At
the FDR maximum, Apertus 1.5 uses layer 0's feed-forward update (FDR .436).

For the historical exhaustive RBF-probe F1 screen, the corresponding matched-panel counts were 88/96 OLMo,
88/96 Apertus 1.5, 77/96 Qwen 9B, 154/192 Qwen 27B, and 0/180 Gemma 4 rows. The nonlinear
boundary detects Qwen 9B but not Gemma 4 after correction; it therefore changes the
model-specific MMLU-Pro result. The corrected cross-benchmark panel has no unadjusted
linear or RBF rejection. Because cuML RBF replay and complement symmetry are not
score-exact, these nonlinear counts are threshold-stability observations rather than exact
randomization evidence.

The now-invalid cross-panel token-mixer/MLP/residual FDR correlations were
`.565/.865/.917` for OLMo, `.570/.003/.015` for Apertus, `.353/.647/.607` for Qwen 9B,
`.633/.883/.922` for Qwen 27B, and `.144/-.051/.228` for Gemma 4. High profile agreement
for OLMo and Qwen. These values are retained only to identify superseded outputs; they are
not scientific evidence.

In the MMLU-Pro 5+5 panel, a post-hoc scalar prompt-length probe did not reject the
task-label null in any condition (`p>=.167` matched and `p>=.905` cross-benchmark for
macro-F1). Those values do not apply to the distinct MMLU 8+8 panel. In MMLU 8+8, mean
knowledge/reasoning prompt-token counts are `179.0/135.3` for OLMo, `172.0/127.1` for both
Qwen sizes, `172.2/129.5` for Gemma, and `206.6/163.5` for Apertus 1.5. An exact word
1--2-gram TF--IDF baseline over the logged prompt text obtains macro-F1 `0.701` with
`p=0.00870`; a character-within-word 3--5-gram baseline obtains `0.768` with
`p=0.000777`. Prompt content therefore predicts the primary partition under the same
task-held-out null. Historical raw generations are unavailable and are not reconstructed.

### Appendix-only heterogeneous multimodal panel

A separate side analysis reuses frozen Qwen 3.5 9B/27B and Gemma 4 31B artifacts for five
historically knowledge-oriented and five historically capability-oriented tasks. Each task
contributes 100 examples. All 30 artifacts pass count, finiteness, semantic exactness, and
prompt-protocol checks. The task is the inferential unit: all 252 balanced oriented
assignments are retained, the Fisher maximum is reselected under each assignment, and the
probe is refit for each of 126 label-complement pairs.

Raw macro-F1 is `.879`, `.858`, and `.899` for Qwen 9B, Qwen 27B, and Gemma, with exact
p-values `.0159`, `.0317`, and `.0159`. The Fisher-maximum p-values are `.0238`, `.0159`,
and `.0476`. None meets the declared `.01` criterion. Surface/modality adjustment lowers
F1 to `.210--.538`; character-TF--IDF adjustment gives `.550--.735`; their combination
gives `.151--.336`. All adjusted p-values are at least `.0635`. Prompt-only word and
character TF--IDF probes each reach approximately `.80` F1 with `p=.0159`.

At the observed Fisher-selected sites, descriptive RBF F1 is `.837/.850/.799`, cosine
silhouette is `.282/.256/.434`, covariance alignment is `.421/.375/.080`, and within-group
effective-rank fraction is `.109/.326/.418` in Qwen 9B, Qwen 27B, and Gemma order. Paired
linear CKA is undefined for all three conditions because the two groups have no declared
cross-group example pairs. It remains missing rather than being computed under an arbitrary
row alignment; covariance alignment is the order-invariant geometry comparison.

This panel is not treated as construct evidence. All five capability-oriented tasks supply
images, versus three of five knowledge-oriented tasks, and domain, benchmark family,
response format, and prompt content also differ. The image-presence control is a task-level
indicator, not a representation of pixels, visual tokens, or image content. The result is
therefore a bounded sensitivity analysis for the ten named tasks only.

## Suitability of the current design

| Claim | Current status |
| --- | --- |
| Complete three-stream extraction | Suitable for the tested OLMo 3, Apertus, Qwen 3.5, and dense Gemma 4 paths |
| Separation of the named task sets | MMLU-Pro evidence disappears under six-feature adjustment; MMLU 8+8 is predictable from prompt text, while 64-component lexical adjustment leaves no model below the declared `.01` criterion but four below a nominal per-model `.05`; corrected cross-panel probes reject nowhere before adjustment |
| Linear separation | Directly testable with task-held-out probes |
| Nonlinear separation | Exhaustive RBF assignments were run, but numerical complement/replay instability must be resolved before exact inference; no cross-panel row rejects |
| General knowledge versus reasoning distinction | Not suitable yet; construct confounds dominate |
| Stable layer or branch localization | Not established; pilot maxima moved |
| Causal importance or pruning safety | Not tested |
| Inference-speed improvement | Not tested |

The central limitation is construct validity rather than activation extraction. One
thousand examples per model/panel improve estimation, but the label is assigned to only ten
independent task groups. Effective inferential diversity is therefore closer to ten tasks
than one thousand examples.

Scaling to more examples from the same tasks cannot solve this limitation. The completed
cross-benchmark panel adds independent tasks, but it does not balance domain, answer type,
difficulty, or input length. A future confirmatory design must add task groups under a
frozen taxonomy and report task-resampling stability. Labeling more benchmarks after
inspecting these results would remain exploratory.

Additional risks are:

- The final prompt-token state can encode prompt surface, subject vocabulary, modality, or
  answer format rather than problem-solving behavior.
- Accuracy, uncertainty, prompt/output length, and task format can separate groups.
- The cross-benchmark panel is text-only but mixes domains, formats, and answer spaces.
- The appendix multimodal panel additionally confounds task label with image presence and
  does not control image content; it is excluded from primary inference.
- The MMLU-Pro control reduces dataset/format variation but retains subject/domain confounds.
- Unpaired benchmark observations do not support ordinary CKA; it must remain missing.
- Token-mixing updates are operationally comparable within model, but Qwen's recurrent or
  linear mixer is not mechanistically identical to Gemma's attention.
- Passive separability does not imply that a component is causally necessary or removable.
- Observer-off/on greedy output parity covers only two examples per model; logit parity and
  full-panel overhead remain open.
- A falling effective rank can raise FDR by shrinking the within-group term. Effective
  rank, stable rank, the mean-distance term, and within-group variance must be read
  together. Dividing FDR by effective rank is not used because it would amplify this case.
- Format variables can support strong probes even under a semantically motivated task
  partition. The scalar prompt-length diagnostic is insufficient; the next sensitivity
  analysis fits multivariable nuisance models inside each task-held-out fold. Its frozen
  prompt-only design uses model-tokenizer token count, character and line counts, option
  count, whitespace-token count, and whitespace-token entropy. It excludes target,
  correctness, response, and output-length variables from prefill residualization because
  those are downstream of the measured state.

## Statistical implementation status

The grouped-fold implementation uses one seeded global task order, filtered under every
balanced label assignment. Fisher and RBF complement invariance are evaluated from the
oriented assignment scores. Linear accuracy and macro-F1 reuse one fitted orientation for
each label-complement pair because exchanging the binary class names leaves these scores
unchanged; their equality is imposed and is not a numerical replay check. A symmetry
failure in an independently evaluated statistic invalidates exact inference. The current
cuML RBF path has documented replay variation and is not presumed deterministic.
The original cross conditions were invalidated by the MuSR audit. Every corrected row and
every valid matched row uses all 251 non-observed assignments. The 252 oriented assignments
form 126 complementary pairs for class-symmetric statistics, so the minimum attainable
p-value is $1/126$.

The frozen extraction manifest requests zero nonlinear permutations. A separately named,
hash-bound sensitivity analysis overrides only that setting and enumerates all 251
non-observed task assignments for every RBF row. It reproduces the observed scores and
joins only nonlinear p/q values to the baseline by model, phase, layer, stream, and module.
CKA remains undefined for ordinary unpaired benchmark examples and is correctly emitted as
missing, not zero.
Benjamini--Hochberg adjustment uses the complete declared layer--stream family. Failed or
undefined hypotheses remain missing but still contribute to the family size, preventing an
anti-conservative reduction of multiplicity after failures.

The prompt-surface sensitivity is a second, separately named analysis over the same
immutable vectors. For each observed or permuted task assignment, a multi-output Ridge
map with fixed alpha 1.0 is fitted on training-task nuisance features and activation
vectors. Held-out tasks are residualized with that fit before fold-local PCA and probing.
The unadjusted probe remains in the same output for direct comparison. In the matched
panel, the linear/RBF rejection counts change from 81/88 to 0/0 for OLMo, 70/88 to 0/0
for Apertus 1.5, 0/77 to 0/0 for Qwen 9B, 92/154 to 0/0 for Qwen 27B, and 88/0 to 0/0
for Gemma 4. The adjusted minimum RBF q-values range from .153 to .190. Thus the original
subject-set prediction is not robust to the six recorded prompt features. Residualization
may also remove construct-correlated variance and does not prove that the measured features
caused the original separation.

The nuisance table
contains only numeric aggregates, is keyed by model/task/sample identity, and is bound by
SHA-256; raw prompts and responses are not copied into analysis outputs. This control does
not make the benchmark labels construct-valid.

The lexical sensitivity fits lowercased character-within-word 3--5-gram TF--IDF and
randomized truncated SVD inside each training-task fold. A multi-output Ridge map removes
linearly predicted activation variance before the existing fold-local PCA and probe. The
unadjusted Fisher maximum is reselected for every balanced assignment; the observed site is
never fixed across the null. Across five models and both 32/64-component settings, all ten
12,870-assignment reports are complete and complement-symmetric. This analysis tests one
finite linear nuisance representation; it neither removes nonlinear lexical information
nor identifies a causal prompt mechanism.

Repeated cuML RBF fits are not bitwise reproducible under concurrent MPS execution. Across
five matched replay conditions, the strict baseline verifier records 145 differences in
RBF scores or p/q fields. No non-RBF common field differs after excluding the already
known silhouette routine, and no RBF q<.05 decision changes. The exact verifier remains
failed; a separate numerical-stability audit records the stable threshold decisions. This
prevents threshold stability from being misreported as exact estimator replay.

Separately, four of 1,320 silhouette values from cuML's convenience
routine differed from the baseline by at most 0.01643.
An explicit batched float64 cosine-distance calculation matched the baseline within
4.44e-15 and repeated exactly. The maintained implementation uses that calculation. The
reported summary preserves baseline geometry and imports only RBF inferential fields from
the nonlinear pass.

RAPIDS/cuML 26.06 is implemented as a separately identified estimator, not a transparent
substitution. On one real 1,000-example row it produced similar PCA/FDR and RBF geometry
but changed raw-space logistic accuracy from 0.879 to 0.898. The frozen cuML path promotes
stored vectors to float64, uses full GPU PCA plus an explicit 95% cumulative-variance
cutoff, and fits QN logistic/RBF probes in PCA space; covariance alignment and cosine
silhouette remain in the original space on GPU. The current five-checkpoint cuML run is complete;
historical scikit-learn/liblinear and older-model outputs remain separate references and are not pooled.
Historical tables fit PCA once on all unlabeled vectors in a layer before grouped
cross-validation. The current analysis fits PCA and standardization on training tasks
inside every fold and records that scope in each row. Historical rows are not pooled with
the current results.

Effective rank uses the entropy of normalized singular values and is computed for the full
row, each task group, and the data after subtracting each group mean. Stable rank and the
raw Fisher parts are also recorded. These are checks on the geometry; they do not create a
new adjusted Fisher score.

## Potential value for model development

The present measurements are a development diagnostic, not yet a control signal. Their
most defensible uses are:

- **Training and release comparison:** track accuracy together with separation, effective
  rank, and within-group variance across checkpoints. This can detect representation-profile
  changes that an aggregate benchmark score does not expose.
- **Benchmark audit:** compare the named partition with arbitrary splits and prompt-adjusted
  analyses. If the evidence disappears, the task taxonomy may be tracking task identity or
  prompt form rather than the intended demand.
- **Intervention prioritization:** use replicated layer--stream profiles to nominate a small
  set of branches for ablation, targeted adaptation, steering, or compression. Localization
  reduces the experimental search space; it does not establish that changing a site is safe
  or useful.
- **Conditional inference, if validated:** a signature that transfers to unseen tasks and
  predicts causal benefit could support conditional compute, retrieval or tool routing, or
  regression alarms between releases.

The current evidence also supplies a useful negative constraint. Selected sites differ
across models, prompt adjustment removes the earlier probe evidence, and the normalized
response pilot has a selected late-layer increase while its depth-mean Fisher ratio falls.
Consequently, the repository does not support a universal knowledge/reasoning router, a
shared layer rule, or pruning policy. Before any such use, require independent held-out
tasks, calibrated false-positive rates, observer parity, and a matched intervention that
measures task quality, latency, throughput, and peak memory.

## Stage and next gates

### Stage A — measurement validation: nearly complete

- Complete: exact semantic streams for all five current model conditions.
- Complete: opt-in observation, cleanup, completeness, finiteness, and sample identity.
- Complete: two-example deterministic output parity for each model.
- Remaining: logit parity, broader output parity, observer latency, and peak-memory overhead.

### Stage B — task-set recoverability: exact prompt comparison and sensitivity complete

- Complete and valid extraction: 50 matched-panel artifacts and five model conditions.
  The historical row-wise probe p-values already include all 251 non-observed assignments,
  but their advertised floor was wrong: class-symmetric statistics have 126 complementary
  outcomes and minimum p-value 1/126.
- Complete on 2026-08-13: selection-aware Fisher reanalysis of all 660 matched-panel rows,
  with every maximum reselected under all 252 assignments. Model-level maximum p-values are
  `0.0159`, `0.0238`, `0.0317`, `0.0317`, and `0.0238` for OLMo, Apertus 1.5, Qwen 9B,
  Qwen 27B, and Gemma 4. All exceed their null 95th percentile, but none meets the
  prespecified `0.01` threshold. The alphabetical 5/5 reference has maximum p-values
  `0.397`--`0.571` and zero rejected rows. Outputs and aligned null vectors are stored under
  `/home/ubuntu/data-imu-us-east3/vlmbehave-artifacts/2026-08-13/`.
- Complete: the optimized selection-aware probe at the Fisher-selected site. Fisher
  selection and grouped probing were repeated over all 12,870 task assignments. Macro-F1
  p-values are `0.00155`, `0.00186`, `0.00186`, `0.00420`, and `0.00404` for OLMo,
  Apertus 1.5, Qwen 9B, Qwen 27B, and Gemma 4. All five reports pass exact complement
  symmetry.
- Complete on MMLU-Pro 5+5: six-feature fold-local prompt sensitivity; every current model
  falls to zero adjusted linear/RBF rows. This is non-robustness to measured cues, not
  causal attribution.
- Complete on MMLU 8+8: length-only adjustment rejects in all five models; six-feature
  adjustment rejects in OLMo, Apertus 1.5, and both Qwen sizes, while Gemma gives
  macro-F1 `.732`, `p=.0137`. These scalar controls do not remove the prompt-only lexical
  alternative and are not mediation estimates.
- Complete on MMLU 8+8: 32/64-component character TF--IDF residualization, with vocabulary,
  SVD, activation Ridge map, PCA, scaling, and probe fit without held-out tasks. At 64
  components macro-F1 is `.613--.698` and exact p-values are `.01197--.06605`; none meets
  `.01`, while four are below a nominal per-model `.05`. The 32-component sensitivity
  reaches the same `.01` decision. All ten reports pass exact support, complement, revision,
  and hash gates.
- Complete on MMLU 8+8: an exact paired representation-minus-prompt test under one
  64-feature cuML estimator family. The activation site is reselected for each of all
  12,870 assignments; both arms use the same grouped folds, scaling, classifier, and
  regularization. Representation macro-F1 exceeds prompt macro-F1 by `.017--.070`, but
  the paired exact p-values are `.177--.440`. This does not establish equivalence; it says
  the observed advantage is not unusually large among balanced partitions under this
  estimator.
- Complete diagnostics: all assignment-specific train/test folds are class balanced.
  Lexical adjustment retains `31--54%` of between-task variance, `57--76%` of total
  variance, and high example-split 16-way task-identity performance at each selected site.
  The sensitivity result therefore is not explained by deletion of all task identity.
- Complete fixed-site positive calibration: an injected activation-only direction aligned
  with an arbitrary 8+8 split is passed through the full lexical-64 adjustment and exact
  probe. Detection is 0/10 directions at standardized mean differences `.5/1/2` and 5/10
  at difference `4`. This shows the adjusted instrument can reject known large signals but
  is direction-dependent; it does not calibrate layer selection or establish a model
  mechanism. A post-hoc `5/6/8` extension was terminated with zero reports because the
  unchanged solver became prohibitively slow near separability. It contributes no evidence.
- Complete: corrected MuSR extraction, task-aware validation, and exact cross-benchmark
  reanalysis. Every unadjusted linear/RBF row is non-significant. Prompt adjustment creates
  66/96 linear rejections only for Apertus 1.5, so residualization is itself a sensitivity.
- Complete in code: training-fold PCA and GPU rank diagnostics.
- Complete for the valid matched panel: uniform fold-local cuML analysis.
- Complete engineering smoke: one record per generated-token update with an explicit decode
  step, 3,072/3,072 finite exact vectors, and observer-off/on response parity. The frozen
  direct-answer outputs provide only prefill and `decode_step=0`; no example reaches the
  next cached update, so the prespecified eight-step contrast is not estimable here.
- Complete long-response protocol smoke: all eight OLMo responses parse, observer-off/on
  outputs match, and 25,344 exact vectors cover prefill plus 32 generated-token positions.
  A 16-task absolute-position extraction produced 811,008 finite records and complete
  32-position coverage, but failed observer-off/on response parity. It was rejected before
  analysis and supplies no separation result.
- Required sensitivity analyses: correctness, prompt/output length, visual-token count,
  pooling, the separately frozen long-response trajectory, batch sensitivity, and task
  influence/leave-one-task-out results.

### Stage C — higher-powered confirmation: primary Fisher analysis complete

- Build a factorial design crossing visual dependence with external-knowledge demand.
- The frozen main-track panel uses eight matched MMLU subjects per category, giving 6,435
  complementary balanced partitions and minimum class-symmetric p near `1.55e-4`. All 80
  model--task artifacts and 660 analysis rows passed. Fisher maximum p-values are `0.0721`
  (OLMo), `0.00435` (Apertus 1.5), `0.00109` (Qwen 9B), `0.000777` (Qwen 27B), and
  `0.01445` (Gemma 4). Thus three of five pass the prespecified `0.01` threshold. The
  selected sites differ across models, and every alphabetical reference split is
  non-significant. This is evidence about the named subject partition, not a universal
  geometry or shared circuit.
- The optimized selection-aware secondary probe is complete. It fits one class orientation
  per complementary pair, reducing computation to 6,435 probe refits per model while
  preserving every oriented score. All five models reject at `p<0.01`, including OLMo and
  Gemma, whose Fisher maxima do not. A word TF--IDF baseline also rejects the primary
  partition (`p=0.00870`), and a character TF--IDF baseline is stronger (`p=.000777`), so
  the unadjusted rejection alone does not establish evidence beyond prompt text. Length-only
  adjustment retains rejection in all five models; six-feature adjustment retains it in
  four, with Gemma at `p=.0137`. These are post-hoc sensitivities;
  correctness-conditioned sensitivity is support-limited and remains non-primary.
- The correctness support audit is complete and hash-bound. A model-wide balanced
  correct/incorrect subset would be capped at 26 examples per task/outcome for OLMo, 13
  for Apertus, 8 for Qwen 9B, 3 for Qwen 27B, and 7 for Gemma. A naive outcome-balanced
  subset would therefore selectively remove power from the strongest Qwen conditions.
  Keep all examples primary; any correctness-conditioned estimator must retain all tasks,
  report these cells, and acknowledge that correctness is downstream of the prefill state.
- Add independent tasks under the frozen taxonomy and report stability as task count grows;
  do not substitute more examples from the same tasks for task-level replication.
- Match or intervene on domain, prompt style, response format, length, difficulty, and
  accuracy rather than treating benchmark names as ground truth constructs.
- Predeclare primary models, sites, metrics, exclusion rules, seeds, and multiplicity family.
- Replicate on held-out tasks and at least one independent model family.

The causal follow-up is now frozen, but not implemented or executed, in
`experiments/representation_dynamics_causal_validation_v1.json`. It fixes each model's
8+8-discovery site, uses 16 non-overlapping MMLU subjects, attenuates one exact branch by
50%, and compares the category interaction with five equally budgeted same-stream random
layers. Because the site is selected without intervention outcomes from the new panel, its
12,870 task-label assignments can test the held-out behavioral interaction without
reselecting the site. A positive result would establish sensitivity to branch attenuation,
not exclusive circuit identity or pruning safety.

Sparse autoencoders are not part of Stage B. Using a token-level encoder for knowledge
examples and a step-level encoder for reasoning examples would make the measurement itself
depend on the label. The current concise MMLU control also has no reasoning trace or step
boundaries. A later sparse-feature study must either train one shared encoder without task
labels on separate data and apply it identically, or define a separate process-trace
experiment with explicit steps.

### Separate study — change across OLMo training

OLMo 3 checkpoints can support a useful follow-up, but they should not be mixed into the
current cross-model table. Start with 7B and keep each training path separate: base
pretraining, SFT, general instruction tuning, RL-Math, and RL-Code are branches, not one
continuous sequence.

- Use the same examples, plain prompts, pooling, precision, and model code at every
  checkpoint. Report task quality because a probe can recognize task wording even when an
  early checkpoint cannot answer correctly.
- Measure finite FDR change per training token, with whole-task resampling intervals.
  Report the Fisher numerator, within-group variance, and effective rank beside FDR; do
  not divide FDR by rank.
- Sample checkpoints densely around stage changes and sparsely within stable stretches.
  A changed intervention effect localizes an interval between checkpoints, not an exact
  training step.
- Train a shared sparse crosscoder only after the geometric change repeats. Require
  held-out reconstruction, matched sparsity, several seeds, and the same feature
  measurement for both task groups.
- Use controlled activation replacement or branch ablation with random-layer and
  matched-magnitude controls before using causal language.

The active `lmms-eval` path now has a plain-text causal-model adapter that uses the same raw
prompt for base and post-trained checkpoints and enters the shared representation context.
The source-grounded OLMo mapping observes `post_attention_layernorm` and
`post_feedforward_layernorm`, the exact residual additions in the installed OLMo 3
implementation. Synthetic tests cover context propagation, exact hook selection, and
complete three-stream collection.

A real `stage1-step10000` 7B smoke passed stricter tracker-off/on checks: forward logits
were bit-identical, generated token IDs were identical, all 32 layers and 96 hooks were
present, and no captures were skipped. A consolidated ten-task, 32-example-per-task debug
run then produced exactly 30,720 vectors and passed the artifact validator. This validates
the early checkpoint and this installed model implementation. The full trajectory retained
the same exact hook mapping and passed all semantic and completeness gates.

The frozen longitudinal manifest uses seven primary base checkpoints. It evaluates three
fixed, disjoint blocks of 32 examples per task at offsets 0, 32, and 64. The blocks are
paired across checkpoints. Mean and sample standard deviation across blocks measure
example-sampling sensitivity, not variation across training seeds. Exact repeated
extractions at selected endpoints measure numerical reproducibility separately.

Do not add nominal generation seeds as if they were independent model runs. Primary
extraction is greedy and deterministic, so the same checkpoint and examples yield the same
prefill states. Different probe-fold seeds are a useful analysis sensitivity check, but
they do not estimate model uncertainty. Training-seed confidence requires independently
trained trajectories; only one OLMo 3 trajectory is currently available. Keep exact
task-label permutation evidence, example-block variability, fold sensitivity, and
training-seed variability as separate quantities.

The runner loads each checkpoint once per block and evaluates all ten tasks before
unloading it. It validates exact task counts, finiteness, three-stream/layer completeness,
matching sample IDs across checkpoints within a block, and disjoint IDs between blocks.
The summary reports every stream separately, depth-mean FDR and its numerator/denominator,
rank fractions, task-held-out probes, covariance alignment, silhouette, task accuracy,
and within-stage FDR velocity. Checkpoint counters reset at each pretraining stage, so
velocity is not computed across stage boundaries until exact cumulative token positions
are available.

The primary trajectory is complete. All 21 checkpoint–block extraction conditions passed:
645,120 vectors, 32 layers, three exact streams, ten tasks, 32 examples/task, no skipped or
non-finite records, identical sample IDs across checkpoints within each block, and
disjoint IDs between blocks. Uniform cuML analysis completed 2,016/2,016
layer–stream–block rows with zero failures and all 251 non-observed task-label assignments
per row. The committed sample-free aggregates are in
`results/olmo3_longitudinal_v1/`; raw artifacts remain local under
`logs/olmo3-longitudinal-v1/`.

The result does **not** support a general monotonic-divergence claim. From the earliest
Stage-1 checkpoint to the final Stage-3 checkpoint, mean knowledge/reasoning accuracy
increased from `0.283/0.279` to `0.585/0.460`, but depth-mean FDR decreased from
`0.197/0.291/0.275` to `0.123/0.145/0.156` for attention/MLP/residual streams.
Cosine silhouette and RBF-probe F1 also decreased overall, and covariance alignment did
not show a common increase in divergence. Linear-probe F1 rose modestly from
`0.811/0.806/0.814` to `0.837/0.842/0.849`; corrected probe-F1 significance repeated in
all three blocks at 0/96 layer–stream rows for every Stage-1 checkpoint, 83/96 and 76/96
for the Stage-2 checkpoints, and 95/96 and 91/96 for Stage 3. This supports increasingly
stable *linear predictability* after Stage 1, not uniformly diverging geometry.

Within-group effective-rank fractions increased overall from `0.669/0.654/0.700` to
`0.694/0.683/0.716`, so the endpoint FDR decrease is not caused by rank collapse. Both the
Fisher numerator and denominator changed sharply at stage boundaries. The early checkpoint
already had probe F1 near `0.8` while task accuracy was near four-choice chance; subject,
domain, or prompt identity therefore remains a plausible source of predictability.

The matched-panel scalar prompt-length diagnostic was regenerated on 2026-08-13 from the
corrected five-model artifact set, including Apertus 1.5 rather than the superseded
Apertus 1.0 condition. Task-level macro-F1 permutation p-values range from `0.167` to
`0.270`. This is only a one-feature linear diagnostic and does not remove prompt-surface,
domain, or difficulty confounding. The CSV and a 50-artifact SHA-256 provenance report are
under `results/representation_dynamics_text_expansion_corrected_v1/`.

### Apertus 1.0 pretraining trajectory

The official `swiss-ai/Apertus-8B-2509` repository now supports a controlled
within-run pretraining study. A live audit on 2026-08-13 found 72 branches and no tags:
44 branches are token-indexed checkpoints from 210B tokens through the 15T pretraining
endpoint, 27 are a separate long-context continuation, and `main` is the released model.
All 11 selected revisions have identical `config.json` and `tokenizer_config.json` hashes.
API tree checks at the earliest, first post-boundary, and final revisions each expose four
weight shards totaling 16.1 GB.

The technical report gives unusually useful intervention points. For the 8B run, Stage 1
lasts to 7T tokens, Stage 2 is skipped, Stage 3 runs to 12T, Stage 4 to 13.5T, and Stage 5
is the 13.5--15T cooldown. Stage 3 increases mathematical data; Stages 4 and 5 further
change data quality and the math/code/task mixture. The frozen manifest therefore selects
11 checkpoints: three within Stage 1, three within Stage 3, two within Stage 4, and three
within Stage 5. It uses cumulative billions of consumed tokens as the position unit and
computes velocity only within a documented stage.

The protocol is in `experiments/apertus1_longitudinal_v1.json`; verify it with:

```bash
.venv/bin/python scripts/check_apertus1_release.py --require-ready
.venv/bin/python scripts/run_olmo_longitudinal_experiment.py \
  --manifest experiments/apertus1_longitudinal_v1.json plan
```

The second command reuses the validated generic mechanics in the historically named OLMo
runner. The full plan is 11 checkpoints × 3 paired, disjoint blocks × 10 tasks × 32
examples/task, or 1,013,760 vectors. The 210B, 7.047T, and 15T checkpoints pass exact stack
inspection and same-process observer-off/on/removed parity: 32 contiguous decoder layers,
96 exact hooks, bit-identical forward logits and greedy token IDs, 384/384 finite float32
vectors, no skipped captures, and zero hooks after cleanup. These checks used BF16 weights
and the Transformers Python xIELU fallback.

A separate full `ai2_arc` check uses the released post-long-context `main` revision and the
official zero-shot probabilistic-scoring task definitions. It obtains 81.94%/52.73% raw
accuracy on ARC-Easy/ARC-Challenge (82.24%/56.48% length-normalized), or 72.29% pooled raw
accuracy over 3,548 examples. The Apertus report gives 72.7%. The YAML task definitions are
byte-identical to the Swiss-AI harness at its last commit before the report, but the report
does not specify how Easy and Challenge are pooled and the values are not identical. This
is evidence that the released model loads and answer-choice scoring works; it is not an
exact reproduction and it does not validate accuracy at the pre-long-context checkpoints.
The direct-answer MMLU scores in the trajectory are study-protocol outcomes, not leaderboard
scores.

The full clean-source run is complete. All 33 checkpoint--block artifacts and their 3,168
layer--stream analysis rows pass, aggregating to 1,056 checkpoint-level summary rows. From 210B to 15T tokens, knowledge/reasoning accuracy changes from
`0.260/0.254` to `0.621/0.433`, while depth-mean Fisher changes from
`0.174/0.273/0.252` to `0.235/0.337/0.323` for mixer/MLP/residual streams. The direction
contrasts with OLMo, where all three endpoint Fisher summaries fall. Apertus is also
non-monotonic: at the Stage-3--4 boundary, both accuracies rise while all three Fisher
summaries fall. Sample-free aggregates are in `results/apertus1_longitudinal_v1/`.

The 27 `longctx-*` branches are not mixed into this curve because context length, RoPE,
and the data mixture change across four phases. The instruct repository's `sft` and `main`
refs likewise require a separate post-training prompt/decode protocol. Apertus 1.0 and 1.5
remain separate: version 1.5 bundles multimodal continued pretraining and revised
post-training and currently exposes only its final ref.

### Apertus 1.5 endpoint anchor and prospective longitudinal study

`swiss-ai/Apertus-v1.5-8B` is integrated as a text-only, non-thinking endpoint candidate at
immutable revision `a411d838600baf0e3635a3daf66fb7c55fc97bb6`. Its current model code has
32 text-decoder layers, and the direct attention and feed-forward outputs are the exact
residual contributions. The runtime is pinned to the official Swiss-AI Transformers commit
`3797303dda74844e3d1f8977ff5518bb91f818b4` until support reaches an upstream release.
The isolated endpoint smoke collected exactly 384 finite float32 vectors from four examples
across 32 layers and three streams, with no skipped captures. Monitoring disabled versus
enabled preserved document IDs, generated answers, parsed scores, and aggregates; a
same-process forward check produced bit-identical logits and confirmed hook cleanup. The
runtime used the Python xIELU fallback because the optional CUDA kernel was absent.

The full endpoint anchor applies the OLMo trajectory's ten MMLU subjects to three disjoint
blocks of 32 examples per task. All 92,160 vectors and 288 layer--stream rows passed with
zero analysis failures. Across blocks, knowledge/reasoning accuracy was
`0.358±0.036/0.306±0.022`; depth-mean Fisher ratios were `0.204±0.016`, `0.241±0.018`,
and `0.240±0.018` for token mixing, feed-forward, and residual streams. Corrected linear
probe-F1 evidence repeated in all three blocks at 87/96 sites. This shows that the subject
partition is repeatedly recoverable at the final endpoint. It does not show whether any
metric increased during Apertus training. The blocks estimate evaluation-example
sensitivity, not variation across independent training runs. Hash-bound, sample-free
aggregates are committed under `results/apertus15_endpoint_anchor_v1/`.

This is not yet a longitudinal result. On 2026-08-11 the Hugging Face repository exposed
only `main` and no tags, while the model card said intermediate checkpoints would be
published later. Apertus 1.0 and 1.5 are not treated as two controlled time points: the
new release bundles continued pretraining, multimodal data, and post-training changes. The
release-gated protocol is frozen in `experiments/apertus15_longitudinal_v1.json`; run
`python scripts/check_apertus15_release.py` to recheck availability. Promotion requires at
least two immutable checkpoints on one documented training branch plus observer parity.

The 21-condition analysis ran for 138.5 minutes under NVIDIA MPS with seven concurrent
checkpoint jobs and two process workers per condition. Five-second telemetry recorded
99.97% mean utilization over samples at or above 90% utilization, 445.5 W mean power,
459.0 W median power, 527.0 W peak power, and 22,141 MiB peak memory. This metric workload
is compute-bound; unused VRAM was not evidence of idle compute.

Three same-starting-point Stage-2 mixture endpoints and six SFT/RL endpoints are frozen as
secondary tiers. They are not part of the default run. The mixture endpoints are a
controlled endpoint contrast, not a temporal series. The post-training tier needs a small
protocol check first: a fixed four-token raw-prompt answer budget may measure response
format compliance rather than task knowledge for thinking-tuned checkpoints.

The final base checkpoint was also checked on complete standard benchmarks. Comparison
with an OLMo-reported value required matching the checkpoint, prompt form, number and
identity of demonstrations, answer scoring, length normalization, and dataset revision. The
[model card](https://huggingface.co/allenai/Olmo-3-1025-7B#evaluation) reports
`ARC MC=89.2`. The official
[OLMES protocol](https://github.com/allenai/olmes#usage) uses curated demonstrations over
the combined train, validation, and test splits. The ordinary `lmms-eval` ARC YAML is
therefore not a direct reproduction. The causal adapter supports batched text
loglikelihood for broad engineering checks, but representation collection is deliberately
rejected on that path: the evaluator creates one request per answer choice, which would
otherwise violate the one-vector-per-example contract.

The broad sanity check is complete on immutable final-main revision
`a81bae42db3975be1671e27b9c9a56da1a9f980f`. With the unmodified zero-shot
`lmms-eval` ARC tasks, the full 2,376-example ARC-Easy test set scored
`acc=0.7984` and `acc_norm=0.7816`; the full 1,172-example ARC-Challenge test set scored
`acc=0.4863` and `acc_norm=0.5239`. The run took 60.3 seconds after setup at batch 128 and
is stored locally under `logs/olmo3-standard-benchmark-full/arc`. These scores are not
close enough to treat the model card's `ARC MC=89.2` as reproduced. That is expected to
reflect the known zero-shot/test-only versus curated-five-shot/all-splits protocol
difference.

The protocol-matched check is now complete with OLMES commit
`5a51f502d463b8cdc4a2dcad7d7096c41ff1197e`. It used `arc:mc::xlarge`, the exact final
model revision above, five OLMES demonstrations with seed 1234, and all dataset splits.
With OLMES-pinned Transformers 4.57.6, ARC-Easy scored `0.935732` on 5,197 examples and
ARC-Challenge scored `0.847490` on 2,590. Their macro average is `0.891611`, which rounds
to the model card's `89.2`. This reproduces the reported value at its published precision
and supports checkpoint loading and likelihood scoring; it is not evidence for
representation separation.

The GH200 is ARM64, while the current OLMES lock selects an x86_64 CUDA wheel. The run
therefore used unchanged ARC task, few-shot, metric, and Transformers code from that
commit but a temporary ARC-only import registry and the workspace PyTorch 2.13 runtime
instead of pinned PyTorch 2.8. A separate run under the workspace's Transformers 5.5.4
scored `0.922840/0.822780` and macro-averaged `0.872810`. This material version
sensitivity is why final benchmark comparisons and longitudinal extraction must record
the runtime and must not mix activations across Transformers versions.

This run also exposed and fixed a general evaluator regression: likelihood tuples were
being passed through the generation-response wrapper and converted to strings before
metric computation. The evaluator now preserves `(loss, is_greedy)` tuples and unwraps
token counts only for generation requests. Focused unit tests and a real two-example ARC
smoke pass after the fix.

### Stage D — causal pruning and efficiency: not started

- Intervene only on sites replicated in Stage C.
- Compare targeted interventions with random and established importance baselines at equal
  budgets.
- Measure task quality, calibration, latency, throughput, and peak memory on the intended
  inference backend.
- Test whether passive separation predicts causal quality loss across interventions.

## Publication rule

The strongest defensible near-term result may be qualified or negative: benchmark-family
identity can be broadly predictable without a stable, construct-specific boundary between
knowledge and reasoning. That is publishable only if the paper reports the confounds,
failed replications, missing quantities, and analysis corrections as results rather than
selecting a favorable layer or model after inspection.
