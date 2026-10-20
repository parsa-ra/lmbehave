# Heterogeneous multimodal side analysis

This directory contains sample-free summaries for a supplementary reanalysis of the
frozen ten-task multimodal panel. It does not replace, extend, or pool with the primary
text-only MMLU analysis.

## Design

- Models: Qwen 3.5 9B/27B and Gemma 4 31B, the current-generation multimodal checkpoints
  available in the frozen archive. Older Gemma 3 conditions are excluded.
- Data: five historically knowledge-oriented and five historically capability-oriented
  tasks, 100 examples per task, with one final-prompt-token vector at every layer and stream.
- Inference: complete-task holdout and all 252 balanced oriented task assignments (126
  label-complement pairs; minimum class-symmetric p-value `1/126 = .00794`). The Fisher
  maximum is reselected for every assignment before the selected-site probe is refit.
- Controls: six prompt-surface covariates; task-level visual-input presence; response-format
  indicators; a fold-local 64-component character TF--IDF activation adjustment; and their
  combination. The visual indicator records only whether the task supplies an image; it
  does not represent pixels, visual tokens, or image content.

## Result

Raw held-out-task macro-F1 is `.858--.899`, but exact p-values are `.0159--.0317`; neither
the raw probes nor the Fisher maxima meet the study's `.01` criterion. After surface and
modality adjustment, F1 is `.210--.538` (`p=.421--.960`); after lexical adjustment it is
`.550--.735` (`p=.0635--.373`); after both it is `.151--.336` (`p=.913--.992`). Prompt-only
word and character TF--IDF baselines reach F1 `.804` and `.802`, respectively, with
`p=.0159` for both.

At the Fisher-selected sites, descriptive RBF F1 is `.799--.850`, cosine silhouette is
`.256--.434`, covariance alignment is `.080--.421`, and within-group effective-rank
fraction is `.109--.418`. These quantities have no task-randomization p-value in this side
analysis. Paired CKA is missing, rather than zero: the benchmark observations have no
declared cross-group `pair_id`, so an arbitrary row alignment would not define a meaningful
CKA comparison. Covariance alignment is the order-invariant geometry statistic available
for these unpaired sets.

The panel is not construct-valid: all five capability-oriented tasks supply images, versus
three of five knowledge-oriented tasks, and domain, benchmark family, answer format, and
prompt content also differ. The high raw F1 therefore establishes only predictability of
these ten named task labels. Its disappearance under measured controls is a sensitivity
result, not proof that image or prompt features causally mediate the raw representation
signal.

`report.json` binds the summaries to the exact Fisher nulls, probe distributions, source
artifact and prompt audits, and nuisance table on persistent storage at:

```text
/home/ubuntu/data-imu-us-east3/vlmbehave-artifacts/2026-08-16/
  representation-dynamics-multimodal-side-v1/
```
