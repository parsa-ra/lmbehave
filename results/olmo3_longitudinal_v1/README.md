# OLMo 3 7B longitudinal aggregate results

These are the compact, sample-free outputs of the frozen primary trajectory in
`experiments/olmo3_longitudinal_v1.json`. Raw activations, prompts, responses, and
per-example records remain outside Git under `logs/olmo3-longitudinal-v1/`.

The run covers seven immutable base-training checkpoints and three paired, disjoint
32-example/task blocks. Extraction passed 21/21 conditions and produced 645,120 vectors.
cuML analysis completed 2,016/2,016 layer–stream–block rows with zero failures and all 251
non-observed balanced task-label assignments per row.

Files:

- `longitudinal_layer_summary.csv`: mean, sample standard deviation, and count across the
  three blocks for every checkpoint, layer, stream, and metric.
- `longitudinal_checkpoint_summary.csv`: depth summaries, task-group accuracy, maxima, rank
  diagnostics, and within-stage FDR velocity for every checkpoint and stream.
- `probe_repeatability_summary.csv`: counts of layers whose BH-adjusted linear-probe F1
  test has `q < 0.05` in all three blocks, at least two blocks, and across the 96
  layer–block tests per stream.
- `report.json`: experiment scope, claim limit, output hashes, and repeat interpretation.

The standard deviations measure sensitivity to example sampling, not training-seed
uncertainty. This is one released model trajectory. The main result is mixed: task
accuracy and repeatable linear-probe evidence increase after Stage 1, while FDR,
silhouette, nonlinear-probe scores, and covariance geometry do not show a common monotonic
increase. These files must not be read as causal evidence.
