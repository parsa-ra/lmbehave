# Main-track instrument validation summary

This directory contains sample-free, hash-bound summaries of the exact MMLU 8+8 controls.
Raw activations, prompts, per-assignment score vectors, and the shared lexical cache remain
on persistent storage.

- `matched_excess_summary.csv`: representation and prompt macro-F1 under one 64-feature
  cuML estimator, their paired difference, and the alphabetical reference split.
- `lexical_diagnostic_summary.csv`: retained total/between/within-task variance, 16-way task
  identity diagnostics, and the exhaustive fold-balance audit.
- `probe_calibration_summary.csv`: detection frequency for the completed injected-signal
  grid. The attempted post-hoc 5/6/8 extension produced no report and is excluded.
- `report.json`: source, manifest, implementation, distribution, and output hashes plus
  the terminated-extension record.

The paired p-values test whether the observed representation-minus-prompt advantage is
unusually large among the balanced task partitions. Non-rejection is not equivalence. The
synthetic calibration is conditional on one fixed OLMo site and does not measure a model
mechanism or calibrate layer selection.
