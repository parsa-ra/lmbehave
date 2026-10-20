# Primary MMLU 8+8 prompt controls

These sample-free tables summarize post-review sensitivity analyses for the five-model
MMLU 8+8 panel. Full per-assignment outputs remain on persistent storage at the path
recorded in `report.json`; SHA-256 bindings cover every source report and compact output.

Key results:

- Prompt text predicts the declared task partition under the same task holdout and 12,870
  balanced assignments. Word TF--IDF gives macro-F1 `0.701` (`p=0.00870`); character
  TF--IDF gives `0.768` (`p=0.000777`).
- The Fisher-selected representation probe remains significant at `p<0.01` in all five
  models after length-only adjustment.
- After adjustment for six scalar prompt features, it remains significant in OLMo 3,
  Apertus 1.5, and both Qwen sizes. Gemma 4 gives macro-F1 `0.732`, `p=0.0137`.
- These results do not identify mediation or a causal prompt feature. The TF--IDF baseline
  captures richer lexical information than the six scalar covariates, while residualizing
  construct-correlated covariates can remove or amplify task-set signal.

The follow-up fold-local 32/64-component lexical activation control is reported separately
under `results/representation_dynamics_main_track_lexical_control_v1/`; do not infer its
outcome from these scalar-control tables.

The directory also records parser-validity, released-trajectory parser checks, post-hoc
adjacent-residual reconstruction, and GPU telemetry. Current-model accuracy is not used as
cross-model evidence because Apertus 1.5 and Gemma 4 have substantial parser failure under
the historical four-token answer protocol.
