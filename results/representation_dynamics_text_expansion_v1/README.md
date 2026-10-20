# Superseded mixed-version summary

This directory is retained only as an audit record of the original current-model text
study. It must not be used as the source of the manuscript's five-model table.

Two scope changes make these files scientifically superseded:

1. Rows named `apertus-8b-instruct` use the Apertus 1.0 checkpoint. The manuscript's
   current-family condition is Apertus 1.5 (`apertus15-8b`) and is summarized under
   `../representation_dynamics_text_expansion_corrected_v1/`.
2. The original heterogeneous panel contained malformed MuSR prompts. The affected rows
   are identified by `protocol_invalidation.json`; corrected cross-panel aggregates are
   also under `../representation_dynamics_text_expansion_corrected_v1/`.

`prompt_length_baseline.csv` likewise belongs to the superseded Apertus 1.0 panel. It may
support historical non-Apertus tokenizer counts, but it is not provenance for the current
Apertus 1.5 condition. Regenerate any scalar length diagnostic from the corrected nuisance
table before citing it.

All `minimum_attainable_p=0.003968...` fields in these historical CSVs record an older,
incorrect metadata convention. For the deterministic class-symmetric statistics used in
the balanced five-versus-five design, complementary assignments are identical and the
true attainable floor is `2/252 = 1/126`. The stored raw p-values already counted the
complement and are not recomputed in this audit directory.
