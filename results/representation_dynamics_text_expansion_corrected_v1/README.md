# Corrected current-model summaries

This directory contains sample-free summaries for the current-family panel after replacing
the Apertus 1.0 substitution with Apertus 1.5 and correcting the malformed MuSR protocol.
`report.json` binds the activation and nuisance-control summaries to their inputs.

`prompt_length_baseline.csv` is a separately regenerated diagnostic for the valid matched
panel. Its dedicated `prompt_length_baseline_report.json` binds all 50 source activation
and record files by SHA-256. It uses the corrected Apertus 1.5 condition and the
complement-aware exact p-value floor of `1/126`. This one-feature grouped probe tests only
whether scalar prompt token count linearly predicts the task partition; it neither removes
prompt confounding nor identifies a knowledge or reasoning mechanism.
