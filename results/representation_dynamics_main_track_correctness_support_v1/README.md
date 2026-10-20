# Correctness support audit

This sample-free audit counts correct and incorrect examples for every model and task in
the completed 8+8 MMLU panel. It is a feasibility check, not a correctness-conditioned
representation result.

A balanced correct/incorrect subset would be limited to 26 examples per task and outcome
for OLMo, 13 for Apertus 1.5, 8 for Qwen 9B, 3 for Qwen 27B, and 7 for Gemma 4. Such a
subset would therefore remove the most data from several high-accuracy conditions and must
not replace the all-example primary analysis. Correctness is also downstream of the
prefill representation; conditioning on it is a sensitivity analysis, not causal nuisance
adjustment.

`correctness_support.csv` contains the task cells. `report.json` binds all 80 evaluation
sample files by SHA-256 and records the model-level support limits. The raw sample files
remain on persistent storage and are not committed here.
