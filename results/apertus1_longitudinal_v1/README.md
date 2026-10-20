# Apertus 1.0 pretraining trajectory

These are sample-free aggregates from eleven immutable checkpoints on one released
Apertus 1.0 8B pretraining run and three paired, disjoint example blocks. All 33 extraction
conditions and their 3,168 layer--stream analysis rows passed validation. Persistent storage
holds the 1,056-row checkpoint-level layer--stream summary hashed in `report.json`; the
committed `checkpoint_summary.csv` collapses it to 11 depth-mean rows, one per checkpoint.

Accuracy increases from 0.260/0.254 to 0.621/0.433. Depth-mean Fisher increases at the
endpoints in all three streams, but is non-monotonic and sometimes falls while both task
accuracies rise. The repetitions measure example-sampling sensitivity, not training-seed
uncertainty. Long-context, SFT/QRPO, and Apertus 1.5 are excluded from this curve.
