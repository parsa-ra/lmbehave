# Selection-aware Fisher calibration

These sample-free files summarize the 2026-08-13 reanalysis of the valid five-model
matched MMLU-Pro panel. For every model and balanced task assignment, the analysis
recomputes all layer--stream Fisher ratios and reselects the maximum. It also calibrates the
rowwise BH rejection count against the same finite assignment distribution.

All 660 rows completed without failure. Maximum-statistic p-values are 0.0159--0.0317, so
none meets the prespecified 0.01 threshold. The alphabetical negative-control split has
p=0.397--0.571 and zero rejected rows. The full aligned null vectors and per-row outputs
remain on persistent storage under
`/home/ubuntu/data-imu-us-east3/vlmbehave-artifacts/2026-08-13/calibrated-matched-panel-v1/`;
`calibrated_model_summary.json` binds them by SHA-256.

This five-versus-five analysis has a class-symmetric p-value floor of 1/126. It does not
replace the frozen eight-versus-eight confirmatory experiment.
