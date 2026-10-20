# Artifact provenance and release

This document states where every number in the paper and in `results/` comes from, how the
raw artifacts were packaged, how they were verified, and how to restore them. It is the
reproducibility contract for *Do Knowledge- and Reasoning-Oriented Benchmarks Induce Stable
Representation Signatures in LLMs?* (NeurIPS 2026 Workshop on Interpretability as a Science).

## Chain of custody

1. **Extraction and analysis** ran on one NVIDIA GH200 under the frozen manifests in
   `experiments/` with the `lmms-eval` fork at commit `1bdf31d` (branch
   `research/representation-dynamics`) and this repository at commit `34c3105`. Raw outputs
   (activation tensors `representations.pt`, per-record `records.csv`, `metadata.json`,
   sample logs, analysis `layer_separability.csv`, `report.json`, and
   `permutation_nulls.json.gz`) were written to dated directories under the persistent volume
   `/lambda/nfs/data-imu-us-east3/vlmbehave-artifacts/` (canonical path in the scripts:
   `/home/ubuntu/data-imu-us-east3/vlmbehave-artifacts/`).
2. **Sample-free summaries** were produced from those outputs by the `scripts/summarize_*.py`
   programs and committed under `results/`. Each `report.json` records the SHA-256 of every
   input file it read and of every CSV it wrote.
3. **Packaging.** On 2026-08-27 the persistent volume was packaged with
   `migration-tools/01_pack_vlmbehave.sh` into the bundle
   `vlmbehave-migration-20260827T000543Z`, at the submitted revisions recorded in its
   `git-state.tsv` (root `34c3105`, `lmms-eval` `1bdf31d`).
4. **Release.** The archive, its per-file manifest, and a text-only extract are published in the
   [Google Drive artifact folder](https://drive.google.com/drive/folders/1eS3xhP3qB_QBep2mUcOfbvddVSj4qHy2?usp=drive_link).
   Everything needed to recompute the paper's tables from the raw outputs is in that folder plus
   the public repositories.

## The released files

The Google Drive folder `lmbehave-release` contains only the artifacts and their hashes:

| File | Bytes | SHA-256 | Content |
| --- | ---: | --- | --- |
| `persistent-data.tar.zst` | 81,699,077,557 | `160ab6cb5cfcf83ac35f43c4b946957cdb2511d33005a617daa71520546d1b22` | `vlmbehave-artifacts/` (2,755 files) and `vlmbehave-runs/` (14 files); 133.7 GB uncompressed |
| `persistent-files.sha256` | 724,807 | `731878d7d3a9bba0153f7d7bf8cdba27010ec31d46f6d206c1f47bd05d31ac52` | SHA-256 of all 2,769 files inside the archive |
| `analysis-outputs-text.tar.gz` | 563,834,133 | `f889e7da4604ddadd2f398e0fe7c8a4946c49f0acd219023c34b9d62e1c33418` | the archive's 2,498 text members (reports, CSVs, logs, manifests, permutation nulls) |
| `SHA256SUMS`, `README.md` | | | hashes of the files above and a short description |

The archive holds 261 `representations.pt` tensors, 953 JSON, 693 JSONL, 343 CSV, 283 log,
and 202 gzip members (including nine nested `.tar.gz` study archives from July and early
August 2026). The code revisions that produced it are recorded in this document (root
`34c3105`, `lmms-eval` `1bdf31d`), not in the folder; the migration bundle's workspace
tarball and local tool-state backups are not part of the release.

## Which results come from which artifacts

Every hash recorded in a `results/<dir>/report.json` was matched against
`persistent-files.sha256` and against git at the submitted revisions. The table lists, for
each committed summary, the bundle directories whose files it hashes (counts are matched
files). Directories whose sources sit inside a nested archive list that archive.

| `results/` directory | Bundle source (`vlmbehave-artifacts/...`) |
| --- | --- |
| `representation_dynamics_main_track_v1` | `2026-08-13/representation-dynamics-main-track-v1` (20), `2026-08-13/representation-dynamics-main-track-apertus15-runtime-v1` (8) |
| `representation_dynamics_main_track_selected_probe_v1` | `2026-08-15/selected-site-probe-main-track-v1` (5) |
| `representation_dynamics_main_track_instrument_validation_v1` | `2026-08-16/representation-dynamics-main-track-instrument-validation-v1` (97) |
| `representation_dynamics_main_track_review_response_v1` | `2026-08-16/representation-dynamics-main-track-review-response-v1` (82), `...instrument-validation-v1` (10), `2026-08-15/selected-site-probe-main-track-v1` (10) |
| `representation_dynamics_main_track_lexical_control_v1` | `2026-08-16/representation-dynamics-main-track-lexical-control-v1` (11) |
| `representation_dynamics_selector_sensitivity_v1` | `2026-08-16/representation-dynamics-selector-sensitivity-v1` (20) |
| `representation_dynamics_main_track_controls_v1` | `2026-08-16/representation-dynamics-main-track-controls-v1` (20), `2026-08-13/representation-dynamics-main-track-v1` (4), `...apertus15-runtime-v1` (1) |
| `representation_dynamics_main_track_correctness_support_v1` | `2026-08-13/representation-dynamics-main-track-v1` (34), `2026-08-16/representation-dynamics-multimodal-side-v1` (30), `...apertus15-runtime-v1` (16), `2026-08-14/main-track-correctness-support-v1` (1) |
| `representation_dynamics_main_track_fisher_v1` | `2026-08-13/calibrated-matched-panel-v1` (21) |
| `representation_dynamics_multimodal_side_v1` | `2026-08-16/representation-dynamics-multimodal-side-v1` (42), `2026-07-27/representation-dynamics-v1-six-model.tar.gz` (1) |
| `representation_dynamics_relative_response_pilot_v2` | `2026-08-15/relative-response-pilot-olmo3-eager-v1` (5) |
| `apertus1_longitudinal_v1` | `2026-08-13/apertus1-longitudinal-v1` (4) |
| `apertus15_endpoint_anchor_v1` | `2026-08-10/apertus15-endpoint-anchor-v1.tar.gz` (nested archive) |
| `olmo3_longitudinal_v1` | `2026-07-27/olmo3-longitudinal-v1.tar.gz` (nested archive, SHA-256 `7b59567d…`; see `docs/REPRODUCIBILITY.md`) |
| `representation_dynamics_text_expansion_corrected_v1` | `2026-08-11/current-corrected-analysis-v1.tar.gz` and `2026-08-03/representation-dynamics-text-expansion-v1-v2.tar.gz` (nested archives) |
| `representation_dynamics_text_expansion_v1` | `2026-08-03/representation-dynamics-text-expansion-v1-v2.tar.gz` (historical; superseded, see its README) |
| `response_trajectory_smoke_v1`, `response_trajectory_long_response_smoke_v1` | `2026-08-13/response-trajectory-smoke-v1` (4), `2026-08-15/response-trajectory-smoke-v4` (3) |

The paper's main-text numbers come from the first eight rows plus the two longitudinal rows;
the multimodal and pilot rows feed appendix-only material.

## Verification performed on 2026-10-15

* **Bundle integrity.** The local `persistent-data.tar.zst` hashes to SHA-256
  `160ab6cb…` (MD5 `2fa8af1d98af34556658e431997d6db3`), equal to the pack-time
  `SHA256SUMS`; the Google Drive copy reports the same SHA-256 server-side.
* **Member integrity.** `scripts/audit_migration_bundle.py` streamed the archive once
  (`zstd -dc | tar`, 133,698,382,505 bytes, 701 s): all 2,769 regular members match
  `persistent-files.sha256`, none is missing, none is unlisted (`audit_report.json`: `PASS`).
* **Credential scan.** Every text member (2,498 files) and every text member inside the nine
  nested `.tar.gz` archives (1,588 of 1,862 members) was scanned for Hugging Face, GitHub,
  OpenAI, AWS, W&B, Slack, Google, bearer, private-key, and generic token/password patterns:
  zero hits. The archive can be released as is.
* **Binding of the committed tables.** Of the 677 SHA-256 values recorded in `results/`,
  435 name files in the bundle; the 426 of those that are text were re-hashed from the
  extracted bytes and all match (the other 9 are tensors or nested archives, verified by the
  manifest). The 39 self-hashes of committed CSVs and experiment manifests, and the 12 hashes of
  workspace files (manifests, analysis modules, summary scripts), match the repository at
  `34c3105`/`1bdf31d`.
* **Recomputation.** Running the committed summary scripts on the extracted text members
  reproduced all 25 CSV files of nine results directories byte for byte
  (`representation_dynamics_main_track_v1`, `..._controls_v1`, `..._fisher_v1`,
  `..._instrument_validation_v1`, `..._review_response_v1`, `..._lexical_control_v1`,
  `..._selector_sensitivity_v1`, `..._multimodal_side_v1`, and
  `..._relative_response_pilot_v2`). Four scripts follow absolute paths recorded inside the
  artifacts' own reports (`/lambda/nfs/data-imu-us-east3/...`), so the verification run
  redirected those paths to the extraction directory without modifying any file; the
  multimodal script takes the study's `analysis/` subdirectory as its source. The two
  longitudinal summaries and the Apertus 1.5 anchor were checked by hash against their nested
  archives rather than re-run.

## Restoring and recomputing

```bash
# 1. verify the download
sha256sum -c SHA256SUMS
# 2. restore into an empty data directory (needs ~140 GB free)
mkdir -p /data/lmbehave && tar --zstd -xpf persistent-data.tar.zst -C /data/lmbehave
(cd /data/lmbehave && sha256sum -c persistent-files.sha256)
# 3. regenerate a committed summary; the artifacts' reports record the original volume paths,
#    which scripts/artifact_path_remap redirects to the restored copy
export PYTHONPATH=scripts/artifact_path_remap LMB_REMAP_TARGET=/data/lmbehave/
python3 scripts/summarize_main_track_instrument_validation.py \
  /data/lmbehave/vlmbehave-artifacts/2026-08-16/representation-dynamics-main-track-instrument-validation-v1 \
  /tmp/regenerated
diff -r /tmp/regenerated results/representation_dynamics_main_track_instrument_validation_v1
```

Full re-extraction of activations requires the model revisions, batch sizes, and manifests in
`experiments/` and the `lmms-eval` fork; see `README.md` (sections "Collect representations"
and "Analyze representations") and `docs/REPRODUCIBILITY.md`.
