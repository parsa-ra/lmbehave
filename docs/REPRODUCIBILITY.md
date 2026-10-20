# Reproducibility and Compute Migration

## Local packaging

The root is a lightweight meta-repository. `lmms-eval` is a real submodule pinned to the private research commit, so its history and changes are developed and reviewed normally. Other independent source histories remain ignored at the root. The dirty Qwen Transformers prototype is still represented by a deterministic binary-capable patch in `overlays/`; clean dependencies are represented by exact commit pins in `repos.lock.json`.

Refresh overlays after an intentional edit to an overlay-managed repository:

```bash
./scripts/snapshot_overlays.sh
```

Do not create an `lmms-eval` overlay: commit and push that work in the submodule, then commit the updated gitlink in the root. After refreshing the remaining overlays, update each checksum and byte size in `repos.lock.json`, review the patches, and run the readiness checks. An overlay preserves file content, symlink targets, and executable bits, but not timestamps, ACLs, or xattrs.

## Recreate on compute

For the normal private-repository workflow:

```bash
GIT_SSH_COMMAND="ssh -i /path/to/private-key -o IdentitiesOnly=yes" \
  git clone https://github.com/parsa-ra/lmbehave.git
cd lmbehave
./scripts/setup_machine.sh /path/to/private-key
```

The submodule URL is private, so GitHub authentication must be available on the compute
host. The root pins the exact `lmms-eval` commit; do not update it implicitly. Setup uses
fast-forward-only pulls, verifies existing ignored repositories instead of overwriting
them, and defaults to uv's automatic PyTorch backend selection. Record the PyTorch/CUDA
line printed by `install.sh` with every experiment.

The bootstrap helper remains available for the non-submodule repositories:

```bash
python3 scripts/bootstrap_workspace.py
```

For an offline replay test using this existing workspace as the object source:

```bash
python3 scripts/bootstrap_workspace.py /private/tmp/vlm-replay --source-root .
```

Pass `--include-optional` to also recreate vLLM, vLLM-Hook, and IVC-Prune. The bootstrap checks out exact bases, validates overlay hashes, applies patches without whitespace rewriting, verifies reverse application, regenerates each base-to-worktree patch byte-for-byte, and runs `git diff --check`. It does not install Python/CUDA dependencies or select between the two Transformers variants.

The current `install.sh` defaults to clean `transformers/`. Qwen visual-token pruning instead requires an explicit `TRANSFORMERS_REPO=transformers_qwenvl ./install.sh`. Keep those environments separate and record which source directory was installed; do not switch this implicitly in a publication run.

## Persistent experiment archives

The historical six-model and OLMo longitudinal trees were archived on 2026-07-27, the
current-model text-only tree on 2026-08-03, and corrected Apertus/MuSR additions on
2026-08-10--11 under dated directories in
`/home/ubuntu/data-imu-us-east3/vlmbehave-artifacts/`:

| Study | Source tree | Archive bytes | SHA-256 |
| --- | --- | ---: | --- |
| Six-model comparison | `logs/representation-dynamics-v1-six-model/` | 16,069,301,143 | `589401e880348a103d0cc985ca57b43f34b720fe4818df3693717fb6384d40e4` |
| OLMo longitudinal | `logs/olmo3-longitudinal-v1/` | 5,016,539,854 | `7b59567d82d05304f482a7f39b1ddd4b5bc8100d1c3bad7f958133f72f468f41` |
| Current-model text-only | `logs/representation-dynamics-text-expansion-v1-v2/` | 12,039,720,218 | `19917aeb92b9ecd0f494986c4140fe331a776d7cedb80e67ba5b353cefca7b0e` |
| Apertus 1.5 current-panel extraction | `logs/apertus15-current-panel-v1/` | 1,492,804,198 | `5271cd5eb55b12014e95049fc01916f0c6175f428ece77a4a70ed812ea147f14` |
| Apertus 1.5 endpoint anchor | `logs/apertus15-endpoint-anchor-v1/` | 716,402,768 | `781232fbce3ba3e57ef8431753e331dc59be73ea4513ee9b04e1dc5a3ce1f823` |
| Corrected MuSR extraction | `logs/musr-protocol-correction-v1/` | 602,861,331 | `ac8f973a38c885de9f05187f26a4fee23ef527a0846fdd7d06294f64b694df5e` |
| Corrected sample-free analysis | corrected analysis outputs | 523,109 | `6fb6230afab9105e4f2dbbed6cdb650cc238aa22b61d2d7499be067d788fec97` |
| Calibrated matched-panel Fisher analysis | five current models, 660 rows | 1,240,853 | `e03390c21b1df72eb616d27749710c1bdad0a1aa31e2c9ef99332ec6c19ff155` |

The provisional Apertus 1.0 observer and ARC-anchor outputs are copied without compression
to `2026-08-13/apertus1-preflight-provisional-v1/`. Its `SHA256SUMS` file covers all 11
files and passes a full `sha256sum -c` replay. It is explicitly provisional because the
first three parity reports predate the source-dirty and observer-script hash fields. Final
clean-source parity reports and the full longitudinal artifacts must use new directories;
do not overwrite this audit trail.

Each gzip-compressed tar archive has a colocated `.sha256` file. A fresh scan found none
of the configured Hugging Face, W&B, or GitHub credentials and no matching generic
credential pattern before transfer. For the current archive this covered all 437
non-binary files; 100 tensor/binary or large files were excluded from text-pattern
scanning. Checksum and gzip verification passed after writing to NFS,
and full archive listings matched the source-tree member counts: 1,318 for the six-model
study, 486 for OLMo, and 1,070 for the current-model study. Each dated archive directory is mode `700`; archives and checksum files
are mode `600`. The 319-GB Hugging Face model/dataset cache is reproducible and was not
included.

All eight pre-existing archives scanned before the calibrated reanalysis on 2026-08-13
passed full SHA-256 verification. The new calibrated-analysis archive passed its colocated
checksum and gzip integrity check separately.
The 12-GB current-model archive restored all 100 declared raw artifacts; the Apertus 1.5
extension restored all 20; and the MuSR correction restored all five. Every restored
`representations.pt` digest matches its completion manifest. Current matched-panel
reanalysis must therefore reuse these activations rather than rerun extraction.
The full verification report is
`2026-08-13/persistent_artifact_verification.json` (SHA-256
`36dbb63fcdb3947c8a5ff977a58caa6a649f3074059882ab972928088f996d9c`).

Verify and restore into an empty staging directory rather than over an active workspace:

```bash
BACKUP_DIR=/home/ubuntu/data-imu-us-east3/vlmbehave-artifacts/2026-07-27
sha256sum -c "$BACKUP_DIR/representation-dynamics-v1-six-model.tar.gz.sha256"
sha256sum -c "$BACKUP_DIR/olmo3-longitudinal-v1.tar.gz.sha256"

mkdir -p /path/to/empty-restore
tar -xzf "$BACKUP_DIR/representation-dynamics-v1-six-model.tar.gz" \
  -C /path/to/empty-restore
tar -xzf "$BACKUP_DIR/olmo3-longitudinal-v1.tar.gz" \
  -C /path/to/empty-restore

CURRENT_BACKUP_DIR=/home/ubuntu/data-imu-us-east3/vlmbehave-artifacts/2026-08-03
sha256sum -c "$CURRENT_BACKUP_DIR/representation-dynamics-text-expansion-v1-v2.tar.gz.sha256"
tar -xzf "$CURRENT_BACKUP_DIR/representation-dynamics-text-expansion-v1-v2.tar.gz" \
  -C /path/to/empty-restore
```

### Migration bundle and public release (2026-08-27)

The complete persistent volume was packaged once experiments ended, at the submitted
revisions (root `34c3105`, `lmms-eval` `1bdf31d`), with
`migration-tools/01_pack_vlmbehave.sh`. The bundle `vlmbehave-migration-20260827T000543Z`
contains `persistent-data.tar.zst` (81,699,077,557 bytes, SHA-256
`160ab6cb5cfcf83ac35f43c4b946957cdb2511d33005a617daa71520546d1b22`; 2,769 files) and the
per-file manifest `persistent-files.sha256`. `scripts/audit_migration_bundle.py` verifies the
archive member by member against that manifest in one streaming pass, scans every text and
nested-archive member for credentials, and extracts the text reports; `docs/ARTIFACTS.md`
records the verification outcome, the mapping from `results/` to bundle paths, and the
release folder.

## Before syncing or migrating the private repositories

1. Revoke and rotate the exposed Hugging Face token. Review remote W&B/HF artifacts as potentially exposed.
2. Run `python3 scripts/check_private_ready.py`. Warnings for site-specific paths are expected until scheduler scripts are parameterized; credential or size findings are fatal.
3. Review `git status --short`, every root diff, the submodule commit, and each remaining overlay. Do not use `git add .` inside child repositories without reviewing untracked source files.
4. Push the `lmms-eval` research commit before pushing a root commit that references it; otherwise fresh submodule clones will fail.
5. Perform a fresh-clone/bootstrap smoke test before launching expensive jobs.

## Publication-grade run contract

The supported orchestrator is `scripts/run_representation_experiment.py`. The completed
study uses `experiments/representation_dynamics_v1.json`; the separate current-model text
extension uses `experiments/representation_dynamics_text_expansion_v1.json`. Run `plan`
and the `check-tasks` and `check-models` gates first. The latter resolves every immutable
Hugging Face revision and requires actual weight files, rather than accepting a repository
name and an unrelated commit hash. Manifests that set `require_frozen_sample_ids` must then run
`freeze-samples` before `run`, `validate`, and `analyze`. Final extraction refuses dirty
root or `lmms-eval` trees. Debug extraction may opt into `--allow-dirty`, but its source
fingerprint is locked on the first task and any attempted resume after a source/environment
change is rejected. Task runs always set `WANDB_MODE=disabled`.

The run directory contains three provenance layers, plus a pre-extraction selection lock
when required:

- `sample_selection.json`: selected split, available capacity, exact sample IDs and their
  digest, and full/selected dataset fingerprints for every unique task, bound to the clean
  root and lmms-eval source states;

- `run_manifest.json`: immutable experiment, sample count, model revisions, hardware and
  runtime environment, full package inventory and digest, import paths, and root/submodule
  source states;
- `completed_manifest.json`: validated task artifacts, result/task hashes, sample-ID-set
  hashes, selected stacks, and extraction fingerprints;
- per-model `analysis_manifest.json`: analysis code state/settings and exact input/output
  hashes.

Validation compares document IDs across models within each task and requires exact,
finite, complete three-stream captures. Analysis rechecks validated artifact hashes before
reading them. The earlier reference estimator is scikit-learn/liblinear on CPU. The GPU
path uses the separately named cuML 26.06 backend with float64 analysis of the stored
float32 vectors. It now fits PCA using training tasks inside each probe fold and computes
effective/stable rank on GPU. These estimator families must not be pooled. Use process
workers for cuML only under NVIDIA MPS; the validated current run used two workers per
condition and five concurrent conditions. Do not use thread workers for parallel CUDA
analysis.

The prompt-surface sensitivity uses a separate frozen manifest and the same cuML 26.06
estimator family. `build_representation_nuisance_table.py` joins tracker identities to
logged inputs and emits numeric prompt aggregates only. `run_full_nuisance_control.sh`
fits the six-variable Ridge map inside every observed and permuted task-held-out fold,
then applies fold-local PCA and the unchanged probes. The launcher uses five concurrent
model processes with two MPS workers each and records GPU telemetry. A resume is accepted
only when the parent manifest, sensitivity manifest, nuisance table, estimator settings,
and all representation hashes match. The non-adjusted fields must reproduce the archived
nonlinear analysis before adjusted values are interpreted; exhaustive assignments are not
called exact inference unless complement-symmetry and replay checks also pass.

On the single-GH200 development server, use
`scripts/run_full_representation_experiment.sh`; pass `--manifest` for any protocol other
than completed v1. It composes the same runner stages without
changing prompts, tasks, pooling, precision, or analysis settings; adds clean-tree,
GPU/disk/import/task-capacity and sample-selection gates; resumes only complete extraction
pairs; serializes the selected model/panel analyses; and records five-second GPU telemetry.
W&B remains disabled during every task. A scalar-only W&B upload is a separate explicit
`--phase report --upload-wandb --wandb-project PROJECT` action after hash and failure gates.

The full common-support count is 100 examples/task. At the pinned MMLU revision,
`abstract_algebra` and `college_mathematics` have 100 test examples, the minimum among the
ten selected subjects. The earlier 512 default was infeasible for an exactly matched
panel and has been corrected rather than allowing unequal counts. Tracker-enabled DocVQA
calibration on the 94.5-GiB GH200 selected Gemma batch 100 (55.7-GiB peak) and Qwen batch
20 (74.7-GiB peak); Qwen batch 32 OOMed. Generated responses matched the batch-4 quality
smoke on all shared DocVQA examples. Pooled vectors were not bit-identical: mean relative
L2 change was 0.79% for Gemma and 2.37% for Qwen, worst-row change was 16.6% and 6.75%, and
minimum cosine similarity was 0.9875 and 0.9977. Batch-4 smoke activations and full-run
activations are therefore distinct experimental conditions.

The current design has two panels. `cross_benchmark_discovery` is intentionally
heterogeneous; its project-specific MedQA, GPQA, and MMMU adapters preserve source datasets
and scoring while standardizing deterministic direct-answer generation. These are study
protocols, not standard leaderboard protocols, and the panel can support only a claim about
its named task sets. `matched_text_mmlu_control` uses ten project-specific concise
MMLU tasks with a common prompt, A/B/C/D output, and four-token budget. It controls dataset
family and format and pins `hails/mmlu_no_train` at
`b2e1ec9aa795adafe68e8e983248dbd4b52a1c60`, but still uses imperfect subject-level
construct labels. Treat it as a control, not as a standard MMLU leaderboard protocol or
final construct validation.

### Current-model text-only checkpoint

The main paper comparison is the clean run
`logs/representation-dynamics-text-expansion-v1-v2/` at 100 examples/task. It uses only
OLMo 3 7B Instruct, Apertus 1.5 8B, Qwen 3.5 9B/27B, and Gemma 4 31B. The matched
MMLU-Pro panel and cross-benchmark stress test share 18 frozen task selections. Validation
originally passed all 100 artifacts and 1,320,000 exact float32 vectors. A later task-aware
prompt audit invalidated the five MuSR artifacts because the adapter rendered the pinned
dataset's string-encoded choice list character by character. The 50 matched-panel
artifacts and analyses remain valid. Corrected MuSR extraction passed for all five models,
and the combined cross-benchmark panel was reanalyzed over every task-label assignment. The
original malformed-MuSR rows remain provenance only and must not be interpreted.

The run manifest SHA-256 is
`4a00ed6d28fbed3f52b05ecde389953650fb55c8a2aaa896e52e2b4df3e75f56`; the completed
manifest SHA-256 is
`6d22fbad68588feab32bd47fc9f6c388d3c3f53154cca5133f1fdebd6e4ddd9c`.
Sample-free summaries are committed under
`results/representation_dynamics_text_expansion_v1/`. Regenerate them with
`scripts/summarize_representation_experiment.py`; it verifies manifest, artifact, result,
and analysis-output hashes. Pass baseline analysis name `analysis-cuml-26.06` and nonlinear
analysis name `analysis-cuml-26.06-nonlinear-251`; the historical summary keeps baseline
geometry and joins exhaustive RBF p/q fields by the full row key. The prompt-length baseline is a separate post-hoc sklearn
diagnostic and is not pooled with the cuML activation rows.

Four silhouette values in the concurrent nonlinear pass differed from the baseline by at
most 0.01643, while every other observed metric reproduced. Direct batched float64 cosine
distance matched the baseline within 4.44e-15 and was repeatable; the maintained analysis
now uses this deterministic path. Do not replace baseline geometry with the four
schedule-sensitive values.

The existing 12-GB activation archive remains
`vlmbehave-artifacts/2026-08-03/representation-dynamics-text-expansion-v1-v2.tar.gz` on the
persistent volume. The ten exhaustive nonlinear analysis directories and their three provenance
manifests are separately archived as
`representation-dynamics-text-expansion-v1-v2-nonlinear-251.tar.gz` (SHA-256
`363a95137262bb97f542501c695562cd628450e262c0c9b4ad688a9058e10b4b`). The archive has
43 members and contains ten analysis manifests, CSVs, and reports; it does not duplicate
raw activations.

Full-launcher telemetry recorded 364.9 W mean power, 612.6 W peak power, 100% peak GPU
utilization, and 65,082 MiB peak memory. W&B remains aggregate-only: never upload raw
activations, records, prompts, responses, or per-sample tables. The reviewed report is W&B
run `byibgfcd`; W&B recorded 0 media and 0 artifact files.

### Historical v1 checkpoints

#### Validated quality-smoke checkpoint

The clean run `quality-smoke-repdyn-v1-20260721` used 32 examples/task for both panels and
both models at root commit `79e3104e1fdaec26015a6cfe12b0ed73665f4466` and `lmms-eval`
commit `6ac4b5849ec6ca4ba759d4c51d9837528cf61619`. Pair-validation passed all 40 artifacts;
all four analyses used 251 non-observed balanced task-label assignments and produced zero
failed rows. The local run directory remains ignored. One scalar-only W&B summary was
uploaded as run `1ywp6db3`; W&B reported zero media and artifact uploads. Do not merge this
quality smoke with earlier dirty-source/debug artifacts or present it as the final study.

#### Full common-support checkpoint

The frozen 100-example/task extraction completed on 2026-07-21 for Gemma 3 4B and Qwen 3.5
9B across both panels. All 40 artifacts passed exact-count, finite-value, complete
three-stream, semantic-exactness, and cross-model sample-ID validation. Corrected analysis
at `lmms-eval` commit `0fa7b1dc` completed 396/396 rows with zero failures. Every row used
all 251 non-observed balanced task-label assignments under one deterministic,
permutation-symmetric grouped-fold policy. Preserve the hash-bound analysis outputs under
`logs/representation-dynamics-v1-common-support/`; CKA is missing because these benchmarks
are unpaired, and nonlinear probe inference remains disabled.

#### Scaling-extension checkpoint

On 2026-07-22, the same panels and exact 100-example task subsets were extracted for three
pinned open-weight conditions: Gemma 3 12B (`96b6f1ec`, batch 16), Gemma 3 27B
(`005ad340`, batch 8), and Qwen 3.5 27B (`fc05daec`, batch 8). Joint validation passed all
60 artifacts and 1,044,000 float32 vectors: 288,000 at 48 layers/width 3,840, 372,000 at 62
layers/width 5,376, and 384,000 at 64 layers/width 5,120. The run manifest SHA-256 is
`1c9fd9df3abbbb3f24c7acc92631dbcda4fd8900ff0161d22cf7b5d016db11d4`; its experiment
manifest SHA-256 is `70ac9f0caa520b384b15a305c7199a5bbd282c99f2ad48f01ebbb043604b07ba`.

Qwen used the Transformers PyTorch fallback because optional fused linear-attention and
causal-convolution packages were absent. Do not install those kernels into the condition
or combine later fused-kernel artifacts without a new backend declaration and parity run.
Gemma logged that `top_k` may be ignored; Qwen logged that temperature, top-p, and top-k
may be ignored under deterministic generation. These warnings are preserved as protocol
metadata and were not ``fixed'' mid-run. Uniform cuML analysis completed all 1,044 scaling
rows without failure; no simple monotonic model-size pattern emerged.

RAPIDS/cuML 26.06 is an explicit estimator family in an isolated environment. On one real
1,000-example row its raw-space probe changed logistic accuracy from 0.879 to 0.898. The
frozen cuML comparison instead promotes to float64, fits full GPU PCA, selects the smallest
integer component count reaching 95% cumulative variance, and fits QN logistic/RBF probes
in that PCA space. Covariance alignment and cosine silhouette use the original activation
space on GPU. Backend, version, dtype, and probe feature space are recorded.
All five historical model conditions were rerun with this same backend. The hash-gated scalar-only
summaries are W&B runs `96g5fep0` (4B/9B) and `5ax24h0s` (12B/27B); W&B reported zero
media and artifact uploads. Earlier scikit-learn outputs remain a separate estimator family.
Those tables used one shared unlabeled PCA basis per row. Current code fits PCA inside each
training fold and adds effective-rank diagnostics, so the historical tables must not be
described as results from the corrected analysis.

Every future run should record:

- all repository SHAs, dirty flag, and overlay/diff SHA-256;
- model ID and immutable model revision;
- task name/hash, split, exact selected document IDs, and sample count;
- tokenizer/processor revisions and image/video preprocessing limits;
- complete generation configuration and all random seeds;
- tracker schema version, phase (`prefill` or `decode`), hook module role, threshold, stride, and attention backend;
- Python, CUDA, PyTorch, Transformers, lmms-eval, hardware, and distributed topology;
- wall time, synchronized latency, peak memory, tokens, and raw/filtered outputs.

Run one task per process. Use deterministic decoding for primary comparisons and repeat sampled conditions with prespecified seeds. Separate prefill from decode and weight at the sample/token level rather than averaging hook calls.

## Minimum causal study

Use a factorial or continuous design rather than treating benchmark names as ground-truth binary labels:

- visual dependence: image-required, image-ablated/shuffled, and text-only controls;
- external knowledge demand: instance-level annotations or matched low/high knowledge subsets;
- matched input length, visual token count, output format/length, sample count, decoding, and model size.

For each model/task condition, distinguish:

1. normal backend, tracker off;
2. eager backend, tracker off;
3. eager backend, tracker on;
4. causal pruning/ablation at prespecified budgets.

Compare logits and greedy outputs for the first three conditions before interpreting tracker differences. Report confidence intervals across examples/seeds and validate whether proposed importance scores predict the accuracy/latency effect of actual pruning.
