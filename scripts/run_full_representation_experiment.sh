#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$ROOT/.venv/bin/python"
RUNNER="$ROOT/scripts/run_representation_experiment.py"
MANIFEST="$ROOT/experiments/representation_dynamics_v1.json"
WANDB_REPORTER="$ROOT/scripts/log_quality_smoke_wandb.py"

RUN_DIR="$ROOT/logs/representation-dynamics-v1-six-model"
RUN_DIR_SET=0
PHASE="all"
ANALYSIS_JOBS=2
ANALYSIS_CONDITION_JOBS=12
ANALYSIS_BACKEND="cuml"
ANALYSIS_PYTHON="${VLM_ANALYSIS_PYTHON:-$PYTHON}"
ANALYSIS_NAME="analysis-cuml-26.06"
ANALYSIS_NONLINEAR_PERMUTATIONS=""
MIN_FREE_DISK_GIB=100
MIN_FREE_VRAM_GIB=80
DRY_RUN=0
UPLOAD_WANDB=0
WANDB_PROJECT="${WANDB_PROJECT:-}"
WANDB_NAME="representation-dynamics-v1-six-model"
WANDB_NAME_SET=0
GPU_MONITOR_PID=""
GPU_TELEMETRY=""
GPU_TELEMETRY_START_LINE=1
MPS_OWNED=0
MPS_ROOT=""

usage() {
    cat <<'EOF'
Usage: scripts/run_full_representation_experiment.sh [options]

Run a frozen manifest-driven representation experiment on one GH200.

Options:
  --manifest PATH         Experiment manifest (default: completed v1 manifest).
  --run-dir PATH          Output directory under this workspace's logs/ directory.
  --phase PHASE           preflight, extract, validate, analyze, report, or all (default).
  --analysis-jobs N       Layer/stream workers per model/panel analysis (default: 2).
  --analysis-condition-jobs N
                          Concurrent model/panel analyses (default: 12).
  --analysis-backend NAME cuml or sklearn (default: cuml).
  --analysis-python PATH  Python containing the backend (default: VLM_ANALYSIS_PYTHON or .venv).
  --analysis-name NAME    Separate result-directory basename (default: analysis-cuml-26.06).
  --nonlinear-permutations N
                          Override the manifest's nonlinear task-label permutations for a
                          separately named sensitivity analysis.
  --min-free-disk-gib N   Preflight free-disk requirement (default: 100).
  --min-free-vram-gib N   Preflight free-VRAM requirement (default: 80).
  --dry-run               Print all extraction commands without launching models.
  --upload-wandb          After all gates pass, upload one scalar-only W&B summary.
  --wandb-project NAME    Required W&B project for --upload-wandb; WANDB_PROJECT is fallback.
  --wandb-name NAME       W&B run name used only with --upload-wandb.
  -h, --help              Show this help.

Rerunning --phase all is safe for completed extraction tasks: the manifest runner verifies
the locked source/environment and skips only task directories containing exactly one valid
representation/result pair. Use --phase validate, analyze, or report to resume later gates.
EOF
}

die() {
    echo "ERROR: $*" >&2
    exit 1
}

while (($#)); do
    case "$1" in
        --manifest)
            (($# >= 2)) || die "--manifest requires a value"
            MANIFEST="$2"
            shift 2
            ;;
        --run-dir)
            (($# >= 2)) || die "--run-dir requires a value"
            RUN_DIR="$2"
            RUN_DIR_SET=1
            shift 2
            ;;
        --phase)
            (($# >= 2)) || die "--phase requires a value"
            PHASE="$2"
            shift 2
            ;;
        --analysis-jobs)
            (($# >= 2)) || die "--analysis-jobs requires a value"
            ANALYSIS_JOBS="$2"
            shift 2
            ;;
        --analysis-condition-jobs)
            (($# >= 2)) || die "--analysis-condition-jobs requires a value"
            ANALYSIS_CONDITION_JOBS="$2"
            shift 2
            ;;
        --analysis-backend)
            (($# >= 2)) || die "--analysis-backend requires a value"
            ANALYSIS_BACKEND="$2"
            shift 2
            ;;
        --analysis-python)
            (($# >= 2)) || die "--analysis-python requires a value"
            ANALYSIS_PYTHON="$2"
            shift 2
            ;;
        --analysis-name)
            (($# >= 2)) || die "--analysis-name requires a value"
            ANALYSIS_NAME="$2"
            shift 2
            ;;
        --nonlinear-permutations)
            (($# >= 2)) || die "--nonlinear-permutations requires a value"
            ANALYSIS_NONLINEAR_PERMUTATIONS="$2"
            shift 2
            ;;
        --min-free-disk-gib)
            (($# >= 2)) || die "--min-free-disk-gib requires a value"
            MIN_FREE_DISK_GIB="$2"
            shift 2
            ;;
        --min-free-vram-gib)
            (($# >= 2)) || die "--min-free-vram-gib requires a value"
            MIN_FREE_VRAM_GIB="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --upload-wandb)
            UPLOAD_WANDB=1
            shift
            ;;
        --wandb-project)
            (($# >= 2)) || die "--wandb-project requires a value"
            WANDB_PROJECT="$2"
            shift 2
            ;;
        --wandb-name)
            (($# >= 2)) || die "--wandb-name requires a value"
            WANDB_NAME="$2"
            WANDB_NAME_SET=1
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "unknown argument: $1"
            ;;
    esac
done

MANIFEST="$(realpath -m -- "$MANIFEST")"
[[ -f "$MANIFEST" ]] || die "missing manifest: $MANIFEST"
if ((RUN_DIR_SET == 0)) && [[ "$MANIFEST" != "$ROOT/experiments/representation_dynamics_v1.json" ]]; then
    experiment_id="$($PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["experiment_id"])' "$MANIFEST")"
    RUN_DIR="$ROOT/logs/$experiment_id"
    if ((WANDB_NAME_SET == 0)); then
        WANDB_NAME="$experiment_id"
    fi
fi

case "$PHASE" in
    preflight|extract|validate|analyze|report|all) ;;
    *) die "invalid --phase: $PHASE" ;;
esac
[[ "$ANALYSIS_JOBS" =~ ^[1-9][0-9]*$ ]] || die "--analysis-jobs must be a positive integer"
[[ "$ANALYSIS_CONDITION_JOBS" =~ ^[1-9][0-9]*$ ]] || die "--analysis-condition-jobs must be a positive integer"
[[ "$ANALYSIS_BACKEND" == "sklearn" || "$ANALYSIS_BACKEND" == "cuml" ]] || die "invalid --analysis-backend"
[[ "$ANALYSIS_NAME" =~ ^[A-Za-z0-9._-]+$ && "$ANALYSIS_NAME" != "." && "$ANALYSIS_NAME" != ".." ]] || die "invalid --analysis-name"
if [[ -n "$ANALYSIS_NONLINEAR_PERMUTATIONS" ]]; then
    [[ "$ANALYSIS_NONLINEAR_PERMUTATIONS" =~ ^[0-9]+$ ]] || die "--nonlinear-permutations must be a non-negative integer"
fi
# Keep a virtual environment's Python symlink intact; resolving it to the base
# interpreter drops the environment's site-packages.
ANALYSIS_PYTHON="$(realpath -ms -- "$ANALYSIS_PYTHON")"
[[ -x "$ANALYSIS_PYTHON" ]] || die "analysis Python is not executable: $ANALYSIS_PYTHON"
[[ "$MIN_FREE_DISK_GIB" =~ ^[1-9][0-9]*$ ]] || die "--min-free-disk-gib must be a positive integer"
[[ "$MIN_FREE_VRAM_GIB" =~ ^[1-9][0-9]*$ ]] || die "--min-free-vram-gib must be a positive integer"
if ((UPLOAD_WANDB)) && [[ "$PHASE" != "all" && "$PHASE" != "report" ]]; then
    die "--upload-wandb is valid only with --phase all or --phase report"
fi

RUN_DIR="$(realpath -m -- "$RUN_DIR")"
case "$RUN_DIR/" in
    "$ROOT/logs/"*) ;;
    *) die "--run-dir must resolve below $ROOT/logs" ;;
esac
[[ "$RUN_DIR" != "$ROOT/logs" ]] || die "--run-dir cannot be the logs root"

on_error() {
    local status=$?
    echo "FAILED at line ${BASH_LINENO[0]} with status $status" >&2
    printf 'Resume after correcting the failure: %q --run-dir %q --phase all\n' "$0" "$RUN_DIR" >&2
    exit "$status"
}
trap on_error ERR

cleanup() {
    if ((MPS_OWNED)); then
        printf 'quit\n' | CUDA_MPS_PIPE_DIRECTORY="$CUDA_MPS_PIPE_DIRECTORY" \
            nvidia-cuda-mps-control >/dev/null 2>&1 || true
        MPS_OWNED=0
    fi
    if [[ -n "$GPU_MONITOR_PID" ]] && kill -0 "$GPU_MONITOR_PID" 2>/dev/null; then
        kill "$GPU_MONITOR_PID" 2>/dev/null || true
        wait "$GPU_MONITOR_PID" 2>/dev/null || true
    fi
    if [[ -n "$GPU_TELEMETRY" && -s "$GPU_TELEMETRY" ]]; then
        tail -n +"$GPU_TELEMETRY_START_LINE" "$GPU_TELEMETRY" | awk -F',' '
            {
                for (i = 3; i <= 6; i++) {
                    gsub(/^[[:space:]]+|[[:space:]]+$/, "", $i)
                    gsub(/\r/, "", $i)
                }
            }
            $6 ~ /^[0-9]+([.][0-9]+)?$/ {
                samples++
                utilization = $3 + 0
                memory = $4 + 0
                power = $6 + 0
                power_sum += power
                if (utilization > utilization_max) utilization_max = utilization
                if (memory > memory_max) memory_max = memory
                if (power > power_max) power_max = power
            }
            END {
                if (samples > 0) {
                    printf "GPU telemetry: mean_power=%.1f W peak_power=%.1f W peak_utilization=%.0f%% peak_memory=%.0f MiB samples=%d\n",
                        power_sum / samples, power_max, utilization_max, memory_max, samples
                }
            }
        ' "$GPU_TELEMETRY" || true
    fi
}
trap cleanup EXIT

start_mps() {
    [[ "$ANALYSIS_BACKEND" == "cuml" ]] || return 0
    command -v nvidia-cuda-mps-control >/dev/null || die "parallel cuML analysis requires nvidia-cuda-mps-control"
    command -v pgrep >/dev/null || die "parallel cuML analysis requires pgrep"
    # Linux exposes both MPS binaries with the 15-character comm name
    # "nvidia-cuda-mps"; matching the longer executable names never succeeds.
    if pgrep -x nvidia-cuda-mps >/dev/null; then
        die "an NVIDIA MPS service is already running; stop it or run the analysis manually with its pipe directory"
    fi
    MPS_ROOT="$(mktemp -d /tmp/vlmbehave-mps.XXXXXX)"
    export CUDA_MPS_PIPE_DIRECTORY="$MPS_ROOT/pipe"
    export CUDA_MPS_LOG_DIRECTORY="$MPS_ROOT/log"
    mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
    nvidia-cuda-mps-control -d
    MPS_OWNED=1
}

stop_mps() {
    if ((MPS_OWNED)); then
        printf 'quit\n' | CUDA_MPS_PIPE_DIRECTORY="$CUDA_MPS_PIPE_DIRECTORY" \
            nvidia-cuda-mps-control >/dev/null
        MPS_OWNED=0
    fi
}

timestamp() {
    date -u +'%Y-%m-%dT%H:%M:%SZ'
}

stage() {
    echo
    echo "[$(timestamp)] $*"
}

require_clean_tree() {
    local root_status child_status
    root_status="$(git -C "$ROOT" status --porcelain=v1 --untracked-files=all)"
    child_status="$(git -C "$ROOT/lmms-eval" status --porcelain=v1 --untracked-files=all)"
    [[ -z "$root_status" ]] || die "root repository is dirty; commit/review changes before a full run"
    [[ -z "$child_status" ]] || die "lmms-eval is dirty; commit/review changes before a full run"
}

preflight() {
    stage "Preflight"
    local require_gpu=0
    [[ -x "$PYTHON" ]] || die "missing executable environment: $PYTHON"
    command -v git >/dev/null || die "git is unavailable"
    command -v realpath >/dev/null || die "realpath is unavailable"
    command -v flock >/dev/null || die "flock is unavailable"
    command -v nvidia-smi >/dev/null || die "nvidia-smi is unavailable"
    command -v tee >/dev/null || die "tee is unavailable"
    require_clean_tree
    git -C "$ROOT" diff --check
    bash -n "$ROOT/scripts/run_full_representation_experiment.sh"
    "$PYTHON" "$ROOT/scripts/check_private_ready.py"
    "$PYTHON" -m pytest -q \
        "$ROOT/tests/test_full_experiment_contract.py" \
        "$ROOT/tests/test_run_representation_experiment.py"
    (
        cd "$ROOT/lmms-eval"
        "$PYTHON" -m pytest -q \
            test/eval/test_representation_dynamics.py \
            test/eval/test_cli_parse_args.py -k representation
    )

    local available_kib required_kib
    available_kib="$(df -Pk "$ROOT" | awk 'NR == 2 {print $4}')"
    required_kib=$((MIN_FREE_DISK_GIB * 1024 * 1024))
    ((available_kib >= required_kib)) || die "less than ${MIN_FREE_DISK_GIB} GiB disk is free"

    if ((DRY_RUN == 0)) && {
        [[ "$PHASE" == "preflight" || "$PHASE" == "all" || "$PHASE" == "extract" ]] ||
        [[ "$PHASE" == "analyze" && "$ANALYSIS_BACKEND" == "cuml" ]]
    }; then
        require_gpu=1
    fi
    VLM_FULL_ROOT="$ROOT" \
    VLM_REQUIRE_GPU="$require_gpu" \
    VLM_MIN_FREE_VRAM_GIB="$MIN_FREE_VRAM_GIB" \
    "$PYTHON" - <<'PY'
import os
from pathlib import Path

import lmms_eval
import torch
import transformers

root = Path(os.environ["VLM_FULL_ROOT"]).resolve()
lmms_path = Path(lmms_eval.__file__).resolve()
transformers_path = Path(transformers.__file__).resolve()
if not lmms_path.is_relative_to(root / "lmms-eval"):
    raise SystemExit(f"lmms_eval imports from the wrong checkout: {lmms_path}")
if not transformers_path.is_relative_to(root / "transformers"):
    raise SystemExit(f"transformers imports from the wrong checkout: {transformers_path}")

require_gpu = os.environ["VLM_REQUIRE_GPU"] == "1"
if require_gpu:
    if torch.cuda.device_count() != 1:
        raise SystemExit(f"expected exactly one visible GPU, found {torch.cuda.device_count()}")
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    free_gib = free_bytes / 2**30
    required_gib = int(os.environ["VLM_MIN_FREE_VRAM_GIB"])
    if free_gib < required_gib:
        raise SystemExit(f"only {free_gib:.1f} GiB GPU memory is free; require {required_gib} GiB")
    print(
        f"GPU: {torch.cuda.get_device_name(0)}; free={free_gib:.1f} GiB; "
        f"total={total_bytes / 2**30:.1f} GiB; torch={torch.__version__}; CUDA={torch.version.cuda}"
    )
print(f"lmms_eval: {lmms_path}")
print(f"transformers: {transformers_path}")
PY

    if [[ "$PHASE" == "preflight" || "$PHASE" == "all" || "$PHASE" == "analyze" ]]; then
        VLM_FULL_ROOT="$ROOT" VLM_ANALYSIS_BACKEND="$ANALYSIS_BACKEND" \
        "$ANALYSIS_PYTHON" - <<'PY'
import os
from pathlib import Path

import lmms_eval

root = Path(os.environ["VLM_FULL_ROOT"]).resolve()
lmms_path = Path(lmms_eval.__file__).resolve()
if not lmms_path.is_relative_to(root / "lmms-eval"):
    raise SystemExit(f"analysis lmms_eval imports from the wrong checkout: {lmms_path}")
backend = os.environ["VLM_ANALYSIS_BACKEND"]
if backend == "cuml":
    import cupy
    import cuml

    print(f"analysis backend: cuML={cuml.__version__}; CuPy={cupy.__version__}")
else:
    import sklearn

    print(f"analysis backend: scikit-learn={sklearn.__version__}")
print(f"analysis lmms_eval: {lmms_path}")
PY
    fi

    if ((DRY_RUN == 0)) && [[ "$PHASE" == "preflight" || "$PHASE" == "all" || "$PHASE" == "extract" ]]; then
        [[ -n "${HF_TOKEN:-}" ]] || die "HF_TOKEN is not exported; invoke through bash -ic or export it first"
    fi
    if ((UPLOAD_WANDB)); then
        [[ -n "${WANDB_API_KEY:-}" ]] || die "WANDB_API_KEY is not exported"
        [[ -n "$WANDB_PROJECT" ]] || die "set --wandb-project or export WANDB_PROJECT"
    fi

    "$PYTHON" "$RUNNER" --manifest "$MANIFEST" check-tasks
    "$PYTHON" "$RUNNER" --manifest "$MANIFEST" check-models
    "$PYTHON" "$RUNNER" --manifest "$MANIFEST" plan

    # Freeze sample selection only before extraction. Validation and analysis
    # consume the hash-bound selection in run_manifest.json, allowing their
    # versioned code to improve without pretending the extraction source changed.
    if [[ "$PHASE" == "preflight" || "$PHASE" == "all" || "$PHASE" == "extract" ]]; then
        "$PYTHON" "$RUNNER" --manifest "$MANIFEST" freeze-samples --run-dir "$RUN_DIR"
    fi
}

extract() {
    stage "Extraction"
    local command=(
        "$PYTHON" "$RUNNER" --manifest "$MANIFEST" run
        --run-dir "$RUN_DIR" --skip-existing
    )
    if ((DRY_RUN)); then
        command+=(--dry-run)
    fi
    WANDB_MODE=disabled \
    TOKENIZERS_PARALLELISM=false \
    PYTHONHASHSEED=0 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        "${command[@]}"
}

validate() {
    stage "Artifact and cross-model sample-ID validation"
    "$PYTHON" "$RUNNER" --manifest "$MANIFEST" validate --run-dir "$RUN_DIR"
}

analyze() {
    stage "Exact task-label permutation analysis"
    local sensitivity_args=()
    if [[ -n "$ANALYSIS_NONLINEAR_PERMUTATIONS" ]]; then
        sensitivity_args+=(--nonlinear-permutations "$ANALYSIS_NONLINEAR_PERMUTATIONS")
    fi
    if ((ANALYSIS_CONDITION_JOBS == 1)); then
        start_mps
        "$PYTHON" "$RUNNER" --manifest "$MANIFEST" --python "$ANALYSIS_PYTHON" analyze \
            --run-dir "$RUN_DIR" --jobs "$ANALYSIS_JOBS" --parallel-backend process \
            --analysis-backend "$ANALYSIS_BACKEND" --analysis-name "$ANALYSIS_NAME" \
            "${sensitivity_args[@]}"
        stop_mps
        return
    fi

    command -v xargs >/dev/null || die "parallel condition analysis requires xargs"
    start_mps
    local pairs=()
    mapfile -t pairs < <("$PYTHON" - "$MANIFEST" <<'PY'
import json
import sys

manifest = json.load(open(sys.argv[1], encoding="utf-8"))
for panel in manifest["panels"]:
    for model in manifest["models"]:
        print(f"{panel}:{model}")
PY
)
    export PYTHON RUNNER MANIFEST ANALYSIS_PYTHON RUN_DIR ANALYSIS_JOBS ANALYSIS_BACKEND ANALYSIS_NAME
    export ANALYSIS_NONLINEAR_PERMUTATIONS
    printf '%s\n' "${pairs[@]}" | xargs -P "$ANALYSIS_CONDITION_JOBS" -n 1 bash -c '
        pair="$1"
        panel="${pair%%:*}"
        model="${pair#*:}"
        sensitivity_args=()
        if [[ -n "$ANALYSIS_NONLINEAR_PERMUTATIONS" ]]; then
            sensitivity_args+=(--nonlinear-permutations "$ANALYSIS_NONLINEAR_PERMUTATIONS")
        fi
        "$PYTHON" "$RUNNER" --manifest "$MANIFEST" --python "$ANALYSIS_PYTHON" analyze \
            --panel "$panel" --model "$model" --run-dir "$RUN_DIR" \
            --jobs "$ANALYSIS_JOBS" --parallel-backend process \
            --analysis-backend "$ANALYSIS_BACKEND" --analysis-name "$ANALYSIS_NAME" \
            "${sensitivity_args[@]}"
    ' _
    stop_mps
}

report() {
    stage "Aggregate-only reporting gate"
    local sample_count preview
    sample_count="$($PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["extraction"]["sample_count_per_task"])' "$MANIFEST")"
    preview="$RUN_DIR/wandb_aggregate_preview.json"
    "$PYTHON" "$WANDB_REPORTER" "$RUN_DIR" \
        --analysis-name "$ANALYSIS_NAME" \
        --kind full-experiment \
        --minimum-samples-per-task "$sample_count" \
        --project "$WANDB_PROJECT" \
        --name "$WANDB_NAME" \
        --dry-run >"$preview"
    echo "Validated scalar-only payload: $preview"
    if ((UPLOAD_WANDB)); then
        "$PYTHON" "$WANDB_REPORTER" "$RUN_DIR" \
            --analysis-name "$ANALYSIS_NAME" \
            --kind full-experiment \
            --minimum-samples-per-task "$sample_count" \
            --project "$WANDB_PROJECT" \
            --name "$WANDB_NAME"
    else
        echo "W&B upload disabled. After reviewing the preview, re-run --phase report"
        echo "with --upload-wandb --wandb-project PROJECT. No default project is assumed."
    fi
}

preflight
if [[ "$PHASE" == "preflight" ]]; then
    exit 0
fi

if ((DRY_RUN)); then
    [[ "$PHASE" == "all" || "$PHASE" == "extract" ]] || die "--dry-run is valid only with --phase all or extract"
    extract
    exit 0
fi

mkdir -p "$RUN_DIR"
exec 9>"$RUN_DIR/full_experiment.lock"
flock -n 9 || die "another full-experiment launcher holds $RUN_DIR/full_experiment.lock"
exec > >(tee -a "$RUN_DIR/launcher.log") 2>&1

if [[ "$PHASE" == "all" || "$PHASE" == "extract" || "$PHASE" == "analyze" ]]; then
    GPU_TELEMETRY="$RUN_DIR/gpu_telemetry.csv"
    if [[ -f "$GPU_TELEMETRY" ]]; then
        GPU_TELEMETRY_START_LINE=$(( $(wc -l <"$GPU_TELEMETRY") + 1 ))
    fi
    nvidia-smi \
        --query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total,power.draw \
        --format=csv,noheader,nounits --loop=5 >>"$GPU_TELEMETRY" &
    GPU_MONITOR_PID=$!
    echo "GPU telemetry: $GPU_TELEMETRY (5-second sampling)"
fi

stage "Locked full run"
echo "run_dir=$RUN_DIR"
echo "root_commit=$(git -C "$ROOT" rev-parse HEAD)"
echo "lmms_eval_commit=$(git -C "$ROOT/lmms-eval" rev-parse HEAD)"
echo "phase=$PHASE analysis_backend=$ANALYSIS_BACKEND analysis_jobs=$ANALYSIS_JOBS analysis_condition_jobs=$ANALYSIS_CONDITION_JOBS nonlinear_permutations=${ANALYSIS_NONLINEAR_PERMUTATIONS:-manifest} upload_wandb=$UPLOAD_WANDB"

case "$PHASE" in
    extract)
        extract
        ;;
    validate)
        validate
        ;;
    analyze)
        analyze
        ;;
    report)
        report
        ;;
    all)
        extract
        validate
        analyze
        report
        ;;
esac

stage "Completed phase: $PHASE"
