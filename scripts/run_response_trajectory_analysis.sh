#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  printf 'Usage: %s RUN_DIR\n' "$0" >&2
  exit 2
fi

workspace=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python="$workspace/.venv/bin/python"
run_dir=$(realpath -m "$1")
persistent_root=$(realpath -m /home/ubuntu/data-imu-us-east3/vlmbehave-artifacts)
jobs=${TRAJECTORY_ANALYSIS_JOBS:-16}
case "$run_dir/" in
  "$persistent_root"/*) ;;
  *) printf 'RUN_DIR must remain under %s\n' "$persistent_root" >&2; exit 3 ;;
esac
[[ "$jobs" =~ ^[1-9][0-9]*$ ]] || { printf 'TRAJECTORY_ANALYSIS_JOBS must be positive\n' >&2; exit 4; }

report="$run_dir/report.json"
"$python" - "$report" <<'PY'
import json
import sys
report = json.load(open(sys.argv[1], encoding="utf-8"))
if report.get("status") != "PASS":
    raise SystemExit("Full trajectory extraction report is not PASS")
if int(report.get("common_decode_prefix", 0)) < 8:
    raise SystemExit("Full trajectory extraction lacks the required common prefix")
PY

mapfile -t artifacts < <(find "$run_dir/representations" -name representations.pt -type f | sort)
if [[ ${#artifacts[@]} -ne 1 ]]; then
  printf 'Expected one representation artifact, found %s\n' "${#artifacts[@]}" >&2
  exit 5
fi
artifact=${artifacts[0]}
analysis_dir="$run_dir/analysis-cuml"
inference_dir="$run_dir/trajectory-inference"
if [[ -f "$inference_dir/report.json" ]]; then
  printf 'Existing trajectory inference: %s\n' "$inference_dir/report.json"
  exit 0
fi
if pgrep -x nvidia-cuda-mps >/dev/null; then
  printf 'Another NVIDIA MPS owner is active; wait for it before this analysis.\n' >&2
  exit 6
fi

mps_root=$(mktemp -d /tmp/vlmbehave-trajectory-mps.XXXXXX)
export CUDA_MPS_PIPE_DIRECTORY="$mps_root/pipe"
export CUDA_MPS_LOG_DIRECTORY="$mps_root/log"
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
telemetry="$run_dir/gpu_telemetry_analysis.csv"
telemetry_pid=""
mps_owned=0
cleanup() {
  if ((mps_owned)); then
    printf 'quit\n' | CUDA_MPS_PIPE_DIRECTORY="$CUDA_MPS_PIPE_DIRECTORY" \
      nvidia-cuda-mps-control >/dev/null 2>&1 || true
    mps_owned=0
  fi
  if [[ -n "$telemetry_pid" ]]; then
    kill "$telemetry_pid" 2>/dev/null || true
    wait "$telemetry_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM
nvidia-cuda-mps-control -d
mps_owned=1
nvidia-smi \
  --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,power.limit,temperature.gpu \
  --format=csv,noheader,nounits --loop=10 >>"$telemetry" &
telemetry_pid=$!

knowledge=repdyn_response_mmlu_anatomy,repdyn_response_mmlu_astronomy,repdyn_response_mmlu_clinical_knowledge,repdyn_response_mmlu_high_school_world_history,repdyn_response_mmlu_professional_law,repdyn_response_mmlu_high_school_biology,repdyn_response_mmlu_medical_genetics,repdyn_response_mmlu_prehistory
reasoning=repdyn_response_mmlu_abstract_algebra,repdyn_response_mmlu_college_mathematics,repdyn_response_mmlu_formal_logic,repdyn_response_mmlu_high_school_mathematics,repdyn_response_mmlu_logical_fallacies,repdyn_response_mmlu_college_physics,repdyn_response_mmlu_conceptual_physics,repdyn_response_mmlu_high_school_statistics

cd "$workspace"
if [[ ! -f "$analysis_dir/report.json" ]]; then
  "$python" -m lmms_eval.representations.analysis "$artifact" \
    --knowledge-tasks "$knowledge" \
    --reasoning-tasks "$reasoning" \
    --output-dir "$analysis_dir" \
    --cv-folds 5 \
    --permutations 0 \
    --fdr-permutations 12869 \
    --nonlinear-permutations 0 \
    --pca-variance 0.95 \
    --silhouette-max-samples 5000 \
    --seed 0 \
    --analysis-backend cuml \
    --jobs "$jobs" \
    --parallel-backend process
fi

"$python" scripts/analyze_response_trajectory_maximum.py \
  --permutation-nulls "$analysis_dir/permutation_nulls.json.gz" \
  --coverage-report "$report" \
  --output-dir "$inference_dir"
