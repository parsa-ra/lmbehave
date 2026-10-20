#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
MANIFEST="$ROOT_DIR/experiments/apertus1_longitudinal_v1.json"
PERSISTENT_ROOT="/home/ubuntu/data-imu-us-east3/vlmbehave-artifacts"
CANONICAL_PERSISTENT_ROOT="$(realpath -m "$PERSISTENT_ROOT")"
RUN_DIR="${1:-$PERSISTENT_ROOT/2026-08-13/apertus1-longitudinal-v1}"
ANALYSIS_JOBS="${APERTUS_ANALYSIS_JOBS:-4}"
ANALYSIS_CONDITION_JOBS="${APERTUS_ANALYSIS_CONDITION_JOBS:-4}"
ANALYSIS_ONLY="${APERTUS_ANALYSIS_ONLY:-0}"
TELEMETRY_PATH="${APERTUS_TELEMETRY_PATH:-$RUN_DIR/gpu_telemetry_analysis_resume_20260815.csv}"
PREFLIGHT="$ROOT_DIR/results/apertus1_preflight_v1/report.json"
TELEMETRY_PID=""
MPS_OWNED=0
MPS_ROOT=""

cleanup() {
  if ((MPS_OWNED)); then
    printf 'quit\n' | CUDA_MPS_PIPE_DIRECTORY="$CUDA_MPS_PIPE_DIRECTORY" \
      nvidia-cuda-mps-control >/dev/null 2>&1 || true
    MPS_OWNED=0
  fi
  if [[ -n "$TELEMETRY_PID" ]]; then
    kill "$TELEMETRY_PID" 2>/dev/null || true
    wait "$TELEMETRY_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

start_mps() {
  ((ANALYSIS_JOBS * ANALYSIS_CONDITION_JOBS > 1)) || return 0
  command -v nvidia-cuda-mps-control >/dev/null || {
    echo "Parallel cuML analysis requires nvidia-cuda-mps-control." >&2
    exit 1
  }
  command -v pgrep >/dev/null || {
    echo "Parallel cuML analysis requires pgrep." >&2
    exit 1
  }
  if pgrep -x nvidia-cuda-mps >/dev/null; then
    echo "An NVIDIA MPS service is already running without this launcher's pipe directory." >&2
    exit 1
  fi
  MPS_ROOT="$(mktemp -d /tmp/vlmbehave-apertus-mps.XXXXXX)"
  export CUDA_MPS_PIPE_DIRECTORY="$MPS_ROOT/pipe"
  export CUDA_MPS_LOG_DIRECTORY="$MPS_ROOT/log"
  mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
  nvidia-cuda-mps-control -d
  MPS_OWNED=1
}

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing environment Python: $PYTHON_BIN" >&2
  exit 1
fi

case "$(realpath -m "$RUN_DIR")" in
  "$CANONICAL_PERSISTENT_ROOT"/*) ;;
  *)
    echo "Final Apertus artifacts must remain under $PERSISTENT_ROOT: $RUN_DIR" >&2
    exit 1
    ;;
esac

if ! [[ "$ANALYSIS_JOBS" =~ ^[1-9][0-9]*$ ]]; then
  echo "APERTUS_ANALYSIS_JOBS must be a positive integer: $ANALYSIS_JOBS" >&2
  exit 1
fi
if ! [[ "$ANALYSIS_CONDITION_JOBS" =~ ^[1-9][0-9]*$ ]]; then
  echo "APERTUS_ANALYSIS_CONDITION_JOBS must be a positive integer: $ANALYSIS_CONDITION_JOBS" >&2
  exit 1
fi
if [[ "$ANALYSIS_ONLY" != "0" && "$ANALYSIS_ONLY" != "1" ]]; then
  echo "APERTUS_ANALYSIS_ONLY must be 0 or 1: $ANALYSIS_ONLY" >&2
  exit 1
fi

cd "$ROOT_DIR"
mkdir -p "$RUN_DIR"
if [[ "$ANALYSIS_ONLY" == "0" ]]; then
  "$PYTHON_BIN" scripts/check_apertus1_release.py --require-ready
  "$PYTHON_BIN" scripts/summarize_apertus1_preflight.py --output "$PREFLIGHT"
  "$PYTHON_BIN" - "$PREFLIGHT" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
if report.get("status") != "PREFLIGHT_COMPLETE_WITH_NONEXACT_BENCHMARK_ANCHOR":
    raise SystemExit(f"Apertus preflight is not final-run ready: {report.get('status')}")
PY
  "$PYTHON_BIN" scripts/run_olmo_longitudinal_experiment.py \
    --manifest "$MANIFEST" plan
  "$PYTHON_BIN" scripts/run_olmo_longitudinal_experiment.py \
    --manifest "$MANIFEST" run --run-dir "$RUN_DIR" --skip-existing
else
  echo "Analysis-only resume: release checks and checkpoint extraction are skipped."
fi

"$PYTHON_BIN" scripts/run_olmo_longitudinal_experiment.py \
  --manifest "$MANIFEST" validate --run-dir "$RUN_DIR"
nvidia-smi \
  --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,power.limit,temperature.gpu \
  --format=csv,noheader,nounits \
  --loop=10 >> "$TELEMETRY_PATH" &
TELEMETRY_PID="$!"

start_mps
"$PYTHON_BIN" scripts/run_olmo_longitudinal_experiment.py \
  --manifest "$MANIFEST" analyze --run-dir "$RUN_DIR" --analysis-backend cuml \
  --jobs "$ANALYSIS_JOBS" --condition-jobs "$ANALYSIS_CONDITION_JOBS"
"$PYTHON_BIN" scripts/run_olmo_longitudinal_experiment.py \
  --manifest "$MANIFEST" summarize --run-dir "$RUN_DIR" --analysis-backend cuml

cleanup
TELEMETRY_PID=""
echo "Apertus longitudinal study completed under $RUN_DIR"
