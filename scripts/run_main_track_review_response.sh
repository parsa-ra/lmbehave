#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$WORKSPACE/.venv/bin/python}"
SOURCE_ROOT="${SOURCE_ROOT:-/home/ubuntu/data-imu-us-east3/vlmbehave-artifacts/2026-08-13/representation-dynamics-main-track-v1/matched_mmlu_8x8_confirmatory}"
APERTUS_SOURCE_ROOT="${APERTUS_SOURCE_ROOT:-/home/ubuntu/data-imu-us-east3/vlmbehave-artifacts/2026-08-13/representation-dynamics-main-track-apertus15-runtime-v1/matched_mmlu_8x8_confirmatory}"
MATCHED_ROOT="${MATCHED_ROOT:-/home/ubuntu/data-imu-us-east3/vlmbehave-artifacts/2026-08-16/representation-dynamics-main-track-instrument-validation-v1/matched-excess}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/ubuntu/data-imu-us-east3/vlmbehave-artifacts/2026-08-16/representation-dynamics-main-track-review-response-v1}"
REDUCTIONS="${REDUCTIONS:-fixed:16 fixed:256 raw}"
MAX_PARALLEL="${MAX_PARALLEL:-10}"
TELEMETRY_INTERVAL="${TELEMETRY_INTERVAL:-30}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python environment not found: $PYTHON_BIN" >&2
  exit 2
fi
if [[ ! -d "$SOURCE_ROOT" || ! -d "$MATCHED_ROOT" ]]; then
  echo "Required persistent artifact root is missing" >&2
  exit 2
fi
if [[ "$MAX_PARALLEL" -lt 1 ]]; then
  echo "MAX_PARALLEL must be positive" >&2
  exit 2
fi

mkdir -p "$OUTPUT_ROOT/logs"
export PYTHONPATH="$WORKSPACE/lmms-eval${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"

models=(
  olmo3-7b-instruct
  apertus15-8b
  qwen3_5-9b
  qwen3_5-27b
  gemma4-31b
)

reduction_slug() {
  case "$1" in
    fixed:*) printf 'fixed%s' "${1#fixed:}" ;;
    raw) printf 'raw' ;;
    *) echo "Unsupported reduction: $1" >&2; return 2 ;;
  esac
}

run_condition() {
  local reduction="$1"
  local model="$2"
  local slug condition_root permutation matched_report matched_distribution
  slug="$(reduction_slug "$reduction")"
  condition_root="$SOURCE_ROOT/$model"
  if [[ "$model" == "apertus15-8b" ]]; then
    condition_root="$APERTUS_SOURCE_ROOT/$model"
  fi
  permutation="$condition_root/analysis-cuml-26.06/permutation_nulls.json.gz"
  matched_report="$MATCHED_ROOT/$model/report.json"
  matched_distribution="$MATCHED_ROOT/$model/matched_excess_distribution.json.gz"
  local -a artifacts=()
  mapfile -t artifacts < <(find "$condition_root/tasks" -type f -name representations.pt | sort)
  if [[ "${#artifacts[@]}" -ne 16 ]]; then
    echo "$model: expected 16 artifacts, found ${#artifacts[@]}" >&2
    return 3
  fi
  for path in "$permutation" "$matched_report" "$matched_distribution"; do
    if [[ ! -f "$path" ]]; then
      echo "$model: missing input $path" >&2
      return 3
    fi
  done

  "$PYTHON_BIN" -m lmms_eval.representations.representation_width_sensitivity \
    "${artifacts[@]}" \
    --permutation-nulls "$permutation" \
    --matched-report "$matched_report" \
    --matched-distribution "$matched_distribution" \
    --output-dir "$OUTPUT_ROOT/width-sensitivity/$slug/$model" \
    --activation-reduction "$reduction" \
    --analysis-backend cuml \
    --checkpoint-every 25 \
    >"$OUTPUT_ROOT/logs/${model}.${slug}.log" 2>&1
}

telemetry_pid=""
if command -v nvidia-smi >/dev/null 2>&1; then
  (
    while true; do
      nvidia-smi --query-gpu=timestamp,index,utilization.gpu,power.draw,memory.used,memory.total \
        --format=csv,noheader,nounits
      sleep "$TELEMETRY_INTERVAL"
    done
  ) >>"$OUTPUT_ROOT/logs/gpu-width-sensitivity.csv" 2>&1 &
  telemetry_pid="$!"
fi
cleanup() {
  if [[ -n "$telemetry_pid" ]]; then
    kill "$telemetry_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

failures=0
running=0
read -r -a reduction_list <<<"$REDUCTIONS"
for reduction in "${reduction_list[@]}"; do
  reduction_slug "$reduction" >/dev/null
  for model in "${models[@]}"; do
    run_condition "$reduction" "$model" &
    running=$((running + 1))
    if [[ "$running" -ge "$MAX_PARALLEL" ]]; then
      if ! wait -n; then
        failures=$((failures + 1))
      fi
      running=$((running - 1))
    fi
  done
done
while [[ "$running" -gt 0 ]]; do
  if ! wait -n; then
    failures=$((failures + 1))
  fi
  running=$((running - 1))
done

if [[ "$failures" -ne 0 ]]; then
  echo "Width sensitivity failed in $failures process(es); inspect $OUTPUT_ROOT/logs" >&2
  exit 1
fi

"$PYTHON_BIN" - "$OUTPUT_ROOT" "$REDUCTIONS" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
reductions = sys.argv[2].split()
models = ("olmo3-7b-instruct", "apertus15-8b", "qwen3_5-9b", "qwen3_5-27b", "gemma4-31b")
for reduction in reductions:
    slug = f"fixed{reduction.split(':', 1)[1]}" if reduction.startswith("fixed:") else reduction
    for model in models:
        path = root / "width-sensitivity" / slug / model / "report.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        assert report["status"] == "PASS", (path, report["status"])
print(json.dumps({"status": "PASS", "reductions": reductions, "models": len(models)}, indent=2))
PY
