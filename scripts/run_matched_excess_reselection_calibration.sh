#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$WORKSPACE/.venv/bin/python}"
SOURCE_ROOT="${SOURCE_ROOT:-/home/ubuntu/data-imu-us-east3/vlmbehave-artifacts/2026-08-13/representation-dynamics-main-track-v1/matched_mmlu_8x8_confirmatory}"
MATCHED_ROOT="${MATCHED_ROOT:-/home/ubuntu/data-imu-us-east3/vlmbehave-artifacts/2026-08-16/representation-dynamics-main-track-instrument-validation-v1/matched-excess}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/ubuntu/data-imu-us-east3/vlmbehave-artifacts/2026-08-16/representation-dynamics-main-track-review-response-v1}"
MAX_PARALLEL="${MAX_PARALLEL:-10}"
MEAN_DIFFERENCES="${MEAN_DIFFERENCES:-0.5 1 2 4}"
TELEMETRY_INTERVAL="${TELEMETRY_INTERVAL:-20}"
model="olmo3-7b-instruct"
condition_root="$SOURCE_ROOT/$model"
permutation="$condition_root/analysis-cuml-26.06/permutation_nulls.json.gz"
matched_report="$MATCHED_ROOT/$model/report.json"
matched_distribution="$MATCHED_ROOT/$model/matched_excess_distribution.json.gz"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python environment not found: $PYTHON_BIN" >&2
  exit 2
fi
if [[ "$MAX_PARALLEL" -lt 1 ]]; then
  echo "MAX_PARALLEL must be positive" >&2
  exit 2
fi
local_missing=0
for path in "$condition_root" "$permutation" "$matched_report" "$matched_distribution"; do
  if [[ ! -e "$path" ]]; then
    echo "Missing calibration input: $path" >&2
    local_missing=1
  fi
done
if [[ "$local_missing" -ne 0 ]]; then
  exit 2
fi

local_artifacts=()
mapfile -t local_artifacts < <(find "$condition_root/tasks" -type f -name representations.pt | sort)
if [[ "${#local_artifacts[@]}" -ne 16 ]]; then
  echo "Expected 16 activation artifacts, found ${#local_artifacts[@]}" >&2
  exit 2
fi

mkdir -p "$OUTPUT_ROOT/logs"
export PYTHONPATH="$WORKSPACE/lmms-eval${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"

run_condition() {
  local difference="$1"
  local direction_seed="$2"
  local slug="d${difference}.seed${direction_seed}"
  "$PYTHON_BIN" -m lmms_eval.representations.matched_excess_calibration \
    "${local_artifacts[@]}" \
    --permutation-nulls "$permutation" \
    --matched-report "$matched_report" \
    --matched-distribution "$matched_distribution" \
    --output-dir "$OUTPUT_ROOT/matched-delta-calibration/$model/difference-$difference/seed-$direction_seed" \
    --direction-seed "$direction_seed" \
    --standardized-mean-difference "$difference" \
    --analysis-backend cuml \
    >"$OUTPUT_ROOT/logs/${model}.matched-delta-calibration.${slug}.log" 2>&1
}

telemetry_pid=""
if command -v nvidia-smi >/dev/null 2>&1; then
  (
    while true; do
      nvidia-smi --query-gpu=timestamp,index,utilization.gpu,power.draw,memory.used,memory.total \
        --format=csv,noheader,nounits
      sleep "$TELEMETRY_INTERVAL"
    done
  ) >>"$OUTPUT_ROOT/logs/gpu-matched-delta-calibration.csv" 2>&1 &
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
read -r -a differences <<<"$MEAN_DIFFERENCES"
for difference in "${differences[@]}"; do
  for direction_seed in {0..9}; do
    run_condition "$difference" "$direction_seed" &
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
  echo "Matched-delta calibration failed in $failures process(es); inspect logs" >&2
  exit 1
fi

"$PYTHON_BIN" - "$OUTPUT_ROOT" "$MEAN_DIFFERENCES" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
differences = sys.argv[2].split()
for difference in differences:
    for seed in range(10):
        path = root / "matched-delta-calibration" / "olmo3-7b-instruct" / f"difference-{difference}" / f"seed-{seed}" / "report.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        assert report["status"] == "PASS", (path, report["status"])
print(json.dumps({"status": "PASS", "differences": differences, "directions_each": 10}, indent=2))
PY
