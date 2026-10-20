#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$WORKSPACE/.venv/bin/python}"
SOURCE_ROOT="${SOURCE_ROOT:-/home/ubuntu/data-imu-us-east3/vlmbehave-artifacts/2026-08-13/representation-dynamics-main-track-v1/matched_mmlu_8x8_confirmatory}"
APERTUS_SOURCE_ROOT="${APERTUS_SOURCE_ROOT:-/home/ubuntu/data-imu-us-east3/vlmbehave-artifacts/2026-08-13/representation-dynamics-main-track-apertus15-runtime-v1/matched_mmlu_8x8_confirmatory}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/ubuntu/data-imu-us-east3/vlmbehave-artifacts/2026-08-16/representation-dynamics-selector-sensitivity-v1}"
MAX_PARALLEL="${MAX_PARALLEL:-5}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python environment not found: $PYTHON_BIN" >&2
  exit 2
fi
mkdir -p "$OUTPUT_ROOT/logs"
export PYTHONPATH="$WORKSPACE/lmms-eval${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"

models=(olmo3-7b-instruct apertus15-8b qwen3_5-9b qwen3_5-27b gemma4-31b)
run_model() {
  local model="$1"
  local condition_root="$SOURCE_ROOT/$model"
  if [[ "$model" == "apertus15-8b" ]]; then
    condition_root="$APERTUS_SOURCE_ROOT/$model"
  fi
  local permutation="$condition_root/analysis-cuml-26.06/permutation_nulls.json.gz"
  local -a artifacts=()
  mapfile -t artifacts < <(find "$condition_root/tasks" -type f -name representations.pt | sort)
  if [[ "${#artifacts[@]}" -ne 16 || ! -f "$permutation" ]]; then
    echo "$model: incomplete selector inputs" >&2
    return 3
  fi
  "$PYTHON_BIN" -m lmms_eval.representations.whitened_selector \
    "${artifacts[@]}" \
    --permutation-nulls "$permutation" \
    --output-dir "$OUTPUT_ROOT/$model" \
    --ridge-alphas 0.01,0.1,1 \
    --pca-variance 0.95 \
    --analysis-backend cuml \
    >"$OUTPUT_ROOT/logs/$model.log" 2>&1
}

telemetry_pid=""
if command -v nvidia-smi >/dev/null 2>&1; then
  (
    while true; do
      nvidia-smi --query-gpu=timestamp,index,utilization.gpu,power.draw,memory.used,memory.total \
        --format=csv,noheader,nounits
      sleep 20
    done
  ) >>"$OUTPUT_ROOT/logs/gpu.csv" 2>&1 &
  telemetry_pid="$!"
fi
cleanup() {
  if [[ -n "$telemetry_pid" ]]; then
    kill "$telemetry_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

running=0
failures=0
for model in "${models[@]}"; do
  run_model "$model" &
  running=$((running + 1))
  if [[ "$running" -ge "$MAX_PARALLEL" ]]; then
    if ! wait -n; then failures=$((failures + 1)); fi
    running=$((running - 1))
  fi
done
while [[ "$running" -gt 0 ]]; do
  if ! wait -n; then failures=$((failures + 1)); fi
  running=$((running - 1))
done
if [[ "$failures" -ne 0 ]]; then
  echo "Whitened selector failed in $failures model process(es)" >&2
  exit 1
fi

"$PYTHON_BIN" - "$OUTPUT_ROOT" <<'PY'
import json
import pathlib
import sys
root = pathlib.Path(sys.argv[1])
models = ("olmo3-7b-instruct", "apertus15-8b", "qwen3_5-9b", "qwen3_5-27b", "gemma4-31b")
for model in models:
    report = json.loads((root / model / "report.json").read_text(encoding="utf-8"))
    assert report["status"] == "PASS", (model, report["status"])
print(json.dumps({"status": "PASS", "models": len(models)}, indent=2))
PY
