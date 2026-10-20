#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$WORKSPACE/.venv/bin/python}"
SOURCE_ROOT="${SOURCE_ROOT:-/home/ubuntu/data-imu-us-east3/vlmbehave-artifacts/2026-08-13/representation-dynamics-main-track-v1/matched_mmlu_8x8_confirmatory}"
APERTUS_SOURCE_ROOT="${APERTUS_SOURCE_ROOT:-/home/ubuntu/data-imu-us-east3/vlmbehave-artifacts/2026-08-13/representation-dynamics-main-track-apertus15-runtime-v1/matched_mmlu_8x8_confirmatory}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/ubuntu/data-imu-us-east3/vlmbehave-artifacts/2026-08-16/representation-dynamics-main-track-instrument-validation-v1}"
PARALLEL_MODELS="${PARALLEL_MODELS:-1}"
CALIBRATION_PARALLEL="${CALIBRATION_PARALLEL:-12}"
CALIBRATION_THREADS="${CALIBRATION_THREADS:-1}"
LEXICAL_CACHE_WORKERS="${LEXICAL_CACHE_WORKERS:-24}"
CALIBRATION_SHIFTS="${CALIBRATION_SHIFTS:-0 0.25 0.5 1 2}"
PHASE="${PHASE:-all}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python environment not found: $PYTHON_BIN" >&2
  exit 2
fi
if [[ ! -d "$SOURCE_ROOT" ]]; then
  echo "Primary artifact root not found: $SOURCE_ROOT" >&2
  exit 2
fi
if [[ "$PARALLEL_MODELS" -lt 1 ]]; then
  echo "PARALLEL_MODELS must be positive" >&2
  exit 2
fi
if [[ "$CALIBRATION_PARALLEL" -lt 1 ]]; then
  echo "CALIBRATION_PARALLEL must be positive" >&2
  exit 2
fi
if [[ "$CALIBRATION_THREADS" -lt 1 ]]; then
  echo "CALIBRATION_THREADS must be positive" >&2
  exit 2
fi
if [[ "$LEXICAL_CACHE_WORKERS" -lt 1 ]]; then
  echo "LEXICAL_CACHE_WORKERS must be positive" >&2
  exit 2
fi
if [[ "$PHASE" != "all" && "$PHASE" != "matched" && "$PHASE" != "diagnostics" && "$PHASE" != "calibration" ]]; then
  echo "PHASE must be all, matched, diagnostics, or calibration" >&2
  exit 2
fi

mkdir -p "$OUTPUT_ROOT/logs"
export PYTHONPATH="$WORKSPACE/lmms-eval${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

models=(
  olmo3-7b-instruct
  apertus15-8b
  qwen3_5-9b
  qwen3_5-27b
  gemma4-31b
)

run_condition() {
  local model="$1"
  local condition_root="$SOURCE_ROOT/$model"
  if [[ "$model" == "apertus15-8b" ]]; then
    condition_root="$APERTUS_SOURCE_ROOT/$model"
  fi
  local permutation="$condition_root/analysis-cuml-26.06/permutation_nulls.json.gz"
  local -a artifacts=()
  mapfile -t artifacts < <(find "$condition_root/tasks" -type f -name representations.pt | sort)
  if [[ "${#artifacts[@]}" -ne 16 ]]; then
    echo "$model: expected 16 activation artifacts, found ${#artifacts[@]}" >&2
    return 3
  fi
  if [[ ! -f "$permutation" ]]; then
    echo "$model: missing Fisher null artifact: $permutation" >&2
    return 3
  fi

  if [[ "$PHASE" == "all" || "$PHASE" == "matched" ]]; then
    "$PYTHON_BIN" -m lmms_eval.representations.matched_excess \
      "${artifacts[@]}" \
      --permutation-nulls "$permutation" \
      --sample-root "$condition_root" \
      --output-dir "$OUTPUT_ROOT/matched-excess/$model" \
      --analysis-backend cuml \
      --components 64 \
      --checkpoint-every 25 \
      >"$OUTPUT_ROOT/logs/${model}.matched.log" 2>&1
  fi

  if [[ "$PHASE" == "all" || "$PHASE" == "diagnostics" ]]; then
    "$PYTHON_BIN" -m lmms_eval.representations.lexical_diagnostics \
      "${artifacts[@]}" \
      --permutation-nulls "$permutation" \
      --sample-root "$condition_root" \
      --output-dir "$OUTPUT_ROOT/lexical-diagnostics/$model" \
      --analysis-backend cuml \
      --lexical-components 64 \
      --task-probe-components 64 \
      >"$OUTPUT_ROOT/logs/${model}.diagnostics.log" 2>&1
  fi
}

run_calibration_condition() {
  local shift="$1"
  local direction_seed="$2"
  local model="olmo3-7b-instruct"
  local condition_root="$SOURCE_ROOT/$model"
  local permutation="$condition_root/analysis-cuml-26.06/permutation_nulls.json.gz"
  local -a artifacts=()
  mapfile -t artifacts < <(find "$condition_root/tasks" -type f -name representations.pt | sort)
  env OMP_NUM_THREADS="$CALIBRATION_THREADS" \
    OPENBLAS_NUM_THREADS="$CALIBRATION_THREADS" \
    MKL_NUM_THREADS="$CALIBRATION_THREADS" \
    "$PYTHON_BIN" -m lmms_eval.representations.probe_calibration \
    "${artifacts[@]}" \
    --permutation-nulls "$permutation" \
    --sample-root "$condition_root" \
    --output-dir "$OUTPUT_ROOT/probe-calibration/$model/shift-$shift/seed-$direction_seed" \
    --analysis-backend cuml \
    --lexical-components 64 \
    --lexical-cache "$OUTPUT_ROOT/cache/olmo3-7b-instruct-char64.npz" \
    --lexical-cache-workers "$LEXICAL_CACHE_WORKERS" \
    --pca-variance 0.95 \
    --direction-seeds "$direction_seed" \
    --standardized-half-shifts "$shift" \
    --threshold 0.01 \
    >"$OUTPUT_ROOT/logs/${model}.calibration-shift-$shift-seed-$direction_seed.log" 2>&1
}

failures=0
if [[ "$PHASE" != "calibration" ]]; then
  running=0
  for model in "${models[@]}"; do
    run_condition "$model" &
    running=$((running + 1))
    if [[ "$running" -ge "$PARALLEL_MODELS" ]]; then
      if ! wait -n; then
        failures=$((failures + 1))
      fi
      running=$((running - 1))
    fi
  done
  while [[ "$running" -gt 0 ]]; do
    if ! wait -n; then
      failures=$((failures + 1))
    fi
    running=$((running - 1))
  done
fi
if [[ "$PHASE" == "all" || "$PHASE" == "calibration" ]]; then
  calibration_running=0
  read -r -a calibration_shifts <<<"$CALIBRATION_SHIFTS"
  if [[ "${#calibration_shifts[@]}" -eq 0 ]]; then
    echo "CALIBRATION_SHIFTS must contain at least one nonnegative half-shift" >&2
    exit 2
  fi
  for shift in "${calibration_shifts[@]}"; do
    if [[ "$shift" == "0" ]]; then
      seeds=(0)
    else
      seeds=(0 1 2 3 4 5 6 7 8 9)
    fi
    for direction_seed in "${seeds[@]}"; do
      run_calibration_condition "$shift" "$direction_seed" &
      calibration_running=$((calibration_running + 1))
      if [[ "$calibration_running" -ge "$CALIBRATION_PARALLEL" ]]; then
        if ! wait -n; then
          failures=$((failures + 1))
        fi
        calibration_running=$((calibration_running - 1))
      fi
    done
  done
  while [[ "$calibration_running" -gt 0 ]]; do
    if ! wait -n; then
      failures=$((failures + 1))
    fi
    calibration_running=$((calibration_running - 1))
  done
fi

if [[ "$failures" -ne 0 ]]; then
  echo "Instrument validation failed in $failures model process(es); inspect $OUTPUT_ROOT/logs" >&2
  exit 1
fi

"$PYTHON_BIN" - "$OUTPUT_ROOT" "$PHASE" "$CALIBRATION_SHIFTS" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
phase = sys.argv[2]
calibration_shifts = sys.argv[3].split()
models = ("olmo3-7b-instruct", "apertus15-8b", "qwen3_5-9b", "qwen3_5-27b", "gemma4-31b")
families = []
if phase in {"all", "matched"}:
    families.append("matched-excess")
if phase in {"all", "diagnostics"}:
    families.append("lexical-diagnostics")
for family in families:
    for model in models:
        path = root / family / model / "report.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        assert report["status"] == "PASS", (path, report["status"])
if phase in {"all", "calibration"}:
    for shift in calibration_shifts:
        seeds = (0,) if shift == "0" else tuple(range(10))
        for direction_seed in seeds:
            path = (
                root / "probe-calibration" / "olmo3-7b-instruct" / f"shift-{shift}"
                / f"seed-{direction_seed}" / "report.json"
            )
            report = json.loads(path.read_text(encoding="utf-8"))
            assert report["status"] == "PASS", (path, report["status"])
print(json.dumps({"status": "PASS", "phase": phase, "models": len(models)}, indent=2))
PY
