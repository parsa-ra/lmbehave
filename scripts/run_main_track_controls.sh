#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
LMMS_PYTHON="$ROOT_DIR/.venv/bin/python"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/home/ubuntu/data-imu-us-east3/vlmbehave-artifacts}"
STANDARD_ROOT="${STANDARD_ROOT:-$ARTIFACT_ROOT/2026-08-13/representation-dynamics-main-track-v1/matched_mmlu_8x8_confirmatory}"
APERTUS_ROOT="${APERTUS_ROOT:-$ARTIFACT_ROOT/2026-08-13/representation-dynamics-main-track-apertus15-runtime-v1/matched_mmlu_8x8_confirmatory}"
OUTPUT_ROOT="${1:-$ARTIFACT_ROOT/2026-08-16/representation-dynamics-main-track-controls-v1}"
NUISANCE_TABLE="$OUTPUT_ROOT/nuisance_features.csv"
MPS_ROOT=""
TELEMETRY_PID=""
CHILD_PIDS=()

cleanup() {
  local pid
  for pid in "${CHILD_PIDS[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
  if [[ -n "$TELEMETRY_PID" ]]; then
    kill "$TELEMETRY_PID" 2>/dev/null || true
    wait "$TELEMETRY_PID" 2>/dev/null || true
  fi
  if [[ -n "$MPS_ROOT" ]]; then
    printf 'quit\n' | nvidia-cuda-mps-control >/dev/null 2>&1 || true
    rm -r "$MPS_ROOT"
  fi
}
trap cleanup EXIT INT TERM

[[ -x "$PYTHON_BIN" ]] || { printf 'Missing Python: %s\n' "$PYTHON_BIN" >&2; exit 2; }
[[ -z "$(git -C "$ROOT_DIR/lmms-eval" status --porcelain --untracked-files=all)" ]] || {
  printf 'lmms-eval must be clean so reports bind to one implementation.\n' >&2
  exit 3
}
mkdir -p "$OUTPUT_ROOT/logs"

if [[ ! -f "$NUISANCE_TABLE" ]]; then
  "$PYTHON_BIN" "$ROOT_DIR/scripts/build_representation_nuisance_table.py" \
    "$STANDARD_ROOT" "$APERTUS_ROOT" --output "$NUISANCE_TABLE"
fi

sample_root="$STANDARD_ROOT/olmo3-7b-instruct"
for baseline in word_tfidf char_tfidf; do
  output="$OUTPUT_ROOT/prompt_${baseline}.json"
  if [[ -f "$output" ]]; then
    continue
  fi
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    "$LMMS_PYTHON" -m lmms_eval.representations.prompt_baseline \
      --sample-root "$sample_root" \
      --manifest "$ROOT_DIR/experiments/representation_dynamics_main_track_v1.json" \
      --baseline "$baseline" --output "$output" --jobs 24 \
      >"$OUTPUT_ROOT/logs/prompt_${baseline}.log" 2>&1 &
  CHILD_PIDS+=("$!")
done

if pgrep -x nvidia-cuda-mps >/dev/null; then
  [[ -n "${CUDA_MPS_PIPE_DIRECTORY:-}" && -d "$CUDA_MPS_PIPE_DIRECTORY" ]] || {
    printf 'An MPS service is running, but its pipe directory is not available in this shell.\n' >&2
    exit 5
  }
else
  MPS_ROOT="$(mktemp -d /tmp/vlmbehave-main-controls.XXXXXX)"
  export CUDA_MPS_PIPE_DIRECTORY="$MPS_ROOT/pipe"
  export CUDA_MPS_LOG_DIRECTORY="$MPS_ROOT/log"
  mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
  nvidia-cuda-mps-control -d
fi
nvidia-smi \
  --query-gpu=timestamp,utilization.gpu,memory.used,power.draw,power.limit \
  --format=csv,noheader,nounits --loop=10 >"$OUTPUT_ROOT/gpu_telemetry.csv" &
TELEMETRY_PID="$!"

conditions=(
  "olmo3-7b-instruct|$STANDARD_ROOT/olmo3-7b-instruct"
  "apertus15-8b|$APERTUS_ROOT/apertus15-8b"
  "qwen3_5-9b|$STANDARD_ROOT/qwen3_5-9b"
  "qwen3_5-27b|$STANDARD_ROOT/qwen3_5-27b"
  "gemma4-31b|$STANDARD_ROOT/gemma4-31b"
)
controls=(
  "length_only|prompt_token_count"
  "six_surface_features|prompt_token_count,prompt_char_count,prompt_line_count,option_count,whitespace_token_count,whitespace_token_entropy"
)
for control in "${controls[@]}"; do
  control_name="${control%%|*}"
  columns="${control#*|}"
  for condition in "${conditions[@]}"; do
    name="${condition%%|*}"
    input="${condition#*|}"
    output="$OUTPUT_ROOT/$control_name/$name"
    if [[ -f "$output/report.json" ]] && "$PYTHON_BIN" - "$output/report.json" <<'PY'
import json, sys
raise SystemExit(0 if json.load(open(sys.argv[1], encoding="utf-8")).get("status") == "PASS" else 1)
PY
    then
      continue
    fi
    mapfile -t artifacts < <(find "$input/tasks" -type f -name representations.pt | sort)
    [[ "${#artifacts[@]}" -eq 16 ]] || {
      printf 'Expected 16 task artifacts for %s, found %s.\n' "$name" "${#artifacts[@]}" >&2
      exit 4
    }
    mkdir -p "$output"
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
      "$LMMS_PYTHON" -m lmms_eval.representations.selected_site_probe \
        "${artifacts[@]}" \
        --permutation-nulls "$input/analysis-cuml-26.06/permutation_nulls.json.gz" \
        --output-dir "$output" --cv-folds 5 --pca-variance 0.95 --seed 0 \
        --analysis-backend cuml --checkpoint-every 10 \
        --nuisance-table "$NUISANCE_TABLE" --nuisance-columns "$columns" \
        >"$OUTPUT_ROOT/logs/${control_name}_${name}.log" 2>&1 &
    CHILD_PIDS+=("$!")
  done
done

status=0
for pid in "${CHILD_PIDS[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
CHILD_PIDS=()
exit "$status"
