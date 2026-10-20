#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/home/ubuntu/data-imu-us-east3/vlmbehave-artifacts}"
STANDARD_ROOT="${STANDARD_ROOT:-$ARTIFACT_ROOT/2026-08-13/representation-dynamics-main-track-v1/matched_mmlu_8x8_confirmatory}"
APERTUS_ROOT="${APERTUS_ROOT:-$ARTIFACT_ROOT/2026-08-13/representation-dynamics-main-track-apertus15-runtime-v1/matched_mmlu_8x8_confirmatory}"
OUTPUT_ROOT="${1:-$ARTIFACT_ROOT/2026-08-15/selected-site-probe-main-track-v1}"
CANONICAL_ARTIFACT_ROOT="$(realpath -m "$ARTIFACT_ROOT")"
TELEMETRY_PID=""

cleanup() {
  if [[ -n "$TELEMETRY_PID" ]]; then
    kill "$TELEMETRY_PID" 2>/dev/null || true
    wait "$TELEMETRY_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

[[ -x "$PYTHON_BIN" ]] || {
  printf 'Missing environment Python: %s\n' "$PYTHON_BIN" >&2
  exit 2
}
[[ -z "$(git -C "$ROOT_DIR/lmms-eval" status --porcelain --untracked-files=all)" ]] || {
  printf 'lmms-eval must be clean before analysis.\n' >&2
  exit 3
}
case "$(realpath -m "$OUTPUT_ROOT")" in
  "$CANONICAL_ARTIFACT_ROOT"/*) ;;
  *) printf 'Output must remain under %s: %s\n' "$ARTIFACT_ROOT" "$OUTPUT_ROOT" >&2; exit 4 ;;
esac

mkdir -p "$OUTPUT_ROOT"
nvidia-smi \
  --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,power.limit,temperature.gpu \
  --format=csv,noheader,nounits \
  --loop=10 >"$OUTPUT_ROOT/gpu-telemetry.csv" &
TELEMETRY_PID="$!"

conditions=(
  "olmo3-7b-instruct|$STANDARD_ROOT/olmo3-7b-instruct"
  "apertus15-8b|$APERTUS_ROOT/apertus15-8b"
  "qwen3_5-9b|$STANDARD_ROOT/qwen3_5-9b"
  "qwen3_5-27b|$STANDARD_ROOT/qwen3_5-27b"
  "gemma4-31b|$STANDARD_ROOT/gemma4-31b"
)

pids=()
names=()
for condition in "${conditions[@]}"; do
  name="${condition%%|*}"
  input="${condition#*|}"
  output="$OUTPUT_ROOT/$name"
  report="$output/report.json"
  if [[ -f "$report" ]] && "$PYTHON_BIN" - "$report" <<'PY'
import json
import sys
raise SystemExit(0 if json.load(open(sys.argv[1], encoding="utf-8")).get("status") == "PASS" else 1)
PY
  then
    printf 'SKIP validated %s\n' "$name"
    continue
  fi
  permutation="$input/analysis-cuml-26.06/permutation_nulls.json.gz"
  [[ -f "$permutation" ]] || { printf 'Missing permutation null: %s\n' "$permutation" >&2; exit 5; }
  mapfile -t artifacts < <(find "$input/tasks" -type f -name representations.pt | sort)
  [[ "${#artifacts[@]}" -eq 16 ]] || {
    printf 'Expected 16 task artifacts for %s, found %s.\n' "$name" "${#artifacts[@]}" >&2
    exit 6
  }
  mkdir -p "$output"
  printf 'START %s (%s artifacts)\n' "$name" "${#artifacts[@]}"
  "$PYTHON_BIN" -m lmms_eval.representations.selected_site_probe \
    "${artifacts[@]}" \
    --permutation-nulls "$permutation" \
    --output-dir "$output" \
    --cv-folds 5 \
    --pca-variance 0.95 \
    --seed 0 \
    --analysis-backend cuml \
    --checkpoint-every 10 \
    >"$output/run.log" 2>&1 &
  pids+=("$!")
  names+=("$name")
done

status=0
for index in "${!pids[@]}"; do
  if wait "${pids[$index]}"; then
    printf 'DONE %s\n' "${names[$index]}"
  else
    printf 'FAILED %s\n' "${names[$index]}" >&2
    status=1
  fi
done
cleanup
TELEMETRY_PID=""
exit "$status"
