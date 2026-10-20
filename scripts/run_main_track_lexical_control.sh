#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/home/ubuntu/data-imu-us-east3/vlmbehave-artifacts}"
STANDARD_ROOT="${STANDARD_ROOT:-$ARTIFACT_ROOT/2026-08-13/representation-dynamics-main-track-v1/matched_mmlu_8x8_confirmatory}"
APERTUS_ROOT="${APERTUS_ROOT:-$ARTIFACT_ROOT/2026-08-13/representation-dynamics-main-track-apertus15-runtime-v1/matched_mmlu_8x8_confirmatory}"
OUTPUT_ROOT="${1:-$ARTIFACT_ROOT/2026-08-16/representation-dynamics-main-track-lexical-control-v1}"
EXPECTED_LMMS_HEAD="0b69f69e455a00b748d799a517939a033338bdd8"
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
[[ "$(git -C "$ROOT_DIR/lmms-eval" rev-parse HEAD)" == "$EXPECTED_LMMS_HEAD" ]] || {
  printf 'Unexpected lmms-eval revision. Expected %s.\n' "$EXPECTED_LMMS_HEAD" >&2
  exit 4
}
mkdir -p "$OUTPUT_ROOT/logs"

if pgrep -x nvidia-cuda-mps >/dev/null; then
  [[ -n "${CUDA_MPS_PIPE_DIRECTORY:-}" && -d "$CUDA_MPS_PIPE_DIRECTORY" ]] || {
    printf 'An MPS service is running, but its pipe directory is unavailable.\n' >&2
    exit 5
  }
else
  MPS_ROOT="$(mktemp -d /tmp/vlmbehave-lexical-control.XXXXXX)"
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
for components in 32 64; do
  for condition in "${conditions[@]}"; do
    name="${condition%%|*}"
    input="${condition#*|}"
    output="$OUTPUT_ROOT/char_tfidf_svd${components}/$name"
    if [[ -f "$output/report.json" ]] && "$PYTHON_BIN" - "$output/report.json" <<'PY'
import json, sys
report = json.load(open(sys.argv[1], encoding="utf-8"))
raise SystemExit(0 if report.get("status") == "PASS" else 1)
PY
    then
      continue
    fi
    mapfile -t artifacts < <(find "$input/tasks" -type f -name representations.pt | sort)
    [[ "${#artifacts[@]}" -eq 16 ]] || {
      printf 'Expected 16 task artifacts for %s, found %s.\n' "$name" "${#artifacts[@]}" >&2
      exit 6
    }
    mkdir -p "$output"
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
      "$PYTHON_BIN" -m lmms_eval.representations.selected_site_probe \
        "${artifacts[@]}" \
        --permutation-nulls "$input/analysis-cuml-26.06/permutation_nulls.json.gz" \
        --output-dir "$output" --cv-folds 5 --pca-variance 0.95 --seed 0 \
        --analysis-backend cuml --checkpoint-every 10 \
        --lexical-nuisance-sample-root "$input" \
        --lexical-nuisance-kind char_tfidf \
        --lexical-nuisance-components "$components" \
        >"$OUTPUT_ROOT/logs/char_tfidf_svd${components}_${name}.log" 2>&1 &
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
[[ "$status" -eq 0 ]] || exit "$status"

"$PYTHON_BIN" - "$OUTPUT_ROOT" "$EXPECTED_LMMS_HEAD" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
expected_head = sys.argv[2]
for components in (32, 64):
    for model in ("olmo3-7b-instruct", "apertus15-8b", "qwen3_5-9b", "qwen3_5-27b", "gemma4-31b"):
        path = root / f"char_tfidf_svd{components}" / model / "report.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        lexical = report["configuration"]["nuisance_control"]["lexical"]
        implementation = report["configuration"]["implementation"]
        assert report["status"] == "PASS", path
        assert report["assignments"] == 12870 and report["probe_refits"] == 6435, path
        assert implementation["git_head"] == expected_head and not implementation["git_dirty"], path
        assert lexical["requested_components"] == components, path
        assert lexical["vectorizer_fit_scope"] == "training_tasks_per_fold", path
        assert report["lexical_nuisance_fit"]["fit_is_inductive"], path
        assert report["lexical_nuisance_fit"]["unique_task_folds"] == 229, path
print("PASS: ten lexical-adjusted reports")
PY
