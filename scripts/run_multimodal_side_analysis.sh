#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/home/ubuntu/data-imu-us-east3/vlmbehave-artifacts}"
SOURCE_ROOT="${SOURCE_ROOT:-$ARTIFACT_ROOT/2026-08-16/representation-dynamics-multimodal-side-v1/input/logs/representation-dynamics-v1-six-model/cross_benchmark_discovery}"
SOURCE_ARCHIVE="${SOURCE_ARCHIVE:-$ARTIFACT_ROOT/2026-07-27/representation-dynamics-v1-six-model.tar.gz}"
OUTPUT_ROOT="${1:-$ARTIFACT_ROOT/2026-08-16/representation-dynamics-multimodal-side-v1/analysis}"
MANIFEST="$ROOT_DIR/experiments/representation_dynamics_v1.json"
EXPECTED_LMMS_HEAD="0b69f69e455a00b748d799a517939a033338bdd8"
EXPECTED_SOURCE_SHA256="589401e880348a103d0cc985ca57b43f34b720fe4818df3693717fb6384d40e4"
KNOWLEDGE_TASKS="ok_vqa_val2014_lite,repdyn_medqa,repdyn_mmmu_val,ai2d_lite,repdyn_gpqa_main"
REASONING_TASKS="pope_random,mme,textvqa_val_lite,ocrbench_v2,docvqa_val_lite"
SURFACE_COLUMNS="prompt_token_count,prompt_char_count,prompt_line_count,option_count,whitespace_token_count,whitespace_token_entropy,has_visual_input,format_binary,format_short_answer,format_mixed"
MPS_ROOT=""
TELEMETRY_PID=""
CHILD_PIDS=()
TELEMETRY_PATH="$OUTPUT_ROOT/gpu_telemetry.csv"

start_telemetry() {
  if [[ -z "$TELEMETRY_PID" ]]; then
    nvidia-smi \
      --query-gpu=timestamp,utilization.gpu,memory.used,power.draw,power.limit \
      --format=csv,noheader,nounits --loop=10 >>"$TELEMETRY_PATH" &
    TELEMETRY_PID="$!"
  fi
}

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
[[ -d "$SOURCE_ROOT" ]] || { printf 'Missing restored source: %s\n' "$SOURCE_ROOT" >&2; exit 3; }
[[ -f "$SOURCE_ARCHIVE" ]] || { printf 'Missing source archive: %s\n' "$SOURCE_ARCHIVE" >&2; exit 3; }
[[ "$(sha256sum "$SOURCE_ARCHIVE" | cut -d' ' -f1)" == "$EXPECTED_SOURCE_SHA256" ]] || {
  printf 'Source archive SHA-256 does not match the frozen manifest.\n' >&2
  exit 3
}
[[ -z "$(git -C "$ROOT_DIR/lmms-eval" status --porcelain --untracked-files=all)" ]] || {
  printf 'lmms-eval must be clean so reports bind to one implementation.\n' >&2
  exit 4
}
[[ "$(git -C "$ROOT_DIR/lmms-eval" rev-parse HEAD)" == "$EXPECTED_LMMS_HEAD" ]] || {
  printf 'Unexpected lmms-eval revision; expected %s.\n' "$EXPECTED_LMMS_HEAD" >&2
  exit 5
}

models=(qwen3_5-9b qwen3_5-27b gemma4-31b)
mkdir -p "$OUTPUT_ROOT/logs" "$OUTPUT_ROOT/preflight"
run_roots=()
source_artifacts=()
for model in "${models[@]}"; do
  condition="$SOURCE_ROOT/$model"
  run_roots+=("$condition")
  mapfile -t artifacts < <(find "$condition/tasks" -type f -name representations.pt | sort)
  [[ "${#artifacts[@]}" -eq 10 ]] || {
    printf 'Expected 10 artifacts for %s, found %s.\n' "$model" "${#artifacts[@]}" >&2
    exit 6
  }
  source_artifacts+=("${artifacts[@]}")
  "$PYTHON_BIN" "$ROOT_DIR/scripts/audit_text_protocol_samples.py" "$condition" \
    --output "$OUTPUT_ROOT/preflight/${model}_prompt_audit.json" >/dev/null
done

SOURCE_AUDIT="$OUTPUT_ROOT/preflight/source_artifact_validation.json"
if [[ ! -f "$SOURCE_AUDIT" ]]; then
  "$PYTHON_BIN" "$ROOT_DIR/scripts/validate_representation_artifact.py" \
    "${source_artifacts[@]}" --require-exact >"$SOURCE_AUDIT"
fi

NUISANCE_TABLE="$OUTPUT_ROOT/multimodal_nuisance_features.csv"
if [[ ! -f "$NUISANCE_TABLE" ]]; then
  "$PYTHON_BIN" "$ROOT_DIR/scripts/build_multimodal_nuisance_table.py" \
    "${run_roots[@]}" --manifest "$MANIFEST" --panel cross_benchmark_discovery \
    --output "$NUISANCE_TABLE"
fi

for baseline in word_tfidf char_tfidf; do
  output="$OUTPUT_ROOT/prompt_${baseline}.json"
  if [[ ! -f "$output" ]]; then
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
      "$PYTHON_BIN" -m lmms_eval.representations.prompt_baseline \
        --sample-root "$SOURCE_ROOT/qwen3_5-9b" --manifest "$MANIFEST" \
        --panel cross_benchmark_discovery --baseline "$baseline" --output "$output" \
        --cv-folds 5 --seed 0 --jobs 24 \
        >"$OUTPUT_ROOT/logs/prompt_${baseline}.log" 2>&1 &
    CHILD_PIDS+=("$!")
  fi
done

if pgrep -x nvidia-cuda-mps >/dev/null; then
  [[ -n "${CUDA_MPS_PIPE_DIRECTORY:-}" && -d "$CUDA_MPS_PIPE_DIRECTORY" ]] || {
    printf 'An MPS service is running, but its pipe directory is unavailable.\n' >&2
    exit 7
  }
else
  MPS_ROOT="$(mktemp -d /tmp/vlmbehave-multimodal-side.XXXXXX)"
  export CUDA_MPS_PIPE_DIRECTORY="$MPS_ROOT/pipe"
  export CUDA_MPS_LOG_DIRECTORY="$MPS_ROOT/log"
  mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
  nvidia-cuda-mps-control -d
fi
for model in "${models[@]}"; do
  condition="$SOURCE_ROOT/$model"
  output="$OUTPUT_ROOT/fisher_nulls/$model"
  if [[ -f "$output/report.json" && -f "$output/permutation_nulls.json.gz" ]]; then
    continue
  fi
  mapfile -t artifacts < <(find "$condition/tasks" -type f -name representations.pt | sort)
  mkdir -p "$output"
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    "$PYTHON_BIN" -m lmms_eval.representations.analysis "${artifacts[@]}" \
      --knowledge-tasks "$KNOWLEDGE_TASKS" --reasoning-tasks "$REASONING_TASKS" \
      --output-dir "$output" --phase prefill --cv-folds 5 --permutations 0 \
      --fdr-permutations 251 --nonlinear-permutations 0 --pca-variance 0.95 \
      --silhouette-max-samples 5000 --seed 0 --analysis-backend cuml --jobs 1 \
      >"$OUTPUT_ROOT/logs/fisher_${model}.log" 2>&1 &
  CHILD_PIDS+=("$!")
done

if (( ${#CHILD_PIDS[@]} > 0 )); then
  start_telemetry
fi

status=0
for pid in "${CHILD_PIDS[@]}"; do
  if ! wait "$pid"; then status=1; fi
done
CHILD_PIDS=()
[[ "$status" -eq 0 ]] || exit "$status"

controls=(raw surface_modality lexical_64 combined_64)
for model in "${models[@]}"; do
  condition="$SOURCE_ROOT/$model"
  mapfile -t artifacts < <(find "$condition/tasks" -type f -name representations.pt | sort)
  nulls="$OUTPUT_ROOT/fisher_nulls/$model/permutation_nulls.json.gz"
  [[ -f "$nulls" ]] || { printf 'Missing Fisher nulls for %s.\n' "$model" >&2; exit 8; }
  for control in "${controls[@]}"; do
    output="$OUTPUT_ROOT/probes/$control/$model"
    if [[ -f "$output/report.json" ]] && "$PYTHON_BIN" - "$output/report.json" <<'PY'
import json, sys
raise SystemExit(0 if json.load(open(sys.argv[1], encoding="utf-8")).get("status") == "PASS" else 1)
PY
    then
      continue
    fi
    extra=()
    if [[ "$control" == surface_modality || "$control" == combined_64 ]]; then
      extra+=(--nuisance-table "$NUISANCE_TABLE" --nuisance-columns "$SURFACE_COLUMNS")
    fi
    if [[ "$control" == lexical_64 || "$control" == combined_64 ]]; then
      extra+=(--lexical-nuisance-sample-root "$condition" --lexical-nuisance-kind char_tfidf --lexical-nuisance-components 64)
    fi
    mkdir -p "$output"
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
      "$PYTHON_BIN" -m lmms_eval.representations.selected_site_probe \
        "${artifacts[@]}" --permutation-nulls "$nulls" --output-dir "$output" \
        --cv-folds 5 --pca-variance 0.95 --seed 0 --analysis-backend cuml \
        --checkpoint-every 10 "${extra[@]}" \
        >"$OUTPUT_ROOT/logs/probe_${control}_${model}.log" 2>&1 &
    CHILD_PIDS+=("$!")
  done
done

if (( ${#CHILD_PIDS[@]} > 0 )); then
  start_telemetry
fi

status=0
for pid in "${CHILD_PIDS[@]}"; do
  if ! wait "$pid"; then status=1; fi
done
CHILD_PIDS=()
[[ "$status" -eq 0 ]] || exit "$status"

"$PYTHON_BIN" - "$OUTPUT_ROOT" "$EXPECTED_LMMS_HEAD" <<'PY'
import gzip, hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
expected = sys.argv[2]
models = ("qwen3_5-9b", "qwen3_5-27b", "gemma4-31b")
for model in models:
    fisher_path = root / "fisher_nulls" / model / "report.json"
    fisher = json.loads(fisher_path.read_text(encoding="utf-8"))
    assert not fisher["failures"], fisher_path
    assert fisher["fdr_group_permutations_requested"] == 251, fisher_path
    assert fisher["selection_aware_randomization_by_model"], fisher_path
    null_path = fisher_path.with_name("permutation_nulls.json.gz")
    assert hashlib.sha256(null_path.read_bytes()).hexdigest() == fisher["permutation_null_artifact"]["sha256"]
    with gzip.open(null_path, "rt", encoding="utf-8") as handle:
        nulls = json.load(handle)
    assert len(nulls["rows"]) == fisher["num_rows"], null_path
    for control in ("raw", "surface_modality", "lexical_64", "combined_64"):
        path = root / "probes" / control / model / "report.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        implementation = report["configuration"]["implementation"]
        assert report["status"] == "PASS", path
        assert report["assignments"] == 252 and report["probe_refits"] == 126, path
        assert report["selection_recomputed_for_every_assignment"], path
        assert report["macro_f1"]["complement_symmetry"]["verified"], path
        assert implementation["git_head"] == expected and not implementation["git_dirty"], path
print("PASS: three Fisher nulls and twelve selected-site probe reports")
PY
