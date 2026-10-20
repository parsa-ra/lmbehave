#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  printf 'Usage: %s RUN_DIR\n' "$0" >&2
  exit 2
fi

workspace=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python="$workspace/.venv/bin/python"
run_dir=$(mkdir -p "$1" && cd "$1" && pwd)
tasks=repdyn_response_mmlu_anatomy,repdyn_response_mmlu_abstract_algebra
model=${TRAJECTORY_SMOKE_MODEL:-hf_causal}
model_args=${TRAJECTORY_SMOKE_MODEL_ARGS:-pretrained=allenai/Olmo-3-7B-Instruct,revision=6e5971d9eba42665f5bd5a0fcf047f299ce1dccc,apply_chat_template=true,use_cache=true}
batch_size=${TRAJECTORY_SMOKE_BATCH_SIZE:-4}
[[ "$batch_size" =~ ^[1-9][0-9]*$ ]] || { printf 'TRAJECTORY_SMOKE_BATCH_SIZE must be positive\n' >&2; exit 5; }

if [[ -f "$run_dir/report.json" ]]; then
  "$python" - "$run_dir/report.json" <<'PY'
import json
import sys
report = json.load(open(sys.argv[1], encoding="utf-8"))
if report.get("status") != "PASS":
    raise SystemExit("Existing smoke report is not PASS")
PY
  printf 'Existing validated smoke: %s\n' "$run_dir/report.json"
  exit 0
fi

if [[ -n "$(git -C "$workspace/lmms-eval" status --porcelain --untracked-files=all)" ]]; then
  printf 'lmms-eval must be clean before the smoke.\n' >&2
  exit 3
fi

common=(
  -m lmms_eval eval
  --model "$model"
  --force_simple
  --model_args "$model_args"
  --tasks "$tasks"
  --log_samples
  --batch_size "$batch_size"
  --limit 4
  --seed 0
  --reasoning_tags none
)

cd "$workspace"
"$python" "${common[@]}" --output_path "$run_dir/unmonitored" \
  >"$run_dir/unmonitored.log" 2>&1

"$python" scripts/validate_response_trajectory_samples.py \
  --samples "$run_dir/unmonitored" \
  --output "$run_dir/unmonitored_parser_gate.json" \
  --task repdyn_response_mmlu_anatomy \
  --task repdyn_response_mmlu_abstract_algebra \
  --expected-examples 4

"$python" "${common[@]}" --output_path "$run_dir/monitored" \
  --representation_dynamics \
  --representation_dynamics_output "$run_dir/representations" \
  --representation_dynamics_pooling last_valid \
  --representation_dynamics_phases both \
  --representation_dynamics_decode_aggregation all_updates \
  --representation_dynamics_decode_max_updates 32 \
  --representation_dynamics_storage_dtype float32 \
  >"$run_dir/monitored.log" 2>&1

mapfile -t artifacts < <(find "$run_dir/representations" -name representations.pt -type f | sort)
if [[ ${#artifacts[@]} -ne 1 ]]; then
  printf 'Expected one monitored artifact, found %s.\n' "${#artifacts[@]}" >&2
  exit 4
fi
artifact=${artifacts[0]}

"$python" scripts/validate_representation_artifact.py "$artifact" \
  --expected-task repdyn_response_mmlu_anatomy=4 \
  --expected-task repdyn_response_mmlu_abstract_algebra=4 \
  --require-exact >"$run_dir/validation.json"

"$python" scripts/summarize_response_trajectory_smoke.py \
  --unmonitored "$run_dir/unmonitored" \
  --monitored "$run_dir/monitored" \
  --artifact "$artifact" \
  --output "$run_dir/report.json" \
  --task repdyn_response_mmlu_anatomy \
  --task repdyn_response_mmlu_abstract_algebra \
  --expected-examples 4 \
  --coverage-threshold 0.95 \
  --minimum-prefix 8
