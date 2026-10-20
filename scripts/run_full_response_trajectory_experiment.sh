#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  printf 'Usage: %s RUN_DIR\n' "$0" >&2
  exit 2
fi

workspace=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python="$workspace/.venv/bin/python"
persistent_root=$(realpath -m /home/ubuntu/data-imu-us-east3/vlmbehave-artifacts)
run_dir=$(realpath -m "$1")
mkdir -p "$run_dir"
case "$run_dir/" in
  "$persistent_root"/*) ;;
  *) printf 'RUN_DIR must remain under %s\n' "$persistent_root" >&2; exit 3 ;;
esac

knowledge_tasks=(
  repdyn_response_mmlu_anatomy
  repdyn_response_mmlu_astronomy
  repdyn_response_mmlu_clinical_knowledge
  repdyn_response_mmlu_high_school_world_history
  repdyn_response_mmlu_professional_law
  repdyn_response_mmlu_high_school_biology
  repdyn_response_mmlu_medical_genetics
  repdyn_response_mmlu_prehistory
)
reasoning_tasks=(
  repdyn_response_mmlu_abstract_algebra
  repdyn_response_mmlu_college_mathematics
  repdyn_response_mmlu_formal_logic
  repdyn_response_mmlu_high_school_mathematics
  repdyn_response_mmlu_logical_fallacies
  repdyn_response_mmlu_college_physics
  repdyn_response_mmlu_conceptual_physics
  repdyn_response_mmlu_high_school_statistics
)
tasks=("${knowledge_tasks[@]}" "${reasoning_tasks[@]}")
tasks_csv=$(IFS=,; printf '%s' "${tasks[*]}")
model_args=pretrained=allenai/Olmo-3-7B-Instruct,revision=6e5971d9eba42665f5bd5a0fcf047f299ce1dccc,apply_chat_template=true,use_cache=true

if [[ -f "$run_dir/report.json" ]]; then
  "$python" - "$run_dir/report.json" <<'PY'
import json
import sys
report = json.load(open(sys.argv[1], encoding="utf-8"))
if report.get("status") != "PASS":
    raise SystemExit("Existing full-trajectory report is not PASS")
PY
  printf 'Existing validated full trajectory: %s\n' "$run_dir/report.json"
  exit 0
fi
if [[ -n "$(git -C "$workspace/lmms-eval" status --porcelain --untracked-files=all)" ]]; then
  printf 'lmms-eval must be clean before full trajectory extraction.\n' >&2
  exit 4
fi

common=(
  -m lmms_eval eval
  --model hf_causal
  --force_simple
  --model_args "$model_args"
  --tasks "$tasks_csv"
  --log_samples
  --batch_size 16
  --limit 16
  --seed 0
  --reasoning_tags none
)

cd "$workspace"
"$python" "${common[@]}" --output_path "$run_dir/unmonitored" \
  >"$run_dir/unmonitored.log" 2>&1

gate_args=()
for task in "${tasks[@]}"; do gate_args+=(--task "$task"); done
"$python" scripts/validate_response_trajectory_samples.py \
  --samples "$run_dir/unmonitored" \
  --output "$run_dir/unmonitored_parser_gate.json" \
  "${gate_args[@]}" \
  --expected-examples 16

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
  exit 5
fi
artifact=${artifacts[0]}

validation_args=()
for task in "${tasks[@]}"; do validation_args+=(--expected-task "$task=16"); done
"$python" scripts/validate_representation_artifact.py "$artifact" \
  "${validation_args[@]}" --require-exact --require-matching-sample-ids \
  >"$run_dir/validation.json"

"$python" scripts/summarize_response_trajectory_smoke.py \
  --unmonitored "$run_dir/unmonitored" \
  --monitored "$run_dir/monitored" \
  --artifact "$artifact" \
  --output "$run_dir/report.json" \
  "${gate_args[@]}" \
  --expected-examples 16 \
  --coverage-threshold 0.95 \
  --minimum-prefix 8
