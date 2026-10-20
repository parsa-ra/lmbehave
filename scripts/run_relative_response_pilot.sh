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
if [[ -n "$(git -C "$workspace/lmms-eval" status --porcelain --untracked-files=all)" ]]; then
  printf 'lmms-eval must be clean before pilot extraction.\n' >&2
  exit 4
fi

knowledge_tasks=(
  repdyn_response_mmlu_anatomy
  repdyn_response_mmlu_astronomy
  repdyn_response_mmlu_clinical_knowledge
  repdyn_response_mmlu_high_school_world_history
)
reasoning_tasks=(
  repdyn_response_mmlu_abstract_algebra
  repdyn_response_mmlu_college_mathematics
  repdyn_response_mmlu_formal_logic
  repdyn_response_mmlu_high_school_mathematics
)
tasks=("${knowledge_tasks[@]}" "${reasoning_tasks[@]}")
tasks_csv=$(IFS=,; printf '%s' "${tasks[*]}")
knowledge_csv=$(IFS=,; printf '%s' "${knowledge_tasks[*]}")
reasoning_csv=$(IFS=,; printf '%s' "${reasoning_tasks[*]}")
model_args=pretrained=allenai/Olmo-3-7B-Instruct,revision=6e5971d9eba42665f5bd5a0fcf047f299ce1dccc,apply_chat_template=true,use_cache=true,attn_implementation=eager

common=(
  -m lmms_eval eval
  --model hf_causal
  --force_simple
  --model_args "$model_args"
  --tasks "$tasks_csv"
  --log_samples
  --batch_size 1
  --limit 4
  --seed 0
  --reasoning_tags none
)

cd "$workspace"
if [[ ! -f "$run_dir/unmonitored_parser_gate.json" ]]; then
  "$python" "${common[@]}" --output_path "$run_dir/unmonitored" \
    >"$run_dir/unmonitored.log" 2>&1
  gate_args=()
  for task in "${tasks[@]}"; do gate_args+=(--task "$task"); done
  "$python" scripts/validate_response_trajectory_samples.py \
    --samples "$run_dir/unmonitored" \
    --output "$run_dir/unmonitored_parser_gate.json" \
    "${gate_args[@]}" --expected-examples 4
fi

if [[ ! -d "$run_dir/representations" ]]; then
  "$python" "${common[@]}" --output_path "$run_dir/monitored" \
    --representation_dynamics \
    --representation_dynamics_output "$run_dir/representations" \
    --representation_dynamics_pooling last_valid \
    --representation_dynamics_phases both \
    --representation_dynamics_decode_aggregation all_updates \
    --representation_dynamics_storage_dtype float32 \
    >"$run_dir/monitored.log" 2>&1
fi

mapfile -t artifacts < <(find "$run_dir/representations" -name representations.pt -type f | sort)
if [[ ${#artifacts[@]} -ne 1 ]]; then
  printf 'Expected one raw representation artifact, found %s.\n' "${#artifacts[@]}" >&2
  exit 5
fi
artifact=${artifacts[0]}
validation_args=()
gate_args=()
for task in "${tasks[@]}"; do
  validation_args+=(--expected-task "$task=4")
  gate_args+=(--task "$task")
done
"$python" scripts/validate_representation_artifact.py "$artifact" \
  "${validation_args[@]}" --require-exact --require-matching-sample-ids \
  >"$run_dir/validation.json"
"$python" scripts/summarize_response_trajectory_smoke.py \
  --unmonitored "$run_dir/unmonitored" \
  --monitored "$run_dir/monitored" \
  --artifact "$artifact" \
  --output "$run_dir/parity_report.json" \
  "${gate_args[@]}" --expected-examples 4 --coverage-threshold 0.95 --minimum-prefix 8

binned="$run_dir/relative_bins/representations.pt"
"$python" scripts/bin_response_trajectory_artifact.py "$artifact" \
  --output "$binned" --report "$run_dir/relative_bins/report.json" --bins 8

analysis_dir="$run_dir/analysis-cuml"
if [[ "${RUN_RELATIVE_PILOT_ANALYSIS:-1}" == 1 && ! -f "$analysis_dir/report.json" ]]; then
  "$python" -m lmms_eval.representations.analysis "$binned" \
    --knowledge-tasks "$knowledge_csv" \
    --reasoning-tasks "$reasoning_csv" \
    --output-dir "$analysis_dir" \
    --phase both \
    --cv-folds 4 \
    --permutations 0 \
    --fdr-permutations 69 \
    --nonlinear-permutations 0 \
    --pca-variance 0.95 \
    --silhouette-max-samples 2000 \
    --seed 0 \
    --analysis-backend cuml \
    --jobs "${RELATIVE_PILOT_ANALYSIS_JOBS:-8}" \
    --parallel-backend process
  "$python" scripts/analyze_response_trajectory_maximum.py \
    --permutation-nulls "$analysis_dir/permutation_nulls.json.gz" \
    --coverage-report "$run_dir/relative_bins/report.json" \
    --output-dir "$run_dir/trajectory-inference"
fi

printf 'Relative-response pilot extraction complete: %s\n' "$run_dir"
