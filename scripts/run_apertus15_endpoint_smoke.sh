#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${APERTUS15_PYTHON:-$ROOT_DIR/.venv-apertus15/bin/python}"
RUN_DIR="${1:-$ROOT_DIR/logs/apertus15-endpoint-smoke}"
MODEL_ID="swiss-ai/Apertus-v1.5-8B"
MODEL_REVISION="a411d838600baf0e3635a3daf66fb7c55fc97bb6"
TASKS="repdyn_mmlu_anatomy,repdyn_mmlu_abstract_algebra"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing Apertus 1.5 Python environment: $PYTHON_BIN" >&2
  exit 2
fi
if [[ -e "$RUN_DIR" ]]; then
  echo "Refusing to mix or overwrite an existing smoke directory: $RUN_DIR" >&2
  exit 2
fi
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is required for the gated Apertus 1.5 weights." >&2
  exit 2
fi

mkdir -p "$RUN_DIR"
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false

COMMON_ARGS=(
  -m lmms_eval eval
  --model apertus1p5
  --force_simple
  --model_args "pretrained=$MODEL_ID,revision=$MODEL_REVISION,enable_thinking=false,use_cache=true"
  --tasks "$TASKS"
  --batch_size 2
  --limit 2
  --seed 0
  --log_samples
)

"$PYTHON_BIN" "${COMMON_ARGS[@]}" \
  --output_path "$RUN_DIR/observer-on/evaluation" \
  --representation_dynamics \
  --representation_dynamics_output "$RUN_DIR/observer-on/representations" \
  --representation_dynamics_pooling last_valid \
  --representation_dynamics_phases prefill \
  --representation_dynamics_storage_dtype float32

"$PYTHON_BIN" "${COMMON_ARGS[@]}" \
  --output_path "$RUN_DIR/observer-off/evaluation"

ARTIFACT="$(find "$RUN_DIR/observer-on/representations" -type f -name representations.pt -print -quit)"
if [[ -z "$ARTIFACT" ]]; then
  echo "The monitored run did not produce representations.pt" >&2
  exit 1
fi
"$PYTHON_BIN" "$ROOT_DIR/scripts/validate_representation_artifact.py" "$ARTIFACT" \
  --expected-task repdyn_mmlu_anatomy=2 \
  --expected-task repdyn_mmlu_abstract_algebra=2 \
  --require-exact >"$RUN_DIR/artifact_validation.json"

"$PYTHON_BIN" - "$RUN_DIR" <<'PY'
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])


def read_samples(root: Path):
    result = {}
    for path in sorted(root.glob("**/*_samples_*.jsonl")):
        task = path.stem.split("_samples_", 1)[1]
        result[task] = [json.loads(line) for line in path.read_text().splitlines()]
    return result


def read_aggregate(root: Path):
    paths = sorted(root.glob("**/*_results.json"))
    if len(paths) != 1:
        raise RuntimeError(f"Expected one aggregate result under {root}, found {len(paths)}")
    return json.loads(paths[0].read_text()).get("results", {})


observer_on = read_samples(run_dir / "observer-on" / "evaluation")
observer_off = read_samples(run_dir / "observer-off" / "evaluation")
aggregate_equal = read_aggregate(
    run_dir / "observer-on" / "evaluation"
) == read_aggregate(run_dir / "observer-off" / "evaluation")
fields = ("doc_id", "target", "filtered_resps", "exact_match")
comparisons = {}
for task in sorted(set(observer_on) | set(observer_off)):
    left = observer_on.get(task, [])
    right = observer_off.get(task, [])
    comparisons[task] = {
        "row_count_on": len(left),
        "row_count_off": len(right),
        **{
            f"{field}_equal": [row.get(field) for row in left]
            == [row.get(field) for row in right]
            for field in fields
        },
    }
all_equal = aggregate_equal and set(observer_on) == set(observer_off) and all(
    all(value for key, value in comparison.items() if key.endswith("_equal"))
    and comparison["row_count_on"] == comparison["row_count_off"]
    for comparison in comparisons.values()
)
report = {
    "schema_version": 1,
    "observer_output_parity": all_equal,
    "task_aggregates_equal": aggregate_equal,
    "comparisons": comparisons,
}
(run_dir / "observer_parity.json").write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n"
)
print(json.dumps(report, indent=2, sort_keys=True))
if not all_equal:
    raise SystemExit("Observer-off/on output parity failed")
PY

echo "Apertus 1.5 endpoint smoke passed: $RUN_DIR"
