#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  printf 'Usage: %s RUN_DIR\n' "$0" >&2
  exit 2
fi

workspace=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python="$workspace/.venv/bin/python"
run_dir=$(realpath -m "$1")
report="$run_dir/report.json"
parser_gate="$run_dir/unmonitored_parser_gate.json"

while [[ ! -f "$report" ]]; do
  if [[ -f "$parser_gate" ]]; then
    "$python" - "$parser_gate" <<'PY'
import json
import sys
report = json.load(open(sys.argv[1], encoding="utf-8"))
if report.get("status") != "PASS":
    raise SystemExit("Parser gate failed; trajectory analysis will not run")
PY
  fi
  if ! pgrep -f "$workspace/scripts/run_full_response_trajectory_experiment.sh" >/dev/null; then
    printf 'Extraction stopped without a passing report: %s\n' "$run_dir" >&2
    exit 3
  fi
  sleep 30
done

while pgrep -x nvidia-cuda-mps >/dev/null; do
  sleep 30
done

exec "$workspace/scripts/run_response_trajectory_analysis.sh" "$run_dir"
