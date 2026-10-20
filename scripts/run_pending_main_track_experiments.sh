#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
PERSISTENT_ROOT="/home/ubuntu/data-imu-us-east3/vlmbehave-artifacts"
CANONICAL_PERSISTENT_ROOT="$(realpath -m "$PERSISTENT_ROOT")"
SELECTED_ROOT="$PERSISTENT_ROOT/2026-08-15/selected-site-probe-main-track-v1"
TRAJECTORY_ROOT="$PERSISTENT_ROOT/2026-08-15/response-trajectory-smoke-v1"
APERTUS_ROOT="$PERSISTENT_ROOT/2026-08-13/apertus1-longitudinal-v1"
UPSTREAM_UNIT="vlm-selected-site-probe-panel.service"

for path in "$SELECTED_ROOT" "$TRAJECTORY_ROOT" "$APERTUS_ROOT"; do
  case "$(realpath -m "$path")" in
    "$CANONICAL_PERSISTENT_ROOT"/*) ;;
    *)
      printf 'Experiment output is outside persistent storage: %s\n' "$path" >&2
      exit 2
      ;;
  esac
done

if [[ ! -x "$PYTHON_BIN" ]]; then
  printf 'Missing environment Python: %s\n' "$PYTHON_BIN" >&2
  exit 3
fi

printf 'Waiting for %s to finish.\n' "$UPSTREAM_UNIT"
while systemctl --user is-active --quiet "$UPSTREAM_UNIT"; do
  sleep 20
done

"$PYTHON_BIN" - "$SELECTED_ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
expected = {
    "olmo3-7b-instruct",
    "apertus15-8b",
    "qwen3_5-9b",
    "qwen3_5-27b",
    "gemma4-31b",
}
reports = {path.parent.name: path for path in root.glob("*/report.json")}
if set(reports) != expected:
    raise SystemExit(
        f"Selected-site panel incomplete: expected {sorted(expected)}, found {sorted(reports)}"
    )
for model, path in sorted(reports.items()):
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("status") != "PASS" or report.get("assignments") != 12870:
        raise SystemExit(f"Selected-site report failed its gate: {model}: {report.get('status')}")
print("Selected-site panel passed: five models × 12,870 assignments")
PY

"$ROOT_DIR/scripts/run_response_trajectory_smoke.sh" "$TRAJECTORY_ROOT"
APERTUS_ANALYSIS_JOBS="${APERTUS_ANALYSIS_JOBS:-4}" \
  "$ROOT_DIR/scripts/run_full_apertus1_longitudinal.sh" "$APERTUS_ROOT"
