#!/usr/bin/env bash
set -euo pipefail

VLM_NUISANCE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VLM_NUISANCE_RUN_ROOT="${1:-$VLM_NUISANCE_ROOT/logs/restored-current-study-full}"
VLM_NUISANCE_TABLE="${2:-$VLM_NUISANCE_ROOT/logs/restored-current-study-metadata/nuisance_features.csv}"
VLM_NUISANCE_OUTPUT="${3:-$VLM_NUISANCE_ROOT/logs/nuisance-control-v1}"
VLM_NUISANCE_PYTHON="$VLM_NUISANCE_ROOT/.venv-analysis/bin/python"
VLM_NUISANCE_MPS_ROOT=""
VLM_NUISANCE_TELEMETRY_PID=""
VLM_NUISANCE_CHILD_PIDS=()

vlm_nuisance_cleanup() {
    local pid
    for pid in "${VLM_NUISANCE_CHILD_PIDS[@]:-}"; do
        kill "$pid" >/dev/null 2>&1 || true
    done
    if [[ -n "$VLM_NUISANCE_TELEMETRY_PID" ]]; then
        kill "$VLM_NUISANCE_TELEMETRY_PID" >/dev/null 2>&1 || true
    fi
    if [[ -n "$VLM_NUISANCE_MPS_ROOT" ]]; then
        printf 'quit\n' | CUDA_MPS_PIPE_DIRECTORY="$CUDA_MPS_PIPE_DIRECTORY" \
            nvidia-cuda-mps-control >/dev/null 2>&1 || true
        rm -r "$VLM_NUISANCE_MPS_ROOT"
    fi
}
trap vlm_nuisance_cleanup EXIT INT TERM

[[ -x "$VLM_NUISANCE_PYTHON" ]] || {
    echo "Missing analysis environment: $VLM_NUISANCE_PYTHON" >&2
    exit 1
}
[[ -d "$VLM_NUISANCE_RUN_ROOT" ]] || {
    echo "Missing extracted run: $VLM_NUISANCE_RUN_ROOT" >&2
    exit 1
}
[[ -f "$VLM_NUISANCE_TABLE" ]] || {
    echo "Missing nuisance table: $VLM_NUISANCE_TABLE" >&2
    exit 1
}
command -v nvidia-cuda-mps-control >/dev/null || {
    echo "nvidia-cuda-mps-control is required" >&2
    exit 1
}
if pgrep -x nvidia-cuda-mps >/dev/null; then
    echo "An NVIDIA MPS service is already running; refusing to take ownership." >&2
    exit 1
fi

mkdir -p "$VLM_NUISANCE_OUTPUT/logs"
VLM_NUISANCE_MPS_ROOT="$(mktemp -d /tmp/vlmbehave-nuisance-mps.XXXXXX)"
export CUDA_MPS_PIPE_DIRECTORY="$VLM_NUISANCE_MPS_ROOT/pipe"
export CUDA_MPS_LOG_DIRECTORY="$VLM_NUISANCE_MPS_ROOT/log"
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
nvidia-cuda-mps-control -d

VLM_NUISANCE_TELEMETRY="$VLM_NUISANCE_OUTPUT/gpu_telemetry.csv"
printf 'timestamp_utc,utilization_gpu_percent,memory_used_mib,power_draw_w\n' > "$VLM_NUISANCE_TELEMETRY"
(
    while true; do
        printf '%s,' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        nvidia-smi \
            --query-gpu=utilization.gpu,memory.used,power.draw \
            --format=csv,noheader,nounits
        sleep 5
    done
) >> "$VLM_NUISANCE_TELEMETRY" 2>/dev/null &
VLM_NUISANCE_TELEMETRY_PID=$!

VLM_NUISANCE_MODELS=(
    olmo3-7b-instruct
    apertus-8b-instruct
    qwen3_5-9b
    qwen3_5-27b
    gemma4-31b
)
for VLM_NUISANCE_MODEL in "${VLM_NUISANCE_MODELS[@]}"; do
    "$VLM_NUISANCE_PYTHON" "$VLM_NUISANCE_ROOT/scripts/run_nuisance_control_sensitivity.py" \
        "$VLM_NUISANCE_RUN_ROOT" \
        --nuisance-table "$VLM_NUISANCE_TABLE" \
        --output-root "$VLM_NUISANCE_OUTPUT" \
        --models "$VLM_NUISANCE_MODEL" \
        --backend cuml \
        --jobs 2 \
        --resume \
        > "$VLM_NUISANCE_OUTPUT/logs/$VLM_NUISANCE_MODEL.log" 2>&1 &
    VLM_NUISANCE_CHILD_PIDS+=("$!")
done

VLM_NUISANCE_STATUS=0
for VLM_NUISANCE_PID in "${VLM_NUISANCE_CHILD_PIDS[@]}"; do
    if ! wait "$VLM_NUISANCE_PID"; then
        VLM_NUISANCE_STATUS=1
    fi
done
VLM_NUISANCE_CHILD_PIDS=()
if ((VLM_NUISANCE_STATUS)); then
    echo "At least one nuisance-control condition failed; inspect $VLM_NUISANCE_OUTPUT/logs." >&2
    exit "$VLM_NUISANCE_STATUS"
fi

echo "Nuisance-control analyses completed under $VLM_NUISANCE_OUTPUT"
