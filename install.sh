#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRANSFORMERS_REPO="${TRANSFORMERS_REPO:-transformers}"
TORCH_BACKEND="${TORCH_BACKEND:-auto}"

if [[ "$TRANSFORMERS_REPO" != "transformers" && "$TRANSFORMERS_REPO" != "transformers_qwenvl" ]]; then
    echo "TRANSFORMERS_REPO must be transformers or transformers_qwenvl." >&2
    exit 2
fi
command -v uv >/dev/null 2>&1 || {
    echo "uv is required; run scripts/setup_machine.sh first." >&2
    exit 2
}
[[ -d "$WORKSPACE_ROOT/$TRANSFORMERS_REPO" ]] || {
    echo "Missing $TRANSFORMERS_REPO; run python3 scripts/bootstrap_workspace.py." >&2
    exit 2
}
[[ -d "$WORKSPACE_ROOT/lmms-eval" ]] || {
    echo "Missing bundled lmms-eval source; restore the complete repository checkout." >&2
    exit 2
}

if [[ ! -x "$WORKSPACE_ROOT/.venv/bin/python" ]]; then
    uv venv --python 3.12 --seed "$WORKSPACE_ROOT/.venv"
fi

# shellcheck disable=SC1091
source "$WORKSPACE_ROOT/.venv/bin/activate"

uv pip install torch torchvision --torch-backend "$TORCH_BACKEND"
uv pip install -e "$WORKSPACE_ROOT/$TRANSFORMERS_REPO"
# The base package contains the active representation study dependencies.
# lmms-eval[all] currently pulls legacy optional packages that do not resolve
# on the supported Python 3.12 environment (notably llvmlite==0.36.0).
uv pip install -e "$WORKSPACE_ROOT/lmms-eval"
# Dependencies exercised by the declared cross-benchmark panel. Keep this
# explicit because the broad lmms-eval[all] extra does not resolve on Python 3.12.
uv pip install jieba distance apted editdistance Levenshtein

python -c 'import torch; print(f"torch={torch.__version__} cuda_runtime={torch.version.cuda} cuda_available={torch.cuda.is_available()} devices={torch.cuda.device_count()}")'
python -m lmms_eval --tasks list >"$WORKSPACE_ROOT/tasks.txt"

printf 'Ready. Run: source %q\n' "$WORKSPACE_ROOT/.venv/bin/activate"
