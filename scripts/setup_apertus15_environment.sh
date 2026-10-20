#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${APERTUS15_ENV:-$ROOT_DIR/.venv-apertus15}"
TRANSFORMERS_REVISION="3797303dda74844e3d1f8977ff5518bb91f818b4"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required; run scripts/setup_machine.sh first." >&2
  exit 2
fi

if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  uv venv "$ENV_DIR" --python 3.12
fi
uv pip install --python "$ENV_DIR/bin/python" torch torchvision --torch-backend auto
uv pip install --python "$ENV_DIR/bin/python" \
  "transformers @ git+https://github.com/swiss-ai/transformers.git@$TRANSFORMERS_REVISION"
uv pip install --python "$ENV_DIR/bin/python" -e "$ROOT_DIR/lmms-eval"

"$ENV_DIR/bin/python" - <<'PY'
import torch
import transformers
from transformers import AutoModelForMultimodalLM, AutoProcessor

print(f"torch={torch.__version__}")
print(f"torch_cuda={torch.version.cuda}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"transformers={transformers.__version__}")
print(f"multimodal_loader={AutoModelForMultimodalLM.__name__}")
print(f"processor_loader={AutoProcessor.__name__}")
PY
