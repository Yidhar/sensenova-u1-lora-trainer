#!/usr/bin/env bash
# Source this from the repo root to use the project venv.
# venv was created with --system-site-packages so torch/transformers/safetensors
# come from the system; only bitsandbytes/peft/accelerate are venv-local.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
# shellcheck disable=SC1091
source "${HERE}/.venv/bin/activate"
export PYTHONPATH="${HERE}:${PYTHONPATH:-}"
echo "[train_u1] venv activated  python=$(python -c 'import sys;print(sys.executable)')"
