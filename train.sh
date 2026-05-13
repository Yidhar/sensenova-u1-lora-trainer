#!/usr/bin/env bash
# train.sh — quick-launch a LoRA training run from a YAML config.
#
# Usage:
#   ./train.sh                                # uses configs/default.yaml
#   ./train.sh configs/my_style.yaml
#   ./train.sh configs/my_style.yaml --steps 12000   # extra args forwarded
#
# Long-running training tip:
#   setsid nohup ./train.sh configs/my_style.yaml </dev/null >run.log 2>&1 &
#   disown
# (a bare `nohup &` can be SIGHUP'd when the SSH/IDE session disconnects).

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${HERE}"

CONFIG="${1:-configs/default.yaml}"
shift || true

if [ ! -f "${CONFIG}" ]; then
    echo "config not found: ${CONFIG}" >&2
    echo "available configs:" >&2
    ls -1 configs/*.yaml 2>/dev/null >&2 || true
    exit 2
fi

export HF_HOME="${HF_HOME:-${HERE}/hf_cache}"
export PYTHONPATH="${HERE}:${PYTHONPATH:-}"
PY="${PY:-${HERE}/.venv/bin/python}"

if [ ! -x "${PY}" ]; then
    PY="$(command -v python3 || command -v python)"
fi

echo "===== train_u1 ====="
echo "  config:   ${CONFIG}"
echo "  python:   ${PY}"
echo "  HF_HOME:  ${HF_HOME}"
echo "  extra:    $*"
echo "===================="

exec "${PY}" -u -m train_u1.scripts.train_bf16_offload \
    --config "${CONFIG}" \
    "$@"
