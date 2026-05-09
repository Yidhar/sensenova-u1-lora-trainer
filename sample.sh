#!/usr/bin/env bash
# sample.sh — sample one image from a trained LoRA checkpoint, using a
# config file to pick up the matching LoRA preset/spec + style trigger.
#
# Usage:
#   ./sample.sh CONFIG.yaml STATE.safetensors --prompt "your prompt here" [--out preview.png]
#
# Common extras:
#   --image-h 1024 --image-w 1024  --num-steps 50  --cfg-scale 4.0
#   --think-mode --think-max-tokens 1024
#
# 8-step distill (stack with our LoRA, keeping fm_head separate):
#   ./sample.sh configs/default.yaml STATE.safetensors --prompt "..." \
#       --upstream-lora-path SenseNova-U1-8B-MoT-LoRA-8step-V1.0.safetensors \
#       --upstream-lora-skip fm_modules.fm_head \
#       --num-steps 8 --cfg-scale 1.0

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${HERE}"

if [ "$#" -lt 2 ]; then
    cat <<EOF >&2
usage: $0 CONFIG.yaml STATE.safetensors [--prompt "..."] [other sample_t2i_offload flags]

minimum example:
  $0 configs/default.yaml artifacts/my_run/trainable_state.safetensors \\
     --prompt "anime girl in dark kimono" --out preview.png
EOF
    exit 2
fi

CONFIG="$1"; shift
STATE="$1"; shift

if [ ! -f "${CONFIG}" ]; then
    echo "config not found: ${CONFIG}" >&2; exit 2
fi
if [ ! -f "${STATE}" ]; then
    echo "trainable state not found: ${STATE}" >&2; exit 2
fi

export HF_HOME="${HF_HOME:-${HERE}/hf_cache}"
export PYTHONPATH="${HERE}:${PYTHONPATH:-}"
PY="${PY:-${HERE}/.venv/bin/python}"

if [ ! -x "${PY}" ]; then
    PY="$(command -v python3 || command -v python)"
fi

exec "${PY}" -u -m train_u1.scripts.sample_t2i_offload \
    --config "${CONFIG}" \
    --load-trainable-state-from "${STATE}" \
    "$@"
