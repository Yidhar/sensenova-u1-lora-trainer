#!/usr/bin/env bash
# Post-hoc periodic preview: iterate over every checkpoint*.safetensors in
# CKPT_DIR and sample 4 prompts × OUT_RES at NUM_STEPS Euler. Writes to
# ${PREVIEW_DIR}/step_NNNNNN/0{0..3}.png.
#
# Used when the training run did NOT produce in-process periodic samples
# (e.g., D v13 first-pass before the periodic-sample bug fix in
# train_fm_mvp.py:387). Each checkpoint contains the full trainable_state
# at that training step, so post-hoc sampling is equivalent to mid-training
# sampling — just shifted in time.
#
# Usage:
#   CKPT_DIR=artifacts/exp_d_v13_long_lora/checkpoints \
#   PREVIEW_DIR=artifacts/exp_d_v13_long_lora/preview \
#   PROMPTS=artifacts/exp_d_prompts.txt \
#   STYLE_TRIGGER="hayateluc style" LORA_R=64 LORA_ALPHA=64 \
#   IMAGE_H=1024 IMAGE_W=1024 NUM_STEPS=30 SAMPLE_SEED=42 \
#   bash train_u1/scripts/sample_checkpoints.sh

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
cd "${HERE}"

CKPT_DIR="${CKPT_DIR:-artifacts/exp_d_v13_long_lora/checkpoints}"
PREVIEW_DIR="${PREVIEW_DIR:-artifacts/exp_d_v13_long_lora/preview}"
PROMPTS="${PROMPTS:-artifacts/exp_d_prompts.txt}"
STYLE_TRIGGER="${STYLE_TRIGGER:-}"
LORA_R="${LORA_R:-0}"
LORA_ALPHA="${LORA_ALPHA:-32}"
IMAGE_H="${IMAGE_H:-1024}"
IMAGE_W="${IMAGE_W:-1024}"
NUM_STEPS="${NUM_STEPS:-30}"
SAMPLE_SEED="${SAMPLE_SEED:-42}"
QUANT="${QUANT:-bf16}"   # bf16 (default, uses sample_t2i_offload.py) | 8bit (legacy, distorts ts/ns atmosphere)
# Optional sidecar `H W` per prompt — overrides IMAGE_H/IMAGE_W per-prompt.
SAMPLE_BUCKETS_FILE="${SAMPLE_BUCKETS_FILE:-}"
# Wrap mlp_mot_gen MLP with LoRA at sample time (must match training scenario).
LORA_ON_MLP="${LORA_ON_MLP:-0}"
LORA_ON_MLP_FLAG=""
if [ "${LORA_ON_MLP}" = "1" ]; then LORA_ON_MLP_FLAG="--lora-on-mlp"; fi

export HF_HOME="${HF_HOME:-${HERE}/hf_cache}"
export PYTHONPATH="${HERE}:${PYTHONPATH:-}"
PY="${PY:-.venv/bin/python}"

mapfile -t PROMPTS_ARR < <(grep -v '^[[:space:]]*$' "${PROMPTS}")

declare -a BH BW
if [ -n "${SAMPLE_BUCKETS_FILE}" ] && [ -f "${SAMPLE_BUCKETS_FILE}" ]; then
    while read -r line; do
        case "${line}" in ''|'#'*) continue ;; esac
        BH+=("$(echo "${line}" | awk '{print $1}')")
        BW+=("$(echo "${line}" | awk '{print $2}')")
    done < "${SAMPLE_BUCKETS_FILE}"
    if [ "${#BH[@]}" -ne "${#PROMPTS_ARR[@]}" ]; then
        echo "[warn] SAMPLE_BUCKETS_FILE has ${#BH[@]} entries, prompts ${#PROMPTS_ARR[@]} — fallback to IMAGE_H/IMAGE_W"
        BH=(); BW=()
    fi
fi
get_h() { if [ ${#BH[@]} -gt 0 ]; then echo "${BH[$1]}"; else echo "${IMAGE_H}"; fi; }
get_w() { if [ ${#BW[@]} -gt 0 ]; then echo "${BW[$1]}"; else echo "${IMAGE_W}"; fi; }

echo "===== sample_checkpoints (${#PROMPTS_ARR[@]} prompts, quant=${QUANT}, ${NUM_STEPS} step) ====="
echo "  ckpt_dir=${CKPT_DIR}"
echo "  preview_dir=${PREVIEW_DIR}"
echo "  style_trigger=${STYLE_TRIGGER:-<none>}  lora_r=${LORA_R}"
echo "  buckets=${SAMPLE_BUCKETS_FILE:-<fixed ${IMAGE_H}×${IMAGE_W}>}"

shopt -s nullglob
ckpts=("${CKPT_DIR}"/step_*.safetensors)
if [ ${#ckpts[@]} -eq 0 ]; then
    echo "no step_*.safetensors found in ${CKPT_DIR}" >&2; exit 2
fi

for ckpt in "${ckpts[@]}"; do
    step_label=$(basename "${ckpt}" .safetensors)  # step_000400
    out_dir="${PREVIEW_DIR}/${step_label}"
    mkdir -p "${out_dir}"
    if [ "$(ls "${out_dir}" 2>/dev/null | wc -l)" -ge ${#PROMPTS_ARR[@]} ]; then
        echo "[skip] ${step_label} already populated"
        continue
    fi
    echo "===== ${step_label} ====="
    for i in "${!PROMPTS_ARR[@]}"; do
        idx=$(printf "%02d" "$i")
        out="${out_dir}/${idx}.png"
        if [ -f "${out}" ]; then
            echo "[skip] ${step_label}/${idx}.png exists"
            continue
        fi
        h_i="$(get_h $i)"; w_i="$(get_w $i)"
        echo "[${step_label}-${idx}] ${h_i}x${w_i} $(echo "${PROMPTS_ARR[$i]}" | head -c 60)..."
        if [ "${QUANT}" = "bf16" ]; then
            $PY -u -m train_u1.scripts.sample_t2i_offload \
                --lora-r "${LORA_R}" --lora-alpha "${LORA_ALPHA}" ${LORA_ON_MLP_FLAG} \
                --style-trigger "${STYLE_TRIGGER}" \
                --load-trainable-state-from "${ckpt}" \
                --prompt "${PROMPTS_ARR[$i]}" \
                --image-h "${h_i}" --image-w "${w_i}" \
                --num-steps "${NUM_STEPS}" --cfg-scale 4.0 \
                --timestep-shift 3.0 --cfg-norm none \
                --seed "${SAMPLE_SEED}" \
                --out "${out}"
        else
            $PY -u -m train_u1.scripts.sample_t2i \
                --quant "${QUANT}" \
                --lora-r "${LORA_R}" --lora-alpha "${LORA_ALPHA}" ${LORA_ON_MLP_FLAG} \
                --style-trigger "${STYLE_TRIGGER}" \
                --load-trainable-state-from "${ckpt}" \
                --prompt "${PROMPTS_ARR[$i]}" \
                --image-h "${h_i}" --image-w "${w_i}" \
                --num-steps "${NUM_STEPS}" --cfg-scale 4.0 \
                --timestep-shift 3.0 --cfg-norm none \
                --seed "${SAMPLE_SEED}" \
                --out "${out}"
        fi
    done
done

echo "===== DONE ====="
