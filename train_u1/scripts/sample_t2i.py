"""Full t2i_generate sampling — drives upstream `NEOChatModel.t2i_generate`
end-to-end so we can compare image quality before vs after a training run.

Why this is separate from `eval_one_step.py`:
- `eval_one_step` measures one-step x0 reconstruction on `z_t = t·x0 + (1-t)·eps`
  built from the *true* x0. The model never has to bridge the gap from
  random Gaussian noise to a coherent image.
- This script starts from `noise_scale * randn(B, 3, H, W)` (per upstream
  `t2i_generate`) and runs the full Euler loop (default 30 steps + CFG).
  This is the actual product surface; it's how a one-step-overfit failure
  mode (e.g. trivial-mean reconstruction) reveals itself.

Usage:
    python -m train_u1.scripts.sample_t2i \
        --prompt "a photograph of a red apple on a wooden table" \
        --image-h 512 --image-w 512 --num-steps 30 --cfg-scale 4 \
        --out artifacts/sample_t2i/baseline.png

This script intentionally avoids any wrapper-side trainable state — it
loads the canonical `NEOChatModel` and drives its public `t2i_generate`.
To compare *trained* models, save trainable state via the planned
checkpointing hooks and re-run with the loaded state.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

from train_u1.constants import MODEL_ID, MODEL_SHA
from train_u1.model.loader import (
    _resolve_local_snapshot,
    load_neo_chat,
    load_neo_chat_4bit,
    load_neo_chat_8bit,
)


def _save_image(tensor: torch.Tensor, path: str | os.PathLike) -> None:
    """tensor: (B, 3, H, W) in [-1, 1] or [0, 1] → PNG."""
    import numpy as np
    from PIL import Image

    if tensor.dim() == 4:
        tensor = tensor[0]
    arr = tensor.detach().to("cpu", torch.float32).numpy()
    if arr.min() < -0.01:
        arr = (arr + 1.0) * 0.5  # [-1, 1] → [0, 1]
    arr = arr.clip(0, 1)
    arr = (arr.transpose(1, 2, 0) * 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default=os.environ.get("HF_HOME"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--image-h", type=int, default=512)
    ap.add_argument("--image-w", type=int, default=512)
    # Defaults match the official inference recipe (u1_src/examples/t2i/inference.py).
    ap.add_argument("--num-steps", type=int, default=50)
    ap.add_argument("--cfg-scale", type=float, default=4.0)
    ap.add_argument("--timestep-shift", type=float, default=3.0)
    # Upstream t2i_generate L1580: assert cfg_norm in
    #   {'cfg_zero_star', 'global', 'none', 'channel'}.
    ap.add_argument(
        "--cfg-norm",
        default="none",
        choices=("none", "global", "channel", "cfg_zero_star"),
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, help="output PNG path")
    ap.add_argument(
        "--quant",
        choices=("bf16", "8bit", "4bit"),
        default="4bit",
        help=(
            "bf16  = full bfloat16 base (~35 GB; needs offload on 32 GB cards). "
            "8bit  = int8 quantization (~17 GB; better quality than 4bit). "
            "4bit  = nf4 4-bit (~9 GB; cheapest but accumulates artifacts in "
            "        pixel-space FM sampling)."
        ),
    )
    ap.add_argument(
        "--load-trainable-state-from",
        default=None,
        help=(
            "Optional path to a safetensors file produced by train_fm_mvp.py "
            "--save-trainable-state-to. Trainable params (fm_head, ts/ns_embed, "
            "vision_model_mot_gen, mot_gen norms) are loaded into matching "
            "modules of the freshly-loaded base. Lets you sample under a quant "
            "different from the one used in training without in-process reload."
        ),
    )
    ap.add_argument(
        "--lora-r",
        type=int,
        default=0,
        help=(
            "If > 0, wrap `_mot_gen` q/k/v/o projections with LoraAdapter of "
            "rank `r` BEFORE loading trainable state. Required when sampling "
            "models trained with scenario=lora_attn — the saved state contains "
            "lora_A/lora_B weights that must land into wrapped modules."
        ),
    )
    ap.add_argument("--lora-alpha", type=float, default=32.0)
    ap.add_argument(
        "--lora-on-mlp",
        action="store_true",
        help=(
            "If set, also wrap mlp_mot_gen.{gate,up,down}_proj with LoRA. "
            "Required when sampling models trained with scenario=lora_attn_mlp "
            "or lora_attn_mlp_plus_embedders so saved lora_A/lora_B land into "
            "wrapped MLP modules."
        ),
    )
    ap.add_argument(
        "--style-trigger",
        default="",
        help="Optional trigger text prepended to prompt (must match training-time trigger).",
    )
    args = ap.parse_args()

    print(f"[sample] loading model {args.quant}...", flush=True)
    if args.quant == "4bit":
        model = load_neo_chat_4bit(cache_dir=args.cache_dir, device_map=args.device)
    elif args.quant == "8bit":
        model = load_neo_chat_8bit(cache_dir=args.cache_dir, device_map=args.device)
    else:
        model = load_neo_chat(cache_dir=args.cache_dir, device_map=args.device, dtype=torch.bfloat16)

    if args.lora_r > 0:
        from train_u1.model.lora import (
            lora_param_count,
            wrap_mot_gen_attention,
            wrap_mot_gen_mlp,
        )

        n_attn = wrap_mot_gen_attention(
            model, r=args.lora_r, alpha=args.lora_alpha,
        )
        n_mlp = 0
        if args.lora_on_mlp:
            n_mlp = wrap_mot_gen_mlp(
                model, r=args.lora_r, alpha=args.lora_alpha,
            )
        print(
            f"[sample] wrapped {n_attn} attn + {n_mlp} mlp projections with LoRA "
            f"r={args.lora_r}, lora_params={lora_param_count(model):,}"
        )

    if args.load_trainable_state_from:
        from safetensors.torch import load_file as st_load_file

        loaded = st_load_file(args.load_trainable_state_from)
        target_state = {n: p for n, p in model.named_parameters()}
        n_loaded = 0
        n_missing = []
        for name, t in loaded.items():
            if name not in target_state:
                n_missing.append(name)
                continue
            tgt = target_state[name]
            tgt.data.copy_(t.to(tgt.device, tgt.dtype))
            n_loaded += 1
        print(f"[sample] injected {n_loaded}/{len(loaded)} trainable tensors "
              f"from {args.load_trainable_state_from}")
        if n_missing:
            print(f"[warn] {len(n_missing)} tensors not found in target model; "
                  f"first 3: {n_missing[:3]}")

    from transformers import AutoTokenizer

    local = _resolve_local_snapshot(args.cache_dir, MODEL_ID, MODEL_SHA)
    tok = AutoTokenizer.from_pretrained(
        local or MODEL_ID,
        revision=None if local else MODEL_SHA,
        trust_remote_code=True,
    )

    # Apply style trigger identical to training-time format
    # (`f"{trigger}, {original_prompt}"`); this fires the trained LoRA pattern.
    effective_prompt = (
        f"{args.style_trigger}, {args.prompt}"
        if args.style_trigger else args.prompt
    )
    if args.style_trigger:
        print(f"[sample] style_trigger={args.style_trigger!r}")
    print(f"[sample] prompt={effective_prompt!r}")
    print(f"[sample] H,W=({args.image_h},{args.image_w})  steps={args.num_steps}  cfg={args.cfg_scale}")

    # Upstream's `t2i_generate` returns the final image tensor and prints
    # progress; we wrap in no_grad just to be defensive even though the
    # function itself runs eval-mode forwards.
    with torch.no_grad():
        image = model.t2i_generate(
            tokenizer=tok,
            prompt=effective_prompt,
            cfg_scale=args.cfg_scale,
            cfg_norm=args.cfg_norm,
            timestep_shift=args.timestep_shift,
            image_size=(args.image_w, args.image_h),  # upstream uses (W, H)
            num_steps=args.num_steps,
            seed=args.seed,
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _save_image(image, out_path)
    print(f"[sample] saved {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
