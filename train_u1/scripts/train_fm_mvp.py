"""24GB MVP training entry — fm_head only, x0-MSE flow-matching overfit.

Goal (per report §10 plan A): demonstrate that the wrapper + freeze policy
+ cache pipeline can drive `fm_head` through a 1/4/16-sample overfit at
512² or smaller. Success criterion: loss drops monotonically and we can
bit-flip the same-sample loss after a few hundred steps.

This is a *correctness-first* runner: synthetic data, batch=1, no LR
schedule. Plug into a real dataset by swapping `SyntheticT2ITinyDataset`
for `FilesystemT2ITinyDataset`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

from train_u1.constants import MODEL_ID, MODEL_SHA
from train_u1.data.collators import CollatorConfig, SenseNovaU1Collator, to_device
from train_u1.data.datasets import PairedFolderT2IDataset, SyntheticT2ITinyDataset
from train_u1.model.loader import (
    _resolve_local_snapshot,
    load_neo_chat,
    load_neo_chat_4bit,
    load_neo_chat_8bit,
)
from train_u1.model.losses import fm_loss_x0
from train_u1.model.params import (
    FREEZE_REGEX_AUX_NO_HEAD,
    FREEZE_REGEX_GEN_VISION,
    FREEZE_REGEX_MVP,
    TRAINABLE_REGEX_AUX_NO_HEAD,
    TRAINABLE_REGEX_GEN_VISION,
    TRAINABLE_REGEX_MVP,
    TRAINABLE_REGEX_MVP_AUX,
    set_requires_grad_by_regex,
)
from train_u1.model.wrapper import TrainingWrapper


def _trainable_params(model: nn.Module) -> list[nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default=os.environ.get("HF_HOME"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument(
        "--data-dir",
        default=None,
        help="folder of paired {stem}.jpg + {stem}.txt. If unset, uses synthetic data.",
    )
    ap.add_argument(
        "--cap-max-pixels",
        type=int,
        default=512 * 512,
        help="VRAM-friendly upper bound on resized image pixels (real images go through smart_resize).",
    )
    ap.add_argument("--n-samples", type=int, default=4, help="cap on dataset size when iterating")
    ap.add_argument("--image-h", type=int, default=256, help="synthetic-only: square image H")
    ap.add_argument("--image-w", type=int, default=256, help="synthetic-only: square image W")
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument(
        "--scenario",
        choices=(
            "mvp", "mvp_aux", "gen_vision", "aux_no_head",
            "lora_attn",
            "lora_attn_mlp",
            "lora_attn_plus_embedders",
            "lora_attn_mlp_plus_embedders",
        ),
        default="mvp_aux",
        help=(
            "mvp                            = fm_head only. "
            "mvp_aux                        = fm_head + ts/ns embedders + mot_gen norms. "
            "gen_vision                     = vision_model_mot_gen + ts/ns + mot_gen norms, no fm_head. "
            "aux_no_head                    = ts/ns + mot_gen norms only. "
            "lora_attn                      = LoRA on `_mot_gen` q/k/v/o (D v13/v14 production). "
            "lora_attn_mlp                  = LoRA on `_mot_gen` q/k/v/o + mlp_mot_gen gate/up/down (D v15a). "
            "lora_attn_plus_embedders       = LoRA on attn + unfreeze ts/ns embedders (D v15b). "
            "lora_attn_mlp_plus_embedders   = LoRA on attn + MLP + unfreeze ts/ns embedders (D v15d)."
        ),
    )
    ap.add_argument(
        "--quant",
        choices=("bf16", "8bit", "4bit"),
        default="4bit",
        help=(
            "Base-model quantization. 4-bit nf4 introduces visible artifacts "
            "in pixel-space FM sampling (累积 ~50 step × 42 layer); 8-bit (int8) "
            "is materially cleaner for ~2x VRAM. bf16 needs cpu_offload on 32 GB."
        ),
    )
    ap.add_argument(
        "--sample-quant",
        choices=("same", "bf16", "8bit", "4bit"),
        default="same",
        help=(
            "Reload base at this quant level for the AFTER-training sampling "
            "pass. 'same' reuses the training-time quant. NOTE: in-process "
            "quant reload + 32 GB card OOMs reliably (bnb buffers + GC closure "
            "cycles leave ~14 GB residual). Use --save-trainable-state-to + "
            "sample_t2i.py --load-trainable-state-from for separate-process "
            "training and clean 8bit sampling."
        ),
    )
    ap.add_argument(
        "--save-trainable-state-to",
        default=None,
        help=(
            "Path to save trainable parameters (safetensors) at end of training. "
            "Use with sample_t2i.py --load-trainable-state-from for clean "
            "sampling under a different quant config."
        ),
    )
    # Periodic checkpoint + sampling (standard LoRA workflow):
    # save trainable state every N optimizer steps → study evolution curve
    # of fine-tune signal across training.
    ap.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help="If > 0, save trainable state to {ckpt-dir}/step_{N:06d}.safetensors every N optimizer steps.",
    )
    ap.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Directory for periodic checkpoints (only used if --checkpoint-every > 0).",
    )
    ap.add_argument(
        "--periodic-sample-every",
        type=int,
        default=0,
        help="If > 0, sample 4 prompts in-process every N optimizer steps using current trainable params.",
    )
    ap.add_argument(
        "--periodic-sample-out-dir",
        default=None,
        help="Directory for periodic-sample preview PNGs (only used if --periodic-sample-every > 0).",
    )
    ap.add_argument(
        "--periodic-sample-image-h",
        type=int,
        default=1024,
        help="Resolution of periodic preview samples (lower = faster; 1024 typical for previews vs 2048 final).",
    )
    ap.add_argument(
        "--periodic-sample-image-w",
        type=int,
        default=1024,
    )
    ap.add_argument(
        "--periodic-sample-num-steps",
        type=int,
        default=30,
        help="Euler steps for periodic preview (30 typical, 50 for final quality).",
    )
    ap.add_argument("--lora-r", type=int, default=16, help="LoRA rank (only for scenario=lora_attn)")
    ap.add_argument("--lora-alpha", type=float, default=32.0, help="LoRA alpha (effective scale = alpha/r)")
    ap.add_argument("--lora-dropout", type=float, default=0.0)
    ap.add_argument(
        "--style-trigger",
        default="",
        help=(
            "Optional trigger text prepended to every caption (e.g. "
            "'hayateluc style'). Standard SDXL/Flux LoRA style-training "
            "practice — anchors the learned style to a specific text pattern."
        ),
    )
    ap.add_argument(
        "--prompt-template",
        choices=("raw", "official"),
        default="raw",
        help=(
            "raw      = tokenize caption directly (matches Phase-4 / experiment A). "
            "official = wrap caption via NEOChatModel._build_t2i_query (chat template; "
            "           matches `t2i_generate` inference distribution)."
        ),
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-path", default="artifacts/train_fm_mvp_log.jsonl")
    ap.add_argument(
        "--shuffle",
        action="store_true",
        help="randomly index into the dataset each step instead of step%%n",
    )
    ap.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help=(
            "enable gradient checkpointing on the language_model side (Qwen3Model). "
            "Required for 2048² training on 32 GB cards (saves attention activations "
            "via per-layer recomputation in backward)."
        ),
    )
    ap.add_argument(
        "--snap-bucket",
        action="store_true",
        help=(
            "Snap each training image to the nearest official supported bucket "
            "(constants.OFFICIAL_BUCKETS_HW), instead of plain smart_resize. "
            "Use when training data has arbitrary aspect ratios — keeps train and "
            "inference shape distributions aligned. Overrides --cap-max-pixels."
        ),
    )
    # Experiment A: one-step denoise visualization
    ap.add_argument(
        "--eval-panel-out",
        default=None,
        help="if set, render `before` and `after` 4-panel PNGs for each training sample",
    )
    ap.add_argument(
        "--eval-t-values",
        default="0.3,0.5,0.7",
        help="comma-separated t values for eval (e.g. 0.3,0.5,0.7)",
    )
    # Experiment D: full t2i_generate sampling
    ap.add_argument(
        "--sample-prompts-file",
        default=None,
        help="path to a text file with one raw prompt per line; before/after sampling runs",
    )
    ap.add_argument("--sample-out-dir", default=None, help="dir to save before/after PNGs")
    ap.add_argument("--sample-image-h", type=int, default=512)
    ap.add_argument("--sample-image-w", type=int, default=512)
    # Defaults below match the *official* inference recipe (u1_src/examples/t2i/inference.py
    # + examples/README.md): num_steps=50, cfg_scale=4.0, timestep_shift=3.0, cfg_norm=none.
    # The model's t2i_generate signature defaults are looser (num_steps=30,
    # cfg_scale=1, timestep_shift=1) but will produce noticeably noisier output.
    ap.add_argument("--sample-num-steps", type=int, default=50)
    ap.add_argument("--sample-cfg-scale", type=float, default=4.0)
    ap.add_argument("--sample-timestep-shift", type=float, default=3.0)
    # Upstream t2i_generate L1580: assert cfg_norm in
    #   {'cfg_zero_star', 'global', 'none', 'channel'}.
    ap.add_argument(
        "--sample-cfg-norm",
        default="none",
        choices=("none", "global", "channel", "cfg_zero_star"),
    )
    ap.add_argument("--sample-seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    print(f"[mvp] loading model {args.quant} (scenario={args.scenario})...", flush=True)
    if args.quant == "4bit":
        model = load_neo_chat_4bit(cache_dir=args.cache_dir, device_map=args.device)
    elif args.quant == "8bit":
        model = load_neo_chat_8bit(cache_dir=args.cache_dir, device_map=args.device)
    else:
        model = load_neo_chat(cache_dir=args.cache_dir, device_map=args.device, dtype=torch.bfloat16)

    if args.scenario == "mvp":
        freeze_pats, train_pats = FREEZE_REGEX_MVP, TRAINABLE_REGEX_MVP
    elif args.scenario == "mvp_aux":
        freeze_pats, train_pats = FREEZE_REGEX_MVP, TRAINABLE_REGEX_MVP_AUX
    elif args.scenario == "gen_vision":
        freeze_pats, train_pats = FREEZE_REGEX_GEN_VISION, TRAINABLE_REGEX_GEN_VISION
    elif args.scenario == "aux_no_head":
        freeze_pats, train_pats = FREEZE_REGEX_AUX_NO_HEAD, TRAINABLE_REGEX_AUX_NO_HEAD
    elif args.scenario in (
        "lora_attn",
        "lora_attn_mlp",
        "lora_attn_plus_embedders",
        "lora_attn_mlp_plus_embedders",
    ):
        from train_u1.model.lora import (
            lora_param_count,
            wrap_mot_gen_attention,
            wrap_mot_gen_mlp,
        )

        n_attn = wrap_mot_gen_attention(
            model, r=args.lora_r, alpha=args.lora_alpha,
            dropout=args.lora_dropout,
        )
        n_mlp = 0
        if args.scenario in ("lora_attn_mlp", "lora_attn_mlp_plus_embedders"):
            n_mlp = wrap_mot_gen_mlp(
                model, r=args.lora_r, alpha=args.lora_alpha,
                dropout=args.lora_dropout,
            )
        n_lora = lora_param_count(model)
        print(
            f"[mvp] wrapped {n_attn} attn + {n_mlp} mlp projections with LoRA "
            f"r={args.lora_r}, alpha={args.lora_alpha}, lora_params={n_lora:,}"
        )
        freeze_pats = (r".*",)  # default-deny
        train_pats = [
            r"\.lora_A\.weight$",
            r"\.lora_B\.weight$",
        ]
        if args.scenario in (
            "lora_attn_plus_embedders",
            "lora_attn_mlp_plus_embedders",
        ):
            train_pats += [
                r"^fm_modules\.timestep_embedder\.",
                r"^fm_modules\.noise_scale_embedder\.",
            ]
        train_pats = tuple(train_pats)
    else:
        raise ValueError(f"unknown scenario: {args.scenario!r}")
    rep = set_requires_grad_by_regex(
        model,
        freeze_patterns=freeze_pats,
        trainable_patterns=train_pats,
        default=False,
        strict=True,
    )
    if args.gradient_checkpointing:
        # Upstream Qwen3Model.forward (modeling_qwen3.py L1027+) does NOT call
        # `self._gradient_checkpointing_func` per layer — i.e., the standard
        # HF `gradient_checkpointing_enable()` flag is honored at the
        # `supports_gradient_checkpointing=True` declaration but not actually
        # plumbed into the layer loop. So we monkey-patch each decoder_layer
        # to wrap its forward in `torch.utils.checkpoint.checkpoint`.
        from torch.utils.checkpoint import checkpoint as _ckpt

        layers = model.language_model.model.layers
        for layer in layers:
            _orig = layer.forward
            def _make_gc_forward(orig):
                def gc_forward(hidden_states, **kw):
                    # Only `hidden_states` flows through checkpoint; everything
                    # else is closed over so checkpoint doesn't try to save it.
                    def _fn(h):
                        return orig(h, **kw)
                    return _ckpt(_fn, hidden_states, use_reentrant=False)
                return gc_forward
            layer.forward = _make_gc_forward(_orig)
        # Required for GC under bnb-4bit base: makes the input embeddings
        # require grad so gradients flow backward through frozen embedding.
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        print(f"[mvp] gradient_checkpointing wrapped on {len(layers)} decoder layers")
    print(f"[mvp] trainable params: {rep.n_trainable:,}")
    for k, v in sorted(rep.bucket_trainable.items(), key=lambda kv: -kv[1]):
        print(f"      {k:<24s} {v:>14,d}")

    # Tokenizer (local snapshot)
    from transformers import AutoTokenizer

    local = _resolve_local_snapshot(args.cache_dir, MODEL_ID, MODEL_SHA)
    tok = AutoTokenizer.from_pretrained(
        local or MODEL_ID,
        revision=None if local else MODEL_SHA,
        trust_remote_code=True,
    )

    # Dataset selection: paired-folder if --data-dir given, else synthetic.
    if args.data_dir:
        ds_full = PairedFolderT2IDataset(
            args.data_dir,
            cap_max_pixels=args.cap_max_pixels,
            snap_bucket=args.snap_bucket,
        )
        n_use = min(args.n_samples, len(ds_full))
        ds = ds_full
        collator_cfg = CollatorConfig(
            image_hw=None,
            seed=args.seed,
            enforce_batch_one=True,
            prompt_template=args.prompt_template,
            style_trigger=args.style_trigger,
        )
        collator = SenseNovaU1Collator(
            tok, cfg=collator_cfg, model=model if args.prompt_template == "official" else None
        )
        print(f"[mvp] paired-folder dataset: {len(ds_full)} pairs (using {n_use})")
    else:
        ds = SyntheticT2ITinyDataset(
            n=args.n_samples, image_hw=(args.image_h, args.image_w), base_seed=args.seed + 100
        )
        n_use = args.n_samples
        collator_cfg = CollatorConfig(
            image_hw=(args.image_h, args.image_w),
            seed=args.seed,
            prompt_template=args.prompt_template,
            style_trigger=args.style_trigger,
        )
        collator = SenseNovaU1Collator(
            tok, cfg=collator_cfg, model=model if args.prompt_template == "official" else None
        )

    wrapper = TrainingWrapper(model)

    # 8-bit AdamW from bitsandbytes (per report §4.2 / §10A)
    import bitsandbytes as bnb  # type: ignore

    optimizer = bnb.optim.AdamW8bit(_trainable_params(model), lr=args.lr, betas=(0.9, 0.95))

    log_path = Path(args.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("")  # truncate

    # Experiment A: snapshot eval samples + run "before" panels.
    eval_samples = None
    rows_before = None
    if args.eval_panel_out:
        from train_u1.scripts.eval_one_step import run_eval_panel

        # Use the same first n_use samples as training (deterministic).
        eval_samples = [ds[i] for i in range(n_use)]
        t_values = tuple(float(x) for x in args.eval_t_values.split(","))
        print(f"\n[mvp] eval BEFORE training: {len(eval_samples)} samples × {len(t_values)} t values")
        rows_before = run_eval_panel(
            wrapper=wrapper,
            collator=collator,
            samples=eval_samples,
            t_values=t_values,
            out_dir=args.eval_panel_out,
            label="before",
            device=args.device,
        )

    # Load sample prompts independently of `--sample-out-dir`; the prompt list
    # is also needed by periodic in-process sampling (which writes to its own
    # `--periodic-sample-out-dir`, separate from the final BEFORE/AFTER dir).
    sample_prompts: list[str] | None = None
    if args.sample_prompts_file:
        sp_path = Path(args.sample_prompts_file)
        sample_prompts = [
            ln.strip() for ln in sp_path.read_text().splitlines() if ln.strip()
        ]
        if not sample_prompts:
            raise SystemExit(f"--sample-prompts-file {sp_path} has no usable lines")

    # Experiment D: BEFORE-training full-sampling baseline (only when an
    # explicit before/after output dir is given).
    if args.sample_prompts_file and args.sample_out_dir:
        from train_u1.scripts.sample_t2i import _save_image as _save_t2i_image

        out_root = Path(args.sample_out_dir)
        (out_root / "before").mkdir(parents=True, exist_ok=True)
        (out_root / "after").mkdir(parents=True, exist_ok=True)
        # record prompts + config
        (out_root / "config.json").write_text(json.dumps({
            "prompts": sample_prompts,
            "image_hw": [args.sample_image_h, args.sample_image_w],
            "num_steps": args.sample_num_steps,
            "cfg_scale": args.sample_cfg_scale,
            "timestep_shift": args.sample_timestep_shift,
            "cfg_norm": args.sample_cfg_norm,
            "seed": args.sample_seed,
            "training_scenario": args.scenario,
            "training_prompt_template": args.prompt_template,
            "training_steps": args.steps,
            "training_lr": args.lr,
            "training_n_samples": n_use,
        }, indent=2))

        sample_quant_before = args.quant if args.sample_quant == "same" else args.sample_quant
        # If we'll sample AFTER at a different quant, do BEFORE at the SAME
        # target quant — otherwise before/after aren't directly comparable.
        # We do this by reloading the base now, sampling, then reloading
        # again at training-quant for the actual training. Two extra reloads
        # but cleanest comparison.
        if sample_quant_before != args.quant:
            print(f"\n[mvp] reloading base for BEFORE sampling at {sample_quant_before}")
            import gc
            del wrapper, model, optimizer
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.empty_cache()
            if sample_quant_before == "8bit":
                model = load_neo_chat_8bit(cache_dir=args.cache_dir, device_map=args.device)
            elif sample_quant_before == "4bit":
                model = load_neo_chat_4bit(cache_dir=args.cache_dir, device_map=args.device)
            else:
                model = load_neo_chat(cache_dir=args.cache_dir, device_map=args.device, dtype=torch.bfloat16)

        print(f"\n[mvp] sampling BEFORE training: {len(sample_prompts)} prompts × "
              f"{args.sample_num_steps} steps × CFG={args.sample_cfg_scale} (quant={sample_quant_before})")
        for i, prompt in enumerate(sample_prompts):
            print(f"[sample-before] {i}: {prompt[:80]}{'...' if len(prompt)>80 else ''}", flush=True)
            with torch.no_grad():
                img = model.t2i_generate(
                    tokenizer=tok,
                    prompt=prompt,
                    cfg_scale=args.sample_cfg_scale,
                    cfg_norm=args.sample_cfg_norm,
                    timestep_shift=args.sample_timestep_shift,
                    image_size=(args.sample_image_w, args.sample_image_h),
                    num_steps=args.sample_num_steps,
                    seed=args.sample_seed,
                )
            _save_t2i_image(img, out_root / "before" / f"{i:02d}.png")

        if sample_quant_before != args.quant:
            print(f"\n[mvp] reloading base back to {args.quant} for training")
            import gc
            del model
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            if args.quant == "4bit":
                model = load_neo_chat_4bit(cache_dir=args.cache_dir, device_map=args.device)
            elif args.quant == "8bit":
                model = load_neo_chat_8bit(cache_dir=args.cache_dir, device_map=args.device)
            else:
                model = load_neo_chat(cache_dir=args.cache_dir, device_map=args.device, dtype=torch.bfloat16)
            # Re-apply freeze policy and (if requested) GC to the freshly-loaded model
            set_requires_grad_by_regex(
                model, freeze_patterns=freeze_pats, trainable_patterns=train_pats,
                default=False, strict=True,
            )
            if args.gradient_checkpointing:
                from torch.utils.checkpoint import checkpoint as _ckpt
                for layer in model.language_model.model.layers:
                    _orig = layer.forward
                    def _make_gc_forward(orig):
                        def gc_forward(hidden_states, **kw):
                            def _fn(h):
                                return orig(h, **kw)
                            return _ckpt(_fn, hidden_states, use_reentrant=False)
                        return gc_forward
                    layer.forward = _make_gc_forward(_orig)
                if hasattr(model, "enable_input_require_grads"):
                    model.enable_input_require_grads()
            # Rebuild wrapper + optimizer to point at the new model
            wrapper = TrainingWrapper(model)
            optimizer = bnb.optim.AdamW8bit(_trainable_params(model), lr=args.lr, betas=(0.9, 0.95))

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    print(f"\n[mvp] starting overfit ({args.steps} steps, n_samples={n_use})", flush=True)
    t0 = time.time()
    losses: list[float] = []
    rng = torch.Generator().manual_seed(args.seed)
    for step in range(args.steps):
        if args.shuffle:
            idx = int(torch.randint(0, n_use, (1,), generator=rng).item())
        else:
            idx = step % n_use
        sample = ds[idx]
        batch = collator([sample])
        batch = to_device(batch, args.device, dtype=torch.bfloat16)
        token_h, token_w = batch["token_hw"]

        out = wrapper.forward_t2i_step(batch)
        loss = fm_loss_x0(out.x_pred, batch["x0_patch"]) / args.grad_accum
        loss.backward()

        do_step = ((step + 1) % args.grad_accum) == 0
        if do_step:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        losses.append(loss.item() * args.grad_accum)
        if step < 5 or step % 10 == 0 or step == args.steps - 1:
            print(
                f"[mvp] step={step:4d}  loss={losses[-1]:.4f}  "
                f"sample={sample.sample_id} hw=({batch['noisy_pixel_values'].shape[2]},"
                f"{batch['noisy_pixel_values'].shape[3]}) tokens={token_h*token_w}",
                flush=True,
            )
        with open(log_path, "a") as f:
            f.write(json.dumps({
                "step": step,
                "sample_idx": idx,
                "sample_id": sample.sample_id,
                "loss": losses[-1],
                "token_h": int(token_h),
                "token_w": int(token_w),
            }) + "\n")

        # ---- Periodic checkpoint save ----
        if args.checkpoint_every > 0 and (step + 1) % args.checkpoint_every == 0:
            ckpt_dir_path = Path(args.checkpoint_dir or args.save_trainable_state_to or ".").resolve()
            if args.checkpoint_dir:
                ckpt_dir_path = Path(args.checkpoint_dir)
            else:
                ckpt_dir_path = Path(args.save_trainable_state_to).parent / "checkpoints"
            ckpt_dir_path.mkdir(parents=True, exist_ok=True)
            ckpt_path = ckpt_dir_path / f"step_{step+1:06d}.safetensors"
            from safetensors.torch import save_file as st_save_file
            ckpt_state = {
                name: p.detach().to("cpu", torch.bfloat16).contiguous()
                for name, p in model.named_parameters()
                if p.requires_grad
            }
            st_save_file(ckpt_state, str(ckpt_path))
            print(f"[mvp] checkpoint saved → {ckpt_path}", flush=True)
            del ckpt_state

        # ---- Periodic in-process sampling ----
        if (
            args.periodic_sample_every > 0
            and (step + 1) % args.periodic_sample_every == 0
            and sample_prompts is not None
        ):
            sample_dir = Path(args.periodic_sample_out_dir or args.sample_out_dir or "preview")
            step_dir = sample_dir / f"step_{step+1:06d}"
            step_dir.mkdir(parents=True, exist_ok=True)
            from train_u1.scripts.sample_t2i import _save_image as _save_t2i_image_inline  # noqa
            from train_u1.scripts.sample_t2i_offload import _save_image as _save_t2i_image
            print(
                f"\n[mvp] periodic sample @ step {step+1}: {len(sample_prompts)} prompts × "
                f"{args.periodic_sample_num_steps} step Euler @ "
                f"{args.periodic_sample_image_h}×{args.periodic_sample_image_w}",
                flush=True,
            )
            t_sample0 = time.time()
            for pi, prompt_text in enumerate(sample_prompts):
                eff_prompt = (
                    f"{args.style_trigger}, {prompt_text}"
                    if args.style_trigger else prompt_text
                )
                with torch.no_grad():
                    img = model.t2i_generate(
                        tokenizer=tok,
                        prompt=eff_prompt,
                        cfg_scale=args.sample_cfg_scale,
                        cfg_norm=args.sample_cfg_norm,
                        timestep_shift=args.sample_timestep_shift,
                        image_size=(args.periodic_sample_image_w, args.periodic_sample_image_h),
                        num_steps=args.periodic_sample_num_steps,
                        seed=args.sample_seed,
                    )
                _save_t2i_image(img, step_dir / f"{pi:02d}.png")
            print(
                f"[mvp]   periodic sample @ step {step+1} done in {time.time()-t_sample0:.1f}s",
                flush=True,
            )

    elapsed = time.time() - t0
    peak_vram = (torch.cuda.max_memory_allocated() / 1e9) if torch.cuda.is_available() else 0.0
    print(
        f"\n[mvp] done in {elapsed:.1f}s ({elapsed/args.steps:.2f}s/step)  "
        f"peak_vram={peak_vram:.2f} GB"
    )
    if len(losses) >= 10:
        first = sum(losses[:5]) / 5
        last = sum(losses[-5:]) / 5
        ratio = last / first if first > 0 else float("nan")
        print(f"[mvp] mean(loss[:5])={first:.4f}  mean(loss[-5:])={last:.4f}  ratio={ratio:.3f}")

    # Experiment A: run "after" panels and summarize.
    if args.eval_panel_out and eval_samples is not None and rows_before is not None:
        from train_u1.scripts.eval_one_step import run_eval_panel, summarize

        t_values = tuple(float(x) for x in args.eval_t_values.split(","))
        print(f"\n[mvp] eval AFTER training: {len(eval_samples)} samples × {len(t_values)} t values")
        rows_after = run_eval_panel(
            wrapper=wrapper,
            collator=collator,
            samples=eval_samples,
            t_values=t_values,
            out_dir=args.eval_panel_out,
            label="after",
            device=args.device,
        )
        table = summarize(rows_before, rows_after)
        print("\n" + table)
        Path(args.eval_panel_out, "summary.txt").write_text(table)
        print(f"\n[mvp] eval panels + summary saved under {args.eval_panel_out}")

    # Save trainable state to disk (separate-process workflow).
    if args.save_trainable_state_to:
        from safetensors.torch import save_file as st_save_file

        out_path = Path(args.save_trainable_state_to)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        trainable = {
            name: p.detach().to("cpu", torch.bfloat16).contiguous()
            for name, p in model.named_parameters()
            if p.requires_grad
        }
        st_save_file(trainable, str(out_path))
        print(f"\n[mvp] saved {len(trainable)} trainable tensors → {out_path}")

    # Experiment D: AFTER-training full-sampling.
    # Only run if user supplied both --sample-prompts-file AND --sample-out-dir.
    # (The runner shells do their own out-of-process AFTER sampling stage, so
    #  they pass --sample-prompts-file for the periodic preview hook but leave
    #  --sample-out-dir unset.)
    if sample_prompts is not None and args.sample_out_dir is not None:
        from train_u1.scripts.sample_t2i import _save_image as _save_t2i_image

        out_root = Path(args.sample_out_dir)

        # If --sample-quant differs from --quant, harvest trainable params,
        # release the training model, reload base at sample_quant, and inject
        # the trainable state. Lets us train at 4bit (cheap) + sample at 8bit
        # (clean) on a 32 GB card.
        sample_quant = args.quant if args.sample_quant == "same" else args.sample_quant
        if sample_quant != args.quant:
            print(
                f"\n[mvp] reloading base for sampling: "
                f"train_quant={args.quant} → sample_quant={sample_quant}"
            )
            trainable_state = {
                name: p.detach().to("cpu").clone()
                for name, p in model.named_parameters()
                if p.requires_grad
            }
            print(f"[mvp] harvested {len(trainable_state)} trainable tensors")
            # Aggressive cleanup: GC closures (the per-layer GC monkey-patch
            # holds refs to original forward bound methods → reference cycles)
            # + bnb 8-bit AdamW state buffers don't release on `del` alone.
            import gc
            del wrapper, model, optimizer
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.empty_cache()
            print(f"[mvp] post-cleanup VRAM used: "
                  f"{torch.cuda.memory_allocated() / 1e9:.2f} GB", flush=True)
            torch.cuda.reset_peak_memory_stats()
            if sample_quant == "8bit":
                model = load_neo_chat_8bit(cache_dir=args.cache_dir, device_map=args.device)
            elif sample_quant == "4bit":
                model = load_neo_chat_4bit(cache_dir=args.cache_dir, device_map=args.device)
            else:
                model = load_neo_chat(cache_dir=args.cache_dir, device_map=args.device, dtype=torch.bfloat16)
            # Re-inject trainable state. The trainable modules (fm_head, ts/ns
            # embedders, vision_model_mot_gen, mot_gen norms) are in
            # `keep_modules_in_fp` for both quant configs, so their weights
            # land at fp16/bf16 — directly load_state_dict-compatible.
            target_state = {n: p for n, p in model.named_parameters()}
            n_loaded = 0
            for name, t in trainable_state.items():
                if name not in target_state:
                    print(f"[warn] trainable tensor missing in sample model: {name}")
                    continue
                tgt = target_state[name]
                tgt.data.copy_(t.to(tgt.device, tgt.dtype))
                n_loaded += 1
            print(f"[mvp] re-injected {n_loaded}/{len(trainable_state)} trainable tensors")

        print(
            f"\n[mvp] sampling AFTER training: {len(sample_prompts)} prompts × "
            f"{args.sample_num_steps} steps × CFG={args.sample_cfg_scale} (quant={sample_quant})"
        )
        for i, prompt in enumerate(sample_prompts):
            print(f"[sample-after] {i}: {prompt[:80]}{'...' if len(prompt)>80 else ''}", flush=True)
            with torch.no_grad():
                img = model.t2i_generate(
                    tokenizer=tok,
                    prompt=prompt,
                    cfg_scale=args.sample_cfg_scale,
                    cfg_norm=args.sample_cfg_norm,
                    timestep_shift=args.sample_timestep_shift,
                    image_size=(args.sample_image_w, args.sample_image_h),
                    num_steps=args.sample_num_steps,
                    seed=args.sample_seed,
                )
            _save_t2i_image(img, out_root / "after" / f"{i:02d}.png")
        print(f"[mvp] sampled before/after PNGs saved under {out_root}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
