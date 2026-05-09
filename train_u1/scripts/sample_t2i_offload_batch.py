"""Batch bf16 offload sampler — load model once, iterate (state × prompt) pairs.

Motivation: post-hoc evaluation often needs N prompts × M checkpoints (e.g. 8
prompts × 5 last ckpts = 40 samples). Naive loop = 40 × (90s model load +
135s sample) ≈ 2h30m. This script amortizes by:
- Loading bf16 base ONCE, wrapping LoRA ONCE.
- For each ckpt: re-inject trainable state (~5s) into the same wrapped model.
- For each prompt under that ckpt: standard offload sample (Phase A swap →
  prefix forward → Phase B swap → 50-step Euler → save).

Total ≈ load_once (90s) + lora_wrap_once (10s) + N_ckpts × inject (5s each)
+ N_ckpts × N_prompts × sample (~135s each). For 5 × 8 = 40 it's ~92 min,
roughly 1.6× faster than naive single-prompt invocation.

Output layout: `{out_dir}/{ckpt_label}/{idx:02d}.png` where `ckpt_label` is
the basename of the safetensors file without extension (e.g. `step_006000`).

Usage:
    HF_HOME=/workspace/senesNovenove/hf_cache PYTHONPATH=. \\
    .venv/bin/python -m train_u1.scripts.sample_t2i_offload_batch \\
        --prompts-file artifacts/exp_d_held8_prompts.txt \\
        --buckets-file artifacts/exp_d_held8_buckets.txt \\
        --state-paths artifacts/exp_d_v15d_long/checkpoints/step_003600.safetensors,\\
                      artifacts/exp_d_v15d_long/checkpoints/step_004200.safetensors,... \\
        --out-dir artifacts/exp_d_v15d_long/sweep \\
        --lora-r 64 --lora-alpha 64 --lora-on-mlp --style-trigger "hayateluc style"
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import torch

from train_u1.constants import MODEL_ID, MODEL_SHA, SFT_MODEL_ID, SFT_MODEL_SHA
from train_u1.model.loader import _resolve_local_snapshot, load_neo_chat
from train_u1.scripts.sample_t2i_offload import (
    _save_image,
    classify_module_paths,
    move_param_set,
    t2i_generate_offload,
)


def _read_prompts(path: Path) -> list[str]:
    out: list[str] = []
    for line in path.read_text().splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            out.append(s)
    return out


def _read_buckets(path: Path | None, n: int) -> list[tuple[int, int]] | None:
    if path is None or not path.exists():
        return None
    pairs: list[tuple[int, int]] = []
    for line in path.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        h, w = s.split()
        pairs.append((int(h), int(w)))
    if len(pairs) != n:
        print(f"[warn] buckets file has {len(pairs)} entries but prompts has {n} — ignoring buckets")
        return None
    return pairs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default=os.environ.get("HF_HOME"))
    ap.add_argument("--prompts-file", required=True)
    ap.add_argument("--buckets-file", default=None,
                    help="Per-prompt `H W` lines; falls back to --image-h/--image-w.")
    ap.add_argument("--state-paths", default="",
                    help="Comma-separated list of trainable_state safetensors paths. "
                         "If empty, runs a single baseline pass (no LoRA inject) — "
                         "useful for evaluating untrained base on the same prompts.")
    ap.add_argument("--out-dir", required=True,
                    help="Root dir; outputs go to {out_dir}/{ckpt_label}/{idx:02d}.png")
    ap.add_argument("--image-h", type=int, default=2048)
    ap.add_argument("--image-w", type=int, default=2048)
    ap.add_argument("--num-steps", type=int, default=50)
    ap.add_argument("--cfg-scale", type=float, default=4.0)
    ap.add_argument("--timestep-shift", type=float, default=3.0)
    ap.add_argument("--cfg-norm", default="none",
                    choices=("none", "global", "channel", "cfg_zero_star"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--config", default=None,
                    help="Optional training-run YAML (used only to read lora.preset/spec + style.trigger).")
    ap.add_argument("--lora-preset", default=None)
    ap.add_argument("--lora-spec", default=None)
    # Legacy uniform-rank flags.
    ap.add_argument("--lora-r", type=int, default=0)
    ap.add_argument("--lora-alpha", type=float, default=32.0)
    ap.add_argument("--lora-on-mlp", action="store_true")
    ap.add_argument("--upstream-lora-path", default=None,
                    help="Bake-in merge an upstream-format LoRA before any sample.")
    ap.add_argument("--upstream-lora-skip", action="append", default=[],
                    help="Substring filter for upstream LoRA bake-in. Repeatable.")
    ap.add_argument("--style-trigger", default=None)
    ap.add_argument("--use-sft-base", action="store_true",
                    help="Load the public SFT release (sensenova/SenseNova-U1-8B-MoT-SFT) "
                         "instead of the post-RL release. For baseline comparison only.")
    ap.add_argument("--think-mode", action="store_true",
                    help="Enable chain-of-thought reasoning before image generation.")
    ap.add_argument("--think-max-tokens", type=int, default=1024)
    args = ap.parse_args()

    prompts = _read_prompts(Path(args.prompts_file))
    if not prompts:
        print(f"[batch] no prompts in {args.prompts_file}"); return 2
    buckets = _read_buckets(Path(args.buckets_file) if args.buckets_file else None, len(prompts))
    state_paths = [Path(p.strip()) for p in args.state_paths.split(",") if p.strip()]
    for sp in state_paths:
        if not sp.exists():
            print(f"[batch] missing state path: {sp}"); return 2
    baseline_mode = len(state_paths) == 0

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    if baseline_mode:
        print(f"[batch] BASELINE mode: {len(prompts)} prompts × untrained base (no state inject)")
    else:
        print(f"[batch] {len(prompts)} prompts × {len(state_paths)} ckpts = {len(prompts) * len(state_paths)} samples")
    print(f"[batch] out_root={out_root}")

    base_id = SFT_MODEL_ID if args.use_sft_base else MODEL_ID
    base_sha = SFT_MODEL_SHA if args.use_sft_base else MODEL_SHA
    print(f"[batch] loading bf16 model to CPU (entire 35 GB)... id={base_id} sha={base_sha[:8]}", flush=True)
    t0 = time.time()
    model = load_neo_chat(
        cache_dir=args.cache_dir,
        device_map="cpu",
        dtype=torch.bfloat16,
        model_id=base_id,
        revision=base_sha,
    )
    print(f"[batch] model load {time.time()-t0:.1f}s")

    # Resolve LoRA spec: --lora-spec > --lora-preset > config > legacy uniform-rank.
    lora_specs = []
    config_trigger = None
    if args.config:
        from train_u1.config import load_train_config
        cfg_obj = load_train_config(args.config)
        if not args.lora_spec and not args.lora_preset:
            lora_specs = cfg_obj.lora.resolved_specs()
        config_trigger = cfg_obj.style.trigger or None
    if args.lora_spec:
        from train_u1.model.lora import parse_lora_spec_str
        lora_specs = parse_lora_spec_str(args.lora_spec)
    elif args.lora_preset:
        from train_u1.model.lora import resolve_preset
        lora_specs = resolve_preset(args.lora_preset)
    elif args.lora_r > 0:
        from train_u1.model.lora import LoRASpec, ATTN_TARGETS, MLP_TARGETS
        lora_specs = [LoRASpec(target=t, r=args.lora_r, alpha=args.lora_alpha) for t in ATTN_TARGETS]
        if args.lora_on_mlp:
            lora_specs += [LoRASpec(target=t, r=args.lora_r, alpha=args.lora_alpha) for t in MLP_TARGETS]
    if args.style_trigger is None:
        args.style_trigger = config_trigger or ""

    if args.upstream_lora_path:
        from train_u1.model.lora_io import merge_upstream_lora
        merge_upstream_lora(
            model, args.upstream_lora_path,
            skip_targets=args.upstream_lora_skip,
        )

    if lora_specs:
        from train_u1.model.lora import apply_lora_specs
        report = apply_lora_specs(model, lora_specs)
        print(f"[batch] {report}")

    from transformers import AutoTokenizer
    local = _resolve_local_snapshot(args.cache_dir, base_id, base_sha)
    tok = AutoTokenizer.from_pretrained(
        local or base_id,
        revision=None if local else base_sha,
        trust_remote_code=True,
    )

    # Pre-compute module classification once (cheap; same across all samples).
    # We need it to move gen tower back to CPU after each sample, otherwise the
    # next sample's Phase A (move prefix → cuda) OOMs because the previous
    # sample's gen tower (16+ GB) still occupies the GPU.
    classify = classify_module_paths(model)

    # Baseline mode: synthesize a single sentinel "ckpt" path with label
    # "baseline" so the rest of the loop reads naturally.
    if baseline_mode:
        state_paths = [Path("baseline.safetensors")]  # virtual; never read

    for state_path in state_paths:
        ckpt_label = "baseline" if baseline_mode else state_path.stem  # e.g. step_006000
        ckpt_out_dir = out_root / ckpt_label
        ckpt_out_dir.mkdir(parents=True, exist_ok=True)

        if baseline_mode:
            print(f"\n===== {ckpt_label}  (untrained base, no inject) =====", flush=True)
        else:
            print(f"\n===== {ckpt_label}  state={state_path} =====", flush=True)
            t_inj = time.time()
            from train_u1.model.lora_io import load_lora_state
            load_lora_state(model, state_path)
            print(f"[batch] inject took {time.time()-t_inj:.1f}s")
            del loaded

        for i, prompt in enumerate(prompts):
            idx = f"{i:02d}"
            out_path = ckpt_out_dir / f"{idx}.png"
            if out_path.exists():
                print(f"[skip] {ckpt_label}/{idx}.png exists"); continue

            h_i, w_i = (buckets[i] if buckets else (args.image_h, args.image_w))
            effective_prompt = (
                f"{args.style_trigger}, {prompt}" if args.style_trigger else prompt
            )
            print(f"[{ckpt_label}-{idx}] {h_i}x{w_i}  {prompt[:60]}...", flush=True)
            t_smp = time.time()
            img = t2i_generate_offload(
                model, tok, effective_prompt,
                cfg_scale=args.cfg_scale,
                cfg_norm=args.cfg_norm,
                timestep_shift=args.timestep_shift,
                image_size=(w_i, h_i),
                num_steps=args.num_steps,
                seed=args.seed,
                think_mode=args.think_mode,
                think_max_tokens=args.think_max_tokens,
            )
            _save_image(img, out_path)
            print(f"[batch] saved {out_path}  in {time.time()-t_smp:.1f}s")
            # Critical: move gen tower (and any GPU-resident params) back to
            # CPU so the next sample's Phase A swap has GPU headroom for the
            # prefix tower. Without this, prompt-2 onward OOMs.
            move_param_set(model, classify["gen"], "cpu")
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

    print("\n===== BATCH DONE =====")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
