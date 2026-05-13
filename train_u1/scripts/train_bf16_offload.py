"""bf16 training with static prefix-KV cache + extreme offload.

Strategy:
1. Load full 35 GB bf16 model to CPU.
2. Phase 0 (one-time): move ordinary tower (17.45 GB) to GPU, precompute
   prefix KV for all training samples, move ordinary tower → CPU permanently.
3. Phase 1 (training loop): gen tower (16.37 GB) + fm_modules permanently on
   GPU. For each step, fetch the sample's pre-computed KV from CPU, run
   gen forward + backward + optimizer step, optionally move KV back to CPU.
4. Phase 2 (save): collect trainable_state.safetensors.

VRAM budget at 2048² + bf16:
  gen tower               16.37 GB
  fm_modules (trainable)   0.08 GB
  prefix KVs on GPU        0.56 GB (8 × 70 MB) or stream
  bnb 8-bit AdamW state    0.13 GB
  gradient buffers         0.13 GB
  GC activations + attn    ~3-4 GB peak
  intermediate buffers     ~0.5 GB
  ──────────────────────  ────
  total peak              ~21 GB    (fits in 32 GB)

Why static prefix cache works:
  Prefix tokens (system + chat + prompt + <think></think><img>) depend only
  on the sample's caption — fixed across all training visits to that sample.
  At training time we use 8 samples, each visited ~15× → 8 prefix forwards
  total instead of 120, saving 112 ordinary-tower swaps.

Trainable scenario: mvp_aux (fm_head + ts/ns_embed + mot_gen norms).
None of these are part of the ordinary tower, so it can stay on CPU forever.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
import time
from pathlib import Path

import torch

from train_u1.config import TrainRunConfig, load_train_config
from train_u1.constants import MODEL_ID, MODEL_SHA
from train_u1.data.collators import CollatorConfig, SenseNovaU1Collator, to_device
from train_u1.data.datasets import ArrowT2IDataset, PairedFolderT2IDataset
from train_u1.model.loader import _resolve_local_snapshot, load_neo_chat
from train_u1.model.losses import VALID_LOSS_TYPES, compute_v_target, fm_loss, fm_loss_x0, fm_loss_v
from train_u1.model.lora import (
    LORA_PRESETS,
    apply_lora_specs,
    lora_param_count,
    parse_lora_spec_str,
    resolve_preset,
)
from train_u1.model.lora_io import merge_upstream_lora, save_trainable_state
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
from train_u1.scripts.sample_t2i_offload import (
    _bytes_for,
    _cache_to,
    classify_module_paths,
    move_param_set,
)


def _trainable_params(model):
    return [p for p in model.parameters() if p.requires_grad]


def precompute_prefix_kvs(
    model,
    samples,
    collator,
    *,
    cuda_device: str,
    cpu_device: str,
    classify: dict,
    include_uncond: bool,
    verbose: bool = True,
) -> dict[str, object]:
    """Run prefix forward for conditional and optional unconditional CFG prefixes.

    Returns a dict with:
    - ``cond``: per-sample DynamicCache list aligned with ``samples`` order
    - ``uncond``: one shared unconditional DynamicCache when ``include_uncond``
      is true, otherwise ``None``
    """
    if verbose:
        sz_prefix = _bytes_for(model, classify["prefix"]) / 1e9
        print(f"[bf16-offload] precompute: moving prefix tower ({sz_prefix:.2f} GB) → {cuda_device}", flush=True)

    t0 = time.time()
    move_param_set(model, classify["prefix"], cuda_device)
    if verbose:
        print(
            f"[bf16-offload]   moved in {time.time()-t0:.1f}s, "
            f"GPU mem alloc={torch.cuda.memory_allocated()/1e9:.2f} GB",
            flush=True,
        )

    # Prefix precompute only needs text inputs; collator still builds full FM
    # batches, so preserve its training RNG streams around these calls.
    collator_gen_state = collator._gen.get_state()
    collator_cond_gen_state = collator._cond_gen.get_state()

    cond_kvs = []
    uncond_kv = None
    for i, sample in enumerate(samples):
        batch = collator([sample], condition_modes=["none"])
        # Force prefix-relevant inputs to GPU for the no_grad forward.
        input_ids = batch["input_ids"].to(cuda_device)
        text_indexes = batch["text_indexes"].to(cuda_device)
        attn_mask_prefix = batch["attn_mask_prefix"].to(cuda_device)

        with torch.no_grad():
            prefix_out = model.language_model.model(
                input_ids=input_ids,
                indexes=text_indexes,
                attention_mask={"full_attention": attn_mask_prefix},
                use_cache=True,
            )
        kv = prefix_out.past_key_values
        # Move KV to CPU (small — ~70 MB per sample at L_text~400)
        _cache_to(kv, cpu_device)
        cond_kvs.append(kv)
        if verbose:
            print(
                f"[bf16-offload]   precomputed prefix KV for sample {i+1}/{len(samples)} "
                f"({sample.sample_id}, L_text={input_ids.shape[1]})",
                flush=True,
            )

    if include_uncond and samples:
        # The unconditional CFG prefix is prompt-independent, so cache it once
        # and share it across all samples/steps.
        batch = collator([samples[0]], condition_modes=["text"])
        input_ids = batch["input_ids"].to(cuda_device)
        text_indexes = batch["text_indexes"].to(cuda_device)
        attn_mask_prefix = batch["attn_mask_prefix"].to(cuda_device)
        with torch.no_grad():
            prefix_out = model.language_model.model(
                input_ids=input_ids,
                indexes=text_indexes,
                attention_mask={"full_attention": attn_mask_prefix},
                use_cache=True,
            )
        uncond_kv = prefix_out.past_key_values
        _cache_to(uncond_kv, cpu_device)
        if verbose:
            print(
                f"[bf16-offload]   precomputed shared uncond prefix KV "
                f"(L_text={input_ids.shape[1]})",
                flush=True,
            )

    if verbose:
        print(f"[bf16-offload] precompute total: {time.time()-t0:.1f}s", flush=True)
    collator._gen.set_state(collator_gen_state)
    collator._cond_gen.set_state(collator_cond_gen_state)
    return {"cond": cond_kvs, "uncond": uncond_kv}


def _select_prefix_kv(prefix_kvs: dict[str, object], idx: int, prefix_key: str):
    if prefix_key == "cond":
        return prefix_kvs["cond"][idx]
    if prefix_key == "uncond":
        kv = prefix_kvs.get("uncond")
        if kv is None:
            raise RuntimeError("missing unconditional prefix KV; condition dropout requires include_uncond=True")
        return kv
    raise ValueError(f"unknown prefix_cache_key {prefix_key!r}")


def _guard_static_prefix_unfreeze(unfreeze_patterns: list[str], classify: dict[str, list[str]]) -> None:
    """Static prefix KVs make prefix/unused params effectively non-trainable."""
    if not unfreeze_patterns:
        return
    compiled = [(pat, re.compile(pat)) for pat in unfreeze_patterns]
    blocked_names = classify.get("prefix", []) + classify.get("unused", [])
    hits: list[tuple[str, str]] = []
    for pat, cre in compiled:
        for name in blocked_names:
            if cre.search(name):
                hits.append((pat, name))
                break
    if hits:
        details = "; ".join(f"{pat!r} matched {name}" for pat, name in hits[:5])
        raise ValueError(
            "unfreeze patterns cannot target prefix/unused tower params while static prefix KV cache is enabled; "
            f"{details}. Restrict unfreeze to gen-side modules or use a non-cached training path."
        )


def evict_ordinary_load_gen(model, classify, cuda_device, cpu_device, verbose=True):
    """Move ordinary tower → CPU permanently, gen tower → GPU permanently."""
    t0 = time.time()
    if verbose:
        sz_gen = _bytes_for(model, classify["gen"]) / 1e9
        print(f"[bf16-offload] swap: ordinary → CPU, gen ({sz_gen:.2f} GB) → {cuda_device}", flush=True)
    move_param_set(model, classify["prefix"], cpu_device)
    move_param_set(model, classify["unused"], cpu_device)  # already there but be explicit
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    move_param_set(model, classify["gen"], cuda_device)
    move_param_set(model, classify["shared"], cuda_device)
    if verbose:
        print(
            f"[bf16-offload]   swap done in {time.time()-t0:.1f}s, "
            f"GPU mem alloc={torch.cuda.memory_allocated()/1e9:.2f} GB",
            flush=True,
        )


def apply_gc_to_decoder_layers(model, skip_last: int = 0):
    """Wrap each decoder layer's forward with torch.utils.checkpoint, except
    the last `skip_last` layers (which run without GC for faster backward at
    the cost of more activation memory).

    Same monkey-patch as train_fm_mvp.py — upstream Qwen3Model doesn't honor
    `gradient_checkpointing_enable()` in its forward loop body.

    Returns (n_wrapped, n_skipped).
    """
    layers = model.language_model.model.layers
    n_total = len(layers)
    skip_last = max(0, min(skip_last, n_total))
    n_gc = n_total - skip_last

    from torch.utils.checkpoint import checkpoint as _ckpt
    n_wrapped = 0
    for i, layer in enumerate(layers):
        if i >= n_gc:
            continue  # last skip_last layers run without GC
        _orig = layer.forward
        def _make_gc_forward(orig):
            def gc_forward(hidden_states, **kw):
                def _fn(h):
                    return orig(h, **kw)
                return _ckpt(_fn, hidden_states, use_reentrant=False)
            return gc_forward
        layer.forward = _make_gc_forward(_orig)
        n_wrapped += 1
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    return n_wrapped, skip_last


def _build_trainable_regex(model, unfreeze_patterns: list[str]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Compose freeze/trainable regex patterns for the LoRA + unfreeze surface.

    Always trainable: every `*.lora_down.weight` / `*.lora_up.weight`.
    User-specified full-finetune: any param whose qualified name matches a
    pattern in `unfreeze_patterns` (regexes).
    Everything else: frozen.
    """
    freeze_pats = (r".*",)
    train_pats = [r"\.lora_down\.weight$", r"\.lora_up\.weight$"]
    train_pats.extend(unfreeze_patterns)
    return freeze_pats, tuple(train_pats)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap = argparse.ArgumentParser(
        description="bf16 LoRA / partial-FT trainer for SenseNova-U1-8B-MoT (config-driven).",
    )
    ap.add_argument("--config", default=None,
                    help="Path to a training-run YAML config (see configs/default.yaml). "
                         "All other CLI flags are optional overrides.")
    ap.add_argument("--cache-dir", default=os.environ.get("HF_HOME"))
    ap.add_argument("--device", default=None,
                    help="Override runtime.device.")
    ap.add_argument("--cpu-device", default=None,
                    help="Override runtime.cpu_device.")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--cap-max-pixels", type=int, default=None)
    ap.add_argument("--n-samples", type=int, default=None,
                    help="Cap on dataset size (default: use entire data_dir).")
    ap.add_argument("--use-think-labels", action="store_true", default=None,
                    help="Use embedded/sidecar think labels during training.")
    ap.add_argument("--no-use-think-labels", dest="use_think_labels",
                    action="store_false", default=None,
                    help="Ignore embedded/sidecar think labels during training.")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--grad-accum", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--shuffle", action="store_true", default=None)
    ap.add_argument("--no-shuffle", dest="shuffle", action="store_false", default=None)
    ap.add_argument("--prompt-template", default=None, choices=(None, "raw", "plain", "official"))
    ap.add_argument("--log-path", default=None)
    ap.add_argument("--save-trainable-state-to", default=None,
                    help="Final state path. Default: artifacts/{run_name}/trainable_state.safetensors")
    ap.add_argument("--run-name", default=None,
                    help="Override run_name from config.")
    # Legacy scenarios (kept for back-compat; new code should use --config + lora.preset).
    ap.add_argument(
        "--scenario", default=None,
        choices=("mvp", "mvp_aux", "gen_vision", "aux_no_head",
                 "lora_attn", "lora_attn_mlp",
                 "lora_attn_plus_embedders", "lora_attn_mlp_plus_embedders",
                 "lora_attn_mlp_plus_embedders_plus_vision_head",
                 "lora_attn_mlp_plus_embedders_plus_vision_head_plus_norms"),
        help="LEGACY: pre-config-era scenarios. Prefer --config or --lora-preset.",
    )
    # New per-spec LoRA controls.
    ap.add_argument("--lora-preset", default=None,
                    choices=tuple(LORA_PRESETS.keys()),
                    help=f"Named LoRA preset. Available: {list(LORA_PRESETS)}")
    ap.add_argument("--lora-spec", default=None,
                    help="Per-target LoRA spec, e.g. "
                         "'attn=r64a64;mlp=r64a64;fm_head=r64a64;mlp_mot_gen.down_proj=off'. "
                         "Overrides --lora-preset.")
    ap.add_argument("--lora-dropout", type=float, default=None)
    ap.add_argument("--unfreeze", action="append", default=None,
                    help="Regex pattern for full-finetune (non-LoRA). Repeatable.")
    ap.add_argument("--style-trigger", default=None,
                    help="Text prepended to every caption (must match sample-time trigger).")
    ap.add_argument("--snap-bucket", action="store_true", default=None,
                    help="Snap each training image to the nearest official bucket.")
    ap.add_argument("--no-snap-bucket", dest="snap_bucket", action="store_false", default=None)
    ap.add_argument("--checkpoint-every", type=int, default=None)
    ap.add_argument("--checkpoint-dir", default=None)
    ap.add_argument("--gc-skip-last", type=int, default=None,
                    help="Skip GC on the last N decoder layers (trades VRAM for backward speed).")
    ap.add_argument("--keep-kvs-on-gpu", action="store_true", default=None,
                    help="Keep all prefix KVs on GPU permanently.")
    ap.add_argument("--no-keep-kvs-on-gpu", dest="keep_kvs_on_gpu",
                    action="store_false", default=None)
    ap.add_argument("--upstream-lora-path", default=None,
                    help="Bake-in merge an upstream-format LoRA into the bf16 base before training. "
                         "Useful for stacking: train new style on top of the 8-step distill LoRA.")
    # ---- Official-consistency knobs (report Eq. (5) / Table 2). ----
    ap.add_argument("--loss-type", default=None,
                    choices=(None, *VALID_LOSS_TYPES),
                    help="FM loss objective: x0 (legacy) | v (official) | x0_huber | v_huber.")
    ap.add_argument("--huber-delta", type=float, default=None,
                    help="Delta for x0_huber / v_huber.")
    ap.add_argument("--t-dist", default=None,
                    choices=(None, "uniform", "logit_normal"),
                    help="t-sampling distribution. Official: logit_normal (mean=-0.8, std=0.8).")
    ap.add_argument("--t-logit-mean", type=float, default=None,
                    help="Mean for logit_normal t-sampling (default: -0.8 per report).")
    ap.add_argument("--t-logit-std", type=float, default=None,
                    help="Std for logit_normal t-sampling (default: 0.8 per report).")
    ap.add_argument("--cond-dropout-text", type=float, default=None,
                    help="Train-time CFG dropout probability for text-only condition drop. Official: 0.10.")
    ap.add_argument("--cond-dropout-both", type=float, default=None,
                    help="Additional unconditional bucket for text+image condition drop. Official: 0.10; "
                         "pure T2I maps this to the unconditional prompt prefix.")
    args = ap.parse_args()

    # ---- Resolve config: load YAML if given, then overlay CLI flags ----
    if args.config:
        cfg = load_train_config(args.config)
        print(f"[bf16-offload] loaded config: {args.config}", flush=True)
    else:
        cfg = TrainRunConfig()

    def _override(section, name: str, val):
        if val is not None:
            setattr(section, name, val)

    _override(cfg, "run_name", args.run_name)
    _override(cfg.data, "data_dir", args.data_dir)
    _override(cfg.data, "cap_max_pixels", args.cap_max_pixels)
    _override(cfg.data, "snap_bucket", args.snap_bucket)
    _override(cfg.data, "n_samples", args.n_samples)
    _override(cfg.data, "use_think_labels", args.use_think_labels)
    _override(cfg.style, "trigger", args.style_trigger)
    _override(cfg.style, "prompt_template", args.prompt_template)
    _override(cfg.lora, "preset", args.lora_preset)
    _override(cfg.lora, "spec", args.lora_spec)
    _override(cfg.lora, "dropout", args.lora_dropout)
    if args.unfreeze:
        cfg.unfreeze = list(args.unfreeze)
    _override(cfg.train, "steps", args.steps)
    _override(cfg.train, "lr", args.lr)
    _override(cfg.train, "seed", args.seed)
    _override(cfg.train, "shuffle", args.shuffle)
    _override(cfg.train, "grad_accum", args.grad_accum)
    _override(cfg.train, "checkpoint_every", args.checkpoint_every)
    _override(cfg.train, "checkpoint_dir", args.checkpoint_dir)
    _override(cfg.runtime, "keep_kvs_on_gpu", args.keep_kvs_on_gpu)
    _override(cfg.runtime, "gc_skip_last", args.gc_skip_last)
    _override(cfg.runtime, "device", args.device)
    _override(cfg.runtime, "cpu_device", args.cpu_device)
    _override(cfg.runtime, "upstream_lora_path", args.upstream_lora_path)
    _override(cfg.train, "loss_type", args.loss_type)
    _override(cfg.train, "huber_delta", args.huber_delta)
    _override(cfg.train, "t_dist", args.t_dist)
    _override(cfg.train, "t_logit_mean", args.t_logit_mean)
    _override(cfg.train, "t_logit_std", args.t_logit_std)
    _override(cfg.train, "cond_dropout_text", args.cond_dropout_text)
    _override(cfg.train, "cond_dropout_both", args.cond_dropout_both)

    # `data_dir` is the only truly required field.
    if not cfg.data.data_dir or cfg.data.data_dir == "dataset/my_style":
        if args.data_dir is None and not args.config:
            ap.error("Must specify --data-dir or --config <yaml>")

    # Default save path uses run_name.
    save_path = (args.save_trainable_state_to
                 or f"artifacts/{cfg.run_name}/trainable_state.safetensors")
    log_path = args.log_path or f"artifacts/{cfg.run_name}/train_log.jsonl"

    # Default device fallbacks
    if cfg.runtime.device == "cuda" and not torch.cuda.is_available():
        cfg.runtime.device = "cpu"
    device = cfg.runtime.device
    cpu_device = cfg.runtime.cpu_device

    # Compatibility shim: package the resolved config in an Args-like object so
    # the body below can reference args.steps etc. without an exhaustive rename.
    class _Args:
        pass

    a = _Args()
    a.cache_dir = args.cache_dir
    a.device = device
    a.cpu_device = cpu_device
    a.data_dir = cfg.data.data_dir
    a.cap_max_pixels = cfg.data.cap_max_pixels
    a.n_samples = cfg.data.n_samples
    a.use_think_labels = cfg.data.use_think_labels
    a.steps = cfg.train.steps
    a.lr = cfg.train.lr
    a.grad_accum = cfg.train.grad_accum
    a.seed = cfg.train.seed
    a.shuffle = cfg.train.shuffle
    a.prompt_template = cfg.style.prompt_template if cfg.style.prompt_template != "plain" else "raw"
    a.log_path = log_path
    a.save_trainable_state_to = save_path
    a.scenario = args.scenario
    a.style_trigger = cfg.style.trigger
    a.snap_bucket = cfg.data.snap_bucket
    a.checkpoint_every = cfg.train.checkpoint_every
    a.checkpoint_dir = cfg.train.checkpoint_dir or cfg.checkpoint_dir
    a.gc_skip_last = cfg.runtime.gc_skip_last
    a.keep_kvs_on_gpu = cfg.runtime.keep_kvs_on_gpu
    a.upstream_lora_path = cfg.runtime.upstream_lora_path
    a.lora_specs = cfg.lora.resolved_specs()
    a.unfreeze_patterns = list(cfg.unfreeze)
    a.loss_type = cfg.train.loss_type
    a.huber_delta = cfg.train.huber_delta
    a.t_dist = cfg.train.t_dist
    a.t_logit_mean = cfg.train.t_logit_mean
    a.t_logit_std = cfg.train.t_logit_std
    a.cond_dropout_text = cfg.train.cond_dropout_text
    a.cond_dropout_both = cfg.train.cond_dropout_both
    args = a  # rebind so the body below uses the resolved config

    if args.loss_type not in VALID_LOSS_TYPES:
        ap.error(f"loss_type must be one of {VALID_LOSS_TYPES}, got {args.loss_type!r}")
    if args.t_dist not in ("uniform", "logit_normal"):
        ap.error(f"t_dist must be 'uniform' or 'logit_normal', got {args.t_dist!r}")
    if args.cond_dropout_text < 0 or args.cond_dropout_both < 0:
        ap.error("condition dropout probabilities must be non-negative")
    if args.cond_dropout_text + args.cond_dropout_both > 1.0:
        ap.error("cond_dropout_text + cond_dropout_both must be <= 1.0")
    print(f"[bf16-offload] loss_type={args.loss_type}  t_dist={args.t_dist}"
          + (f" (mean={args.t_logit_mean}, std={args.t_logit_std})"
             if args.t_dist == "logit_normal" else ""), flush=True)
    print(f"[bf16-offload] cond_dropout: text={args.cond_dropout_text:.3f}  "
          f"both={args.cond_dropout_both:.3f}", flush=True)

    torch.manual_seed(args.seed)

    # ---- Phase 0: load bf16 model to CPU ----
    print("[bf16-offload] loading bf16 model to CPU (entire 35 GB)...", flush=True)
    t_load = time.time()
    model = load_neo_chat(
        cache_dir=args.cache_dir,
        device_map="cpu",
        dtype=torch.bfloat16,
    )
    print(f"[bf16-offload] loaded in {time.time()-t_load:.1f}s", flush=True)

    # tokenizer
    from transformers import AutoTokenizer
    local = _resolve_local_snapshot(args.cache_dir, MODEL_ID, MODEL_SHA)
    tok = AutoTokenizer.from_pretrained(
        local or MODEL_ID,
        revision=None if local else MODEL_SHA,
        trust_remote_code=True,
    )

    # dataset + collator. Auto-dispatch on data_dir suffix:
    #   .parquet     → ArrowT2IDataset (single shard or directory)
    #   anything else, including a `.parquet`-less directory → PairedFolderT2IDataset
    _data_path = Path(args.data_dir)
    if _data_path.suffix == ".parquet" or (_data_path.is_dir() and any(_data_path.glob("*.parquet"))):
        ds = ArrowT2IDataset(
            _data_path,
            cap_max_pixels=args.cap_max_pixels,
            snap_bucket=args.snap_bucket,
            use_think_labels=args.use_think_labels,
        )
    else:
        ds = PairedFolderT2IDataset(
            args.data_dir,
            cap_max_pixels=args.cap_max_pixels,
            snap_bucket=args.snap_bucket,
            use_think_labels=args.use_think_labels,
        )
    n_use = len(ds) if args.n_samples is None else min(args.n_samples, len(ds))
    samples = [ds[i] for i in range(n_use)]
    collator = SenseNovaU1Collator(
        tok,
        cfg=CollatorConfig(
            image_hw=None,
            seed=args.seed,
            enforce_batch_one=True,
            prompt_template=args.prompt_template,
            style_trigger=args.style_trigger,
            t_dist=args.t_dist,
            t_logit_mean=args.t_logit_mean,
            t_logit_std=args.t_logit_std,
            cond_dropout_text=args.cond_dropout_text,
            cond_dropout_both=args.cond_dropout_both,
        ),
        model=model if args.prompt_template == "official" else None,
    )
    print(f"[bf16-offload] dataset: {len(ds)} pairs (using {n_use})  "
          f"snap_bucket={args.snap_bucket}  use_think_labels={args.use_think_labels}  "
          f"style_trigger={args.style_trigger!r}", flush=True)

    # ---- Optional: bake-in merge an upstream-format LoRA into the base
    # before our wrap, so we train on top of (e.g.) the 8-step distill LoRA. ----
    if args.upstream_lora_path:
        merge_upstream_lora(model, args.upstream_lora_path)

    # ---- LoRA wrap (must happen BEFORE classify_module_paths so wrapped
    # `_mot_gen` lora_down/lora_up params are bucketed into the gen tower) ----
    if args.lora_specs:
        report = apply_lora_specs(model, args.lora_specs)
        print(f"[bf16-offload] {report}", flush=True)
    elif args.scenario and args.scenario.startswith("lora_"):
        # LEGACY scenario dispatch — translate to specs once for back-compat.
        from train_u1.model.lora import LoRASpec, ATTN_TARGETS, MLP_TARGETS

        legacy_specs: list[LoRASpec] = [LoRASpec(target=t, r=64, alpha=64)
                                        for t in ATTN_TARGETS]
        if "_mlp" in args.scenario:
            legacy_specs += [LoRASpec(target=t, r=64, alpha=64) for t in MLP_TARGETS]
        report = apply_lora_specs(model, legacy_specs)
        print(f"[bf16-offload] (legacy scenario) {report}", flush=True)
    else:
        report = None
        print(f"[bf16-offload] no LoRA wraps configured", flush=True)

    # tower classification
    classify = classify_module_paths(model)
    legacy_static_scenarios = {"mvp", "mvp_aux", "gen_vision", "aux_no_head"}
    if args.scenario not in legacy_static_scenarios:
        try:
            _guard_static_prefix_unfreeze(args.unfreeze_patterns, classify)
        except ValueError as e:
            ap.error(str(e))
    move_param_set(model, classify["unused"], args.cpu_device)
    move_param_set(model, classify["shared"], args.device)

    # ---- Phase 1: precompute prefix KVs (ordinary tower briefly on GPU) ----
    prefix_kvs_cpu = precompute_prefix_kvs(
        model, samples, collator,
        cuda_device=args.device, cpu_device=args.cpu_device,
        classify=classify,
        include_uncond=(args.cond_dropout_text + args.cond_dropout_both) > 0.0,
    )

    # ---- Phase 2: ordinary → CPU permanently, gen → GPU permanently ----
    evict_ordinary_load_gen(
        model, classify,
        cuda_device=args.device, cpu_device=args.cpu_device,
    )

    # ---- Phase 3: freeze policy ----
    # New world: trainable surface = all LoRA tensors + user `unfreeze` regexes.
    # Legacy: scenarios mvp / mvp_aux / gen_vision / aux_no_head still dispatch
    # to their preset regex pairs for backward compat.
    if args.scenario == "mvp":
        freeze_pats, train_pats = FREEZE_REGEX_MVP, TRAINABLE_REGEX_MVP
    elif args.scenario == "mvp_aux":
        freeze_pats, train_pats = FREEZE_REGEX_MVP, TRAINABLE_REGEX_MVP_AUX
    elif args.scenario == "gen_vision":
        freeze_pats, train_pats = FREEZE_REGEX_GEN_VISION, TRAINABLE_REGEX_GEN_VISION
    elif args.scenario == "aux_no_head":
        freeze_pats, train_pats = FREEZE_REGEX_AUX_NO_HEAD, TRAINABLE_REGEX_AUX_NO_HEAD
    else:
        freeze_pats, train_pats = _build_trainable_regex(model, args.unfreeze_patterns)

    rep = set_requires_grad_by_regex(
        model,
        freeze_patterns=freeze_pats,
        trainable_patterns=train_pats,
        default=False,
        strict=True,
    )
    print(f"[bf16-offload] trainable params: {rep.n_trainable:,}  "
          f"(LoRA={lora_param_count(model):,})")
    for k, v in sorted(rep.bucket_trainable.items(), key=lambda kv: -kv[1]):
        print(f"      {k:<24s} {v:>14,d}")

    # ---- Phase 4: gradient checkpointing on decoder layers ----
    n_wrapped, n_skipped = apply_gc_to_decoder_layers(model, skip_last=args.gc_skip_last)
    print(f"[bf16-offload] gradient_checkpointing wrapped on {n_wrapped} decoder layers, "
          f"{n_skipped} last layers run without GC")

    # If requested, move all prefix KVs to GPU permanently
    if args.keep_kvs_on_gpu:
        for kv in prefix_kvs_cpu["cond"]:
            _cache_to(kv, args.device)
        if prefix_kvs_cpu["uncond"] is not None:
            _cache_to(prefix_kvs_cpu["uncond"], args.device)
        n_prefix_kv = len(prefix_kvs_cpu["cond"]) + (1 if prefix_kvs_cpu["uncond"] is not None else 0)
        print(f"[bf16-offload] all {n_prefix_kv} prefix KVs kept on GPU")

    # ---- Phase 5: optimizer + training loop ----
    wrapper = TrainingWrapper(model)
    import bitsandbytes as bnb
    # PagedAdamW8bit empirically beats non-paged on this workload — bnb's
    # 8bit→bf16 dequant kernels seem to favor the paged code path (likely
    # better overlap of state load with compute). Measured: 2.21 s/step paged
    # vs 2.57 s/step non-paged on the same v16c config.
    try:
        optimizer = bnb.optim.PagedAdamW8bit(
            _trainable_params(model), lr=args.lr, betas=(0.9, 0.95),
        )
        print("[bf16-offload] optimizer = bnb.optim.PagedAdamW8bit", flush=True)
    except AttributeError:
        optimizer = bnb.optim.AdamW8bit(
            _trainable_params(model), lr=args.lr, betas=(0.9, 0.95),
        )
        print("[bf16-offload] optimizer = bnb.optim.AdamW8bit (paged variant unavailable)", flush=True)

    log_path = Path(args.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    print(f"\n[bf16-offload] starting training ({args.steps} steps, n_samples={n_use})", flush=True)
    t0 = time.time()
    losses: list[float] = []
    loss_tensors_buf: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    log_records_buf: list[dict] = []            # paired JSON record per buffered loss
    json_lines_buf: list[str] = []              # serialized JSON awaiting batched file write
    rng = torch.Generator().manual_seed(args.seed)

    # ---- Pre-compute index sequence so the prefetch thread can iterate
    # deterministically without sharing the main RNG ----
    indices = []
    for step in range(args.steps):
        if args.shuffle:
            idx = int(torch.randint(0, n_use, (1,), generator=rng).item())
        else:
            idx = step % n_use
        indices.append(idx)

    # ---- CPU-side prefetch worker: overlaps collator + H2D transfer with GPU
    # forward+backward of previous step. Without prefetch, GPU drops to ~0%
    # util / 90W between steps while CPU rebuilds the batch (collator is
    # 100-200ms per call) and synchronously copies tensors to GPU (~10ms more).
    # Worker pushes GPU-resident batches into the queue (uses a dedicated CUDA
    # stream + non_blocking H2D so H2D overlaps with main-thread compute). ----
    import threading
    import queue as _queue

    prefetch_q: "_queue.Queue[tuple[int, dict] | None]" = _queue.Queue(maxsize=2)
    stop_flag = threading.Event()
    h2d_stream = torch.cuda.Stream() if torch.cuda.is_available() else None

    def _prefetch_worker():
        for prefetch_step, prefetch_idx in enumerate(indices):
            if stop_flag.is_set():
                break
            try:
                sample = samples[prefetch_idx]
                batch = collator([sample])
                # Issue H2D transfer on a dedicated stream so it overlaps with
                # the main thread's current compute. The main thread will
                # implicitly wait via the default-stream barrier when it first
                # touches a tensor (PyTorch records the producing stream).
                if h2d_stream is not None:
                    with torch.cuda.stream(h2d_stream):
                        batch = to_device(batch, args.device, dtype=torch.bfloat16)
                else:
                    batch = to_device(batch, args.device, dtype=torch.bfloat16)
                prefetch_q.put((prefetch_idx, batch), timeout=300)
            except Exception as e:
                print(f"[bf16-offload] prefetch error at step {prefetch_step}: {e}", flush=True)
                stop_flag.set()
                break
        prefetch_q.put(None)  # sentinel

    prefetch_thread = threading.Thread(target=_prefetch_worker, daemon=True)
    prefetch_thread.start()

    for step in range(args.steps):
        item = prefetch_q.get()
        if item is None:
            print(f"[bf16-offload] unexpected end of prefetch queue at step {step}", flush=True)
            break
        idx, batch = item
        # Sync the producing stream with the default compute stream so any
        # downstream forward sees a fully-arrived batch on GPU.
        if h2d_stream is not None:
            torch.cuda.current_stream().wait_stream(h2d_stream)
        sample = samples[idx]
        token_h, token_w = batch["token_hw"]

        # Fetch this sample's pre-computed prefix KV. Condition dropout uses a
        # shared unconditional prefix; normal samples use their per-sample cond KV.
        prefix_key = batch.get("prefix_cache_key", ["cond"])[0]
        cond_drop_mode = batch.get("cond_drop_mode", ["none"])[0]
        kv = _select_prefix_kv(prefix_kvs_cpu, idx, prefix_key)
        if not args.keep_kvs_on_gpu:
            _cache_to(kv, args.device)

        out = wrapper.forward_t2i_step(batch, prefix_kv=kv)
        # v target for both training and diagnostics. Cheap (one /).
        v_target = compute_v_target(batch["x0_patch"], out.z_t, batch["t"], t_eps=wrapper.t_eps)
        loss = fm_loss(
            loss_type=args.loss_type,
            x_pred=out.x_pred,
            x0_patch=batch["x0_patch"],
            v_pred=out.v_pred,
            v_target=v_target,
            huber_delta=args.huber_delta,
        ) / args.grad_accum
        loss.backward()
        # Diagnostic: always compute the *other* MSE (no grad) so the log
        # carries both `x0_mse` and `v_mse` regardless of which one we train.
        with torch.no_grad():
            x0_mse_t = fm_loss_x0(out.x_pred, batch["x0_patch"]).detach()
            v_mse_t = fm_loss_v(out.v_pred, v_target).detach()
            t_vec = batch["t"].detach()

        do_step = ((step + 1) % args.grad_accum) == 0
        if do_step:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        if not args.keep_kvs_on_gpu:
            _cache_to(kv, args.cpu_device)

        # Defer loss.item() (forces GPU→CPU sync) — keep tensor on GPU,
        # call .item() only at log-print boundaries. Tensor list is tiny.
        loss_tensors_buf.append((
            loss.detach() * args.grad_accum,   # un-scaled active loss
            x0_mse_t, v_mse_t, t_vec,
        ))
        log_records_buf.append({
            "step": step, "sample_idx": idx, "sample_id": sample.sample_id,
            "token_h": int(token_h), "token_w": int(token_w),
            "cond_drop_mode": cond_drop_mode,
            "prefix_cache_key": prefix_key,
        })

        is_log_boundary = (step < 5 or step % 10 == 0 or step == args.steps - 1)
        if is_log_boundary:
            # Sync materialize buffered losses + diagnostics in one shot.
            new_losses: list[float] = []
            new_x0: list[float] = []
            new_v: list[float] = []
            new_t_stats: list[tuple[float, float, float, float]] = []
            for loss_t, x0_t, vt, t_vec_t in loss_tensors_buf:
                new_losses.append(loss_t.item())
                new_x0.append(x0_t.item())
                new_v.append(vt.item())
                tv = t_vec_t.float()
                new_t_stats.append((
                    tv.mean().item(),
                    tv.std(unbiased=False).item() if tv.numel() > 1 else 0.0,
                    tv.min().item(),
                    tv.max().item(),
                ))
            losses.extend(new_losses)
            for rec, lv, x0v, vv, tstat in zip(
                log_records_buf, new_losses, new_x0, new_v, new_t_stats
            ):
                rec["loss"] = lv
                rec["x0_mse"] = x0v
                rec["v_mse"] = vv
                rec["t_mean"], rec["t_std"], rec["t_min"], rec["t_max"] = tstat
                json_lines_buf.append(json.dumps(rec))
            loss_tensors_buf.clear()
            log_records_buf.clear()

            elapsed = time.time() - t0
            cur_h = batch['noisy_pixel_values'].shape[2]
            cur_w = batch['noisy_pixel_values'].shape[3]
            print(
                f"[bf16-offload] step={step:4d}  loss={losses[-1]:.4f}  "
                f"x0={new_x0[-1]:.4f}  v={new_v[-1]:.4f}  "
                f"t̄={new_t_stats[-1][0]:.3f}  "
                f"sample={sample.sample_id}  hw=({cur_h},{cur_w}) tokens={token_h*token_w}  "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )

        # Buffered JSON write — flush every 50 steps or at end of run.
        if len(json_lines_buf) >= 50 or step == args.steps - 1:
            with open(log_path, "a") as f:
                f.write("\n".join(json_lines_buf))
                if json_lines_buf:
                    f.write("\n")
            json_lines_buf.clear()

        # Periodic checkpoint
        if (
            args.checkpoint_every > 0
            and (step + 1) % args.checkpoint_every == 0
            and do_step
        ):
            ckpt_dir = Path(args.checkpoint_dir or args.save_trainable_state_to or ".")
            if ckpt_dir.suffix == ".safetensors":
                ckpt_dir = ckpt_dir.parent / "checkpoints"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = ckpt_dir / f"step_{step+1:06d}.safetensors"
            save_trainable_state(model, ckpt_path,
                                 extra_metadata={"step": str(step + 1)})
            print(f"[bf16-offload] checkpoint saved → {ckpt_path}", flush=True)

    stop_flag.set()
    # Drain the queue so worker doesn't block on put.
    while True:
        try:
            prefetch_q.get_nowait()
        except _queue.Empty:
            break
    prefetch_thread.join(timeout=5)

    elapsed = time.time() - t0
    peak_vram = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    print(
        f"\n[bf16-offload] training done in {elapsed:.1f}s ({elapsed/args.steps:.2f}s/step)  "
        f"peak_vram={peak_vram:.2f} GB"
    )
    if len(losses) >= 10:
        first = sum(losses[:5]) / 5
        last = sum(losses[-5:]) / 5
        print(f"[bf16-offload] mean(loss[:5])={first:.4f}  mean(loss[-5:])={last:.4f}  ratio={last/first:.3f}")

    # ---- Phase 6: save trainable state (upstream format: lora_down/lora_up/.alpha) ----
    state_path = Path(args.save_trainable_state_to)
    state = save_trainable_state(model, state_path)
    print(f"\n[bf16-offload] saved {len(state)} tensors → {state_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
