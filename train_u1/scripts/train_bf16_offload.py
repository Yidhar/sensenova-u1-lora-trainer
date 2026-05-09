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
import sys
import time
from pathlib import Path

import torch

from train_u1.config import TrainRunConfig, load_train_config
from train_u1.constants import MODEL_ID, MODEL_SHA
from train_u1.data.collators import CollatorConfig, SenseNovaU1Collator, to_device
from train_u1.data.datasets import PairedFolderT2IDataset
from train_u1.model.loader import _resolve_local_snapshot, load_neo_chat
from train_u1.model.losses import fm_loss_x0
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
    tokenizer,
    collator,
    *,
    cuda_device: str,
    cpu_device: str,
    classify: dict,
    verbose: bool = True,
) -> list:
    """Run prefix forward for each sample with ordinary tower on GPU.

    Returns list[ DynamicCache (with K/V tensors on `cpu_device`) ] aligned
    with `samples` order.
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

    kvs = []
    for i, sample in enumerate(samples):
        batch = collator([sample])
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
        kvs.append(kv)
        if verbose:
            print(
                f"[bf16-offload]   precomputed prefix KV for sample {i+1}/{len(samples)} "
                f"({sample.sample_id}, L_text={input_ids.shape[1]})",
                flush=True,
            )

    if verbose:
        print(f"[bf16-offload] precompute total: {time.time()-t0:.1f}s", flush=True)
    return kvs


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
    args = a  # rebind so the body below uses the resolved config

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

    # dataset + collator
    ds = PairedFolderT2IDataset(
        args.data_dir,
        cap_max_pixels=args.cap_max_pixels,
        snap_bucket=args.snap_bucket,
    )
    n_use = min(args.n_samples, len(ds))
    samples = [ds[i] for i in range(n_use)]
    collator = SenseNovaU1Collator(
        tok,
        cfg=CollatorConfig(
            image_hw=None,
            seed=args.seed,
            enforce_batch_one=True,
            prompt_template=args.prompt_template,
            style_trigger=args.style_trigger,
        ),
        model=model if args.prompt_template == "official" else None,
    )
    print(f"[bf16-offload] dataset: {len(ds)} pairs (using {n_use})  "
          f"snap_bucket={args.snap_bucket}  style_trigger={args.style_trigger!r}", flush=True)

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
    move_param_set(model, classify["unused"], args.cpu_device)
    move_param_set(model, classify["shared"], args.device)

    # ---- Phase 1: precompute prefix KVs (ordinary tower briefly on GPU) ----
    prefix_kvs_cpu = precompute_prefix_kvs(
        model, samples, tok, collator,
        cuda_device=args.device, cpu_device=args.cpu_device,
        classify=classify,
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
        for kv in prefix_kvs_cpu:
            _cache_to(kv, args.device)
        print(f"[bf16-offload] all {len(prefix_kvs_cpu)} prefix KVs kept on GPU")

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
    loss_tensors_buf: list[torch.Tensor] = []   # deferred .item() — flushed at log boundaries
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

        # Fetch this sample's pre-computed prefix KV
        kv = prefix_kvs_cpu[idx]
        if not args.keep_kvs_on_gpu:
            _cache_to(kv, args.device)

        out = wrapper.forward_t2i_step(batch, prefix_kv=kv)
        loss = fm_loss_x0(out.x_pred, batch["x0_patch"]) / args.grad_accum
        loss.backward()

        do_step = ((step + 1) % args.grad_accum) == 0
        if do_step:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        if not args.keep_kvs_on_gpu:
            _cache_to(kv, args.cpu_device)

        # Defer loss.item() (forces GPU→CPU sync) — keep tensor on GPU,
        # call .item() only at log-print boundaries. Tensor list is tiny.
        loss_tensors_buf.append(loss.detach() * args.grad_accum)  # un-scaled
        log_records_buf.append({
            "step": step, "sample_idx": idx, "sample_id": sample.sample_id,
            "token_h": int(token_h), "token_w": int(token_w),
        })

        is_log_boundary = (step < 5 or step % 10 == 0 or step == args.steps - 1)
        if is_log_boundary:
            # Sync materialize buffered losses (cheap; PyTorch batches the H2D)
            new_losses = [t.item() for t in loss_tensors_buf]
            losses.extend(new_losses)
            for rec, lv in zip(log_records_buf, new_losses):
                rec["loss"] = lv
                json_lines_buf.append(json.dumps(rec))
            loss_tensors_buf.clear()
            log_records_buf.clear()

            elapsed = time.time() - t0
            cur_h = batch['noisy_pixel_values'].shape[2]
            cur_w = batch['noisy_pixel_values'].shape[3]
            print(
                f"[bf16-offload] step={step:4d}  loss={losses[-1]:.4f}  "
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
