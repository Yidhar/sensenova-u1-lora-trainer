"""bf16 sampling with phase-aware tower offload.

Motivation: U1's t2i sampling has two non-overlapping compute phases:
- prefix forward (1×, uses ORDINARY tower via forward_und)
- gen forward (50×, uses `_mot_gen` tower via forward_gen)

Both towers are ~16 GB at bf16 → 32 GB GPU can host only one at a time.
This script keeps the full bf16 model resident on CPU (~944 GB RAM available
on the dev box) and shuttles whichever tower is currently active to GPU.

Engineering bound (per sample at 2048²):
- 2 PCIe transfers of ~16 GB each (~3-5 sec each on PCIe 4.0 x16)
- KV cache CPU↔GPU roundtrip: ~70 MB (negligible)
- Total overhead: ~10 sec on top of ~2-3 min sampling time

Usage:
    HF_HOME=/workspace/senesNovenove/hf_cache PYTHONPATH=. \\
    .venv/bin/python -m train_u1.scripts.sample_t2i_offload \\
        --prompt "..." --image-h 2048 --image-w 2048 \\
        --num-steps 50 --cfg-scale 4.0 --timestep-shift 3.0 --cfg-norm none \\
        --seed 42 --out artifacts/baseline_2048_apple_bf16.png

Reuses the upstream `model.t2i_generate(...)` semantics by reimplementing
the body with manual offload between phases. The actual computation
(per-layer forward, RoPE, attention, fm_head projection, Euler step) is
the SAME public code path as `t2i_generate`; only weight residency policy
differs.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import torch

from train_u1.constants import (
    MODEL_ID,
    MODEL_SHA,
    NOISE_SCALE_BASE_IMAGE_SEQ_LEN,
    NOISE_SCALE_MAX,
    PATCH_SIZE,
)
from train_u1.model.loader import _resolve_local_snapshot, load_neo_chat


# --------------------------------------------------------------------------- #
# Module classification                                                       #
# --------------------------------------------------------------------------- #


def classify_module_paths(model: torch.nn.Module) -> dict[str, list[str]]:
    """Bucket *parameter* names by the phase that consumes them.

    Phases:
      `prefix` → moved to GPU before _t2i_prefix_forward, back to CPU after
      `gen`    → moved to GPU before per-step gen, stays during 50 steps
      `shared` → on GPU permanently (small modules used by both phases:
                 rotary_emb buffers etc.)
      `unused` → never touched by t2i (vision_model, lm_head, embed_tokens
                 unless prefix uses it)
    """
    prefix: list[str] = []
    gen: list[str] = []
    shared: list[str] = []
    unused: list[str] = []

    for name, _ in model.named_parameters():
        if name.startswith("vision_model."):
            unused.append(name); continue
        if name.startswith("language_model.lm_head"):
            unused.append(name); continue
        if name.startswith("language_model.model.embed_tokens"):
            # text prefix uses embed_tokens to look up token embeddings
            prefix.append(name); continue
        if name.startswith("fm_modules."):
            # fm_head + ts/ns embedders + vision_model_mot_gen — gen phase
            gen.append(name); continue
        # language_model.model.layers.X.* and language_model.model.norm{,_mot_gen}
        if name.endswith("model.norm.weight"):
            prefix.append(name); continue
        if name.endswith("model.norm_mot_gen.weight"):
            gen.append(name); continue
        if "_mot_gen" in name:
            gen.append(name); continue
        # everything else under language_model.model.layers.X is ordinary tower
        prefix.append(name)

    # Buffers: rotary_emb has a (small) inv_freq buffer per attention module.
    # Keep them shared on GPU permanently.
    for name, _ in model.named_buffers():
        if "rotary_emb" in name:
            shared.append(name)
        # Other buffers tied to specific tower follow that tower; cheap so
        # we can also keep them shared if small.

    return {"prefix": prefix, "gen": gen, "shared": shared, "unused": unused}


def _set_param_device(model: torch.nn.Module, name: str, device) -> None:
    parts = name.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    leaf = parts[-1]
    obj = getattr(parent, leaf)
    if isinstance(obj, torch.nn.Parameter):
        obj.data = obj.data.to(device, non_blocking=True)
    elif isinstance(obj, torch.Tensor):
        setattr(parent, leaf, obj.to(device, non_blocking=True))


def move_param_set(model: torch.nn.Module, names: list[str], device) -> None:
    """Move a batch of named parameters/buffers to `device` in place."""
    for n in names:
        _set_param_device(model, n, device)


def _bytes_for(model: torch.nn.Module, names: list[str]) -> int:
    total = 0
    for n in names:
        parts = n.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        obj = getattr(parent, parts[-1])
        if isinstance(obj, torch.Tensor):
            total += obj.numel() * obj.element_size()
    return total


# --------------------------------------------------------------------------- #
# KV cache helpers                                                            #
# --------------------------------------------------------------------------- #


def _cache_to(cache, device) -> None:
    """Move a transformers Cache (DynamicCache) tensors to `device` in place."""
    if cache is None:
        return
    for layer in getattr(cache, "layers", []):
        if getattr(layer, "keys", None) is not None:
            layer.keys = layer.keys.to(device, non_blocking=True)
        if getattr(layer, "values", None) is not None:
            layer.values = layer.values.to(device, non_blocking=True)


# --------------------------------------------------------------------------- #
# Reimplementation of t2i_generate body with phase-aware offload              #
# --------------------------------------------------------------------------- #


def t2i_generate_offload(
    model,
    tokenizer,
    prompt: str,
    *,
    cfg_scale: float = 4.0,
    cfg_norm: str = "none",
    timestep_shift: float = 3.0,
    enable_timestep_shift: bool = True,
    image_size: tuple[int, int] = (2048, 2048),
    num_steps: int = 50,
    cfg_interval: tuple[float, float] = (0.0, 1.0),
    seed: int = 42,
    cuda_device: str = "cuda",
    cpu_device: str = "cpu",
    verbose: bool = True,
    think_mode: bool = False,
    think_max_tokens: int = 1024,
) -> torch.Tensor:
    """Mirror of `model.t2i_generate` (single-batch, no think_mode) with manual
    weight residency control between prefix and gen phases.
    """
    import time

    classify = classify_module_paths(model)
    if verbose:
        sz_prefix = _bytes_for(model, classify["prefix"]) / 1e9
        sz_gen = _bytes_for(model, classify["gen"]) / 1e9
        sz_shared = _bytes_for(model, classify["shared"]) / 1e9
        sz_unused = _bytes_for(model, classify["unused"]) / 1e9
        print(
            f"[offload] tower sizes (bf16):  prefix={sz_prefix:.2f} GB  "
            f"gen={sz_gen:.2f} GB  shared={sz_shared:.2f} GB  unused={sz_unused:.2f} GB"
        )

    # Unused stays on CPU forever; shared lives on GPU forever.
    move_param_set(model, classify["unused"], cpu_device)
    move_param_set(model, classify["shared"], cuda_device)

    # ---- Phase A: prefix forward at GPU (ordinary tower hot) ----
    t0 = time.time()
    if verbose:
        print(f"[offload] Phase A: moving prefix tower → {cuda_device}", flush=True)
    move_param_set(model, classify["prefix"], cuda_device)
    if verbose:
        print(
            f"[offload]   moved in {time.time()-t0:.1f}s, "
            f"GPU mem alloc={torch.cuda.memory_allocated()/1e9:.1f} GB",
            flush=True,
        )

    # Build cond / uncond queries via _build_t2i_query.
    # think_mode=False → cond ends at `<think>\n\n</think>\n\n<img>` (closed think).
    # think_mode=True  → cond ends at `<think>\n` (open think); model autoregressively
    #                    generates think text + closing `</think>` + `\n\n<img>`.
    from importlib import import_module
    _utils = import_module(f"{model.__class__.__module__.rsplit('.', 1)[0]}.utils")
    SYSTEM_MESSAGE_FOR_GEN = _utils.SYSTEM_MESSAGE_FOR_GEN
    IMG_START_TOKEN = "<img>"
    if think_mode:
        think_content = "<think>\n"  # open — autoregressive will fill
        # When think_mode, lm_head must be on GPU for the autoregressive loop.
        # Move it from "unused" → cuda for the duration of Phase A; will move
        # back to CPU before Phase B starts.
        lm_head_names = [n for n in classify["unused"] if "lm_head" in n]
        move_param_set(model, lm_head_names, cuda_device)
        if verbose:
            print(f"[offload]   think_mode: moved {len(lm_head_names)} lm_head params to {cuda_device}", flush=True)
    else:
        think_content = f"<think>\n\n</think>\n\n{IMG_START_TOKEN}"
        lm_head_names = []

    cond_query = model._build_t2i_query(
        prompt, system_message=SYSTEM_MESSAGE_FOR_GEN, append_text=think_content
    )
    uncond_query = model._build_t2i_query("", append_text=IMG_START_TOKEN)

    # `_build_t2i_text_inputs` lands input_ids on `model.device`, which is
    # ambiguous when params are split across CPU/CUDA. Force everything to
    # cuda_device before calling prefix forward.
    def _to_cuda_inputs(ids, idx, attn):
        ids = ids.to(cuda_device)
        idx = idx.to(cuda_device)
        if isinstance(attn, dict):
            attn = {k: (v.to(cuda_device) if torch.is_tensor(v) else v) for k, v in attn.items()}
        elif torch.is_tensor(attn):
            attn = attn.to(cuda_device)
        return ids, idx, attn

    cond_ids, cond_idx, cond_attn = model._build_t2i_text_inputs(tokenizer, cond_query)
    uncond_ids, uncond_idx, uncond_attn = model._build_t2i_text_inputs(tokenizer, uncond_query)
    cond_ids, cond_idx, cond_attn = _to_cuda_inputs(cond_ids, cond_idx, cond_attn)
    uncond_ids, uncond_idx, uncond_attn = _to_cuda_inputs(uncond_ids, uncond_idx, uncond_attn)

    think_text = ""
    cond_t_idx_extra = 0  # how many extra tokens were appended after prefix in think mode
    with torch.no_grad():
        if think_mode:
            # Reimplemented `_generate_think` body with explicit cuda_device.
            # Upstream's version uses `self.device` for tensor placement, which
            # under our offload setup returns CPU (vision_model is first param,
            # still on CPU) — that mismatches embed_tokens which we already
            # moved to cuda. So we run the autoregressive loop directly here.
            cond_outputs = model.language_model(
                input_ids=cond_ids,
                indexes=cond_idx,
                attention_mask=cond_attn,
                use_cache=True,
                output_hidden_states=False,
            )
            cond_kv = cond_outputs.past_key_values
            cond_t_idx = int(cond_idx[0].max().item())
            t_idx_local = cond_t_idx

            from importlib import import_module as _imp
            _conv = _imp(f"{model.__class__.__module__.rsplit('.', 1)[0]}.conversation")
            template = _conv.get_conv_template(model.template)
            eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())
            think_end_token_id = tokenizer.convert_tokens_to_ids("</think>")

            think_token_ids = []
            next_token = torch.argmax(cond_outputs.logits[:, -1, :], dim=-1)
            for _ in range(think_max_tokens):
                token_item = int(next_token.item())
                if token_item == eos_token_id:
                    break
                hit_end = (token_item == think_end_token_id)
                think_token_ids.append(token_item)
                model.language_model.model.current_index = t_idx_local
                outputs = model.language_model(
                    input_ids=next_token.unsqueeze(0).to(cuda_device),
                    past_key_values=cond_kv,
                    use_cache=True,
                )
                cond_kv = outputs.past_key_values
                t_idx_local += 1
                if hit_end:
                    break
                next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1)

            # Append "\n\n<img>" tokens to KV (matches upstream end of _generate_think).
            append_ids = tokenizer(
                "\n\n" + IMG_START_TOKEN,
                return_tensors="pt",
                add_special_tokens=False,
            )["input_ids"].to(cuda_device)
            t_idx_local = model._append_text_tokens_to_cache(cond_kv, t_idx_local, append_ids)

            cond_t_idx_extra = t_idx_local - cond_t_idx
            think_text = tokenizer.decode(think_token_ids, skip_special_tokens=False)
            if verbose:
                tt = think_text.strip().replace("\n", " ")
                print(f"[offload]   think generated +{cond_t_idx_extra} tokens (full): {tt!r}", flush=True)
        else:
            cond_kv, _ = model._t2i_prefix_forward(cond_ids, cond_idx, cond_attn)
        uncond_kv, _ = model._t2i_prefix_forward(uncond_ids, uncond_idx, uncond_attn)
    if verbose:
        print(f"[offload]   prefix forward done in {time.time()-t0:.1f}s", flush=True)

    # ---- Tower swap: prefix → CPU, KV → CPU, gen → GPU ----
    t1 = time.time()
    _cache_to(cond_kv, cpu_device)
    _cache_to(uncond_kv, cpu_device)
    move_param_set(model, classify["prefix"], cpu_device)
    if lm_head_names:
        move_param_set(model, lm_head_names, cpu_device)
    torch.cuda.empty_cache()
    move_param_set(model, classify["gen"], cuda_device)
    _cache_to(cond_kv, cuda_device)
    _cache_to(uncond_kv, cuda_device)
    if verbose:
        print(
            f"[offload] Phase B prep: tower swap done in {time.time()-t1:.1f}s, "
            f"GPU mem alloc={torch.cuda.memory_allocated()/1e9:.1f} GB",
            flush=True,
        )

    # ---- Phase B: 50-step Euler loop using `_mot_gen` tower ----
    # Closely mirrors `t2i_generate` L1656+. We replicate just enough state
    # to call `_t2i_predict_v` per step and apply Euler updates.
    merge_size = int(1 / model.downsample_ratio)  # = 2

    grid_h = image_size[1] // model.patch_size
    grid_w = image_size[0] // model.patch_size
    token_h = grid_h // merge_size
    token_w = grid_w // merge_size
    image_token_num = token_h * token_w
    grid_hw = torch.tensor([[grid_h, grid_w]], device=cuda_device)

    # Resolution-dependent noise scale (matches L1656).
    if model.noise_scale_mode in ("resolution", "dynamic", "dynamic_sqrt"):
        scale = math.sqrt((grid_h * grid_w) / (merge_size**2) / float(NOISE_SCALE_BASE_IMAGE_SEQ_LEN))
        noise_scale = scale * float(model.noise_scale)
        if model.noise_scale_mode == "dynamic_sqrt":
            noise_scale = math.sqrt(noise_scale)
    else:
        noise_scale = float(model.noise_scale)
    noise_scale = min(noise_scale, NOISE_SCALE_MAX)

    generator = torch.Generator(cuda_device).manual_seed(seed)
    dtype = next(model.fm_modules["fm_head"].parameters()).dtype
    image_prediction = noise_scale * torch.randn(
        (1, 3, image_size[1], image_size[0]),
        device=cuda_device, dtype=dtype, generator=generator,
    )

    # Image-span indexes (per-cond/per-uncond text length differs slightly).
    # In think_mode, cond's effective text length grew by `cond_t_idx_extra`
    # (the autoregressively-generated think tokens + appended `\n\n<img>`).
    indexes_image_cond = model._build_t2i_image_indexes(
        token_h, token_w, cond_ids.shape[1] + cond_t_idx_extra, device=cuda_device,
    )
    indexes_image_uncond = model._build_t2i_image_indexes(
        token_h, token_w, uncond_ids.shape[1], device=cuda_device,
    )
    attention_mask_cond = {"full_attention": None}
    attention_mask_uncond = {"full_attention": None}

    timesteps = torch.linspace(0.0, 1.0, num_steps + 1, device=cuda_device)
    if enable_timestep_shift:
        timesteps = model._apply_time_schedule(timesteps, image_token_num, timestep_shift)

    # Optional flash-cache prep (matches L1640).
    try:
        from importlib import import_module
        _qwen3 = import_module(f"{model.__class__.__module__.rsplit('.', 1)[0]}.modeling_qwen3")
        prepare_flash_kv_cache = getattr(_qwen3, "prepare_flash_kv_cache", None)
        if prepare_flash_kv_cache is not None:
            prepare_flash_kv_cache(cond_kv, current_len=image_token_num, batch_size=1)
            prepare_flash_kv_cache(uncond_kv, current_len=image_token_num, batch_size=1)
    except Exception as e:
        if verbose:
            print(f"[offload]   prepare_flash_kv_cache skipped: {e}", flush=True)

    if verbose:
        print(
            f"[offload] Phase B: {num_steps}-step Euler @ {image_size[1]}×{image_size[0]} "
            f"(noise_scale={noise_scale:.3f}, image_token_num={image_token_num})",
            flush=True,
        )

    t2 = time.time()
    with torch.no_grad():
        for step_i in range(num_steps):
            t = timesteps[step_i]
            t_next = timesteps[step_i + 1]

            z = model.patchify(image_prediction, model.patch_size * merge_size)
            image_input = model.patchify(image_prediction, model.patch_size, channel_first=True)
            image_embeds = model.extract_feature(
                image_input.view(1 * grid_h * grid_w, -1),
                gen_model=True, grid_hw=grid_hw,
            ).view(1, image_token_num, -1)

            t_expanded = t.expand(1 * image_token_num)
            timestep_embeddings = model.fm_modules["timestep_embedder"](t_expanded).view(1, image_token_num, -1)
            if model.add_noise_scale_embedding:
                noise_scale_tensor = torch.full_like(t_expanded, noise_scale / NOISE_SCALE_MAX)
                noise_embeddings = model.fm_modules["noise_scale_embedder"](noise_scale_tensor).view(1, image_token_num, -1)
                timestep_embeddings = timestep_embeddings + noise_embeddings
            image_embeds = image_embeds + timestep_embeddings

            v_cond = model._t2i_predict_v(
                image_embeds, indexes_image_cond, attention_mask_cond,
                cond_kv, t, z, image_token_num=image_token_num,
                timestep_embeddings=timestep_embeddings, image_size=image_size,
            )

            use_cfg = (t > cfg_interval[0] and t < cfg_interval[1]) or cfg_interval[0] == 0
            if use_cfg and cfg_scale != 1.0:
                v_uncond = model._t2i_predict_v(
                    image_embeds, indexes_image_uncond, attention_mask_uncond,
                    uncond_kv, t, z, image_token_num=image_token_num,
                    timestep_embeddings=timestep_embeddings, image_size=image_size,
                )
                # Mirror upstream `t2i_generate` L1693-1718 cfg_norm branches
                # exactly. Previously this offload script only implemented the
                # `none` form even though CLI exposed `cfg_zero_star`.
                if cfg_norm == "cfg_zero_star":
                    # Pull `optimized_scale` from upstream so dot-product /
                    # squared-norm is bit-faithful.
                    from importlib import import_module
                    _mnc = import_module(
                        f"{model.__class__.__module__.rsplit('.', 1)[0]}.modeling_neo_chat"
                    )
                    optimized_scale = _mnc.optimized_scale
                    batch_size = v_cond.shape[0]
                    positive_flat = v_cond.reshape(batch_size, -1)
                    negative_flat = v_uncond.reshape(batch_size, -1)
                    alpha = optimized_scale(positive_flat, negative_flat)
                    alpha = alpha.view(batch_size, *([1] * (v_cond.dim() - 1)))
                    alpha = alpha.to(positive_flat.dtype)
                    if step_i <= 0:
                        v_pred = v_cond * 0.0
                    else:
                        v_pred = v_uncond * alpha + cfg_scale * (v_cond - v_uncond * alpha)
                else:
                    v_pred = v_uncond + cfg_scale * (v_cond - v_uncond)
                    if cfg_norm == "global":
                        norm_v_cond = torch.norm(v_cond, dim=(1, 2), keepdim=True)
                        norm_v_cfg = torch.norm(v_pred, dim=(1, 2), keepdim=True)
                        scale_n = (norm_v_cond / (norm_v_cfg + 1e-8)).clamp(min=0, max=1.0)
                        v_pred = v_pred * scale_n
                    elif cfg_norm == "channel":
                        norm_v_cond = torch.norm(v_cond, dim=-1, keepdim=True)
                        norm_v_cfg = torch.norm(v_pred, dim=-1, keepdim=True)
                        scale_n = (norm_v_cond / (norm_v_cfg + 1e-8)).clamp(min=0, max=1.0)
                        v_pred = v_pred * scale_n
                    # cfg_norm == "none" → no extra scaling
            else:
                v_pred = v_cond

            # Euler step in patchified space
            dt = t_next - t
            z = z + dt * v_pred
            image_prediction = model.unpatchify(
                z, model.patch_size * merge_size, image_size[1], image_size[0],
            )

            if verbose and (step_i % 10 == 0 or step_i == num_steps - 1):
                print(
                    f"[offload]   step {step_i+1:2d}/{num_steps}  t={t.item():.3f}  "
                    f"z|.l1={z.abs().mean().item():.3f}",
                    flush=True,
                )

    if verbose:
        print(f"[offload] Phase B done in {time.time()-t2:.1f}s", flush=True)

    # Cleanup: free flash cache if used
    try:
        from importlib import import_module
        _qwen3 = import_module(f"{model.__class__.__module__.rsplit('.', 1)[0]}.modeling_qwen3")
        clear_flash_kv_cache = getattr(_qwen3, "clear_flash_kv_cache", None)
        if clear_flash_kv_cache is not None:
            clear_flash_kv_cache(cond_kv); clear_flash_kv_cache(uncond_kv)
    except Exception:
        pass

    return image_prediction


def _save_image(tensor: torch.Tensor, path: str | os.PathLike) -> None:
    """tensor: (1, 3, H, W) in [-1, 1] or [0, 1] → PNG. Matches sample_t2i._save_image."""
    import numpy as np
    from PIL import Image

    if tensor.dim() == 4:
        tensor = tensor[0]
    arr = tensor.detach().to("cpu", torch.float32).numpy()
    if arr.min() < -0.01:
        arr = (arr + 1.0) * 0.5  # [-1, 1] → [0, 1] (matches official _denorm with mean=std=0.5)
    arr = arr.clip(0, 1)
    arr = (arr.transpose(1, 2, 0) * 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default=os.environ.get("HF_HOME"))
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--image-h", type=int, default=2048)
    ap.add_argument("--image-w", type=int, default=2048)
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
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True, help="output PNG path")
    ap.add_argument(
        "--load-trainable-state-from",
        default=None,
        help=(
            "Optional safetensors path produced by train_fm_mvp.py "
            "--save-trainable-state-to. Trainable params (fm_head, ts/ns_embed, "
            "mot_gen norms, vision_model_mot_gen) are bf16-injected into the "
            "matching modules of the freshly-loaded bf16 base."
        ),
    )
    ap.add_argument("--config", default=None,
                    help="Optional YAML config (training-run config). Reads lora.preset/spec "
                         "and style.trigger to ensure sample-time wrap matches train-time.")
    ap.add_argument("--lora-preset", default=None,
                    help="Named LoRA preset (must match training preset).")
    ap.add_argument("--lora-spec", default=None,
                    help="Per-target LoRA spec (must match training spec).")
    # LEGACY uniform-rank flags. New code should use --lora-preset / --lora-spec / --config.
    ap.add_argument("--lora-r", type=int, default=0,
                    help="LEGACY: uniform LoRA rank for attn (and MLP if --lora-on-mlp). "
                         "Prefer --lora-preset / --lora-spec / --config.")
    ap.add_argument("--lora-alpha", type=float, default=32.0,
                    help="LEGACY: uniform LoRA alpha (paired with --lora-r).")
    ap.add_argument("--lora-on-mlp", action="store_true",
                    help="LEGACY: also wrap mlp_mot_gen.{gate,up,down}_proj.")
    ap.add_argument("--upstream-lora-path", default=None,
                    help="Bake-in merge an upstream-format LoRA (e.g. the official 8-step "
                         "distill LoRA) before sampling. Use --upstream-lora-skip to keep "
                         "specific modules untouched (e.g. fm_modules.fm_head when stacking "
                         "with our v16c).")
    ap.add_argument("--upstream-lora-skip", action="append", default=[],
                    help="Substring to filter out of the upstream LoRA bake-in merge. "
                         "Repeatable.")
    ap.add_argument(
        "--style-trigger",
        default=None,
        help="Optional trigger text prepended to prompt (must match training-time trigger).",
    )
    ap.add_argument(
        "--think-mode",
        action="store_true",
        help="Enable chain-of-thought reasoning before image generation.",
    )
    ap.add_argument("--think-max-tokens", type=int, default=1024)
    args = ap.parse_args()

    # Resolve LoRA spec source: explicit --lora-spec > --lora-preset > config > legacy.
    lora_specs = []
    config_trigger = None
    if args.config:
        from train_u1.config import load_train_config
        cfg = load_train_config(args.config)
        if not args.lora_spec and not args.lora_preset:
            lora_specs = cfg.lora.resolved_specs()
        config_trigger = cfg.style.trigger or None
    if args.lora_spec:
        from train_u1.model.lora import parse_lora_spec_str
        lora_specs = parse_lora_spec_str(args.lora_spec)
    elif args.lora_preset:
        from train_u1.model.lora import resolve_preset
        lora_specs = resolve_preset(args.lora_preset)
    elif args.lora_r > 0:
        # Legacy fallback.
        from train_u1.model.lora import LoRASpec, ATTN_TARGETS, MLP_TARGETS
        lora_specs = [LoRASpec(target=t, r=args.lora_r, alpha=args.lora_alpha) for t in ATTN_TARGETS]
        if args.lora_on_mlp:
            lora_specs += [LoRASpec(target=t, r=args.lora_r, alpha=args.lora_alpha) for t in MLP_TARGETS]
    if args.style_trigger is None:
        args.style_trigger = config_trigger or ""

    print("[offload] loading bf16 model to CPU (entire 35 GB)...", flush=True)
    # Explicit CPU placement so trust_remote_code construction doesn't try to
    # land 35 GB on GPU.
    model = load_neo_chat(
        cache_dir=args.cache_dir,
        device_map="cpu",
        dtype=torch.bfloat16,
    )

    # Optional: bake-in merge an upstream-format LoRA into base. Filtering
    # substrings via --upstream-lora-skip lets us stack (e.g.) the 8-step
    # LoRA on top of v16c without conflicting on fm_head.
    if args.upstream_lora_path:
        from train_u1.model.lora_io import merge_upstream_lora
        merge_upstream_lora(
            model, args.upstream_lora_path,
            skip_targets=args.upstream_lora_skip,
        )

    if lora_specs:
        from train_u1.model.lora import apply_lora_specs, lora_param_count
        report = apply_lora_specs(model, lora_specs)
        print(f"[offload] {report}")

    if args.load_trainable_state_from:
        from train_u1.model.lora_io import load_lora_state
        load_lora_state(model, args.load_trainable_state_from)

    from transformers import AutoTokenizer
    local = _resolve_local_snapshot(args.cache_dir, MODEL_ID, MODEL_SHA)
    tok = AutoTokenizer.from_pretrained(
        local or MODEL_ID,
        revision=None if local else MODEL_SHA,
        trust_remote_code=True,
    )

    effective_prompt = (
        f"{args.style_trigger}, {args.prompt}"
        if args.style_trigger else args.prompt
    )
    if args.style_trigger:
        print(f"[offload] style_trigger={args.style_trigger!r}", flush=True)
    print(f"[offload] sampling: prompt={effective_prompt[:70]!r}...", flush=True)
    img = t2i_generate_offload(
        model, tok, effective_prompt,
        cfg_scale=args.cfg_scale,
        cfg_norm=args.cfg_norm,
        timestep_shift=args.timestep_shift,
        image_size=(args.image_w, args.image_h),
        num_steps=args.num_steps,
        seed=args.seed,
        think_mode=args.think_mode,
        think_max_tokens=args.think_max_tokens,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _save_image(img, out_path)
    print(f"[offload] saved {out_path}")
    if torch.cuda.is_available():
        print(f"[offload] peak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
