"""LoRA save/load + upstream-format inter-op.

Three responsibilities:

1. **`save_lora_state`** — emit a safetensors file in the upstream layout:
       <key>.lora_down.weight    fp32  (rank, in_features)
       <key>.lora_up.weight      fp32  (out_features, rank)
       <key>.alpha               int32 ()  scalar

2. **`load_lora_state`** — restore wrapped LoraAdapters from either:
       - the new upstream layout above, or
       - the legacy `.lora_A.weight` / `.lora_B.weight` layout produced by
         pre-2026-05-09 versions of this trainer.

3. **`merge_upstream_lora`** — bake-in merge of any upstream-format LoRA into
   a (potentially unwrapped) base model, mirroring the loader at
   `OpenSenseNova/SenseNova-U1` `src/sensenova_u1/utils/lora.py` (commit `8b9220e`).
   Use this to consume the official 8-step distill LoRA, or to compose two
   LoRAs by progressively merging them into the base.

This module also handles **full-finetune trainable params** (vision_model_mot_gen,
ts/ns embedders, fm_head when not LoRA-wrapped) — those are emitted under their
plain qualified names alongside the LoRA tensors. `save_trainable_state` is
the public entry point used by the training loop.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
from safetensors import safe_open
from safetensors.torch import save_file as st_save_file


# --------------------------------------------------------------------------- #
# Save                                                                        #
# --------------------------------------------------------------------------- #


def save_trainable_state(
    model: nn.Module,
    out_path: str | Path,
    *,
    lora_dtype: torch.dtype = torch.float32,
    full_dtype: torch.dtype | None = None,
    extra_metadata: dict[str, str] | None = None,
) -> dict[str, torch.Tensor]:
    """Save every `requires_grad=True` param + every LoraAdapter buffer.

    LoraAdapter parameters land at `<wrap>.lora_down.weight` /
    `<wrap>.lora_up.weight` (cast to `lora_dtype`, default fp32 for
    upstream compatibility) and `<wrap>.alpha` (int32 scalar).

    Other trainable params (full-finetune surfaces) land at their qualified
    names unchanged. `full_dtype=None` keeps their native dtype; pass
    `torch.float32` to force fp32.
    """
    from train_u1.model.lora import LoraAdapter  # local import to avoid cycle

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    state: dict[str, torch.Tensor] = {}

    # Trainable params first (covers both LoRA tensors and full-FT surfaces).
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # Translate `*.lora_down.weight` / `*.lora_up.weight` into the dtype
        # we want on disk; everything else keeps its native dtype unless
        # `full_dtype` is given.
        if name.endswith(".lora_down.weight") or name.endswith(".lora_up.weight"):
            state[name] = p.detach().to(dtype=lora_dtype, device="cpu").contiguous()
        else:
            target_dtype = full_dtype if full_dtype is not None else p.dtype
            state[name] = p.detach().to(dtype=target_dtype, device="cpu").contiguous()

    # Then LoraAdapter `.alpha` buffers — saved as int32 scalar to match
    # upstream. We always emit one alpha per wrapped module, even if the
    # `requires_grad` filter above missed it.
    for module_name, module in model.named_modules():
        if isinstance(module, LoraAdapter):
            key = f"{module_name}.alpha"
            state[key] = torch.tensor(int(module.alpha_value), dtype=torch.int32)

    metadata = {
        "format": "train_u1.upstream_lora",
        "naming": "lora_down/lora_up/.alpha",
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    st_save_file(state, str(out_path), metadata=metadata)
    return state


# Alias for clarity at call sites that only want LoRA tensors.
save_lora_state = save_trainable_state


# --------------------------------------------------------------------------- #
# Load                                                                        #
# --------------------------------------------------------------------------- #


def _load_safetensors(path: str | Path) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    state: dict[str, torch.Tensor] = {}
    metadata: dict[str, str] = {}
    with safe_open(str(path), framework="pt", device="cpu") as f:
        meta = f.metadata() or {}
        metadata = dict(meta)
        for k in f.keys():
            state[k] = f.get_tensor(k)
    return state, metadata


def _detect_legacy_naming(state: dict[str, torch.Tensor]) -> bool:
    """Return True iff the state dict uses the pre-2026-05-09 `.lora_A`/`.lora_B` naming."""
    has_legacy = any(k.endswith(".lora_A.weight") or k.endswith(".lora_B.weight") for k in state)
    has_new = any(k.endswith(".lora_down.weight") or k.endswith(".lora_up.weight") for k in state)
    if has_legacy and has_new:
        # Mixed — pick whichever is more numerous and warn.
        n_legacy = sum(1 for k in state if k.endswith(".lora_A.weight") or k.endswith(".lora_B.weight"))
        n_new = sum(1 for k in state if k.endswith(".lora_down.weight") or k.endswith(".lora_up.weight"))
        return n_legacy > n_new
    return has_legacy


def _remap_legacy_keys(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Rename `.lora_A.weight` → `.lora_down.weight`, `.lora_B.weight` → `.lora_up.weight`.

    Drops anything else only if it conflicts with an already-present new-naming key.
    """
    out: dict[str, torch.Tensor] = {}
    for k, v in state.items():
        if k.endswith(".lora_A.weight"):
            new_k = k[: -len(".lora_A.weight")] + ".lora_down.weight"
        elif k.endswith(".lora_B.weight"):
            new_k = k[: -len(".lora_B.weight")] + ".lora_up.weight"
        else:
            new_k = k
        out[new_k] = v
    return out


def _strip_diffusion_model_prefix(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Strip leading `diffusion_model.` from all keys (upstream's training format)."""
    if not any(k.startswith("diffusion_model.") for k in state):
        return state
    return {
        (k[len("diffusion_model."):] if k.startswith("diffusion_model.") else k): v
        for k, v in state.items()
    }


def load_lora_state(
    model: nn.Module,
    path: str | Path,
    *,
    strict_missing: bool = False,
    verbose: bool = True,
) -> tuple[int, list[str]]:
    """Load a safetensors checkpoint into the model in-place.

    Auto-detects both:
      - legacy `.lora_A.weight` / `.lora_B.weight` naming (remaps in memory),
      - leading `diffusion_model.` prefix (strips it).

    Returns `(n_loaded, missing_keys)`. `missing_keys` are state-dict entries
    that did not match any model parameter — typically harmless when the
    state covers a different LoRA scenario.
    """
    state, _meta = _load_safetensors(path)
    state = _strip_diffusion_model_prefix(state)
    if _detect_legacy_naming(state):
        if verbose:
            print(f"[load_lora_state] {path}: legacy .lora_A/.lora_B naming detected, remapping...")
        state = _remap_legacy_keys(state)

    target = {n: p for n, p in model.named_parameters()}
    target_buffers = {n: b for n, b in model.named_buffers()}

    n_loaded = 0
    missing: list[str] = []
    for name, t in state.items():
        if name in target:
            tgt = target[name]
            tgt.data.copy_(t.to(tgt.device, tgt.dtype))
            n_loaded += 1
        elif name in target_buffers:
            tgt = target_buffers[name]
            tgt.data.copy_(t.to(tgt.device, tgt.dtype))
            n_loaded += 1
        else:
            missing.append(name)

    if verbose:
        print(f"[load_lora_state] loaded {n_loaded}/{len(state)} tensors from {path}")
        if missing and verbose:
            print(f"[load_lora_state] {len(missing)} unmatched (first 3): {missing[:3]}")
    if strict_missing and missing:
        raise RuntimeError(f"unmatched keys when loading {path}: {missing[:5]} (... total {len(missing)})")
    return n_loaded, missing


# --------------------------------------------------------------------------- #
# Upstream bake-in merge                                                      #
# --------------------------------------------------------------------------- #


def merge_upstream_lora(
    model: nn.Module,
    lora_path: str | Path,
    *,
    skip_targets: Iterable[str] = (),
    verbose: bool = True,
) -> int:
    """Bake-in merge an upstream-format LoRA into `model`'s parameters.

    Mirrors `OpenSenseNova/SenseNova-U1`'s
    `src/sensenova_u1/utils/lora.py:load_and_merge_lora_weight` (commit
    `8b9220e`):

        scaling = alpha / rank
        delta_W = scaling * (lora_up @ lora_down)
        param.data = (param.data + delta_W).type_as(param.data)

    `skip_targets` is a list of substring patterns; any param whose lora
    triple's base path contains one of these substrings is skipped. Useful
    for stacking the official 8-step LoRA on top of a model that already
    has its own fm_head delta::

        merge_upstream_lora(model, lora_path,
                            skip_targets=("fm_modules.fm_head",))

    Returns the number of base params modified (= number of `lora_down`
    keys consumed). Asserts up/down dtype is fp32 to match upstream.
    """
    state, _meta = _load_safetensors(lora_path)
    state = _strip_diffusion_model_prefix(state)

    target = {n: p for n, p in model.named_parameters()}
    n_merged = 0
    n_skipped = 0
    skip_targets_t = tuple(skip_targets)

    # Iterate over down-weight keys; the corresponding up + alpha keys must
    # exist. Modules are identified by stripping `.lora_down.weight`.
    for k in list(state):
        if not k.endswith(".lora_down.weight"):
            continue
        base_path = k[: -len(".lora_down.weight")]

        if any(s in base_path for s in skip_targets_t):
            n_skipped += 1
            continue

        weight_param_name = base_path + ".weight"
        if weight_param_name not in target:
            # The `base_path` may have a sub-namespace. For LoraAdapter-wrapped
            # modules created by `apply_lora_specs`, the actual param name is
            # `<base_path>.base.weight` — try that fallback.
            alt = base_path + ".base.weight"
            if alt in target:
                weight_param_name = alt
            else:
                if verbose:
                    print(f"[merge_upstream_lora] no base param for {base_path}; skipping")
                continue

        lora_down = state[k]
        lora_up = state[base_path + ".lora_up.weight"]
        alpha_t = state.get(base_path + ".alpha")
        if alpha_t is None:
            raise RuntimeError(f"missing .alpha for {base_path}")

        # Upstream asserts fp32 for both up/down. We mirror that.
        if lora_down.dtype != torch.float32 or lora_up.dtype != torch.float32:
            raise RuntimeError(
                f"upstream LoRA expects fp32 lora_down/lora_up, got "
                f"{lora_down.dtype}/{lora_up.dtype} at {base_path}"
            )

        rank = lora_down.shape[0]
        scaling = float(alpha_t.item()) / float(rank)
        param = target[weight_param_name]
        delta = scaling * torch.matmul(lora_up, lora_down).to(param.device)
        param.data = (param.data + delta.to(param.dtype)).contiguous()
        n_merged += 1

    if verbose:
        print(
            f"[merge_upstream_lora] merged {n_merged} modules from {lora_path}"
            + (f" (skipped {n_skipped} via skip_targets={skip_targets_t})" if n_skipped else "")
        )
    return n_merged
