"""Model loading utilities pinned to the upstream HF revision.

Two entry points:
- `load_neo_chat(...)` : full-precision (bf16/fp16) load — for Phase 2/3
  smoke tests on a beefy GPU or for SFT-final diff probes.
- `load_neo_chat_4bit(...)` : nf4 quantized base via bitsandbytes — for the
  24GB MVP scenario (Phase 4). Only the frozen base is quantized; trainable
  modules (`fm_head`, optional `vision_model_mot_gen`, `_mot_gen` LoRA) stay
  in bf16.

We pin `revision=MODEL_SHA` on every load so the checkpoint we exercise
matches the SHA recorded in `constants.py`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from train_u1.constants import MODEL_ID, MODEL_SHA


@dataclass
class LoadReport:
    model_id: str
    revision: str
    dtype: torch.dtype
    quant: str
    device: str | None
    n_params: int
    n_trainable_before_freeze: int


def _resolve_local_snapshot(cache_dir: str | None, model_id: str, sha: str) -> str | None:
    """Return the local snapshot path if it exists, else None.

    `trust_remote_code` looks for `configuration_*.py` next to `config.json`.
    HF caches those .py only when the repo itself ships them; SenseNova-U1
    keeps the .py in a separate GitHub repo (installed via
    `install_modeling_into_snapshot.py`). Loading by *path* lets the
    dynamic-module loader pick up our local copy directly.
    """
    if not cache_dir:
        return None
    safe = model_id.replace("/", "--")
    snap = Path(cache_dir) / f"models--{safe}" / "snapshots" / sha
    return str(snap) if (snap / "config.json").is_file() else None


def load_neo_chat(
    *,
    model_id: str = MODEL_ID,
    revision: str = MODEL_SHA,
    dtype: torch.dtype = torch.bfloat16,
    device_map: str | dict[str, Any] | None = "auto",
    cache_dir: str | None = None,
    attn_impl: str = "eager",
):
    """Load NEOChatModel at full precision via `trust_remote_code`.

    Prefers the local snapshot dir under `cache_dir` so the modeling .py
    files installed by `install_modeling_into_snapshot.py` are picked up
    without an HTTP roundtrip to the Hub.
    """
    from transformers import AutoModel  # lazy

    local = _resolve_local_snapshot(cache_dir, model_id, revision)
    pretrained = local or model_id
    kwargs: dict[str, Any] = dict(
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map=device_map,
        attn_implementation=attn_impl,
    )
    if local is None:
        kwargs["revision"] = revision
        kwargs["cache_dir"] = cache_dir

    try:
        return AutoModel.from_pretrained(pretrained, **kwargs)
    except FileNotFoundError as e:
        # transformers' dynamic-module copy walks `auto_map` entries but does
        # NOT recurse into indirect relative imports (e.g.
        # `configuration_neo_chat.py` imports `configuration_neo_vit.py` —
        # the latter never gets copied to the hash dir, breaking loads on
        # transformers >= 5.x). Detect that specific failure and fix it
        # in-place by copying every modeling .py from the local snapshot to
        # the missing hash dir, then retry once.
        if local is None or "transformers_modules" not in str(e):
            raise
        import shutil
        from pathlib import Path as _Path
        miss_path = _Path(str(e).split("'")[1] if "'" in str(e) else "")
        target_dir = miss_path.parent if miss_path.parent.exists() else None
        if target_dir is None:
            raise
        snap = _Path(local)
        n = 0
        for src in snap.glob("*.py"):
            dst = target_dir / src.name
            if not dst.exists():
                shutil.copy2(src, dst)
                n += 1
        print(
            f"[loader] copied {n} modeling .py from {snap} → {target_dir} "
            f"(transformers dynamic-cache fixup), retrying load..."
        )
        return AutoModel.from_pretrained(pretrained, **kwargs)


_DEFAULT_KEEP_MODULES_IN_FP: tuple[str, ...] = (
    # fm_head + gen vision + timestep/noise embedders are tiny — keep them in bf16
    # so trainable LoRA / full-param ops are numerically clean.
    "fm_modules.fm_head",
    "fm_modules.timestep_embedder",
    "fm_modules.noise_scale_embedder",
    # vision encoders contain Conv2d that bnb can't quantize anyway;
    # leaving them in compute_dtype avoids fp16/bf16 mismatches against
    # bf16 inputs at the conv boundary.
    "fm_modules.vision_model_mot_gen",
    "vision_model",
    # final norms
    "language_model.model.norm",
    "language_model.model.norm_mot_gen",
)


def load_neo_chat_8bit(
    *,
    model_id: str = MODEL_ID,
    revision: str = MODEL_SHA,
    compute_dtype: torch.dtype = torch.bfloat16,
    device_map: str | dict[str, Any] | None = "auto",
    cache_dir: str | None = None,
    attn_impl: str = "eager",
    keep_modules_in_fp: tuple[str, ...] = _DEFAULT_KEEP_MODULES_IN_FP,
):
    """8-bit (int8) base via bitsandbytes — better precision than 4-bit nf4.

    For pixel-space FM models, 4-bit nf4 introduces visible artifacts that
    accumulate across 42 layers × ~50 sampling steps. 8-bit doubles the
    base model's VRAM (~17 GB vs ~9 GB for 4-bit) but materially improves
    sampling quality.
    """
    from transformers import AutoModel, BitsAndBytesConfig  # lazy

    bnb = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_skip_modules=list(keep_modules_in_fp),
    )

    local = _resolve_local_snapshot(cache_dir, model_id, revision)
    pretrained = local or model_id
    kwargs: dict[str, Any] = dict(
        trust_remote_code=True,
        quantization_config=bnb,
        device_map=device_map,
        attn_implementation=attn_impl,
        torch_dtype=compute_dtype,
    )
    if local is None:
        kwargs["revision"] = revision
        kwargs["cache_dir"] = cache_dir
    return AutoModel.from_pretrained(pretrained, **kwargs)


def load_neo_chat_4bit(
    *,
    model_id: str = MODEL_ID,
    revision: str = MODEL_SHA,
    compute_dtype: torch.dtype = torch.bfloat16,
    device_map: str | dict[str, Any] | None = "auto",
    cache_dir: str | None = None,
    attn_impl: str = "eager",
    keep_modules_in_fp: tuple[str, ...] = _DEFAULT_KEEP_MODULES_IN_FP,
):
    """4-bit nf4 base + bf16 trainable modules.

    Uses transformers' `BitsAndBytesConfig` with `llm_int8_skip_modules` to
    keep the small trainable modules in compute_dtype. Frozen `_mot_gen`
    LLM core can be 4-bit when we're not LoRA-targeting it directly; once
    we attach PEFT LoRA the underlying linear stays 4-bit and only the
    LoRA `A`/`B` floats are trainable in compute_dtype.
    """
    from transformers import AutoModel, BitsAndBytesConfig  # lazy

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
        llm_int8_skip_modules=list(keep_modules_in_fp),
    )

    local = _resolve_local_snapshot(cache_dir, model_id, revision)
    pretrained = local or model_id
    kwargs: dict[str, Any] = dict(
        trust_remote_code=True,
        quantization_config=bnb,
        device_map=device_map,
        attn_implementation=attn_impl,
        # Pin non-quantized modules to compute_dtype so bf16 inputs don't
        # collide with fp16 weights at module boundaries.
        torch_dtype=compute_dtype,
    )
    if local is None:
        kwargs["revision"] = revision
        kwargs["cache_dir"] = cache_dir
    return AutoModel.from_pretrained(pretrained, **kwargs)


def report(model: torch.nn.Module, *, model_id: str, revision: str, quant: str) -> LoadReport:
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    dtype = next((p.dtype for p in model.parameters()), torch.float32)
    device = str(next((p.device for p in model.parameters()), "cpu"))
    return LoadReport(
        model_id=model_id,
        revision=revision,
        dtype=dtype,
        quant=quant,
        device=device,
        n_params=n_params,
        n_trainable_before_freeze=n_trainable,
    )
