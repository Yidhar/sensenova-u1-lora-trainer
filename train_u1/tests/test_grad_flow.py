"""Gradient-flow guardrail: after one backward,
- every `requires_grad=False` param has `.grad is None`
- every `requires_grad=True` param has a non-None, non-NaN `.grad`
- the set of params with non-None grad equals the set of trainable params
- `_mot_gen` LLM core (q/k/v/o etc.) stays untouched in MVP+aux

Heavy test — opt-in.
"""
from __future__ import annotations

import os

import pytest
import torch

from train_u1.constants import MODEL_ID, MODEL_SHA
from train_u1.data.collators import CollatorConfig, SenseNovaU1Collator, to_device
from train_u1.data.datasets import SyntheticT2ITinyDataset
from train_u1.model.losses import fm_loss_x0
from train_u1.model.params import (
    FREEZE_REGEX_MVP,
    TRAINABLE_REGEX_MVP_AUX,
    classify_param,
    set_requires_grad_by_regex,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.skipif(
    not os.environ.get("RUN_HEAVY_TESTS"),
    reason="set RUN_HEAVY_TESTS=1 to opt in",
)
def test_grad_only_on_trainable():
    from train_u1.model.loader import _resolve_local_snapshot, load_neo_chat_4bit
    from train_u1.model.wrapper import TrainingWrapper

    cache_dir = os.environ.get("HF_HOME")
    model = load_neo_chat_4bit(cache_dir=cache_dir, device_map="cuda")
    set_requires_grad_by_regex(
        model,
        freeze_patterns=FREEZE_REGEX_MVP,
        trainable_patterns=TRAINABLE_REGEX_MVP_AUX,
        default=False,
        strict=True,
    )

    from transformers import AutoTokenizer
    local = _resolve_local_snapshot(cache_dir, MODEL_ID, MODEL_SHA)
    tok = AutoTokenizer.from_pretrained(
        local or MODEL_ID, revision=None if local else MODEL_SHA, trust_remote_code=True
    )

    ds = SyntheticT2ITinyDataset(n=1, image_hw=(256, 256))
    collator = SenseNovaU1Collator(tok, cfg=CollatorConfig(image_hw=(256, 256), seed=0))
    batch = to_device(collator([ds[0]]), "cuda", dtype=torch.bfloat16)

    wrapper = TrainingWrapper(model)
    out = wrapper.forward_t2i_step(batch)
    loss = fm_loss_x0(out.x_pred, batch["x0_patch"])
    loss.backward()

    leak_frozen: list[str] = []
    no_grad_trainable: list[str] = []
    nan_grad: list[str] = []
    bucket_with_grad: dict[str, int] = {}

    for name, p in model.named_parameters():
        has_grad = p.grad is not None
        if not p.requires_grad and has_grad and p.grad.abs().sum().item() > 0:
            leak_frozen.append(name)
        if p.requires_grad and not has_grad:
            no_grad_trainable.append(name)
        if has_grad and torch.isnan(p.grad).any().item():
            nan_grad.append(name)
        if has_grad and p.requires_grad:
            b = classify_param(name)
            bucket_with_grad[b] = bucket_with_grad.get(b, 0) + 1

    assert not leak_frozen, f"frozen params received non-zero grad: {leak_frozen[:5]}"
    assert not no_grad_trainable, f"trainable params got no grad: {no_grad_trainable[:5]}"
    assert not nan_grad, f"NaN grad in: {nan_grad[:5]}"

    # MVP+aux trainable buckets: fm_head, ts/ns embedders, mot_gen norms, final_norms
    assert "fm_head" in bucket_with_grad
    assert "timestep_embedder" in bucket_with_grad
    assert "noise_scale_embedder" in bucket_with_grad
    # ordinary path must be entirely silent
    assert "ordinary_llm_core" not in bucket_with_grad
    assert "vision_understanding" not in bucket_with_grad
    assert "vision_mot_gen" not in bucket_with_grad  # not trainable in MVP+aux
    assert "lm_head" not in bucket_with_grad
    assert "token_embeddings" not in bucket_with_grad
    print(f"[grad-flow] buckets with grad: {bucket_with_grad}")
