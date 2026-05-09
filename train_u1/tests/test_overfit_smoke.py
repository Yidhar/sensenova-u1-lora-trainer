"""Phase 4 correctness gate: a 1-sample overfit must drop loss meaningfully.

Designed to run as part of CI on a CUDA box. Skips if no GPU. Exits in
under ~2 min on an RTX 5090 at 256².

This is NOT a model-quality test — it's a wrapper/forward/grad correctness
test. Failure here means the wrapper is broken, not that training itself
is bad.
"""
from __future__ import annotations

import os
import time

import pytest
import torch

from train_u1.constants import MODEL_ID, MODEL_SHA
from train_u1.data.collators import CollatorConfig, SenseNovaU1Collator, to_device
from train_u1.data.datasets import SyntheticT2ITinyDataset
from train_u1.model.losses import fm_loss_x0
from train_u1.model.params import (
    FREEZE_REGEX_MVP,
    TRAINABLE_REGEX_MVP_AUX,
    set_requires_grad_by_regex,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.skipif(
    not os.environ.get("RUN_HEAVY_TESTS"),
    reason="set RUN_HEAVY_TESTS=1 to opt in (loads 4-bit base, ~2 min)",
)
def test_one_sample_overfit_drops_loss():
    from train_u1.model.loader import _resolve_local_snapshot, load_neo_chat_4bit
    from train_u1.model.wrapper import TrainingWrapper

    cache_dir = os.environ.get("HF_HOME")
    model = load_neo_chat_4bit(cache_dir=cache_dir, device_map="cuda")
    rep = set_requires_grad_by_regex(
        model,
        freeze_patterns=FREEZE_REGEX_MVP,
        trainable_patterns=TRAINABLE_REGEX_MVP_AUX,
        default=False,
        strict=True,
    )
    assert rep.n_trainable > 0
    assert rep.n_trainable < 100_000_000  # MVP+aux must stay tiny

    from transformers import AutoTokenizer

    local = _resolve_local_snapshot(cache_dir, MODEL_ID, MODEL_SHA)
    tok = AutoTokenizer.from_pretrained(
        local or MODEL_ID,
        revision=None if local else MODEL_SHA,
        trust_remote_code=True,
    )

    ds = SyntheticT2ITinyDataset(n=1, image_hw=(256, 256))
    collator = SenseNovaU1Collator(tok, cfg=CollatorConfig(image_hw=(256, 256), seed=0))
    sample = ds[0]
    batch = to_device(collator([sample]), "cuda", dtype=torch.bfloat16)

    wrapper = TrainingWrapper(model)

    import bitsandbytes as bnb

    opt = bnb.optim.AdamW8bit(
        [p for p in model.parameters() if p.requires_grad], lr=2e-4
    )

    losses = []
    t0 = time.time()
    for _ in range(40):
        out = wrapper.forward_t2i_step(batch)
        loss = fm_loss_x0(out.x_pred, batch["x0_patch"])
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        losses.append(loss.item())

    first = sum(losses[:5]) / 5
    last = sum(losses[-5:]) / 5
    print(f"[overfit-smoke] first5={first:.4f} last5={last:.4f} took={time.time()-t0:.1f}s")
    # On 1 sample with bf16 + 4-bit base, we expect a clear monotonic drop.
    # Threshold is conservative — failure here means autograd or freeze is broken.
    assert last < first * 0.85, f"loss did not drop enough: {first} -> {last}"
