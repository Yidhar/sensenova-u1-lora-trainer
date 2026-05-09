"""Same-seed determinism: two collator+forward runs must produce identical x_pred.

Catches non-determinism leaks (cudnn nondet kernels, dropout left on,
unseeded RNG paths inside the model, etc.). Heavy test — opt-in.
"""
from __future__ import annotations

import os

import pytest
import torch

from train_u1.constants import MODEL_ID, MODEL_SHA
from train_u1.data.collators import CollatorConfig, SenseNovaU1Collator, to_device
from train_u1.data.datasets import SyntheticT2ITinyDataset
from train_u1.model.params import (
    FREEZE_REGEX_MVP,
    TRAINABLE_REGEX_MVP_AUX,
    set_requires_grad_by_regex,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.skipif(
    not os.environ.get("RUN_HEAVY_TESTS"),
    reason="set RUN_HEAVY_TESTS=1 to opt in",
)
def test_same_seed_bit_exact():
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
        local or MODEL_ID,
        revision=None if local else MODEL_SHA,
        trust_remote_code=True,
    )

    wrapper = TrainingWrapper(model)

    def _one_pass(seed: int) -> torch.Tensor:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        ds = SyntheticT2ITinyDataset(n=1, image_hw=(256, 256), base_seed=seed)
        collator = SenseNovaU1Collator(tok, cfg=CollatorConfig(image_hw=(256, 256), seed=seed))
        batch = to_device(collator([ds[0]]), "cuda", dtype=torch.bfloat16)
        with torch.no_grad():
            out = wrapper.forward_t2i_step(batch)
        return out.x_pred.detach().clone()

    a = _one_pass(seed=7)
    b = _one_pass(seed=7)
    diff = (a.float() - b.float()).abs()
    print(f"[repro] same-seed max_abs={diff.max():.2e} mean_abs={diff.mean():.2e}")
    # Bit-exact tolerance for same seed + same model + same batch.
    assert diff.max().item() == 0.0
