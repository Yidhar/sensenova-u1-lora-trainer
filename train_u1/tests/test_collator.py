"""Test the collator pipeline using a *mock* tokenizer (no model load).

We only need a tokenizer-shaped object that returns deterministic
`input_ids` + an `attention_mask`. This lets us verify shapes, indexes,
and mask invariants without downloading the real HF tokenizer.
"""
from __future__ import annotations

import torch

from train_u1.constants import FM_OUTPUT_DIM, PATCH32
from train_u1.data.collators import CollatorConfig, SenseNovaU1Collator
from train_u1.data.datasets import SyntheticT2ITinyDataset


class _MockTokenizer:
    pad_token_id = 0

    def __call__(self, texts, return_tensors="pt", padding=True):
        # one fake id per character; pad to max len with 0.
        ids = [[ord(c) % 50 + 1 for c in s] for s in texts]
        L = max(len(x) for x in ids)
        for x in ids:
            while len(x) < L:
                x.append(self.pad_token_id)
        t = torch.tensor(ids, dtype=torch.long)
        return {"input_ids": t, "attention_mask": (t != self.pad_token_id).long()}


def _make_collator(image_hw=(64, 64)):
    return SenseNovaU1Collator(
        tokenizer=_MockTokenizer(),
        cfg=CollatorConfig(image_hw=image_hw, seed=0),
    )


def test_collator_shapes_match_forward_contract():
    ds = SyntheticT2ITinyDataset(n=1, image_hw=(64, 64))
    samples = [ds[0]]
    collator = _make_collator(image_hw=(64, 64))

    batch = collator(samples)

    H, W = 64, 64
    token_h, token_w = H // PATCH32, W // PATCH32  # 2, 2
    N = token_h * token_w  # 4

    B = batch["input_ids"].shape[0]
    L_text = batch["input_ids"].shape[1]

    assert B == 1
    assert batch["x0_patch"].shape == (B, N, FM_OUTPUT_DIM)
    assert batch["eps"].shape == (B, N, FM_OUTPUT_DIM)
    assert batch["t"].shape == (B,)
    assert batch["noisy_pixel_values"].shape == (B, 3, H, W)
    assert batch["noisy_grid_hw"].shape == (B, 2)
    assert batch["text_indexes"].shape == (3, L_text)
    assert batch["image_indexes"].shape == (3, N)
    assert batch["position_indexes"].shape == (3, L_text + N)
    assert batch["attn_mask"].shape == (1, 1, L_text + N, L_text + N)
    assert batch["attn_mask_prefix"].shape == (1, 1, L_text, L_text)
    assert batch["cond_drop_mode"] == ["none"]
    assert batch["prefix_cache_key"] == ["cond"]


def test_collator_rejects_batch_gt_one_in_native_mode():
    ds = SyntheticT2ITinyDataset(n=2, image_hw=(64, 64))
    collator = _make_collator(image_hw=(64, 64))
    try:
        collator([ds[0], ds[1]])
    except ValueError as e:
        assert "batch_size=1" in str(e)
    else:
        raise AssertionError("expected ValueError for batch>1 with enforce_batch_one")


def test_collator_t_in_eps_one_range():
    ds = SyntheticT2ITinyDataset(n=1, image_hw=(64, 64))
    collator = _make_collator(image_hw=(64, 64))
    ts = []
    for _ in range(8):
        ts.append(collator([ds[0]])["t"])
    t = torch.cat(ts)
    assert (t > 0.0).all()
    assert (t <= 1.0).all()
    assert (t >= collator.cfg.t_eps - 1e-6).all()


def test_collator_image_hw_divisibility_check():
    ds = SyntheticT2ITinyDataset(n=1, image_hw=(64, 64))
    s = ds[0]
    # mismatched H = 65 — not divisible by 32 → must raise.
    s_bad = type(s)(s.sample_id, s.prompt, torch.zeros(3, 65, 64))
    collator = _make_collator(image_hw=(65, 64))
    try:
        collator([s_bad])
    except ValueError as e:
        assert "PATCH32" in str(e)
    else:
        raise AssertionError("expected ValueError on non-32-aligned image")


def test_native_resolution_mode_picks_up_sample_hw():
    """`image_hw=None` lets the collator infer H,W per sample."""
    ds = SyntheticT2ITinyDataset(n=1, image_hw=(96, 128))
    cfg = CollatorConfig(image_hw=None, seed=0)
    collator = SenseNovaU1Collator(_MockTokenizer(), cfg=cfg)
    batch = collator([ds[0]])
    assert batch["noisy_pixel_values"].shape == (1, 3, 96, 128)
    token_h, token_w = 96 // PATCH32, 128 // PATCH32  # 3, 4
    assert batch["x0_patch"].shape == (1, token_h * token_w, 3 * PATCH32 * PATCH32)


def test_collator_attn_mask_block_invariants():
    """Image span gets full attention; text prefix is causal among itself."""
    ds = SyntheticT2ITinyDataset(n=1, image_hw=(64, 64))
    collator = _make_collator(image_hw=(64, 64))
    batch = collator([ds[0]])

    L_text = batch["input_ids"].shape[1]
    N = batch["x0_patch"].shape[1]
    mask = batch["attn_mask"][0, 0]  # (L, L)
    can_attend = (mask == 0.0)

    image_block = can_attend[L_text:, L_text:]
    assert image_block.all(), "image span must have full mutual attention"

    text_block = can_attend[:L_text, :L_text]
    expected = torch.tril(torch.ones_like(text_block, dtype=text_block.dtype))
    assert (text_block == expected).all(), "text prefix must be causal"

    # text rows are blind to image rows (image t > text t at higher arange)
    assert not can_attend[:L_text, L_text:].any()


def test_condition_dropout_text_mode_uses_uncond_prefix_key():
    ds = SyntheticT2ITinyDataset(n=1, image_hw=(64, 64))
    cfg = CollatorConfig(
        image_hw=(64, 64),
        seed=0,
        cond_dropout_text=1.0,
        cond_dropout_both=0.0,
    )
    collator = SenseNovaU1Collator(_MockTokenizer(), cfg=cfg)

    batch = collator([ds[0]])

    assert batch["cond_drop_mode"] == ["text"]
    assert batch["prefix_cache_key"] == ["uncond"]
    assert batch["cond_drop_text"].tolist() == [True]


def test_condition_dropout_both_mode_is_logged_separately():
    ds = SyntheticT2ITinyDataset(n=1, image_hw=(64, 64))
    cfg = CollatorConfig(
        image_hw=(64, 64),
        seed=0,
        cond_dropout_text=0.0,
        cond_dropout_both=1.0,
    )
    collator = SenseNovaU1Collator(_MockTokenizer(), cfg=cfg)

    batch = collator([ds[0]])

    assert batch["cond_drop_mode"] == ["text_image"]
    assert batch["prefix_cache_key"] == ["uncond"]
    assert batch["cond_drop_text"].tolist() == [True]


def test_condition_dropout_can_be_forced_off():
    ds = SyntheticT2ITinyDataset(n=1, image_hw=(64, 64))
    cfg = CollatorConfig(
        image_hw=(64, 64),
        seed=0,
        cond_dropout_text=0.0,
        cond_dropout_both=0.0,
    )
    collator = SenseNovaU1Collator(_MockTokenizer(), cfg=cfg)

    batch = collator([ds[0]])

    assert batch["cond_drop_mode"] == ["none"]
    assert batch["prefix_cache_key"] == ["cond"]
    assert batch["cond_drop_text"].tolist() == [False]
