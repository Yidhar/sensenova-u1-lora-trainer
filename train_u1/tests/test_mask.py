"""Block-causal mask + THW index correctness tests.

These run on CPU only — no model weights required. They verify the report's
§5.1 / §12.1 invariants:

1. Image-context tokens for a single image share `t = text_len`, giving them
   full mutual attention via the `(idx_j == idx_i)` clause.
2. Text prefix gets standard causal among itself.
3. Image rows can attend to ALL text rows (because text `t` < image `t`,
   the `arange_j <= arange_i` clause is satisfied for image-row indices).
4. Text rows are blind to image rows (image `t` > text `t`, and image rows
   sit at higher `arange` positions).
"""
from __future__ import annotations

import torch

from train_u1.model.masking import (
    build_t2i_image_indexes,
    build_t2i_text_indexes,
    concat_text_image_indexes,
    create_block_causal_mask,
    mask_is_valid_t2i,
)


def test_create_block_causal_mask_text_only_is_pure_causal():
    indexes = build_t2i_text_indexes(text_len=8)
    mask = create_block_causal_mask(indexes[0])
    assert mask.shape == (1, 1, 8, 8)
    can_attend = (mask[0, 0] == 0.0)
    expected = torch.tril(torch.ones(8, 8, dtype=torch.bool))
    assert (can_attend == expected).all()


def test_create_block_causal_mask_image_span_full_attention():
    # all image tokens share the same t-index → full mutual attention block
    L = 6
    image_t = torch.full((L,), 100, dtype=torch.long)
    mask = create_block_causal_mask(image_t)
    # entire (L, L) block should be 0.0 (i.e., all attendable)
    assert (mask[0, 0] == 0.0).all()


def test_t2i_combined_mask_invariants():
    text_len = 5
    token_h, token_w = 4, 6
    image_len = token_h * token_w

    text_idx = build_t2i_text_indexes(text_len=text_len)
    image_idx = build_t2i_image_indexes(token_h=token_h, token_w=token_w, text_len=text_len)
    full_idx = concat_text_image_indexes(text_idx, image_idx)
    assert full_idx.shape == (3, text_len + image_len)

    mask = create_block_causal_mask(full_idx[0])
    diag = mask_is_valid_t2i(mask, text_len=text_len, image_len=image_len)

    assert diag["text_causal_ok"], diag
    assert diag["image_full_ok"], diag
    assert diag["image_sees_all_text"], diag
    assert diag["text_blind_to_image"], diag


def test_image_indexes_h_w_coords_row_major():
    token_h, token_w = 3, 4
    idx = build_t2i_image_indexes(token_h=token_h, token_w=token_w, text_len=10)
    # row 1 = h, row 2 = w
    h_row = idx[1]
    w_row = idx[2]
    expected_h = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2], dtype=torch.long)
    expected_w = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3], dtype=torch.long)
    assert (h_row == expected_h).all()
    assert (w_row == expected_w).all()
    # all image t = text_len = 10
    assert (idx[0] == 10).all()


def test_image_token_count_matches_grid_div_2():
    """公开证据显示: image_token_num = (grid_h // 2) * (grid_w // 2).

    With downsample_ratio=0.5 and patch=16, an HxW image whose H,W are
    multiples of 32 has token_h = H // 32, token_w = W // 32.
    """
    H, W = 512, 768
    expected_tokens = (H // 32) * (W // 32)  # 16 * 24 = 384
    assert expected_tokens == 384

    idx = build_t2i_image_indexes(token_h=H // 32, token_w=W // 32, text_len=20)
    assert idx.shape == (3, 384)
