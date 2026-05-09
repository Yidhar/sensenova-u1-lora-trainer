"""Block-causal mask + THW index helpers.

Mirrors `create_block_causal_mask` (modeling_qwen3.py L152-164) and the
T2I index builders (`_build_t2i_text_inputs` / `_build_t2i_image_indexes`)
from `modeling_neo_chat.py` at commit `df86ca90`.

Key shape contract:
- `indexes` is `(3, L)`: row 0 = `t`, row 1 = `h`, row 2 = `w`.
- block-causal mask is built from `indexes[0]` (the `t` row).
- All image tokens of *one* image share a single `t` value → they get
  full mutual attention via `(idx_j == idx_i)`. Text tokens have unique
  `t` values → they fall back to standard causal via `arange_j <= arange_i`.

公开证据显示 — see upstream functions named above.
"""
from __future__ import annotations

import torch


def create_block_causal_mask(index: torch.Tensor) -> torch.Tensor:
    """index: (L,) → (1, 1, L, L) additive attention mask (0 / -inf).

    Verbatim reproduction of upstream `create_block_causal_mask`. We pick
    dtype/device from `index` for `torch.where`'s scalar branches so the
    mask lands on the same device with a sensible dtype.
    """
    if index.dim() != 1:
        raise ValueError(f"index must be 1-D, got shape {index.shape}")
    L = index.size(0)
    idx_i = index.unsqueeze(1).expand(L, L)
    idx_j = index.unsqueeze(0).expand(L, L)
    arange = torch.arange(L, device=index.device)
    mask = (idx_j == idx_i) | (arange.unsqueeze(0) <= arange.unsqueeze(1))
    zero = torch.tensor(0.0, dtype=torch.float32, device=index.device)
    neg_inf = torch.tensor(float("-inf"), dtype=torch.float32, device=index.device)
    return torch.where(mask[None, None, :, :], zero, neg_inf)


def build_t2i_text_indexes(text_len: int, device: torch.device | str = "cpu") -> torch.Tensor:
    """Mirror of `_build_t2i_text_inputs` index part: (3, text_len)."""
    t_idx = torch.arange(0, text_len, dtype=torch.long, device=device)
    h_idx = torch.zeros_like(t_idx)
    w_idx = torch.zeros_like(t_idx)
    return torch.stack([t_idx, h_idx, w_idx], dim=0)


def build_t2i_image_indexes(
    token_h: int, token_w: int, text_len: int, device: torch.device | str = "cpu"
) -> torch.Tensor:
    """Mirror of `_build_t2i_image_indexes`: (3, token_h * token_w).

    All image-span tokens share `t = text_len` (so the block-causal mask
    gives them full mutual attention). h/w are integer coordinates within
    `[0, token_h) × [0, token_w)` in row-major order.
    """
    n = token_h * token_w
    t_image = torch.full((n,), text_len, dtype=torch.long, device=device)
    idx = torch.arange(n, device=device, dtype=torch.long)
    h_image = idx // token_w
    w_image = idx % token_w
    return torch.stack([t_image, h_image, w_image], dim=0)


def concat_text_image_indexes(
    text_indexes: torch.Tensor, image_indexes: torch.Tensor
) -> torch.Tensor:
    """Stack text-prefix indexes followed by an image span. (3, L_text + L_img)."""
    if text_indexes.shape[0] != 3 or image_indexes.shape[0] != 3:
        raise ValueError("indexes must be (3, L)")
    return torch.cat([text_indexes, image_indexes], dim=1)


# --------------------------------------------------------------------------- #
# Convenience: mask sanity checks for tests                                   #
# --------------------------------------------------------------------------- #


def mask_is_valid_t2i(
    mask: torch.Tensor, text_len: int, image_len: int
) -> dict[str, bool]:
    """Return a dict of boolean diagnostics about a (1,1,L,L) block-causal mask.

    Used by tests to confirm:
    - text rows are causal among themselves
    - image rows have *full* attention to other image rows
    - image rows can attend to all text rows (since text `t` < `text_len`,
      and image `t` == `text_len`, the `arange_j <= arange_i` clause covers it).
    """
    if mask.dim() != 4 or mask.shape[:2] != (1, 1):
        raise ValueError(f"expected (1,1,L,L) mask, got {mask.shape}")
    L = mask.shape[-1]
    assert L == text_len + image_len, "mask length mismatch"
    m = mask[0, 0]  # (L, L)
    # 0.0 means "can attend"; -inf means "blocked".
    can_attend = (m == 0.0)

    text_block = can_attend[:text_len, :text_len]
    text_causal_ok = bool(
        (text_block == torch.tril(torch.ones_like(text_block, dtype=text_block.dtype))).all()
    )
    image_block = can_attend[text_len:, text_len:]
    image_full_ok = bool(image_block.all())
    image_to_text = can_attend[text_len:, :text_len]
    image_sees_all_text = bool(image_to_text.all())
    text_to_image = can_attend[:text_len, text_len:]
    text_blind_to_image = bool((~text_to_image).all())

    return {
        "text_causal_ok": text_causal_ok,
        "image_full_ok": image_full_ok,
        "image_sees_all_text": image_sees_all_text,
        "text_blind_to_image": text_blind_to_image,
    }
