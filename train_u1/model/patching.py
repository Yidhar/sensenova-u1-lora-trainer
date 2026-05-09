"""Patchify / unpatchify helpers.

Exact mirror of `NEOChatModel.patchify` (modeling_neo_chat.py L694) at the
pinned commit `df86ca90`. We reproduce the einsum signatures and shapes so
that downstream consumers (vision_model_mot_gen at patch=16, fm_head output
at patch=32) see byte-compatible tensor layouts.

公开证据显示 — see `_build_t2i_text_inputs`, `_build_t2i_image_indexes`,
`patchify` in modeling_neo_chat.py at the pinned commit.
"""
from __future__ import annotations

import torch

from train_u1.constants import FM_OUTPUT_DIM, PATCH32, PATCH_SIZE


def patchify(images: torch.Tensor, patch_size: int, channel_first: bool = False) -> torch.Tensor:
    """(N, 3, H, W) -> (N, h*w, patch_size**2 * 3).

    Mirrors the upstream einsum exactly (`nchpwq -> nhwpqc` by default).
    `channel_first=True` keeps `[c, p, q]` ordering, used in some vision-
    model branches that prefer channels first inside the patch dim.
    """
    n, c, H, W = images.shape
    if H % patch_size or W % patch_size:
        raise ValueError(
            f"image HxW=({H},{W}) not divisible by patch_size={patch_size}"
        )
    h, w = H // patch_size, W // patch_size
    x = images.reshape(n, c, h, patch_size, w, patch_size)
    if channel_first:
        x = torch.einsum("nchpwq->nhwcpq", x)
    else:
        x = torch.einsum("nchpwq->nhwpqc", x)
    return x.reshape(n, h * w, patch_size * patch_size * c)


def unpatchify(
    patches: torch.Tensor,
    grid_hw: tuple[int, int],
    patch_size: int,
    channels: int = 3,
    channel_first: bool = False,
) -> torch.Tensor:
    """Inverse of `patchify`: (N, h*w, patch**2 * c) -> (N, c, h*patch, w*patch)."""
    n = patches.shape[0]
    h, w = grid_hw
    if patches.shape[1] != h * w:
        raise ValueError(f"patch count {patches.shape[1]} != h*w={h*w}")
    expected_dim = patch_size * patch_size * channels
    if patches.shape[2] != expected_dim:
        raise ValueError(
            f"patch dim {patches.shape[2]} != patch_size**2 * c = {expected_dim}"
        )
    if channel_first:
        x = patches.reshape(n, h, w, channels, patch_size, patch_size)
        x = torch.einsum("nhwcpq->nchpwq", x)
    else:
        x = patches.reshape(n, h, w, patch_size, patch_size, channels)
        x = torch.einsum("nhwpqc->nchpwq", x)
    return x.reshape(n, channels, h * patch_size, w * patch_size)


def patchify_x0(images: torch.Tensor) -> torch.Tensor:
    """Patchify a target image into the fm_head output space (patch=32, dim=3072).

    `fm_head` returns `(N, h*w, FM_OUTPUT_DIM)` where `FM_OUTPUT_DIM = 3 * 32 * 32 = 3072`.
    The MVP loss `MSE(x_pred, x0)` requires `x0` patches to match this
    layout — so we patchify with patch_size = `PATCH32` (= 32).
    """
    out = patchify(images, patch_size=PATCH32)
    if out.shape[-1] != FM_OUTPUT_DIM:
        raise RuntimeError(
            f"x0 patch dim {out.shape[-1]} != FM_OUTPUT_DIM {FM_OUTPUT_DIM}"
        )
    return out


def patchify_for_vision_model_mot_gen(images: torch.Tensor) -> torch.Tensor:
    """Patchify into the vision_model_mot_gen input space (patch=16).

    The gen-side vision model takes `(N, h_grid * w_grid, 3 * 16 * 16)` flat
    patches and projects them through Conv2d(3->1024, k=16, s=16) followed
    by 2D RoPE and a 2x2 dense merge — see `modeling_neo_vit.py`.
    """
    return patchify(images, patch_size=PATCH_SIZE)


# --------------------------------------------------------------------------- #
# z_t construction helpers                                                    #
# --------------------------------------------------------------------------- #


def linear_z_t(x0_patch: torch.Tensor, eps: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Linear-flow interpolation `z_t = t * x0 + (1 - t) * eps`.

    合理推断 — `_t2i_predict_v` divides `(x_pred - z_t) / (1 - t)` to recover
    velocity, which is the standard linear-flow form. See report §0.1 (5).
    Shapes:
        x0_patch : (B, N, FM_OUTPUT_DIM)
        eps      : same as x0_patch
        t        : (B,) in (t_eps, 1]
    """
    if t.dim() != 1:
        raise ValueError(f"expected 1-D t, got {t.shape}")
    t_b = t.to(x0_patch.dtype).reshape(-1, 1, 1)
    return t_b * x0_patch + (1.0 - t_b) * eps


def predict_v_from_x(
    x_pred: torch.Tensor, z_t: torch.Tensor, t: torch.Tensor, t_eps: float = 1e-3
) -> torch.Tensor:
    """`v_pred = (x_pred - z_t) / max(1 - t, t_eps)`.

    公开证据显示 — exactly `_t2i_predict_v` in modeling_neo_chat.py L562-600.
    """
    one_minus_t = (1.0 - t.to(x_pred.dtype)).clamp_min(t_eps).reshape(-1, 1, 1)
    return (x_pred - z_t) / one_minus_t


def velocity_target(
    x0_patch: torch.Tensor, z_t: torch.Tensor, t: torch.Tensor, t_eps: float = 1e-3
) -> torch.Tensor:
    """Target velocity `v_target = (x0 - z_t) / max(1 - t, t_eps)`."""
    one_minus_t = (1.0 - t.to(x0_patch.dtype)).clamp_min(t_eps).reshape(-1, 1, 1)
    return (x0_patch - z_t) / one_minus_t
