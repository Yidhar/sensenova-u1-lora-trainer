"""Training losses for the FM step.

Two primaries (report Eq. (5) / Table 2):
- `fm_loss_x0(x_pred, x0_patch)` — MSE on clean patches (legacy MVP default).
- `fm_loss_v(v_pred, v_target)`  — MSE on velocity (matches the official
   x-predict + v-loss training objective; equivalent to
   `MSE(x_pred - x0) / (1 - t)^2`, i.e. an x0-MSE re-weighted by `(1-t)^-2`).

Plus Huber variants and an `fm_loss` dispatcher that selects by `loss_type`.

CE guardrail kept for the Phase C unified-training scenario.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def fm_loss_x0(x_pred: torch.Tensor, x0_patch: torch.Tensor) -> torch.Tensor:
    """MSE between predicted clean patches and target clean patches."""
    if x_pred.shape != x0_patch.shape:
        raise ValueError(f"shape mismatch x_pred {x_pred.shape} vs x0 {x0_patch.shape}")
    return F.mse_loss(x_pred.float(), x0_patch.float())


def fm_loss_v(v_pred: torch.Tensor, v_target: torch.Tensor) -> torch.Tensor:
    """MSE between predicted velocity and target velocity."""
    if v_pred.shape != v_target.shape:
        raise ValueError(f"shape mismatch v_pred {v_pred.shape} vs v_target {v_target.shape}")
    return F.mse_loss(v_pred.float(), v_target.float())


def fm_loss_x0_huber(x_pred: torch.Tensor, x0_patch: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    return F.huber_loss(x_pred.float(), x0_patch.float(), delta=delta)


def fm_loss_v_huber(v_pred: torch.Tensor, v_target: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    return F.huber_loss(v_pred.float(), v_target.float(), delta=delta)


def compute_v_target(
    x0_patch: torch.Tensor,
    z_t: torch.Tensor,
    t: torch.Tensor,
    *,
    t_eps: float = 1e-3,
) -> torch.Tensor:
    """Closed-form velocity target for rectified-flow / linear-z_t.

    Report Eq. (5):  `v* = (x0 - z_t) / (1 - t)`  with `z_t = t x0 + (1-t) eps`.
    `t` is expected to be a (B,) tensor — we broadcast to (B, 1, 1) to match
    the (B, N, D) patch tensors.
    """
    if x0_patch.shape != z_t.shape:
        raise ValueError(f"shape mismatch x0 {x0_patch.shape} vs z_t {z_t.shape}")
    t = t.to(x0_patch.dtype)
    while t.dim() < x0_patch.dim():
        t = t.unsqueeze(-1)
    denom = (1.0 - t).clamp(min=t_eps)
    return (x0_patch - z_t) / denom


# --------------------------------------------------------------------------- #
# Dispatcher                                                                  #
# --------------------------------------------------------------------------- #

VALID_LOSS_TYPES = ("x0", "v", "x0_huber", "v_huber")


def fm_loss(
    *,
    loss_type: str,
    x_pred: torch.Tensor,
    x0_patch: torch.Tensor,
    v_pred: torch.Tensor | None = None,
    v_target: torch.Tensor | None = None,
    huber_delta: float = 1.0,
) -> torch.Tensor:
    """Single entry point selecting one of the four FM losses.

    `x0` / `x0_huber` only need `x_pred` + `x0_patch`.
    `v` / `v_huber` require `v_pred` + `v_target` (caller computes them via
    `compute_v_target` from the same `(x0, z_t, t)` used to build the batch).
    """
    if loss_type == "x0":
        return fm_loss_x0(x_pred, x0_patch)
    if loss_type == "x0_huber":
        return fm_loss_x0_huber(x_pred, x0_patch, delta=huber_delta)
    if loss_type == "v":
        if v_pred is None or v_target is None:
            raise ValueError("loss_type='v' requires v_pred and v_target")
        return fm_loss_v(v_pred, v_target)
    if loss_type == "v_huber":
        if v_pred is None or v_target is None:
            raise ValueError("loss_type='v_huber' requires v_pred and v_target")
        return fm_loss_v_huber(v_pred, v_target, delta=huber_delta)
    raise ValueError(f"unknown loss_type {loss_type!r}; valid: {VALID_LOSS_TYPES}")


def text_ce_guardrail(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Standard next-token CE for understanding batches.

    Used only in the long-term unified training scenario (Phase C). Caller
    is responsible for shifting labels and masking pad / image-span
    positions to `ignore_index`.
    """
    return F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        ignore_index=ignore_index,
    )
