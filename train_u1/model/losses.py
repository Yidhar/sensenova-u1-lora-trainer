"""Training losses for the FM step.

Two primaries (report §5 / §2.1):
- `fm_loss_x0(x_pred, x0_patch)` — MVP recommended. MSE on clean patches.
- `fm_loss_v(v_pred, v_target)`  — velocity-target ablation.

Plus optional Huber variants for outlier robustness, and a tiny CE
guardrail for the unified-training scenario (Phase C).
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
