"""One-step denoise visualization (experiment A).

For each sample × fixed t × fixed eps, build z_t = t·x0 + (1-t)·eps, run the
wrapper, fetch x_pred (in fm-head output space, i.e. patch=32 RGB patches),
and save a 4-panel PNG: [x0 | z_t | x_pred | |x_pred - x0|].

**Convention**: this codebase uses the upstream linear-flow form
`z_t = t * x0 + (1 - t) * eps` (see `train_u1/model/patching.py`). So:
- `t = 0` → pure noise, hard case (the inference-time start)
- `t = 1` → clean image, trivial case
- t=0.30 is therefore *high-noise* (hard); t=0.70 is *low-noise* (easy).

Used to disambiguate "is the loss drop real learning or a trivial-mean
shortcut?":
- If x_pred at high noise (small t, e.g. t=0.30) still recovers structure
  matching the prompt/x0 → real learning under harder conditions.
- If x_pred is always a global blur regardless of prompt/sample → shortcut.

Caveat: this is **one-step denoise on a noised version of the true x0**.
It is *not* a full random-noise → image sampling test. A model that
overfits one-step x0 reconstruction can still fail completely at
multi-step Euler sampling from random noise (experiment D).

The script is callable both as a CLI on a freshly-loaded model AND as a
library function from train_fm_mvp.py (`run_eval_panel`).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from train_u1.constants import FM_OUTPUT_DIM, PATCH32
from train_u1.data.collators import CollatorConfig, SenseNovaU1Collator, to_device
from train_u1.data.datasets import T2ISample
from train_u1.data.u1_preprocess import X0_MEAN, X0_STD
from train_u1.model.patching import linear_z_t, unpatchify
from train_u1.model.wrapper import TrainingWrapper


# --------------------------------------------------------------------------- #
# Image (de)normalization                                                     #
# --------------------------------------------------------------------------- #


def denorm_x0_to_uint8(chw: torch.Tensor) -> np.ndarray:
    """Reverse the (0.5, 0.5, 0.5)/(0.5, 0.5, 0.5) normalization (i.e.
    [-1, 1] → [0, 1]) that the FM x0 target / fm_head output space lives in.

    Matches `examples/t2i/inference.py::_denorm` exactly.
    """
    mean = torch.tensor(X0_MEAN, device=chw.device, dtype=chw.dtype).view(3, 1, 1)
    std = torch.tensor(X0_STD, device=chw.device, dtype=chw.dtype).view(3, 1, 1)
    rgb = chw * std + mean
    rgb = rgb.clamp(0, 1)
    arr = (rgb.detach().cpu().float().numpy() * 255.0).astype(np.uint8)
    return arr.transpose(1, 2, 0)  # HWC


# Back-compat alias (callers expect this name).
denorm_imagenet_to_uint8 = denorm_x0_to_uint8


def patches_to_image_uint8(
    patches: torch.Tensor, token_h: int, token_w: int
) -> np.ndarray:
    """`(N, 3072)` fm-head patches → uint8 HWC after ImageNet de-normalization.

    fm_head outputs RGB-patch tensors in the same normalized space as the
    input image (since x0 = patchify(image_imagenet_normalized)).
    """
    if patches.dim() == 3:
        patches = patches.squeeze(0)
    chw = unpatchify(
        patches.unsqueeze(0).float(),
        grid_hw=(token_h, token_w),
        patch_size=PATCH32,
    )[0]
    return denorm_imagenet_to_uint8(chw)


# --------------------------------------------------------------------------- #
# Eval                                                                         #
# --------------------------------------------------------------------------- #


@dataclass
class EvalRow:
    sample_id: str
    t: float
    token_h: int
    token_w: int
    mse_x_pred_x0: float
    mse_z_t_x0: float
    image_paths: dict[str, str]


def run_eval_panel(
    *,
    wrapper: TrainingWrapper,
    collator: SenseNovaU1Collator,
    samples: list[T2ISample],
    t_values: tuple[float, ...] = (0.3, 0.5, 0.7),
    out_dir: str | os.PathLike,
    label: str = "after",
    device: str | torch.device = "cuda",
    eps_seed: int = 7,
) -> list[EvalRow]:
    """Render and save one-step denoise panels.

    Saves to `{out_dir}/{label}/{sample_id}_t{t}.png`. Returns a row per
    (sample, t) with quantitative MSEs.
    """
    from PIL import Image

    out_root = Path(out_dir) / label
    out_root.mkdir(parents=True, exist_ok=True)

    # Pre-sample fixed eps in patch-32 space, one per sample, reused across t.
    rng = torch.Generator().manual_seed(eps_seed)
    rows: list[EvalRow] = []

    for sample in samples:
        # Force the collator into "raw" mode so we can override t and eps.
        # Build a single-sample batch via the collator (gets text/indexes/mask),
        # then overwrite x_pred-relevant tensors with our fixed t/eps.
        batch = collator([sample])
        batch = to_device(batch, device, dtype=torch.bfloat16)
        token_h, token_w = batch["token_hw"]
        N = token_h * token_w

        x0 = batch["x0_patch"]  # (1, N, 3072) bf16
        for t_val in t_values:
            t = torch.tensor([float(t_val)], device=x0.device, dtype=x0.dtype)
            eps = torch.randn(1, N, FM_OUTPUT_DIM, generator=rng).to(x0.device, x0.dtype)
            z_t = linear_z_t(x0, eps, t)

            # Re-render noisy_pixel_values from this z_t so vision_model_mot_gen
            # sees the matching pixels (the collator's z_t was sampled with a
            # different t/eps; we override to the fixed eval values).
            from train_u1.model.patching import unpatchify

            noisy_pix = unpatchify(z_t, grid_hw=(token_h, token_w), patch_size=PATCH32).to(
                x0.dtype
            )

            # Build a trimmed copy of the batch with overridden tensors.
            eval_batch = dict(batch)
            eval_batch["x0_patch"] = x0
            eval_batch["eps"] = eps
            eval_batch["t"] = t
            eval_batch["noisy_pixel_values"] = noisy_pix

            with torch.no_grad():
                out = wrapper.forward_t2i_step(eval_batch)

            # Quantitative
            mse_xp = torch.nn.functional.mse_loss(out.x_pred.float(), x0.float()).item()
            mse_zt = torch.nn.functional.mse_loss(z_t.float(), x0.float()).item()

            # Render 4-panel: x0 | z_t | x_pred | abs(diff)
            x0_img = patches_to_image_uint8(x0, int(token_h), int(token_w))
            zt_img = patches_to_image_uint8(z_t, int(token_h), int(token_w))
            xp_img = patches_to_image_uint8(out.x_pred, int(token_h), int(token_w))
            # diff is (x_pred - x0) in normalized space; visualize abs scaled to 0-255
            diff = (out.x_pred - x0).float()
            diff_chw = unpatchify(diff, grid_hw=(int(token_h), int(token_w)), patch_size=PATCH32)[0]
            diff_abs = diff_chw.abs().mean(dim=0)  # collapse channels
            d = diff_abs.detach().cpu().numpy()
            d = (d / max(d.max(), 1e-6) * 255.0).astype(np.uint8)
            diff_img = np.stack([d, d, d], axis=-1)

            panel = np.concatenate([x0_img, zt_img, xp_img, diff_img], axis=1)
            out_path = out_root / f"{sample.sample_id}_t{t_val:.2f}.png"
            Image.fromarray(panel).save(out_path)

            rows.append(
                EvalRow(
                    sample_id=sample.sample_id,
                    t=float(t_val),
                    token_h=int(token_h),
                    token_w=int(token_w),
                    mse_x_pred_x0=mse_xp,
                    mse_z_t_x0=mse_zt,
                    image_paths={"panel": str(out_path)},
                )
            )
    # JSONL log per `label`
    with open(out_root / "rows.jsonl", "w") as f:
        for r in rows:
            f.write(
                json.dumps(
                    {
                        "sample_id": r.sample_id,
                        "t": r.t,
                        "token_h": r.token_h,
                        "token_w": r.token_w,
                        "mse_x_pred_x0": r.mse_x_pred_x0,
                        "mse_z_t_x0": r.mse_z_t_x0,
                        "panel": r.image_paths["panel"],
                    }
                )
                + "\n"
            )
    return rows


def summarize(rows_before: list[EvalRow], rows_after: list[EvalRow]) -> str:
    """Compact text table comparing per-(sample, t) MSE before → after.

    Convention: small t = high noise (hard); large t = low noise (easy).
    The `mse_zt` column is the noisy-baseline MSE: a model that returns
    `z_t` itself (identity) would score this; `after` < `mse_zt` means the
    trained model beats identity.
    """
    by_id_t_before = {(r.sample_id, r.t): r for r in rows_before}
    by_id_t_after = {(r.sample_id, r.t): r for r in rows_after}
    keys = sorted(by_id_t_before.keys() & by_id_t_after.keys())

    lines = [
        f"{'sample':30s} {'t':>5s} {'mse_zt':>9s} {'before':>9s} {'after':>9s} "
        f"{'Δ':>9s} {'rel':>9s} {'beats_zt':>9s}"
    ]
    deltas: list[float] = []
    beats_zt = 0
    for k in keys:
        b = by_id_t_before[k]
        a = by_id_t_after[k]
        d = a.mse_x_pred_x0 - b.mse_x_pred_x0
        rel = d / max(b.mse_x_pred_x0, 1e-9)
        deltas.append(rel)
        bz = a.mse_x_pred_x0 < b.mse_z_t_x0
        beats_zt += int(bz)
        lines.append(
            f"{b.sample_id:30s} {b.t:>5.2f} {b.mse_z_t_x0:>9.4f} "
            f"{b.mse_x_pred_x0:>9.4f} {a.mse_x_pred_x0:>9.4f} {d:>+9.4f} "
            f"{rel:>+9.2%} {('Y' if bz else 'N'):>9s}"
        )
    if deltas:
        lines.append("")
        lines.append(f"mean relative Δ across (sample,t): {sum(deltas)/len(deltas):+.2%}")
        lines.append(f"  min: {min(deltas):+.2%}  max: {max(deltas):+.2%}")
        lines.append(f"after beats noisy-baseline (mse_zt) on {beats_zt}/{len(keys)} (sample,t)")
        # Aggregate beats_zt rate per t value to expose the high-noise regime.
        per_t: dict[float, list[bool]] = {}
        for k in keys:
            a = by_id_t_after[k]; b = by_id_t_before[k]
            per_t.setdefault(b.t, []).append(a.mse_x_pred_x0 < b.mse_z_t_x0)
        lines.append("\nbeats_zt rate by t (small t = high noise = hard):")
        for t_val, flags in sorted(per_t.items()):
            lines.append(f"  t={t_val:.2f}  {sum(flags)}/{len(flags)}")
    return "\n".join(lines)
