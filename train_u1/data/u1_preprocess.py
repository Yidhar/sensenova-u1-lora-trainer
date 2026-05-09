"""Upstream-faithful image preprocessing for U1 training data.

Mirrors `utils.py` (commit `df86ca9`):
- `smart_resize` rounds H/W to multiples of `factor`, total pixels ∈ [min, max]
- We always force resized H/W divisible by `PATCH32` (= 32) so the fm_head
  output grid (`token_h × token_w = H/32 × W/32`) lands on integer counts.

**Normalization convention** (公开证据显示 — examples/t2i/inference.py L17-18,
L52-54):
- `vision_model` (UNDERSTANDING path, ordinary `extract_feature`) consumes
  ImageNet-mean/std normalized inputs.
- `fm_head` OUTPUT space (the x0 target for FM training) is **(0.5, 0.5, 0.5)
  / (0.5, 0.5, 0.5)** — i.e. [-1, 1] symmetric — because the official
  inference's `_to_pil` reverses with `(x * 0.5 + 0.5).clamp(0, 1)`. This
  is the space `t2i_generate.image_prediction` lives in throughout the
  Euler loop. **Training x0 must use this same space** — using ImageNet
  normalize was a previous bug that pushed fm_head outputs into a clipped
  / cyan-shifted regime under Euler's `(x*0.5+0.5).clamp(0,1)` interpretation.

Helper: `load_and_preprocess_image(..., normalize="x0")` for FM training,
`normalize="vision"` for the understanding path.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
from PIL import Image

from train_u1.constants import (
    OFFICIAL_BUCKETS_HW,
    PATCH32,
    SMART_RESIZE_MAX_PIXELS,
    SMART_RESIZE_MIN_PIXELS,
)

# ImageNet (used by ordinary vision_model, NOT for x0 / fm_head training)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# fm_head output / x0 target space — symmetric [-1, 1] (公开证据显示:
# examples/t2i/inference.py NORM_MEAN/NORM_STD = (0.5, 0.5, 0.5)).
X0_MEAN = (0.5, 0.5, 0.5)
X0_STD = (0.5, 0.5, 0.5)


def _round_by_factor(n: float, factor: int) -> int:
    return round(n / factor) * factor


def _ceil_by_factor(n: float, factor: int) -> int:
    return math.ceil(n / factor) * factor


def _floor_by_factor(n: float, factor: int) -> int:
    return math.floor(n / factor) * factor


def smart_resize(
    height: int,
    width: int,
    *,
    factor: int = PATCH32,
    min_pixels: int = SMART_RESIZE_MIN_PIXELS,
    max_pixels: int = SMART_RESIZE_MAX_PIXELS,
) -> tuple[int, int]:
    """Bit-faithful copy of upstream `smart_resize`."""
    if max(height, width) / max(min(height, width), 1) > 200:
        raise ValueError(
            f"absolute aspect ratio must be < 200; got "
            f"{max(height, width) / max(min(height, width), 1)}"
        )
    h_bar = max(factor, _round_by_factor(height, factor))
    w_bar = max(factor, _round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, _floor_by_factor(height / beta, factor))
        w_bar = max(factor, _floor_by_factor(width / beta, factor))
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = _ceil_by_factor(height * beta, factor)
        w_bar = _ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def snap_to_official_bucket(H: int, W: int) -> tuple[int, int]:
    """Pick the official bucket whose aspect ratio is closest to H/W.

    Officially supported buckets are ~2K pixel ratios at 32-aligned shapes
    (公开证据显示: examples/README.md "Supported resolution buckets").
    Any image not landing on these shapes is OOD and may degrade quality —
    snapping ensures all training samples share the same shape distribution
    as inference.
    """
    src_ratio = math.log(W / max(H, 1))  # log-ratio so 16:9 ↔ 9:16 are equidistant
    best = min(
        OFFICIAL_BUCKETS_HW,
        key=lambda hw: abs(math.log(hw[1] / hw[0]) - src_ratio),
    )
    return best


def preprocess_pil_image(
    img: Image.Image,
    *,
    factor: int = PATCH32,
    min_pixels: int = SMART_RESIZE_MIN_PIXELS,
    max_pixels: int = SMART_RESIZE_MAX_PIXELS,
    cap_max_pixels: int | None = None,
    normalize: str = "x0",
    snap_bucket: bool = False,
) -> tuple[torch.Tensor, tuple[int, int]]:
    """In-memory variant of `load_and_preprocess_image` taking a PIL.Image.

    Use this for arrow/parquet datasets that decode image bytes inline.
    Returns `(image_chw, (H, W))` with the same conventions.
    """
    img = img.convert("RGB")
    W, H = img.size
    if snap_bucket:
        H_bar, W_bar = snap_to_official_bucket(H, W)
    else:
        eff_max = min(max_pixels, cap_max_pixels) if cap_max_pixels is not None else max_pixels
        H_bar, W_bar = smart_resize(H, W, factor=factor, min_pixels=min_pixels, max_pixels=eff_max)
    img = img.resize((W_bar, H_bar))

    import numpy as np

    arr = np.asarray(img).astype("float32") / 255.0
    if normalize == "x0":
        mean = np.array(X0_MEAN, dtype="float32")
        std = np.array(X0_STD, dtype="float32")
        arr = (arr - mean) / std
    elif normalize == "vision":
        mean = np.array(IMAGENET_MEAN, dtype="float32")
        std = np.array(IMAGENET_STD, dtype="float32")
        arr = (arr - mean) / std
    elif normalize == "none":
        pass
    else:
        raise ValueError(f"normalize must be 'x0' / 'vision' / 'none', got {normalize!r}")

    chw = torch.from_numpy(arr.transpose(2, 0, 1))
    return chw, (H_bar, W_bar)


def load_and_preprocess_image(
    path: str | Path,
    *,
    factor: int = PATCH32,
    min_pixels: int = SMART_RESIZE_MIN_PIXELS,
    max_pixels: int = SMART_RESIZE_MAX_PIXELS,
    cap_max_pixels: int | None = None,
    normalize: str = "x0",
    snap_bucket: bool = False,
) -> tuple[torch.Tensor, tuple[int, int]]:
    """Read RGB image, smart-resize, return `(image_chw, (H, W))`.

    `normalize`:
    - `"x0"`     → mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5) → output in [-1, 1].
                   Use this for FM training x0 targets (matches fm_head output
                   space + `t2i_generate.image_prediction` Euler state).
    - `"vision"` → ImageNet mean/std. Use this for ordinary `vision_model`
                   inputs (understanding path).
    - `"none"`   → raw [0, 1] (debugging only).

    `cap_max_pixels` further clamps the upstream max (e.g. 4194304 for the
    2048² training bucket).

    `snap_bucket`: if True, pick the closest aspect-ratio match in
    `OFFICIAL_BUCKETS_HW` (overrides smart_resize/min/max/cap). Use this for
    training when you want every sample to land on a known-supported bucket
    shape, eliminating the "trained on arbitrary shape, sampled at 2048²"
    mismatch.
    """
    img = Image.open(path)
    return preprocess_pil_image(
        img, factor=factor, min_pixels=min_pixels, max_pixels=max_pixels,
        cap_max_pixels=cap_max_pixels, normalize=normalize, snap_bucket=snap_bucket,
    )
