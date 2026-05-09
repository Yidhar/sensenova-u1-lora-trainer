"""Online-vs-cached `vit_embeds` equivalence smoke test (report §5.2 / §12.2).

Approach:
1. Pick a synthetic image at a 32-aligned resolution.
2. Patchify (patch=16) per upstream `preprocess_pixel_values`.
3. Run `extract_feature(..., gen_model=False)` ONCE — this is what the
   "cache" would be filled with.
4. Run it AGAIN — same inputs, same model — and compare.
5. Also save the first call's output via `cache_io.write_cache_sample`,
   read it back, and compare to a fresh online extraction.

Pass criteria (BF16 tolerances):
   max_abs_diff < 5e-3
   mean_abs_diff < 1e-4
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile

import torch

from train_u1.constants import MODEL_ID, MODEL_SHA, PATCH_SIZE
from train_u1.data.cache_io import (
    CacheRecord,
    CacheSampleMeta,
    CacheValidity,
    hash_image_tensor,
    read_cache_sample,
    write_cache_sample,
)
from train_u1.model.loader import _resolve_local_snapshot, load_neo_chat_4bit


def _preprocess_pixel_values(pixel_values: torch.Tensor, patch_size: int = PATCH_SIZE):
    """Mirror of upstream `preprocess_pixel_values` (utils.py L94-105)."""
    c, h, w = pixel_values.shape
    grid_h = h // patch_size
    grid_w = w // patch_size
    flat = (
        pixel_values.view(c, grid_h, patch_size, grid_w, patch_size)
        .permute(1, 3, 0, 2, 4)
        .reshape(grid_h * grid_w, c * patch_size**2)
    )
    grid_hw = torch.tensor([[grid_h, grid_w]], device=pixel_values.device)
    return flat, grid_hw


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default=os.environ.get("HF_HOME"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--width", type=int, default=256)
    args = ap.parse_args()

    print("[equiv] loading model (4bit)...", flush=True)
    model = load_neo_chat_4bit(cache_dir=args.cache_dir, device_map=args.device)
    device = next(model.parameters()).device

    # synthetic 3xHxW image in [-1, 1].
    H, W = args.height, args.width
    if H % 32 or W % 32:
        raise SystemExit(f"H,W must be multiples of 32; got ({H}, {W})")
    torch.manual_seed(0)
    img = (torch.rand(3, H, W) * 2.0 - 1.0).to(device, torch.bfloat16)

    flat, grid_hw = _preprocess_pixel_values(img, patch_size=PATCH_SIZE)
    grid_hw = grid_hw.to(device)
    print(f"[equiv] image HxW=({H},{W})  grid_hw={grid_hw.tolist()}  flat={tuple(flat.shape)}")

    # ----- online pass A
    with torch.no_grad():
        online_a = model.extract_feature(flat, gen_model=False, grid_hw=grid_hw)
    # ----- online pass B (deterministic since model is in eval-equivalent mode)
    with torch.no_grad():
        online_b = model.extract_feature(flat, gen_model=False, grid_hw=grid_hw)

    diff_ab = (online_a.float() - online_b.float()).abs()
    print(f"[equiv] online_A vs online_B    max_abs={diff_ab.max():.2e}  mean_abs={diff_ab.mean():.2e}  shape={tuple(online_a.shape)}")

    # ----- cache write / read
    with tempfile.TemporaryDirectory() as td:
        token_h = grid_hw[0, 0].item() // 2
        token_w = grid_hw[0, 1].item() // 2
        rec = CacheRecord(
            meta=CacheSampleMeta(
                sample_id="equiv-test",
                prompt_sha256="-",
                image_sha256=hash_image_tensor(img),
                resized_hw=(H, W),
                grid_hw=tuple(grid_hw[0].tolist()),
                token_hw=(token_h, token_w),
                image_token_num=token_h * token_w,
            ),
            tensors_path="blobs/equiv-test.safetensors",
            validity=CacheValidity(),
        )
        write_cache_sample(td, rec, {"vit_embeds_ref": online_a.detach().to("cpu", torch.bfloat16)})
        rec2, tensors = read_cache_sample(td, "equiv-test")
        cached = tensors["vit_embeds_ref"].to(device, torch.bfloat16)

    diff_ac = (online_a.float() - cached.float()).abs()
    print(f"[equiv] online_A vs cached      max_abs={diff_ac.max():.2e}  mean_abs={diff_ac.mean():.2e}")

    # Pass criteria (loose BF16 tolerances).
    max_thr, mean_thr = 5e-3, 1e-4
    ok = (
        diff_ab.max().item() < max_thr and diff_ab.mean().item() < mean_thr
        and diff_ac.max().item() < max_thr and diff_ac.mean().item() < mean_thr
    )
    if ok:
        print("[equiv] PASS — online deterministic and cache roundtrip equivalent")
        return 0
    print("[equiv] FAIL — diff exceeds threshold")
    return 1


if __name__ == "__main__":
    sys.exit(main())
