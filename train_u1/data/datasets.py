"""Tiny in-memory T2I datasets for overfit smoke tests.

Three flavors:
- `SyntheticT2ITinyDataset` : pure-noise images, fixed prompts. Lets us
  exercise the forward + loss + optimizer step on CPU/GPU without any
  external data download.
- `FilesystemT2ITinyDataset` : reads `(prompt, image_path)` rows from a
  manifest JSONL. Used once we have a real overfit set with manifest.
- `PairedFolderT2IDataset` : reads `{stem}.jpg` + `{stem}.txt` pairs from
  a single folder. The natural format for the Hayateluc dataset.

All three yield `T2ISample` consumed by `SenseNovaU1Collator`. Since real
images come at varying aspect ratios, the *Filesystem*/*PairedFolder*
variants run upstream `smart_resize` per sample — meaning the collator
must be invoked at `batch_size=1` (or with all samples already at the
same H,W).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset


@dataclass
class T2ISample:
    sample_id: str
    prompt: str
    image: torch.Tensor      # (3, H, W) in [0, 1] or normalized — collator decides
    seed: int = 0


class SyntheticT2ITinyDataset(Dataset):
    """N synthetic samples: deterministic noise images + fixed prompt template.

    Useful only for forward-graph smoke testing. Don't expect anything to
    "learn" — the prompt and image have no semantic correlation.
    """

    def __init__(
        self,
        n: int = 4,
        image_hw: tuple[int, int] = (256, 256),
        prompt_template: str = "a synthetic test image, sample {idx}",
        base_seed: int = 1234,
    ):
        self.n = n
        self.image_hw = image_hw
        self.prompt_template = prompt_template
        self.base_seed = base_seed

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> T2ISample:
        gen = torch.Generator().manual_seed(self.base_seed + idx)
        H, W = self.image_hw
        img = torch.rand(3, H, W, generator=gen) * 2.0 - 1.0  # in [-1, 1]
        return T2ISample(
            sample_id=f"synth-{idx:06d}",
            prompt=self.prompt_template.format(idx=idx),
            image=img,
            seed=self.base_seed + idx,
        )


class FilesystemT2ITinyDataset(Dataset):
    """JSONL manifest with rows: {"sample_id", "prompt", "image_path"}."""

    def __init__(self, manifest_path: str | os.PathLike, image_hw: tuple[int, int] = (512, 512)):
        self.manifest_path = Path(manifest_path)
        self.image_hw = image_hw
        with open(self.manifest_path) as f:
            self.rows = [json.loads(line) for line in f if line.strip()]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> T2ISample:
        from PIL import Image

        row = self.rows[idx]
        img = Image.open(row["image_path"]).convert("RGB")
        # naive resize — production code should use the upstream `smart_resize`
        # to match the training-time grid; we keep this simple and document it.
        img = img.resize(self.image_hw[::-1])  # PIL uses (W, H)
        arr = torch.from_numpy(_pil_to_chw_float(img))
        return T2ISample(
            sample_id=row["sample_id"],
            prompt=row["prompt"],
            image=arr,
            seed=row.get("seed", 0),
        )


def _pil_to_chw_float(img):
    import numpy as np

    arr = np.asarray(img).astype("float32") / 255.0  # (H, W, 3) in [0, 1]
    arr = arr.transpose(2, 0, 1)  # (3, H, W)
    arr = arr * 2.0 - 1.0  # to [-1, 1]
    return arr


# --------------------------------------------------------------------------- #
# PairedFolderT2IDataset                                                      #
# --------------------------------------------------------------------------- #


class PairedFolderT2IDataset(Dataset):
    """Folder of `{stem}.jpg` + `{stem}.txt` pairs.

    Reads both per-sample. Image is run through upstream `smart_resize` at
    construction-equivalent time so the collator sees ready-to-patchify
    tensors at H,W divisible by `PATCH32 (=32)`.

    Args:
        folder: directory of paired files
        cap_max_pixels: optional VRAM-friendly cap (e.g. 512*512). Defaults
            to upstream `max_pixels` (16.78 MP).
        prompt_template: optional `.format(caption=...)` template; if None
            the raw caption text is used (matches upstream `_build_t2i_text_inputs`).
        image_extensions: filename suffixes to search for paired captions.
    """

    def __init__(
        self,
        folder: str | os.PathLike,
        *,
        cap_max_pixels: int | None = None,
        prompt_template: str | None = None,
        image_extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp"),
        snap_bucket: bool = False,
    ):
        self.folder = Path(folder)
        if not self.folder.is_dir():
            raise FileNotFoundError(f"dataset folder not found: {self.folder}")
        self.cap_max_pixels = cap_max_pixels
        self.prompt_template = prompt_template
        self.snap_bucket = snap_bucket

        pairs: list[tuple[Path, Path, str]] = []
        for ext in image_extensions:
            for img in sorted(self.folder.glob(f"*{ext}")):
                txt = img.with_suffix(".txt")
                if txt.is_file():
                    pairs.append((img, txt, img.stem))
        if not pairs:
            raise RuntimeError(
                f"no paired (image, .txt) files found under {self.folder}"
            )
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> T2ISample:
        # Local import keeps PIL/numpy/u1_preprocess off the hot import path.
        from train_u1.data.u1_preprocess import load_and_preprocess_image

        img_path, txt_path, stem = self.pairs[idx]
        with open(txt_path, encoding="utf-8") as f:
            caption = f.read().strip()
        if self.prompt_template:
            caption = self.prompt_template.format(caption=caption)
        chw, _hw = load_and_preprocess_image(
            img_path,
            cap_max_pixels=self.cap_max_pixels,
            normalize="x0",  # 公开证据显示 — fm_head output space, [-1, 1] (NOT ImageNet)
            snap_bucket=self.snap_bucket,
        )
        return T2ISample(sample_id=stem, prompt=caption, image=chw, seed=hash(stem) & 0xFFFF)
