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
    # Optional pre-computed `<think>...</think>` reasoning text. When set, the
    # collator embeds it INSIDE the empty think block of the official prompt
    # template, so training distribution matches inference-time `--think-mode`.
    # When None, the empty `<think>\n\n</think>` block is preserved (matches
    # inference-time without `--think-mode`).
    think: str | None = None


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
            raw = f.read()
        caption, think_text = parse_caption_and_think(raw)
        if self.prompt_template:
            caption = self.prompt_template.format(caption=caption)
        # Legacy fallback: `<id>.think.txt` separate sidecar (deprecated;
        # `parse_caption_and_think` is the preferred path).
        if think_text is None:
            think_path = img_path.with_suffix(".think.txt")
            if think_path.is_file():
                with open(think_path, encoding="utf-8") as f:
                    think_text = f.read().strip() or None
        chw, _hw = load_and_preprocess_image(
            img_path,
            cap_max_pixels=self.cap_max_pixels,
            normalize="x0",  # 公开证据显示 — fm_head output space, [-1, 1] (NOT ImageNet)
            snap_bucket=self.snap_bucket,
        )
        return T2ISample(
            sample_id=stem, prompt=caption, image=chw,
            seed=hash(stem) & 0xFFFF, think=think_text,
        )


# Marker used inside a single `.txt` to separate caption from think label.
# Any whitespace tolerated around the marker; case-insensitive on the
# `THINK` keyword. Pick a marker unlikely to appear in natural prose.
THINK_DELIMITER_RE = __import__("re").compile(r"^\s*-{3,}\s*think\s*-{3,}\s*$", flags=__import__("re").IGNORECASE | __import__("re").MULTILINE)


def parse_caption_and_think(raw: str) -> tuple[str, str | None]:
    """Split a caption file into `(caption, think)` parts.

    The new compact format puts both labels in a single `.txt`::

        A natural-language caption embedding the artist style.
        ---think---
        1. **Instruction Understanding:** ...
        ...
        6. **Explicit Prompt:** ...

    Falls back to "whole file is caption, no think" for plain captions.
    Returns `(caption, None)` if no marker is present, else `(caption, think)`
    with both stripped.
    """
    m = THINK_DELIMITER_RE.search(raw)
    if m is None:
        return raw.strip(), None
    caption = raw[: m.start()].strip()
    think = raw[m.end():].strip()
    return caption, (think or None)


# --------------------------------------------------------------------------- #
# ArrowT2IDataset (large-scale)                                               #
# --------------------------------------------------------------------------- #


class ArrowT2IDataset(Dataset):
    """T2I dataset backed by a parquet/arrow shard.

    Schema expected (rows aligned with `T2ISample` fields):
        - `sample_id` : string
        - `caption`   : string
        - `think`     : string (optional; nullable column)
        - `image`     : binary (raw image bytes, e.g. PNG/JPEG) — preferred
                         OR `image_path` : string (path resolved relative to
                         `image_root` if relative, else absolute)

    Reads via `pyarrow` table-of-shards mmap, so memory-efficient even at
    millions of rows. Suitable for the 1M-image scaling experiment (task #53).

    Args:
        path: parquet file or directory of parquet shards
        image_root: base dir to resolve relative `image_path` columns
        cap_max_pixels: optional VRAM-friendly cap (passed to smart_resize)
        prompt_template: optional template (matches PairedFolderT2IDataset)
        snap_bucket: snap to nearest official bucket
    """

    def __init__(
        self,
        path: str | os.PathLike,
        *,
        image_root: str | os.PathLike | None = None,
        cap_max_pixels: int | None = None,
        prompt_template: str | None = None,
        snap_bucket: bool = False,
    ):
        try:
            import pyarrow.parquet as pq  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "ArrowT2IDataset requires pyarrow. Install with `pip install pyarrow>=14`."
            ) from e
        self.path = Path(path)
        self.image_root = Path(image_root) if image_root else None
        self.cap_max_pixels = cap_max_pixels
        self.prompt_template = prompt_template
        self.snap_bucket = snap_bucket
        self._table = None  # lazy-loaded
        self._n: int | None = None

    def _ensure_table(self):
        if self._table is not None:
            return
        import pyarrow.parquet as pq
        if self.path.is_dir():
            # Directory of shards — read combined.
            self._table = pq.read_table(str(self.path))
        else:
            self._table = pq.read_table(str(self.path))
        cols = self._table.column_names
        # Schema sanity
        if "sample_id" not in cols or "caption" not in cols:
            raise RuntimeError(
                f"{self.path}: required columns missing. "
                f"Schema must include `sample_id` and `caption`. Got: {cols}"
            )
        if "image" not in cols and "image_path" not in cols:
            raise RuntimeError(
                f"{self.path}: must have either `image` (bytes) or `image_path` (string). Got: {cols}"
            )
        self._n = self._table.num_rows

    def __len__(self) -> int:
        self._ensure_table()
        return self._n  # type: ignore[return-value]

    def __getitem__(self, idx: int) -> T2ISample:
        from io import BytesIO

        from train_u1.data.u1_preprocess import (
            load_and_preprocess_image,
            preprocess_pil_image,
        )

        self._ensure_table()
        row = self._table.slice(idx, 1).to_pydict()
        sample_id = row["sample_id"][0]
        caption = row["caption"][0]
        think = (row.get("think") or [None])[0] or None

        if "image" in self._table.column_names and row["image"][0] is not None:
            from PIL import Image
            pil = Image.open(BytesIO(row["image"][0])).convert("RGB")
            chw, _hw = preprocess_pil_image(
                pil, cap_max_pixels=self.cap_max_pixels,
                normalize="x0", snap_bucket=self.snap_bucket,
            )
        else:
            img_path_str = row["image_path"][0]
            img_path = Path(img_path_str)
            if not img_path.is_absolute() and self.image_root is not None:
                img_path = self.image_root / img_path
            chw, _hw = load_and_preprocess_image(
                img_path, cap_max_pixels=self.cap_max_pixels,
                normalize="x0", snap_bucket=self.snap_bucket,
            )

        if self.prompt_template:
            caption = self.prompt_template.format(caption=caption)
        return T2ISample(
            sample_id=str(sample_id), prompt=str(caption), image=chw,
            seed=hash(sample_id) & 0xFFFF, think=think,
        )
