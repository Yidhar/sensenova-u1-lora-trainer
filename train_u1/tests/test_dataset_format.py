"""Tests for the unified caption+think .txt format and arrow dataset."""
from __future__ import annotations

from pathlib import Path

import pytest

from train_u1.data.datasets import THINK_DELIMITER_RE, parse_caption_and_think


def test_parse_plain_caption() -> None:
    c, t = parse_caption_and_think("A simple caption with no marker.")
    assert c == "A simple caption with no marker."
    assert t is None


def test_parse_unified_format() -> None:
    raw = (
        "A painterly Hayateluc-style scene of a wisteria trellis.\n"
        "\n"
        "---think---\n"
        "1. **Instruction Understanding:** The subject is a wisteria trellis...\n"
        "6. **Explicit Prompt:** Render in Hayateluc's painterly anime style...\n"
    )
    c, t = parse_caption_and_think(raw)
    assert c == "A painterly Hayateluc-style scene of a wisteria trellis."
    assert t.startswith("1. **Instruction Understanding")
    assert "6. **Explicit Prompt" in t


def test_parse_marker_case_insensitive() -> None:
    raw = "cap\n--- THINK ---\nthink body"
    c, t = parse_caption_and_think(raw)
    assert c == "cap"
    assert t == "think body"


def test_parse_extra_dashes() -> None:
    raw = "cap\n----think----\nthink body"  # 4-dash variant
    c, t = parse_caption_and_think(raw)
    assert c == "cap"
    assert t == "think body"


def test_parse_no_marker_with_think_word() -> None:
    """The literal word `think` (not on its own line, not as marker) is fine."""
    raw = "A scene where the model needs to think hard about composition."
    c, t = parse_caption_and_think(raw)
    assert c == raw.strip()
    assert t is None


def test_parse_empty_think() -> None:
    raw = "cap\n---think---\n   "
    c, t = parse_caption_and_think(raw)
    assert c == "cap"
    assert t is None  # whitespace-only think → None


def test_paired_folder_dataset_unified_txt(tmp_path: Path) -> None:
    """PairedFolderT2IDataset reads the unified `<id>.txt` format."""
    from PIL import Image

    from train_u1.data.datasets import PairedFolderT2IDataset

    img = Image.new("RGB", (64, 64), color=(128, 128, 128))
    img.save(tmp_path / "test_001.jpg")
    (tmp_path / "test_001.txt").write_text(
        "An illustration by Hayateluc.\n---think---\nthink reasoning here\n"
    )

    ds = PairedFolderT2IDataset(tmp_path)
    assert len(ds) == 1
    s = ds[0]
    assert s.sample_id == "test_001"
    assert s.prompt == "An illustration by Hayateluc."
    assert s.think == "think reasoning here"


def test_paired_folder_dataset_legacy_split(tmp_path: Path) -> None:
    """Legacy split format (`<id>.txt` + `<id>.think.txt`) still works."""
    from PIL import Image

    from train_u1.data.datasets import PairedFolderT2IDataset

    img = Image.new("RGB", (64, 64), color=(0, 0, 0))
    img.save(tmp_path / "split_001.jpg")
    (tmp_path / "split_001.txt").write_text("plain caption only")
    (tmp_path / "split_001.think.txt").write_text("legacy think text")

    ds = PairedFolderT2IDataset(tmp_path)
    s = ds[0]
    assert s.prompt == "plain caption only"
    assert s.think == "legacy think text"


def test_paired_folder_can_ignore_think_labels(tmp_path: Path) -> None:
    """Training defaults can keep prefixes short even if sidecar think files exist."""
    from PIL import Image

    from train_u1.data.datasets import PairedFolderT2IDataset

    img = Image.new("RGB", (64, 64), color=(0, 0, 0))
    img.save(tmp_path / "ignore_001.jpg")
    (tmp_path / "ignore_001.txt").write_text("caption\n---think---\nembedded think")
    (tmp_path / "ignore_001.think.txt").write_text("legacy think")

    ds = PairedFolderT2IDataset(tmp_path, use_think_labels=False)
    s = ds[0]
    assert s.prompt == "caption"
    assert s.think is None


def test_arrow_dataset_roundtrip(tmp_path: Path) -> None:
    """ArrowT2IDataset reads back what `dataset_tools pack-arrow` writes."""
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")

    from PIL import Image

    from train_u1.data.datasets import ArrowT2IDataset
    from train_u1.scripts.dataset_tools import cmd_pack_arrow

    folder = tmp_path / "ds"
    folder.mkdir()
    img = Image.new("RGB", (64, 64), color=(255, 255, 255))
    img.save(folder / "a.jpg")
    (folder / "a.txt").write_text("cap a\n---think---\nthink a")

    out = tmp_path / "out.parquet"
    rc = cmd_pack_arrow(folder, out)
    assert rc == 0
    assert out.exists()

    ds = ArrowT2IDataset(out)
    assert len(ds) == 1
    s = ds[0]
    assert s.sample_id == "a"
    assert s.prompt == "cap a"
    assert s.think == "think a"
    assert s.image.shape[0] == 3  # CHW

    ds_no_think = ArrowT2IDataset(out, use_think_labels=False)
    assert ds_no_think[0].think is None


def test_think_delimiter_regex_compiled() -> None:
    """Sanity: regex matches the canonical marker."""
    assert THINK_DELIMITER_RE.search("cap\n---think---\nthink") is not None
    assert THINK_DELIMITER_RE.search("cap\n---tHiNk---\nthink") is not None
    assert THINK_DELIMITER_RE.search("cap with think word") is None
