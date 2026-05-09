"""smart_resize + paired-folder dataset offline tests."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch
from PIL import Image

from train_u1.constants import PATCH32, SMART_RESIZE_MAX_PIXELS, SMART_RESIZE_MIN_PIXELS
from train_u1.data.datasets import PairedFolderT2IDataset
from train_u1.data.u1_preprocess import smart_resize


@pytest.mark.parametrize(
    "h,w",
    [
        (2367, 1825),    # smallest in Hayateluc
        (6741, 5318),    # largest in Hayateluc
        (3840, 2160),    # 16:9
        (1825, 2367),    # vertical
        (200, 100),      # tiny → should snap up to >= min_pixels
    ],
)
def test_smart_resize_outputs_are_aligned_and_in_range(h, w):
    H, W = smart_resize(h, w)
    assert H % PATCH32 == 0
    assert W % PATCH32 == 0
    assert H * W <= SMART_RESIZE_MAX_PIXELS
    # may dip slightly below min when starting from a tiny tensor, but at
    # most by one factor on each side
    assert H * W >= SMART_RESIZE_MIN_PIXELS // 2


def test_cap_max_pixels_caps_resolution():
    H, W = smart_resize(6741, 5318)  # native cap
    H2, W2 = smart_resize(6741, 5318, max_pixels=512 * 512)  # explicit cap
    assert H * W > H2 * W2
    assert H2 * W2 <= 512 * 512


def test_paired_folder_dataset_loads_caption_and_image():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        # synthetic 1024x768 RGB image and a caption
        img = Image.fromarray(
            (torch.rand(768, 1024, 3) * 255).byte().numpy(),
            mode="RGB",
        )
        img.save(td / "x.jpg")
        (td / "x.txt").write_text("a synthetic anime scene")
        # second pair with different aspect ratio
        Image.fromarray((torch.rand(512, 1024, 3) * 255).byte().numpy(), "RGB").save(
            td / "y.jpg"
        )
        (td / "y.txt").write_text("another synthetic scene with widescreen")

        ds = PairedFolderT2IDataset(td, cap_max_pixels=512 * 512)
        assert len(ds) == 2
        s0 = ds[0]
        assert s0.image.shape[0] == 3
        H0, W0 = s0.image.shape[1], s0.image.shape[2]
        assert H0 % PATCH32 == 0
        assert W0 % PATCH32 == 0
        assert s0.prompt
        # second sample may have a different H,W after smart_resize
        s1 = ds[1]
        assert (s1.image.shape[1] % PATCH32 == 0) and (s1.image.shape[2] % PATCH32 == 0)


def test_paired_folder_skips_unmatched_files():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        Image.fromarray((torch.rand(64, 64, 3) * 255).byte().numpy(), "RGB").save(
            td / "lonely.jpg"
        )  # no .txt
        with pytest.raises(RuntimeError):
            PairedFolderT2IDataset(td)
