"""Roundtrip + validity tests for cache_io. No model required."""
from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from train_u1.data.cache_io import (
    CacheRecord,
    CacheSampleMeta,
    CacheValidity,
    assert_validity_compatible,
    hash_image_tensor,
    hash_state_dict_keys,
    read_cache_sample,
    read_manifest,
    write_cache_sample,
)


def _make_record(sample_id: str = "sample-001") -> CacheRecord:
    meta = CacheSampleMeta(
        sample_id=sample_id,
        prompt_sha256="aa" * 32,
        image_sha256="bb" * 32,
        resized_hw=(512, 512),
        grid_hw=(32, 32),
        token_hw=(16, 16),
        image_token_num=256,
    )
    validity = CacheValidity(frozen_modules_hash="cc" * 32)
    return CacheRecord(meta=meta, tensors_path=f"blobs/{sample_id}.safetensors", validity=validity)


def test_roundtrip_write_read():
    with tempfile.TemporaryDirectory() as td:
        rec = _make_record()
        tensors = {
            "vit_embeds_ref": torch.randn(256, 4096, dtype=torch.bfloat16),
            "clean_x0": torch.randn(256, 3072, dtype=torch.bfloat16),
        }
        write_cache_sample(td, rec, tensors)
        rec2, t2 = read_cache_sample(td, "sample-001")
        assert rec2.meta.sample_id == rec.meta.sample_id
        assert tuple(rec2.meta.token_hw) == (16, 16)
        assert torch.allclose(t2["clean_x0"].float(), tensors["clean_x0"].float())


def test_manifest_accumulates_rows():
    with tempfile.TemporaryDirectory() as td:
        for i in range(3):
            rec = _make_record(sample_id=f"s-{i}")
            write_cache_sample(td, rec, {"x": torch.zeros(2)})
        rows = read_manifest(td)
        assert len(rows) == 3
        assert {r["meta"]["sample_id"] for r in rows} == {"s-0", "s-1", "s-2"}


def test_assert_validity_passes_when_aligned():
    v = CacheValidity(frozen_modules_hash="hh")
    assert_validity_compatible(v, expected_frozen_hash="hh")


def test_assert_validity_fails_on_drift():
    v = CacheValidity(model_sha="old-sha")
    try:
        assert_validity_compatible(v, expected_model_sha="new-sha")
    except RuntimeError as e:
        assert "model_sha" in str(e)
    else:
        raise AssertionError("expected validity drift to raise")


def test_hash_helpers_stable_across_clones():
    img = torch.randn(3, 32, 32)
    h1 = hash_image_tensor(img)
    h2 = hash_image_tensor(img.clone())
    assert h1 == h2

    sd = {"a": torch.zeros(2, 3), "b": torch.ones(4, dtype=torch.float16)}
    h_a = hash_state_dict_keys(sd)
    h_b = hash_state_dict_keys({"b": torch.ones(4, dtype=torch.float16), "a": torch.zeros(2, 3)})
    assert h_a == h_b, "state-dict hash should be order-independent"


def test_safetensors_blob_path_relative_to_manifest():
    with tempfile.TemporaryDirectory() as td:
        rec = _make_record()
        write_cache_sample(td, rec, {"x": torch.zeros(2)})
        assert (Path(td) / "blobs" / "sample-001.safetensors").exists()
        assert (Path(td) / "manifest.jsonl").exists()


def test_duplicate_sample_id_default_errors():
    with tempfile.TemporaryDirectory() as td:
        rec = _make_record()
        write_cache_sample(td, rec, {"x": torch.zeros(2)})
        try:
            write_cache_sample(td, rec, {"x": torch.ones(2)})
        except FileExistsError as e:
            assert "sample-001" in str(e)
        else:
            raise AssertionError("expected FileExistsError on duplicate sample_id")


def test_duplicate_sample_id_replace_returns_latest():
    with tempfile.TemporaryDirectory() as td:
        rec = _make_record()
        write_cache_sample(td, rec, {"x": torch.zeros(3)})
        write_cache_sample(td, rec, {"x": torch.ones(3)}, on_duplicate="replace")
        _, tensors = read_cache_sample(td, "sample-001")
        # Latest write wins (the ones-tensor), not the original zeros-tensor.
        assert torch.allclose(tensors["x"].float(), torch.ones(3))
