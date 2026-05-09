"""Static cache I/O — `vit_embeds_ref`, `clean_x0`, prompt/package metadata.

Schema follows report §11. Each cached *sample* lives in its own
safetensors file plus a JSON sidecar; a top-level `manifest.jsonl` indexes
all samples and carries the validity hashes.

Design principles (report §11.3):
- safetensors for tensor blobs (safe pickle, fast sequential read)
- JSON sidecars + JSONL manifest for queryable metadata
- model_sha / code_commit / preprocess_version pinned in every record so
  silent mismatches are impossible

This module deliberately makes *no* assumption that the model is loaded;
the production cache job will call `extract_feature(..., gen_model=False)`
itself. Here we only define read/write primitives.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file as st_load_file
from safetensors.torch import save_file as st_save_file

from train_u1.constants import (
    CACHE_VERSION,
    CODE_COMMIT,
    MODEL_ID,
    MODEL_SHA,
    PREPROCESS_VERSION,
)


# --------------------------------------------------------------------------- #
# Schema dataclasses                                                          #
# --------------------------------------------------------------------------- #


@dataclass
class CacheValidity:
    cache_version: str = CACHE_VERSION
    model_id: str = MODEL_ID
    model_sha: str = MODEL_SHA
    tokenizer_sha: str = MODEL_SHA  # tokenizer ships with the same revision
    code_commit: str = CODE_COMMIT
    preprocess_version: str = PREPROCESS_VERSION
    dtype: str = "bfloat16"
    peft_target_signature: str = "none"
    frozen_modules_hash: str | None = None  # callers fill via `hash_state_dict_keys`


@dataclass
class CacheSampleMeta:
    sample_id: str
    prompt_sha256: str
    image_sha256: str | None
    resized_hw: tuple[int, int]
    grid_hw: tuple[int, int]
    token_hw: tuple[int, int]
    image_token_num: int
    augmentation_seed: int = 0
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class CacheRecord:
    """One sample in the cache (one safetensors blob + metadata)."""

    meta: CacheSampleMeta
    tensors_path: str          # relative to manifest dir
    validity: CacheValidity

    def to_jsonl_row(self) -> dict[str, Any]:
        return {
            "meta": asdict(self.meta),
            "tensors_path": self.tensors_path,
            "validity": asdict(self.validity),
        }


# --------------------------------------------------------------------------- #
# Hashing helpers                                                              #
# --------------------------------------------------------------------------- #


def _sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def hash_image_tensor(image_chw: torch.Tensor) -> str:
    """Stable hash over the contiguous bytes of a CHW image tensor.

    Casts to float32 first because numpy lacks bf16 support; the cost is a
    one-time copy and the hash stays deterministic across dtypes that share
    a value range.
    """
    arr = image_chw.detach().to("cpu", torch.float32).contiguous()
    return _sha256_bytes(arr.numpy().tobytes())


def hash_state_dict_keys(state_dict: dict[str, torch.Tensor]) -> str:
    """Hash of (param_name, dtype, shape) tuples — checks structure not values.

    Used as a quick sanity guard: if `frozen_modules_hash` doesn't match
    the cache record's value, the cache is invalid.
    """
    keys = sorted(state_dict.keys())
    payload = "\n".join(
        f"{k}\t{state_dict[k].dtype}\t{tuple(state_dict[k].shape)}" for k in keys
    )
    return _sha256_text(payload)


# --------------------------------------------------------------------------- #
# Read / write primitives                                                     #
# --------------------------------------------------------------------------- #


def write_cache_sample(
    cache_dir: str | os.PathLike,
    record: CacheRecord,
    tensors: dict[str, torch.Tensor],
    *,
    on_duplicate: str = "error",
) -> Path:
    """Write one sample's tensors as safetensors + append a manifest row.

    `on_duplicate` controls collisions on `sample_id`:
      "error"   → raise (default; safest, surfaces silent overwrite bugs)
      "replace" → blob is overwritten; manifest gains a fresh row.
                  `read_cache_sample` returns the *latest* row for that id.
      "append"  → no check; old rows remain readable but readers always see
                  the latest. Use only when you intentionally want history.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if on_duplicate not in {"error", "replace", "append"}:
        raise ValueError(f"on_duplicate must be error/replace/append, got {on_duplicate!r}")

    if on_duplicate == "error":
        existing = {row["meta"]["sample_id"] for row in read_manifest(cache_dir)}
        if record.meta.sample_id in existing:
            raise FileExistsError(
                f"sample_id={record.meta.sample_id!r} already exists in {cache_dir}; "
                "pass on_duplicate='replace' to overwrite or 'append' to keep history."
            )

    blob_path = cache_dir / record.tensors_path
    blob_path.parent.mkdir(parents=True, exist_ok=True)
    # safetensors requires contiguous CPU tensors.
    cpu_tensors = {k: v.detach().to("cpu").contiguous() for k, v in tensors.items()}
    st_save_file(cpu_tensors, str(blob_path))

    manifest_path = cache_dir / "manifest.jsonl"
    with open(manifest_path, "a") as f:
        f.write(json.dumps(record.to_jsonl_row()) + "\n")
    return blob_path


def read_cache_sample(
    cache_dir: str | os.PathLike,
    sample_id: str,
) -> tuple[CacheRecord, dict[str, torch.Tensor]]:
    """Look up a sample by id; raise if not found.

    When the manifest contains multiple rows for the same `sample_id`
    (allowed only by `on_duplicate='replace'/'append'` writes) we return
    the *latest* row — matching the semantics that the most recent write
    wins, mirroring filesystem replace semantics.
    """
    cache_dir = Path(cache_dir)
    manifest = read_manifest(cache_dir)
    last_row = None
    for row in manifest:
        if row["meta"]["sample_id"] == sample_id:
            last_row = row
    if last_row is None:
        raise KeyError(f"sample_id={sample_id!r} not found in {cache_dir}")
    tensors = st_load_file(str(cache_dir / last_row["tensors_path"]))
    record = CacheRecord(
        meta=CacheSampleMeta(**last_row["meta"]),
        tensors_path=last_row["tensors_path"],
        validity=CacheValidity(**last_row["validity"]),
    )
    return record, tensors


def read_manifest(cache_dir: str | os.PathLike) -> list[dict[str, Any]]:
    cache_dir = Path(cache_dir)
    manifest_path = cache_dir / "manifest.jsonl"
    if not manifest_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


# --------------------------------------------------------------------------- #
# Validity checks                                                             #
# --------------------------------------------------------------------------- #


def assert_validity_compatible(
    record_validity: CacheValidity,
    *,
    expected_model_sha: str = MODEL_SHA,
    expected_code_commit: str = CODE_COMMIT,
    expected_preprocess_version: str = PREPROCESS_VERSION,
    expected_frozen_hash: str | None = None,
) -> None:
    """Hard-fail if any validity field drifts. Cheap to call before training step."""
    mismatches = []
    if record_validity.model_sha != expected_model_sha:
        mismatches.append(("model_sha", record_validity.model_sha, expected_model_sha))
    if record_validity.code_commit != expected_code_commit:
        mismatches.append(("code_commit", record_validity.code_commit, expected_code_commit))
    if record_validity.preprocess_version != expected_preprocess_version:
        mismatches.append(
            ("preprocess_version", record_validity.preprocess_version, expected_preprocess_version)
        )
    if expected_frozen_hash is not None and record_validity.frozen_modules_hash != expected_frozen_hash:
        mismatches.append(
            ("frozen_modules_hash", record_validity.frozen_modules_hash, expected_frozen_hash)
        )
    if mismatches:
        details = "\n".join(
            f"  - {k}: cache={cur!r} runtime={want!r}" for k, cur, want in mismatches
        )
        raise RuntimeError(
            "Cache validity mismatch — refusing to use stale cache:\n" + details
        )
