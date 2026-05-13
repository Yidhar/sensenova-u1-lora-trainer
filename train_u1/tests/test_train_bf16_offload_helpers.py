"""Pure helper tests for the bf16 offload trainer."""
from __future__ import annotations

import pytest

from train_u1.scripts.train_bf16_offload import (
    _guard_static_prefix_unfreeze,
    _select_prefix_kv,
)


def test_select_prefix_kv_routes_cond_and_uncond() -> None:
    cond0 = object()
    cond1 = object()
    uncond = object()
    prefix_kvs = {"cond": [cond0, cond1], "uncond": uncond}

    assert _select_prefix_kv(prefix_kvs, 1, "cond") is cond1
    assert _select_prefix_kv(prefix_kvs, 0, "uncond") is uncond


def test_select_prefix_kv_requires_uncond_cache() -> None:
    with pytest.raises(RuntimeError, match="missing unconditional prefix KV"):
        _select_prefix_kv({"cond": [object()], "uncond": None}, 0, "uncond")


def test_select_prefix_kv_rejects_unknown_key() -> None:
    with pytest.raises(ValueError, match="unknown prefix_cache_key"):
        _select_prefix_kv({"cond": [object()], "uncond": object()}, 0, "bogus")


def test_static_prefix_guard_rejects_prefix_unfreeze() -> None:
    classify = {
        "prefix": ["language_model.model.layers.0.self_attn.q_proj.weight"],
        "unused": ["vision_model.patch_embed.proj.weight"],
        "gen": ["fm_modules.fm_head.0.weight"],
    }
    with pytest.raises(ValueError, match="static prefix KV cache"):
        _guard_static_prefix_unfreeze([r"language_model\.model\.layers"], classify)


def test_static_prefix_guard_rejects_unused_unfreeze() -> None:
    classify = {
        "prefix": ["language_model.model.layers.0.self_attn.q_proj.weight"],
        "unused": ["vision_model.patch_embed.proj.weight"],
        "gen": ["fm_modules.fm_head.0.weight"],
    }
    with pytest.raises(ValueError, match="static prefix KV cache"):
        _guard_static_prefix_unfreeze([r"^vision_model\."], classify)


def test_static_prefix_guard_allows_gen_unfreeze() -> None:
    classify = {
        "prefix": ["language_model.model.layers.0.self_attn.q_proj.weight"],
        "unused": ["vision_model.patch_embed.proj.weight"],
        "gen": ["fm_modules.fm_head.0.weight"],
    }
    _guard_static_prefix_unfreeze([r"^fm_modules\.fm_head\."], classify)

