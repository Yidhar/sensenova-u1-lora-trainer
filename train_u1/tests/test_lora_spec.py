"""Tests for the LoRA spec parser, presets, and YAML config loading."""
from __future__ import annotations

from pathlib import Path

import pytest

from train_u1.config import load_train_config
from train_u1.model.lora import (
    ALL_KNOWN_TARGETS,
    ATTN_TARGETS,
    LORA_PRESETS,
    LoRASpec,
    parse_lora_spec_str,
    resolve_preset,
)


def test_compact_form() -> None:
    specs = parse_lora_spec_str("attn=r64a64")
    assert len(specs) == len(ATTN_TARGETS)
    for s in specs:
        assert s.r == 64
        assert s.alpha == 64.0
        assert s.enabled


def test_alpha_default_to_rank() -> None:
    [s] = parse_lora_spec_str("q_proj_mot_gen=r128")
    assert s.r == 128
    assert s.alpha == 128.0


def test_kv_form() -> None:
    [s] = parse_lora_spec_str("q_proj_mot_gen=r=64,a=32")
    assert s.r == 64
    assert s.alpha == 32.0


def test_off_disables() -> None:
    specs = parse_lora_spec_str("attn=r64a64;q_proj_mot_gen=off")
    by_t = {s.target: s for s in specs}
    assert by_t["q_proj_mot_gen"].enabled is False
    assert by_t["k_proj_mot_gen"].enabled is True


def test_group_expansion_all_three() -> None:
    specs = parse_lora_spec_str("all=r64a64")
    targets = {s.target for s in specs}
    assert targets == set(ALL_KNOWN_TARGETS)


def test_unknown_target_rejected() -> None:
    with pytest.raises(ValueError):
        parse_lora_spec_str("not_a_real_module=r64")


def test_preset_default_matches_official_coverage() -> None:
    """Default preset must match the official 8-step LoRA's module coverage."""
    specs = resolve_preset("default")
    targets = {s.target for s in specs}
    # 168 attn + 126 mlp + 2 fm_head ≡ ALL_KNOWN_TARGETS at the per-target level.
    assert targets == set(ALL_KNOWN_TARGETS)
    # All at rank 64 / alpha 64 (our reduction from upstream's r=128).
    for s in specs:
        assert s.r == 64
        assert s.alpha == 64.0


def test_official_r128_preset() -> None:
    specs = resolve_preset("official_r128")
    for s in specs:
        assert s.r == 128
        assert s.alpha == 128.0


def test_yaml_default_config() -> None:
    cfg = load_train_config(Path(__file__).parent.parent.parent / "configs" / "default.yaml")
    assert cfg.lora.preset == "default"
    assert cfg.style.prompt_template == "official"
    specs = cfg.lora.resolved_specs()
    assert len(specs) == 9   # 4 attn + 3 mlp + 2 fm_head


def test_yaml_v16c_config() -> None:
    cfg = load_train_config(Path(__file__).parent.parent.parent / "configs" / "v16c.yaml")
    # v16c uses an explicit spec (attn+mlp only — fm_head is full-FT)
    assert cfg.lora.spec is not None
    specs = cfg.lora.resolved_specs()
    targets = {s.target for s in specs}
    assert "fm_modules.fm_head.0" not in targets
    assert "q_proj_mot_gen" in targets
    assert "mlp_mot_gen.gate_proj" in targets
    # And vision_model + fm_head are in the full-FT regex list
    assert any("vision_model_mot_gen" in p for p in cfg.unfreeze)
    assert any("fm_head" in p for p in cfg.unfreeze)


def test_dropout_propagates_to_specs() -> None:
    from train_u1.config import LoRAConfig

    cfg = LoRAConfig(preset="default", dropout=0.1)
    specs = cfg.resolved_specs()
    for s in specs:
        assert s.dropout == 0.1


def test_lorspec_invalid_rank() -> None:
    with pytest.raises(ValueError):
        LoRASpec(target="q_proj_mot_gen", r=0)


def test_preset_list_includes_default() -> None:
    assert "default" in LORA_PRESETS
    assert "attn_only" in LORA_PRESETS
    assert "attn_mlp" in LORA_PRESETS
    assert "official_r128" in LORA_PRESETS
