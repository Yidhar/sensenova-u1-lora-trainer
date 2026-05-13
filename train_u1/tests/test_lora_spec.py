"""Tests for the LoRA spec parser, presets, and YAML config loading."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch.nn as nn

from train_u1.config import load_train_config
from train_u1.model.lora import (
    ALL_KNOWN_TARGETS,
    ATTN_TARGETS,
    DENSE_KNOWN_TARGETS,
    GEN_MOE_MLP_TARGETS,
    GEN_MOE_ROUTER_TARGETS,
    LORA_PRESETS,
    LoRASpec,
    LoraAdapter,
    apply_lora_specs,
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
    assert targets == set(DENSE_KNOWN_TARGETS)


def test_unknown_target_rejected() -> None:
    with pytest.raises(ValueError):
        parse_lora_spec_str("not_a_real_module=r64")


def test_preset_default_matches_official_coverage() -> None:
    """Default preset must match the official 8-step LoRA's module coverage."""
    specs = resolve_preset("default")
    targets = {s.target for s in specs}
    # 168 attn + 126 mlp + 2 fm_head: stable 8B dense coverage.
    assert targets == set(DENSE_KNOWN_TARGETS)
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
    assert cfg.lora.preset == "attn_mlp_no_head"
    assert cfg.style.prompt_template == "official"
    specs = cfg.lora.resolved_specs()
    assert len(specs) == 7   # 4 attn + 3 mlp; fm_head is full-FT'd separately




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
    assert "a3b_moe_r8" in LORA_PRESETS


class _DummyExpert(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(8, 4, bias=False)
        self.up_proj = nn.Linear(8, 4, bias=False)
        self.down_proj = nn.Linear(4, 8, bias=False)


class _DummyMoEMLP(nn.Module):
    def __init__(self, n_experts: int = 3) -> None:
        super().__init__()
        self.experts = nn.ModuleList([_DummyExpert() for _ in range(n_experts)])
        self.gate = nn.Linear(8, n_experts, bias=False)


class _DummyAttn(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj_mot_gen = nn.Linear(8, 8, bias=False)
        self.k_proj_mot_gen = nn.Linear(8, 2, bias=False)
        self.v_proj_mot_gen = nn.Linear(8, 2, bias=False)
        self.o_proj_mot_gen = nn.Linear(8, 8, bias=False)


class _DummyLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = _DummyAttn()
        self.mlp_mot_gen = _DummyMoEMLP()


class _DummyInnerModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_DummyLayer(), _DummyLayer()])


class _DummyLanguageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _DummyInnerModel()


class _DummyA3BModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.language_model = _DummyLanguageModel()


def test_moe_target_groups_parse() -> None:
    specs = parse_lora_spec_str("gen_moe_mlp=r8a8;gen_moe_router=r4a4")
    targets = {s.target for s in specs}
    assert targets == set(GEN_MOE_MLP_TARGETS + GEN_MOE_ROUTER_TARGETS)
    assert all(s.r == 8 for s in specs if s.target in GEN_MOE_MLP_TARGETS)
    assert all(s.r == 4 for s in specs if s.target in GEN_MOE_ROUTER_TARGETS)


def test_moe_specific_expert_target_parse() -> None:
    [spec] = parse_lora_spec_str("mlp_mot_gen.experts.0.gate_proj=r2a2")
    assert spec.target == "mlp_mot_gen.experts.0.gate_proj"
    assert spec.r == 2
    assert spec.alpha == 2.0


def test_apply_moe_lora_specs_on_dummy_model() -> None:
    model = _DummyA3BModel()
    specs = parse_lora_spec_str(
        "mlp_mot_gen.experts.*.gate_proj=r2a2;"
        "mlp_mot_gen.experts.0.down_proj=r2a2;"
        "gen_moe_router=r2a2"
    )

    report = apply_lora_specs(model, specs)

    # 2 layers × 3 experts for gate_proj, plus 2 layers × expert 0 down_proj,
    # plus 2 layer routers.
    assert report.n_wrapped == 10
    assert isinstance(model.language_model.model.layers[0].mlp_mot_gen.experts[0].gate_proj, LoraAdapter)
    assert isinstance(model.language_model.model.layers[0].mlp_mot_gen.experts[0].down_proj, LoraAdapter)
    assert isinstance(model.language_model.model.layers[0].mlp_mot_gen.gate, LoraAdapter)
