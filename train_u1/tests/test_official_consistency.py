"""Tests for the SenseNova-U1 official-consistency knobs:
- v-loss / Huber dispatcher
- logit-normal t sampler
- `attn_mlp_no_head` / `attn_only_no_head` LoRA presets
- YAML round-trip of the new TrainConfig fields
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from train_u1.config import (
    TrainRunConfig,
    dump_train_config,
    load_train_config,
)
from train_u1.data.collators import CollatorConfig, SenseNovaU1Collator
from train_u1.model.lora import (
    FM_HEAD_TARGETS,
    LORA_PRESETS,
    resolve_preset,
)
from train_u1.model.losses import (
    VALID_LOSS_TYPES,
    compute_v_target,
    fm_loss,
    fm_loss_v,
    fm_loss_x0,
)
from train_u1.model.patching import linear_z_t


# --------------------------------------------------------------------------- #
# Loss dispatcher                                                             #
# --------------------------------------------------------------------------- #

def _toy_batch(B: int = 1, N: int = 8, D: int = 4, *, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    x0 = torch.randn(B, N, D, generator=g)
    eps = torch.randn(B, N, D, generator=g)
    t = torch.rand(B, generator=g).clamp(min=0.05, max=0.95)
    t_b = t.view(B, 1, 1)
    z_t = linear_z_t(x0, eps, t)
    x_pred = x0 + 0.1 * torch.randn(B, N, D, generator=g)
    v_target = compute_v_target(x0, z_t, t)
    v_pred = (x_pred - z_t) / (1.0 - t_b).clamp(min=1e-3)
    return x0, z_t, t, x_pred, v_pred, v_target


def test_fm_loss_dispatcher_x0() -> None:
    x0, z_t, t, x_pred, v_pred, v_target = _toy_batch()
    l = fm_loss(loss_type="x0", x_pred=x_pred, x0_patch=x0, v_pred=v_pred, v_target=v_target)
    expected = fm_loss_x0(x_pred, x0)
    assert torch.isclose(l, expected)


def test_fm_loss_dispatcher_v() -> None:
    x0, z_t, t, x_pred, v_pred, v_target = _toy_batch()
    l = fm_loss(loss_type="v", x_pred=x_pred, x0_patch=x0, v_pred=v_pred, v_target=v_target)
    expected = fm_loss_v(v_pred, v_target)
    assert torch.isclose(l, expected)


def test_fm_loss_v_equiv_to_reweighted_x0() -> None:
    """`MSE(v) == MSE((x_pred - x0)/(1-t))` — the (1-t)^-2 re-weight identity."""
    x0, z_t, t, x_pred, _v_pred, v_target = _toy_batch(B=1, N=128, D=8, seed=42)
    t_b = t.view(1, 1, 1)
    # Reconstruct v_pred from x_pred and z_t in the exact same way wrapper does:
    v_pred = (x_pred - z_t) / (1.0 - t_b).clamp(min=1e-3)
    lhs = fm_loss_v(v_pred, v_target)
    rhs = ((x_pred - x0) / (1.0 - t_b).clamp(min=1e-3)).pow(2).mean()
    assert torch.isclose(lhs, rhs, atol=1e-6, rtol=1e-5)


def test_v_target_uses_same_t_eps_as_v_pred_near_t_one() -> None:
    """High-t uniform ablations must clamp target and prediction identically."""
    x0 = torch.tensor([[[1.0, -1.0]]])
    z_t = torch.tensor([[[0.5, -0.25]]])
    t = torch.tensor([0.99])
    x_pred = torch.tensor([[[0.75, -0.5]]])
    t_eps = 0.05
    denom = torch.tensor([[[t_eps]]])

    v_target = compute_v_target(x0, z_t, t, t_eps=t_eps)
    v_pred = (x_pred - z_t) / denom

    assert torch.allclose(v_target, (x0 - z_t) / denom)
    assert torch.isfinite(fm_loss_v(v_pred, v_target))


def test_fm_loss_huber_variants() -> None:
    x0, z_t, t, x_pred, v_pred, v_target = _toy_batch()
    lh_x0 = fm_loss(loss_type="x0_huber", x_pred=x_pred, x0_patch=x0,
                    v_pred=v_pred, v_target=v_target, huber_delta=0.5)
    lh_v = fm_loss(loss_type="v_huber", x_pred=x_pred, x0_patch=x0,
                   v_pred=v_pred, v_target=v_target, huber_delta=0.5)
    assert lh_x0.item() >= 0
    assert lh_v.item() >= 0


def test_fm_loss_unknown_raises() -> None:
    with pytest.raises(ValueError, match="unknown loss_type"):
        fm_loss(loss_type="bogus", x_pred=torch.zeros(1, 1, 1),
                x0_patch=torch.zeros(1, 1, 1))


def test_fm_loss_v_requires_v_args() -> None:
    x0 = torch.zeros(1, 1, 1)
    with pytest.raises(ValueError, match="requires v_pred"):
        fm_loss(loss_type="v", x_pred=x0, x0_patch=x0)


def test_compute_v_target_formula() -> None:
    """Spot-check `v* = (x0 - z_t)/(1-t)` is what compute_v_target returns."""
    x0 = torch.tensor([[[1.0, 2.0]]])
    z_t = torch.tensor([[[0.3, 0.5]]])
    t = torch.tensor([0.25])
    v = compute_v_target(x0, z_t, t)
    # (1-0.25) = 0.75
    expected = torch.tensor([[[(1.0 - 0.3) / 0.75, (2.0 - 0.5) / 0.75]]])
    assert torch.allclose(v, expected)


def test_valid_loss_types_constant() -> None:
    assert "x0" in VALID_LOSS_TYPES
    assert "v" in VALID_LOSS_TYPES
    assert "x0_huber" in VALID_LOSS_TYPES
    assert "v_huber" in VALID_LOSS_TYPES


# --------------------------------------------------------------------------- #
# Collator t sampler                                                          #
# --------------------------------------------------------------------------- #

class _FakeTokenizer:
    pad_token_id = 0

    def __call__(self, prompts, return_tensors="pt", padding=True):
        # Return a dummy ids tensor of shape (B, L). Lengths don't matter for
        # _sample_t — the test only inspects t.
        return {"input_ids": torch.zeros(len(prompts), 1, dtype=torch.long)}


def test_t_sampler_uniform_range() -> None:
    cfg = CollatorConfig(t_dist="uniform", seed=123, t_eps=0.01)
    c = SenseNovaU1Collator(_FakeTokenizer(), cfg=cfg)
    t = c._sample_t(2000)
    assert t.min().item() >= cfg.t_eps - 1e-6
    assert t.max().item() <= 1.0 + 1e-6
    # uniform mean ~ (eps + 1) / 2 = 0.505 ; allow a wide tolerance
    assert 0.45 < t.mean().item() < 0.55


def test_t_sampler_logit_normal_stats() -> None:
    cfg = CollatorConfig(
        t_dist="logit_normal", t_logit_mean=-0.8, t_logit_std=0.8, seed=7, t_eps=1e-3,
    )
    c = SenseNovaU1Collator(_FakeTokenizer(), cfg=cfg)
    t = c._sample_t(5000)
    # All in (eps, 1-eps)
    assert t.min().item() >= cfg.t_eps - 1e-6
    assert t.max().item() <= 1.0 - cfg.t_eps + 1e-6
    # Median(sigmoid(N(-0.8, 0.8))) = sigmoid(-0.8) ≈ 0.310 — strongly biased low.
    # With 5000 samples, empirical mean should land near 0.34 (Monte Carlo of
    # E[sigmoid(N(-0.8, 0.8))]); accept anything in [0.28, 0.40].
    assert 0.28 < t.mean().item() < 0.40
    # Median much closer to 0.31 than to 0.5 (uniform mean).
    assert t.median().item() < 0.42


def test_t_sampler_logit_normal_eps_clamp() -> None:
    """Even tiny tail of sigmoid is clipped to [t_eps, 1 - t_eps]."""
    cfg = CollatorConfig(t_dist="logit_normal", t_logit_mean=0.0, t_logit_std=10.0,
                         seed=0, t_eps=0.05)
    c = SenseNovaU1Collator(_FakeTokenizer(), cfg=cfg)
    t = c._sample_t(2000)
    assert t.min().item() >= 0.05 - 1e-6
    assert t.max().item() <= 0.95 + 1e-6


def test_t_sampler_invalid_dist() -> None:
    cfg = CollatorConfig(t_dist="bogus")
    c = SenseNovaU1Collator(_FakeTokenizer(), cfg=cfg)
    with pytest.raises(NotImplementedError):
        c._sample_t(4)


def test_collator_defaults_do_not_enable_condition_dropout() -> None:
    """Diagnostics using CollatorConfig directly stay fully conditional by default."""
    cfg = CollatorConfig()
    assert cfg.cond_dropout_text == 0.0
    assert cfg.cond_dropout_both == 0.0


# --------------------------------------------------------------------------- #
# LoRA presets                                                                #
# --------------------------------------------------------------------------- #

def test_no_head_presets_registered() -> None:
    assert "attn_only_no_head" in LORA_PRESETS
    assert "attn_mlp_no_head" in LORA_PRESETS


def test_no_head_presets_exclude_fm_head() -> None:
    for name in ("attn_only_no_head", "attn_mlp_no_head"):
        specs = resolve_preset(name)
        targets = {s.target for s in specs}
        for fmt in FM_HEAD_TARGETS:
            assert fmt not in targets, f"{name} unexpectedly includes {fmt}"


def test_attn_mlp_no_head_targets() -> None:
    specs = resolve_preset("attn_mlp_no_head")
    targets = {s.target for s in specs}
    assert "q_proj_mot_gen" in targets
    assert "mlp_mot_gen.gate_proj" in targets
    assert all(s.r == 64 and s.alpha == 64.0 for s in specs)


# --------------------------------------------------------------------------- #
# YAML round-trip of new fields                                               #
# --------------------------------------------------------------------------- #

def test_yaml_roundtrip_loss_and_t(tmp_path: Path) -> None:
    cfg = TrainRunConfig()
    cfg.train.loss_type = "v"
    cfg.train.huber_delta = 1.5
    cfg.train.t_dist = "logit_normal"
    cfg.train.t_logit_mean = -0.8
    cfg.train.t_logit_std = 0.8
    cfg.train.cond_dropout_text = 0.2
    cfg.train.cond_dropout_both = 0.05

    path = tmp_path / "rt.yaml"
    dump_train_config(cfg, path)
    cfg2 = load_train_config(path)
    assert cfg2.train.loss_type == "v"
    assert cfg2.train.huber_delta == 1.5
    assert cfg2.train.t_dist == "logit_normal"
    assert math.isclose(cfg2.train.t_logit_mean, -0.8)
    assert math.isclose(cfg2.train.t_logit_std, 0.8)
    assert math.isclose(cfg2.train.cond_dropout_text, 0.2)
    assert math.isclose(cfg2.train.cond_dropout_both, 0.05)


def test_yaml_defaults_use_local_baseline() -> None:
    """A config without explicit FM knobs uses the v16c local baseline."""
    cfg = TrainRunConfig()
    assert cfg.train.loss_type == "x0"
    assert cfg.train.t_dist == "uniform"
    assert math.isclose(cfg.train.t_logit_mean, -0.8)
    assert math.isclose(cfg.train.t_logit_std, 0.8)
    assert math.isclose(cfg.train.cond_dropout_text, 0.0)
    assert math.isclose(cfg.train.cond_dropout_both, 0.0)
    assert cfg.data.use_think_labels is False


def test_default_yaml_uses_local_baseline() -> None:
    """The shipped `configs/default.yaml` uses the v16c local baseline."""
    repo_root = Path(__file__).resolve().parents[2]
    p = repo_root / "configs" / "default.yaml"
    if not p.exists():  # source checkout might not include configs/
        pytest.skip(f"{p} not present")
    cfg = load_train_config(p)
    assert cfg.lora.preset == "attn_mlp_no_head"
    assert cfg.data.use_think_labels is False
    assert cfg.train.loss_type == "x0"
    assert cfg.train.t_dist == "uniform"
    assert math.isclose(cfg.train.cond_dropout_text, 0.0)
    assert math.isclose(cfg.train.cond_dropout_both, 0.0)


def test_official_alignment_yaml_uses_official_knobs() -> None:
    """Official report knobs are still available as an explicit optional config."""
    repo_root = Path(__file__).resolve().parents[2]
    p = repo_root / "configs" / "official_alignment.yaml"
    if not p.exists():
        pytest.skip(f"{p} not present")
    cfg = load_train_config(p)
    assert cfg.data.use_think_labels is True
    assert cfg.train.loss_type == "v"
    assert cfg.train.t_dist == "logit_normal"
    assert math.isclose(cfg.train.t_logit_mean, -0.8)
    assert math.isclose(cfg.train.t_logit_std, 0.8)
    assert math.isclose(cfg.train.cond_dropout_text, 0.10)
    assert math.isclose(cfg.train.cond_dropout_both, 0.10)
