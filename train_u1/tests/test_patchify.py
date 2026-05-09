"""Patchify / unpatchify roundtrip tests."""
from __future__ import annotations

import torch

from train_u1.constants import FM_OUTPUT_DIM, PATCH32, PATCH_SIZE
from train_u1.model.patching import (
    linear_z_t,
    patchify,
    patchify_for_vision_model_mot_gen,
    patchify_x0,
    predict_v_from_x,
    unpatchify,
    velocity_target,
)


def test_patchify_unpatchify_roundtrip_patch16():
    img = torch.randn(2, 3, 64, 96)  # 4x6 grid at patch=16
    p = patchify(img, patch_size=16)
    assert p.shape == (2, 4 * 6, 16 * 16 * 3)
    recon = unpatchify(p, grid_hw=(4, 6), patch_size=16)
    assert torch.allclose(recon, img, atol=1e-6), "patch=16 roundtrip lossy"


def test_patchify_unpatchify_roundtrip_patch32():
    img = torch.randn(2, 3, 96, 64)  # 3x2 grid at patch=32
    p = patchify(img, patch_size=32)
    assert p.shape == (2, 3 * 2, FM_OUTPUT_DIM)
    recon = unpatchify(p, grid_hw=(3, 2), patch_size=32)
    assert torch.allclose(recon, img, atol=1e-6), "patch=32 roundtrip lossy"


def test_patchify_x0_shape_matches_fm_head_output():
    img = torch.randn(1, 3, 256, 256)  # token_hw = (8, 8) at patch=32
    p = patchify_x0(img)
    assert p.shape == (1, 64, FM_OUTPUT_DIM)


def test_patchify_for_vision_model_mot_gen_shape():
    img = torch.randn(1, 3, 512, 512)  # grid_hw = (32, 32) at patch=16
    p = patchify_for_vision_model_mot_gen(img)
    assert p.shape == (1, 32 * 32, PATCH_SIZE * PATCH_SIZE * 3)


def test_patchify_rejects_non_divisible_image():
    img = torch.randn(1, 3, 65, 64)
    try:
        patchify(img, patch_size=16)
    except ValueError as e:
        assert "divisible" in str(e)
    else:
        raise AssertionError("expected ValueError for non-divisible H")


def test_z_t_endpoints_match_eps_and_x0():
    x0 = torch.randn(2, 4, FM_OUTPUT_DIM)
    eps = torch.randn(2, 4, FM_OUTPUT_DIM)
    # at t=0 → all noise
    z = linear_z_t(x0, eps, torch.zeros(2))
    assert torch.allclose(z, eps)
    # at t=1 → exactly x0
    z = linear_z_t(x0, eps, torch.ones(2))
    assert torch.allclose(z, x0)


def test_predict_v_consistent_with_velocity_target():
    """If x_pred == x0, then predict_v_from_x must equal velocity_target."""
    x0 = torch.randn(3, 5, FM_OUTPUT_DIM)
    eps = torch.randn(3, 5, FM_OUTPUT_DIM)
    t = torch.tensor([0.1, 0.5, 0.9])
    z_t = linear_z_t(x0, eps, t)
    v_pred_when_perfect = predict_v_from_x(x0, z_t, t)
    v_target = velocity_target(x0, z_t, t)
    assert torch.allclose(v_pred_when_perfect, v_target, atol=1e-5)


def test_predict_v_at_t_close_to_one_is_clamped():
    x0 = torch.randn(1, 2, FM_OUTPUT_DIM)
    eps = torch.randn_like(x0)
    t = torch.tensor([1.0])
    z_t = linear_z_t(x0, eps, t)
    v = predict_v_from_x(x0, z_t, t, t_eps=1e-3)
    # 1 - t = 0 → clamped to 1e-3, so v is finite
    assert torch.isfinite(v).all()
