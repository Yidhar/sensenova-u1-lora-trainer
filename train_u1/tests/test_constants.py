"""Smoke tests for constants — no model load required."""
from __future__ import annotations

from train_u1 import constants as C


def test_param_reconcile_matches_hf_metadata():
    """公开证据显示 + 公式复核: stored params == 17,552,340,992."""
    assert sum(C.PARAM_COUNTS.values()) == C.PARAM_TOTAL == 17_552_340_992


def test_fm_output_dim():
    """公式复核: fm_head 输出 = 3 * (patch * merge) ** 2 = 3072."""
    assert C.FM_OUTPUT_DIM == 3 * (16 * 2) ** 2 == 3072


def test_branch_share():
    """ordinary 与 mot_gen 各占 ~46%."""
    share = C.PARAM_COUNTS["ordinary_llm_core"] / C.PARAM_TOTAL
    assert 0.46 < share < 0.47


def test_pinned_revisions_present():
    assert len(C.MODEL_SHA) == 40
    assert len(C.SFT_MODEL_SHA) == 40
    assert len(C.CODE_COMMIT) == 40
